use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Result as CandleResult, Shape, Tensor};
use candle_nn::Module;

use crate::weights::VarBuilder;

thread_local! {
    /// When set, [`quantized_linear`] dequantizes every weight to dense F32
    /// even if the WGPU F16 override (`LUX3D_WGPU_DEQUANT_WEIGHTS=f16`) is
    /// active. Used by the TripoSG VAE loader: its 512-token decoder produces
    /// tensors of exactly 1 MiB, a size class the candle WGPU hot-ring recycles
    /// while still bound as a copy source, which trips
    /// `Source and destination cannot be the same buffer`. F32 moves those
    /// tensors out of the 1 MiB class and also matches the reference decoder
    /// precision. Scoped per thread and restored on unwind.
    static FORCE_DENSE_F32: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` with dense-F32 dequantization forced for [`Linear`] construction.
pub(crate) fn with_force_dense_f32<T>(f: impl FnOnce() -> T) -> T {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            FORCE_DENSE_F32.with(|cell| cell.set(false));
        }
    }
    FORCE_DENSE_F32.with(|cell| cell.set(true));
    let _reset = Reset;
    f()
}

fn force_dense_f32() -> bool {
    FORCE_DENSE_F32.with(std::cell::Cell::get)
}

/// Fetch a tensor of the expected shape from a safetensors or GGUF backend.
///
/// Safetensors packages store every tensor at its original rank. Candle's
/// GGUF reader caps tensor rank at four, so the quantizer stores rank>4
/// tensors (3D convolution kernels) as flat vectors; this helper reshapes
/// the dequantized vector back to the expected shape. The regular
/// [`VarBuilder::get`] path is tried first, so safetensors loading is
/// unchanged.
pub(crate) fn get_weight_shaped(
    vb: &VarBuilder,
    shape: impl Into<Shape>,
    name: &str,
) -> CandleResult<Tensor> {
    let shape = shape.into();
    match vb.get(shape.clone(), name) {
        Ok(tensor) => Ok(tensor),
        Err(_) => {
            let tensor = vb.get_unchecked(name)?;
            if tensor.elem_count() != shape.elem_count() {
                candle_core::bail!(
                    "tensor `{name}` has {} elements, expected shape {shape:?}",
                    tensor.elem_count()
                );
            }
            tensor.reshape(&shape)
        }
    }
}

/// Linear layer that accepts either dense or GGUF-packed weights.
///
/// Safetensors packages load a regular [`candle_nn::Linear`]. A quantized
/// GGUF package keeps Q8_0/Q4_0/k-quant weights packed in a [`QMatMul`], so
/// the matmul runs the backend's integer-dot-product kernels and the weights
/// stay compressed in VRAM. F32/F16 GGUF entries are dequantized to a dense
/// layer, which keeps F16 storage as a container-only option for now.
#[derive(Debug, Clone)]
pub(crate) struct Linear {
    inner: LinearInner,
}

#[derive(Debug, Clone)]
enum LinearInner {
    Dense(candle_nn::Linear),
    /// WGPU memory-saving dense path: packed GGUF weights are dequantized
    /// once to F16 instead of F32, halving weight VRAM. Activations are cast
    /// to F16 for the matmul and the result is cast back to the caller's
    /// dtype, so the rest of the pipeline keeps its original precision.
    DenseF16(candle_nn::Linear),
    Quant {
        weight: QMatMul,
        bias: Option<Tensor>,
        in_dim: usize,
        out_dim: usize,
    },
}

impl Linear {
    pub(crate) fn in_dim(&self) -> usize {
        match &self.inner {
            LinearInner::Dense(l) | LinearInner::DenseF16(l) => l.weight().dim(1).unwrap_or(0),
            LinearInner::Quant { in_dim, .. } => *in_dim,
        }
    }

    /// Storage dtype of the layer's parameters. Packed QMatMul layers report
    /// F32 because their kernels consume and return F32 activations.
    pub(crate) fn dtype(&self) -> DType {
        match &self.inner {
            LinearInner::Dense(l) | LinearInner::DenseF16(l) => l.weight().dtype(),
            LinearInner::Quant { bias, .. } => {
                bias.as_ref().map_or(DType::F32, |bias| bias.dtype())
            }
        }
    }

    pub(crate) fn bias(&self) -> Option<&Tensor> {
        match &self.inner {
            LinearInner::Dense(l) | LinearInner::DenseF16(l) => l.bias(),
            LinearInner::Quant { bias, .. } => bias.as_ref(),
        }
    }

    fn dense_weight(&self) -> Option<&Tensor> {
        match &self.inner {
            LinearInner::Dense(l) => Some(l.weight()),
            LinearInner::DenseF16(_) | LinearInner::Quant { .. } => None,
        }
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        match &self.inner {
            LinearInner::Dense(l) => l.forward(xs),
            LinearInner::DenseF16(l) => {
                let out_dtype = xs.dtype();
                let xs = xs.contiguous()?;
                let xs = if out_dtype == DType::F16 {
                    xs
                } else {
                    xs.to_dtype(DType::F16)?
                };
                let ys = l.forward(&xs)?;
                if out_dtype == DType::F16 {
                    Ok(ys)
                } else {
                    ys.to_dtype(out_dtype)
                }
            }
            LinearInner::Quant {
                weight,
                bias,
                out_dim,
                ..
            } => {
                // The quantized matmul kernels across cuda/vulkan/wgpu require a
                // contiguous F32 activation; chunked GeGLU halves and reshaped
                // attention states arrive strided, and an `--dtype f16` pipeline
                // feeds F16 states. Packed weights stay compressed: only the
                // activations are widened, then the result is narrowed back.
                let out_dtype = xs.dtype();
                let xs = xs.contiguous()?;
                let xs = if out_dtype == DType::F32 {
                    xs
                } else {
                    xs.to_dtype(DType::F32)?
                };
                // QMatMul only accepts 2D/3D activations, while dense
                // `candle_nn::Linear` broadcasts over any leading dims. Collapse
                // `[..., in]` to `[rows, in]` and restore the leading dims.
                let dims = xs.dims().to_vec();
                let ys = if dims.len() > 3 {
                    let rows = dims[..dims.len() - 1].iter().product::<usize>();
                    let flat = xs.reshape((rows, dims[dims.len() - 1]))?;
                    let mut out_dims = dims[..dims.len() - 1].to_vec();
                    out_dims.push(*out_dim);
                    weight.forward(&flat)?.reshape(out_dims)?
                } else {
                    weight.forward(&xs)?
                };
                let ys = match bias {
                    Some(bias) => ys.broadcast_add(bias)?,
                    None => ys,
                };
                if out_dtype == DType::F32 {
                    Ok(ys)
                } else {
                    ys.to_dtype(out_dtype)
                }
            }
        }
    }
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        Linear::forward(self, xs)
    }
}

fn load_linear(vb: VarBuilder, in_dim: usize, out_dim: usize, bias: bool) -> CandleResult<Linear> {
    let bias = if bias {
        Some(vb.get(out_dim, "bias")?)
    } else {
        None
    };
    if let Some(qt) = vb.get_quantized("weight")? {
        if let Some(linear) = quantized_linear(qt, &vb, in_dim, out_dim, bias.clone())? {
            return Ok(linear);
        }
    }
    let weight = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear {
        inner: LinearInner::Dense(candle_nn::Linear::new(weight, bias)),
    })
}

/// Keep packed weights for the quantized dtypes; dequantize raw F16/F32 GGUF
/// entries so they behave exactly like the dense safetensors path.
///
/// WGPU override: every packed GGML dtype is dequantized to dense F32 at
/// load, because its packed-weight `quantized_matmul` shader runs at ~2.8
/// GFLOP/s on the 32x32 tile (measured 192 ms per q8 4096x1024x1024 GEMM on
/// an RTX 3060), so a q8 TRELLIS.2 stage spends ~162 s per sampler step in
/// that kernel while the dense f32 warp-tile matmul takes ~18 ms for the same
/// shape. Dequantizing each stage's weights once at load (then dropping the
/// packed QTensor) trades ~4x weight VRAM for ~10x GEMM throughput; stages
/// are loaded and dropped sequentially, so the peak stays inside 12 GiB.
/// `LUX3D_WGPU_DEQUANT_WEIGHTS=0` restores packed weights (including the
/// K-quant path with its Q8K dispatch limit). `=f16` dequantizes packed
/// weights to dense F16 instead, which halves the weight footprint — but
/// candle's WGPU F16 GEMM mishandles batch > 1 (a whole-model TripoSG
/// `--num-shapes 2` run yields one noisy and one empty mesh), so whole-model
/// packages that need a real batch should stay packed (`=0`), which is also
/// the leanest layout; activations stay F32 and the VAE decode is dense F32.
fn quantized_linear(
    qt: QTensor,
    vb: &VarBuilder,
    in_dim: usize,
    out_dim: usize,
    bias: Option<Tensor>,
) -> CandleResult<Option<Linear>> {
    if qt.shape().dims2()? != (out_dim, in_dim) {
        candle_core::bail!(
            "quantized linear weight shape {:?} does not match expected ({out_dim}, {in_dim})",
            qt.shape()
        );
    }
    let wgpu_mode = std::env::var("LUX3D_WGPU_DEQUANT_WEIGHTS").unwrap_or_default();
    let force_f32 = force_dense_f32();
    let packed = !matches!(
        qt.dtype(),
        GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
    );
    let wgpu_dense = vb.device().is_wgpu() && wgpu_mode != "0" && !force_f32;
    // A pipeline loaded with `--dtype f16` requests F16 activations, so every
    // dequantized weight must stay F16; returning a dense F32 layer here
    // produces `dtype mismatch in matmul, lhs: F16, rhs: F32` inside the
    // DINOv2/CLIP/VAE stages whose GGUF entries are plain F16 rather than
    // packed quantized tensors. The explicit WGPU env override selects the
    // same layout for a whole-model q8 package even when the pipeline itself
    // still runs F32.
    let dense_f16 = !force_f32
        && (vb.dtype() == DType::F16
            || (vb.device().is_wgpu() && wgpu_mode.eq_ignore_ascii_case("f16")));
    if dense_f16 && (wgpu_dense || !packed) {
        // F16 dense weights halve the dequantized footprint. This is the
        // memory-safe fast path for whole-model GGUF packages (TripoSG q8:
        // ~8 GiB of F32 weights does not fit a 12 GiB card; ~4 GiB of F16
        // does). Activations are cast per layer and the output is restored to
        // the caller's dtype, so pipeline tensors keep their F32 contract.
        let weight = qt.dequantize(vb.device())?.to_dtype(DType::F16)?;
        let bias = bias.map(|bias| bias.to_dtype(DType::F16)).transpose()?;
        return Ok(Some(Linear {
            inner: LinearInner::DenseF16(candle_nn::Linear::new(weight, bias)),
        }));
    }
    if wgpu_dense || force_f32 || !packed {
        let target = if force_f32 { DType::F32 } else { vb.dtype() };
        let weight = qt.dequantize(vb.device())?.to_dtype(target)?;
        let bias = match bias {
            Some(bias) if bias.dtype() != target => Some(bias.to_dtype(target)?),
            bias => bias,
        };
        return Ok(Some(Linear {
            inner: LinearInner::Dense(candle_nn::Linear::new(weight, bias)),
        }));
    }
    // Packed QMatMul returns F32; keep the bias in the same dtype so the
    // narrowing cast in `forward` is the only conversion.
    let bias = bias.map(|bias| bias.to_dtype(DType::F32)).transpose()?;
    Ok(Some(Linear {
        inner: LinearInner::Quant {
            weight: QMatMul::from_qtensor(qt)?,
            bias,
            in_dim,
            out_dim,
        },
    }))
}

/// Load a linear layer from the legacy `(vb, in, out, bias)` call convention
/// used by most runtime blocks.
pub(crate) fn linear(
    vb: VarBuilder,
    in_dim: usize,
    out_dim: usize,
    bias: bool,
) -> CandleResult<Linear> {
    load_linear(vb, in_dim, out_dim, bias)
}

/// `candle_nn::linear`-compatible constructor.
pub(crate) fn linear_dense(in_dim: usize, out_dim: usize, vb: VarBuilder) -> CandleResult<Linear> {
    load_linear(vb, in_dim, out_dim, true)
}

/// `candle_nn::linear_no_bias`-compatible constructor.
pub(crate) fn linear_no_bias_dense(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
) -> CandleResult<Linear> {
    load_linear(vb, in_dim, out_dim, false)
}

/// Linear forward. The fused bias-epilogue GEMM (MUL_MAT_ADD,
/// `candle_nn::ops::mul_mat_add`) is correct on every shape but measured
/// NET-NEGATIVE on a 12 GiB WDDM card for the dense pi3x shapes (warm
/// criterion 5.13s -> 8.85s median, OOM risk: removing the add dispatches
/// shifts the batch/allocator dynamics over the VRAM edge), so it is
/// opt-in via CANDLE_LUX3D_FUSED_LINEAR=1. Default: plain unfused path.
/// Quantized (QMatMul) layers always take the plain path.
pub(crate) fn linear_fwd(l: &Linear, xs: &Tensor) -> CandleResult<Tensor> {
    match (l.bias(), l.dense_weight()) {
        (Some(bias), Some(weight))
            if std::env::var("CANDLE_LUX3D_FUSED_LINEAR").as_deref() == Ok("1") =>
        {
            candle_nn::ops::mul_mat_add(xs, &weight.t()?, bias)
        }
        _ => l.forward(xs),
    }
}

#[derive(Debug)]
pub(crate) struct LayerScale {
    gamma: Tensor,
}

impl LayerScale {
    pub(crate) fn new(vb: VarBuilder, dim: usize) -> CandleResult<Self> {
        Ok(Self {
            gamma: vb.get(dim, "gamma")?,
        })
    }
}

impl Module for LayerScale {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        xs.broadcast_mul(&self.gamma)
    }
}

#[derive(Debug)]
pub(crate) struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    pub(crate) fn new(vb: VarBuilder, dim: usize) -> CandleResult<Self> {
        Ok(Self {
            fc1: linear(vb.pp("fc1"), dim, dim * 4, true)?,
            fc2: linear(vb.pp("fc2"), dim * 4, dim, true)?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let xs = linear_fwd(&self.fc1, xs)?.gelu_erf()?;
        linear_fwd(&self.fc2, &xs)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GeGlu {
    proj: Linear,
}

impl GeGlu {
    pub(crate) fn load(vb: VarBuilder, dim_in: usize, dim_out: usize) -> CandleResult<Self> {
        Ok(Self {
            proj: linear(vb.pp("proj"), dim_in, dim_out * 2, true)?,
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let parts = linear_fwd(&self.proj, xs)?.chunk(2, candle_core::D::Minus1)?;
        parts[0].broadcast_mul(&parts[1].gelu_erf()?)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FeedForward {
    geglu: GeGlu,
    out: Linear,
}

impl FeedForward {
    pub(crate) fn load(vb: VarBuilder, dim: usize) -> CandleResult<Self> {
        Ok(Self {
            geglu: GeGlu::load(vb.pp("net").pp("0"), dim, dim * 4)?,
            out: linear(vb.pp("net").pp("2"), dim * 4, dim, true)?,
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        self.out.forward(&self.geglu.forward(xs)?)
    }
}
