//! TRELLIS.2 dense sparse-structure decoder
//! (`SparseStructureDecoder`, checkpoint `ss_dec_conv3d_16l8_fp16` from
//! microsoft/TRELLIS-image-large, 74 tensors).
//!
//! Reference: `trellis2/models/sparse_structure_vae.py` +
//! `trellis2/modules/spatial.py` (`pixel_shuffle_3d`).
//!
//! Checkpoint config (the local `ss_dec_conv3d_16l8_fp16.json`): out_channels
//! 1, latent_channels 8, num_res_blocks 2, num_res_blocks_middle 2,
//! channels [512, 128, 32], use_fp16. The `norm_type` arg is absent, so the
//! reference default `"layer"` applies: the block norms are
//! **ChannelLayerNorm32** (LayerNorm over the channel dim, affine, eps 1e-5),
//! not GroupNorm32.
//!
//! Layout: `input_layer` Conv3d(8, 512, 3, pad 1) -> middle 2x ResBlock(512)
//! at 16^3 -> blocks `[Res(512) x2, Up(512->128), Res(128) x2, Up(128->32),
//! Res(32) x2]` (two 2x upsamples -> 64^3) -> `out_layer` [ChannelLayerNorm,
//! SiLU, Conv3d(32, 1, 3, pad 1)] -> logits `[B, 1, 64, 64, 64]`.
//!
//! `ResBlock3d`: `h = conv2(silu(conv1(silu(norm1(x)))));
//! out = h + skip(x)` with `skip = Identity` when channels match (all blocks
//! here). `UpsampleBlock3d`: `Conv3d(ch, ch_out * 8, 3, pad 1)` then
//! `pixel_shuffle_3d(x, 2)`.
//!
//! `pixel_shuffle_3d` channel order (verified against
//! `modules/spatial.py`): `out[b, c_, 2x+s0, 2y+s1, 2z+s2] =
//! in[b, 8*c_ + 4*s0 + 2*s1 + s2, x, y, z]`.
//!
//! The reference pipeline runs this decoder with an F32 stem (`input_layer`,
//! `out_layer`) and F16 torso; per locked decision 3 this port computes
//! everything in F32 (weights are F32/F16 on disk and cast at access).
//!
//! candle has no Conv3d, so `Conv3d` below implements cross-correlation as a
//! sum over the 27 kernel offsets of `[B*XYZ, C] @ [C, O]` GEMMs (im2col-free,
//! zero padding via tensor padding). F32 accumulation order differs from
//! PyTorch's GEMM but agrees to fp32 rounding.

use candle_core::{DType, Result as CandleResult, Tensor};

use crate::weights::VarBuilder;

/// Load a tensor and force F32 (checkpoint stores mixed F16/F32). GGUF
/// packages store rank>4 tensors (3D convolution kernels) flattened; the
/// helper reshapes them back to the expected shape.
fn get_f32(
    vb: &VarBuilder,
    shape: impl Into<candle_core::Shape>,
    name: &str,
) -> CandleResult<Tensor> {
    let tensor = super::nn_blocks::get_weight_shaped(vb, shape, name)?;
    if tensor.dtype() == DType::F32 {
        Ok(tensor)
    } else {
        tensor.to_dtype(DType::F32)
    }
}

/// Dense 3D convolution `[O, I, kD, kH, kW]` (PyTorch layout,
/// cross-correlation, zero padding, stride 1).
#[derive(Debug)]
pub(crate) struct Conv3d {
    /// Per-offset kernels `[in_channels, out_channels]` F32, `k = kd*9 +
    /// kh*3 + kw`, each a standalone contiguous tensor with start offset 0.
    ///
    /// Keeping them as views into one `[27, C, O]` tensor looked cheaper at
    /// load time but was catastrophic on Vulkan: a view with a start offset
    /// above 65535 elements (any `k > 0` at C=512) makes candle's
    /// RHS-transpose fallback materialize the kernel through
    /// `vkCmdCopyBuffer` with one region per element (262k regions for a
    /// 512x512 kernel, ~20 ms per GEMM, ~11 s per SS decode).
    kernels: Vec<Tensor>,
    in_channels: usize,
    out_channels: usize,
    bias: Option<Tensor>,
    padding: usize,
}

impl Conv3d {
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        padding: usize,
        bias: bool,
    ) -> CandleResult<Self> {
        let weight = get_f32(
            &vb,
            (out_channels, in_channels, kernel, kernel, kernel),
            "weight",
        )?;
        // [O, C, kd, kh, kw] -> [27, C, O]: flatten the kernel axes in memory
        // order (kw fastest), then split into per-offset owned `[C, O]`
        // tensors so each matmul consumes the kernel directly.
        let weight27 = weight
            .reshape((out_channels, in_channels, kernel * kernel * kernel))?
            .permute((2, 1, 0))?
            .contiguous()?;
        let kernels = (0..kernel * kernel * kernel)
            .map(|k| weight27.narrow(0, k, 1)?.squeeze(0)?.affine(1.0, 0.0))
            .collect::<CandleResult<Vec<_>>>()?;
        let bias = if bias {
            Some(get_f32(&vb, out_channels, "bias")?)
        } else {
            None
        };
        Ok(Self {
            kernels,
            in_channels,
            out_channels,
            bias,
            padding,
        })
    }

    /// Rewrite the output channels from the reference pixel-shuffle order
    /// `c*8 + s0*4 + s1*2 + s2` to sub-voxel major `(s0*4 + s1*2 + s2)*C + c`.
    ///
    /// Used only by the two upsample blocks. With this order each of the 8
    /// sub-voxel planes is a contiguous `[B, X, Y, Z, C]` slice, and the
    /// shuffle is three `stack`+`reshape` interleaves along Z, Y, X. Those
    /// copies go through `Tensor::cat`'s 2D copy path, whereas the reference
    /// order needs a rank-8 permutation that Vulkan materializes with one
    /// copy per leading index (0.3-0.6 s per upsample).
    fn reorder_output_channels_subvoxel_major(&mut self) -> CandleResult<()> {
        const SUBVOXELS: usize = 8;
        if self.out_channels % SUBVOXELS != 0 {
            candle_core::bail!(
                "pixel-shuffle output channels {} not divisible by 8",
                self.out_channels
            );
        }
        let groups = self.out_channels / SUBVOXELS;
        for kernel in &mut self.kernels {
            let (cin, cout) = kernel.dims2()?;
            *kernel = kernel
                .reshape((cin, groups, SUBVOXELS))?
                .permute((0, 2, 1))?
                .contiguous()?
                .reshape((cin, cout))?;
        }
        if let Some(bias) = self.bias.take() {
            let cout = bias.dim(0)?;
            self.bias = Some(
                bias.reshape((groups, SUBVOXELS))?
                    .permute((1, 0))?
                    .contiguous()?
                    .reshape(cout)?,
            );
        }
        Ok(())
    }

    /// `x: [B, C, X, Y, Z]` -> `[B, O, X, Y, Z]` (stride 1, even padding).
    /// Reference wrapper for the cross-correlation test; the decoder itself
    /// runs [`Self::forward_channels_last`] end-to-end.
    #[cfg(test)]
    pub fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let channels_last = xs.permute((0, 2, 3, 4, 1))?.contiguous()?;
        self.forward_channels_last(&channels_last)?
            .permute((0, 4, 1, 2, 3))
    }

    /// Channels-last 3D convolution: `[B, X, Y, Z, C]` -> `[B, X, Y, Z, O]`.
    ///
    /// The SS decoder keeps every activation in this layout so the 3D
    /// convolution never transposes `[B*XYZ, C]`.
    ///
    /// Vulkan copies strided rank>4 layouts through
    /// `materialize_rank_gt4_compact`, which emits one copy per leading index
    /// (and, for large offsets, one `vkCmdCopyBuffer` region per row). A
    /// direct `[B, X, Y, Z, C]` window therefore cost hundreds of submissions
    /// per offset. Instead, one `affine(1, 0)` materialization per kd slice
    /// resets the storage offset and collapses the spatial axes into a rank-4
    /// `[B*X, Yp, Zp, C]` matrix; each kh/kw window is then a 4D strided view
    /// with a small offset, which Vulkan copies with a single dispatch.
    /// `k = kd*9 + kh*3 + kw` matches `kernels`' order.
    pub fn forward_channels_last(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let (batch, dx, dy, dz, in_channels) = xs.dims5()?;
        if in_channels != self.in_channels || self.kernels.len() != 27 {
            candle_core::bail!(
                "Conv3d kernels ({}, {}) do not match input [B, .., C={in_channels}] with 3^3 kernel",
                self.kernels.len(),
                self.in_channels
            );
        }
        if self.padding > dx || self.padding > dy || self.padding > dz {
            candle_core::bail!("Conv3d padding larger than input dims");
        }
        let (ox, oy, oz) = (dx, dy, dz); // stride 1, symmetric padding keeps size.
        let out_channels = self.out_channels;
        let p = self.padding;
        let padded = xs
            .pad_with_zeros(1, p, p)?
            .pad_with_zeros(2, p, p)?
            .pad_with_zeros(3, p, p)?;
        let (yp, zp) = (oy + 2 * p, oz + 2 * p);
        let rows = batch * ox;
        let tokens = ox * oy * oz;
        let mut acc: Option<Tensor> = None;
        for kd_offset in 0..3 {
            // `affine(1, 0)` yields a fresh contiguous buffer with offset 0.
            // (`copy()` cannot be used here: it keeps the source layout's
            // start offset, which then indexes past the copied storage.)
            // Resetting the offset keeps every rank-4 kh/kw window copy below
            // the copy shader's 16-bit offset limit, so each window is one
            // dispatch instead of one region per row.
            let slab = padded
                .narrow(1, kd_offset, ox)?
                .reshape((rows, yp, zp, in_channels))?
                .affine(1.0, 0.0)?;
            for kh_offset in 0..3 {
                for kw_offset in 0..3 {
                    let columns = slab
                        .narrow(1, kh_offset, oy)?
                        .narrow(2, kw_offset, oz)?
                        .contiguous()?
                        .reshape((batch * tokens, in_channels))?;
                    let k = kd_offset * 9 + kh_offset * 3 + kw_offset;
                    let part = columns.matmul(&self.kernels[k])?; // [B*XYZ, O]
                    acc = Some(match acc {
                        None => part,
                        Some(existing) => existing.add(&part)?,
                    });
                }
            }
        }
        let acc = acc.expect("27 kernel offsets accumulated");
        let mut out = acc.reshape((batch, ox, oy, oz, out_channels))?;
        if let Some(bias) = &self.bias {
            out = out.broadcast_add(&bias.reshape((1, 1, 1, 1, out_channels))?)?;
        }
        Ok(out)
    }
}

/// `ChannelLayerNorm32` (`norm_layer("layer", ..)`): LayerNorm over the
/// channel dim of a `[B, C, X, Y, Z]` tensor with per-channel affine
/// parameters, eps 1e-5.
#[derive(Debug)]
pub(crate) struct ChannelLayerNorm {
    weight: Tensor,
    bias: Tensor,
}

impl ChannelLayerNorm {
    pub fn load(vb: VarBuilder, channels: usize) -> CandleResult<Self> {
        Ok(Self {
            weight: get_f32(&vb, channels, "weight")?,
            bias: get_f32(&vb, channels, "bias")?,
        })
    }

    /// Channels-last variant (`[B, X, Y, Z, C]`, norm over the last axis).
    ///
    /// Uses the backend's fused layer-norm. The manual mean/sub/sqr/div
    /// sequence materializes each rank-5 broadcast through Vulkan's
    /// `materialize_rank_gt4_compact`, which walks one copy per leading index
    /// and made a single 64^3 norm cost ~1.7 s; the fused kernel flattens to
    /// `[rows, C]` and normalizes in one dispatch.
    pub fn forward_channels_last(&self, xs: &Tensor) -> CandleResult<Tensor> {
        candle_nn::ops::layer_norm(xs, &self.weight, &self.bias, 1e-5)
    }
}

/// Decoder residual block (`ResBlock3d`, norm_type "layer", same in/out
/// channels; the 1x1 skip conv is only used when channels differ).
#[derive(Debug)]
pub(crate) struct ResBlock3d {
    norm1: ChannelLayerNorm,
    norm2: ChannelLayerNorm,
    conv1: Conv3d,
    conv2: Conv3d,
}

impl ResBlock3d {
    pub fn load(vb: VarBuilder, channels: usize) -> CandleResult<Self> {
        Ok(Self {
            norm1: ChannelLayerNorm::load(vb.pp("norm1"), channels)?,
            norm2: ChannelLayerNorm::load(vb.pp("norm2"), channels)?,
            conv1: Conv3d::load(vb.pp("conv1"), channels, channels, 3, 1, true)?,
            conv2: Conv3d::load(vb.pp("conv2"), channels, channels, 3, 1, true)?,
        })
    }

    /// Channels-last block (`[B, X, Y, Z, C]`).
    pub fn forward_channels_last(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let h = self.norm1.forward_channels_last(xs)?.silu()?;
        let h = self.conv1.forward_channels_last(&h)?;
        let h = self.norm2.forward_channels_last(&h)?.silu()?;
        let h = self.conv2.forward_channels_last(&h)?;
        h.add(xs)
    }
}

/// `pixel_shuffle_3d(x, 2)` (`modules/spatial.py`), the reference
/// channels-first shuffle pinned by the channel-order test:
/// `out[b, c_, 2x+s0, 2y+s1, 2z+s2] = in[b, 8*c_ + 4*s0 + 2*s1 + s2, x, y, z]`.
#[cfg(test)]
pub(crate) fn pixel_shuffle_3d(xs: &Tensor) -> CandleResult<Tensor> {
    let (batch, channels, dx, dy, dz) = xs.dims5()?;
    if channels % 8 != 0 {
        candle_core::bail!("pixel_shuffle_3d expects channels divisible by 8, got {channels}");
    }
    let out_channels = channels / 8;
    let xs = xs.reshape(&[batch, out_channels, 2, 2, 2, dx, dy, dz])?;
    let xs = xs.permute([0usize, 1, 5, 2, 6, 3, 7, 4])?;
    xs.reshape(&[batch, out_channels, dx * 2, dy * 2, dz * 2])
}

/// Channels-last `pixel_shuffle_3d` for convolution outputs whose channels
/// were reordered by [`Conv3d::reorder_output_channels_subvoxel_major`]:
/// `[B, X, Y, Z, 8*C]` -> `[B, 2X, 2Y, 2Z, C]`.
///
/// Each sub-voxel plane `q = s0*4 + s1*2 + s2` is contiguous, so interleaving
/// along Z, then Y, then X is a `stack` + adjacent-dims `reshape` (metadata
/// only, the copy inside `stack` runs through the backend's 2D cat/copy path).
fn pixel_shuffle_3d_subvoxel_major(xs: &Tensor) -> CandleResult<Tensor> {
    let (batch, dx, dy, dz, channels) = xs.dims5()?;
    if channels % 8 != 0 {
        candle_core::bail!("pixel_shuffle_3d expects channels divisible by 8, got {channels}");
    }
    let c = channels / 8;
    let plane = |q: usize| xs.narrow(4, q * c, c);
    let mut z_planes = Vec::with_capacity(4);
    for s0 in 0..2usize {
        for s1 in 0..2usize {
            let q = s0 * 4 + s1 * 2;
            let low = plane(q)?;
            let high = plane(q + 1)?;
            // [B, X, Y, Z, 2, C] -> [B, X, Y, 2Z, C] (z, s2 interleaved).
            z_planes.push(Tensor::stack(&[&low, &high], 4)?.reshape((batch, dx, dy, dz * 2, c))?);
        }
    }
    // [B, X, Y, 2, 2Z, C] -> [B, X, 2Y, 2Z, C] (y, s1 interleaved).
    let y0 =
        Tensor::stack(&[&z_planes[0], &z_planes[1]], 3)?.reshape((batch, dx, dy * 2, dz * 2, c))?;
    let y1 =
        Tensor::stack(&[&z_planes[2], &z_planes[3]], 3)?.reshape((batch, dx, dy * 2, dz * 2, c))?;
    // [B, X, 2, 2Y, 2Z, C] -> [B, 2X, 2Y, 2Z, C] (x, s0 interleaved).
    Tensor::stack(&[&y0, &y1], 2)?.reshape((batch, dx * 2, dy * 2, dz * 2, c))
}

/// Decoder upsample block (`UpsampleBlock3d`, mode "conv"):
/// `Conv3d(ch_in, ch_out * 8, 3, pad 1)` then `pixel_shuffle_3d(.., 2)`.
#[derive(Debug)]
pub(crate) struct UpsampleBlock3d {
    conv: Conv3d,
}

impl UpsampleBlock3d {
    pub fn load(vb: VarBuilder, in_channels: usize, out_channels: usize) -> CandleResult<Self> {
        let mut conv = Conv3d::load(vb.pp("conv"), in_channels, out_channels * 8, 3, 1, true)?;
        // Reference channel order is `c*8 + s0*4 + s1*2 + s2`; reorder to
        // `(s0*4 + s1*2 + s2)*C + c` so the pixel shuffle becomes three
        // adjacent-axis interleaves instead of a rank-8 permutation.
        conv.reorder_output_channels_subvoxel_major()?;
        Ok(Self { conv })
    }

    /// Channels-last upsample (`[B, X, Y, Z, C]`).
    pub fn forward_channels_last(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let h = self.conv.forward_channels_last(xs)?;
        pixel_shuffle_3d_subvoxel_major(&h)
    }
}

/// Dense sparse-structure decoder. Loads the `ss_dec` stage (74 tensors).
#[derive(Debug)]
pub struct Trellis2SsDecoder {
    input_layer: Conv3d,
    middle_block: Vec<ResBlock3d>,
    blocks: Vec<DecoderBlock>,
    out_norm: ChannelLayerNorm,
    out_conv: Conv3d,
}

#[derive(Debug)]
enum DecoderBlock {
    Res(ResBlock3d),
    Up(UpsampleBlock3d),
}

/// Sparse-structure decoder architecture parameters (checkpoint json args).
#[derive(Debug, Clone, PartialEq)]
pub struct Trellis2SsDecoderConfig {
    /// Output logits channels (1).
    pub out_channels: usize,
    /// Latent input channels (8).
    pub latent_channels: usize,
    /// Residual blocks per level (2).
    pub num_res_blocks: usize,
    /// Middle residual blocks (2).
    pub num_res_blocks_middle: usize,
    /// Channel pyramid, encoder order ([512, 128, 32]).
    pub channels: Vec<usize>,
}

impl Trellis2SsDecoderConfig {
    /// `ss_dec_conv3d_16l8_fp16` checkpoint args (the `norm_type` arg is
    /// absent there, so the reference default `"layer"` applies).
    pub fn ss_dec_16l8() -> Self {
        Self {
            out_channels: 1,
            latent_channels: 8,
            num_res_blocks: 2,
            num_res_blocks_middle: 2,
            channels: vec![512, 128, 32],
        }
    }
}

impl Default for Trellis2SsDecoderConfig {
    fn default() -> Self {
        Self::ss_dec_16l8()
    }
}

impl Trellis2SsDecoder {
    /// Load from a VarBuilder rooted at the decoder tensors (canonical
    /// package: `vb.pp("ss_dec")`). Fails on any missing tensor.
    pub fn load(vb: VarBuilder, config: &Trellis2SsDecoderConfig) -> CandleResult<Self> {
        if config.channels.len() < 2 {
            candle_core::bail!("SS decoder needs at least two channel levels");
        }
        let first = config.channels[0];
        let input_layer = Conv3d::load(
            vb.pp("input_layer"),
            config.latent_channels,
            first,
            3,
            1,
            true,
        )?;
        let middle_block = (0..config.num_res_blocks_middle)
            .map(|index| ResBlock3d::load(vb.pp("middle_block").pp(index.to_string()), first))
            .collect::<CandleResult<Vec<_>>>()?;

        let mut blocks = Vec::new();
        for (level, &channels) in config.channels.iter().enumerate() {
            for _ in 0..config.num_res_blocks {
                blocks.push(DecoderBlock::Res(ResBlock3d::load(
                    vb.pp("blocks").pp(blocks.len().to_string()),
                    channels,
                )?));
            }
            if level + 1 < config.channels.len() {
                blocks.push(DecoderBlock::Up(UpsampleBlock3d::load(
                    vb.pp("blocks").pp(blocks.len().to_string()),
                    channels,
                    config.channels[level + 1],
                )?));
            }
        }
        let last = *config.channels.last().expect("non-empty channels");
        let out_norm = ChannelLayerNorm::load(vb.pp("out_layer").pp("0"), last)?;
        let out_conv = Conv3d::load(
            vb.pp("out_layer").pp("2"),
            last,
            config.out_channels,
            3,
            1,
            true,
        )?;
        Ok(Self {
            input_layer,
            middle_block,
            blocks,
            out_norm,
            out_conv,
        })
    }

    /// `z_s: [B, 8, 16, 16, 16]` -> logits `[B, 1, 64, 64, 64]`.
    pub fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let (_batch, channels, _, _, _) = xs.dims5()?;
        if channels != self.input_layer_weight_in() {
            candle_core::bail!(
                "SS decoder expects {} latent channels, got {channels}",
                self.input_layer_weight_in()
            );
        }
        // Convert to channels-last once for the whole decoder. Every block
        // then stays in `[B, X, Y, Z, C]`, so the convs never transpose.
        let profile = std::env::var_os("LUX3D_SS_DEC_PROFILE").is_some();
        let mut start = std::time::Instant::now();
        let mut h = xs.permute((0, 2, 3, 4, 1))?.contiguous()?;
        if profile {
            eprintln!(
                "[ss-dec] to channels-last: {:.2}s",
                start.elapsed().as_secs_f64()
            );
            start = std::time::Instant::now();
        }
        h = self.input_layer.forward_channels_last(&h)?;
        if profile {
            eprintln!("[ss-dec] input conv: {:.2}s", start.elapsed().as_secs_f64());
            start = std::time::Instant::now();
        }
        for (index, block) in self.middle_block.iter().enumerate() {
            h = block.forward_channels_last(&h)?;
            if profile {
                eprintln!(
                    "[ss-dec] middle {index}: {:.2}s",
                    start.elapsed().as_secs_f64()
                );
                start = std::time::Instant::now();
            }
        }
        for (index, block) in self.blocks.iter().enumerate() {
            h = match block {
                DecoderBlock::Res(res) => res.forward_channels_last(&h)?,
                DecoderBlock::Up(up) => up.forward_channels_last(&h)?,
            };
            if profile {
                eprintln!(
                    "[ss-dec] block {index}: {:.2}s",
                    start.elapsed().as_secs_f64()
                );
                start = std::time::Instant::now();
            }
        }
        let h = self.out_norm.forward_channels_last(&h)?.silu()?;
        if profile {
            eprintln!("[ss-dec] out norm: {:.2}s", start.elapsed().as_secs_f64());
            start = std::time::Instant::now();
        }
        let out = self
            .out_conv
            .forward_channels_last(&h)?
            .permute((0, 4, 1, 2, 3))?;
        if profile {
            eprintln!("[ss-dec] out conv: {:.2}s", start.elapsed().as_secs_f64());
        }
        Ok(out)
    }

    fn input_layer_weight_in(&self) -> usize {
        self.input_layer.in_channels
    }
}

/// Extract occupied-voxel coordinates from decoder logits, replicating the
/// pipeline (`decoded = ss_logits > 0`; optional
/// `F.max_pool3d(decoded.float(), 2, 2, 0) > 0.5`):
///
/// - `target_res == logits res` (64): `argwhere(logits > 0)[:, [0,2,3,4]]`.
/// - `target_res == logits res / 2` (32): max-pool the occupancy by 2 and
///   threshold at 0.5 (i.e. any of the 8 sub-voxels occupied).
///
/// Returns `(b, i, j, k)` in row-major order, matching torch `argwhere`.
pub fn occupancy_coords(logits: &Tensor, target_res: u32) -> CandleResult<Vec<[i32; 4]>> {
    let (batch, channels, dx, dy, dz) = logits.dims5()?;
    if channels != 1 || dx != dy || dy != dz {
        candle_core::bail!(
            "occupancy_coords expects [B, 1, R, R, R] logits, got {batch}, {channels}, \
             {dx}, {dy}, {dz}"
        );
    }
    let values = logits
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    // Row-major strides for [B, 1, X, Y, Z]: last dim stride 1.
    let stride_z = 1usize;
    let stride_y = dz;
    let stride_x = dy * dz;
    let stride_b = dx * dy * dz;
    if target_res as usize == dx {
        let mut coords = Vec::new();
        for b in 0..batch {
            for i in 0..dx {
                for j in 0..dy {
                    for k in 0..dz {
                        let index = b * stride_b + i * stride_x + j * stride_y + k * stride_z;
                        if values[index] > 0.0 {
                            coords.push([b as i32, i as i32, j as i32, k as i32]);
                        }
                    }
                }
            }
        }
        return Ok(coords);
    }
    if target_res as usize * 2 == dx {
        let res = target_res as usize;
        let mut coords = Vec::new();
        for b in 0..batch {
            for i in 0..res {
                for j in 0..res {
                    for k in 0..res {
                        let mut any = false;
                        for s0 in 0..2 {
                            for s1 in 0..2 {
                                for s2 in 0..2 {
                                    let index = b * stride_b
                                        + (i * 2 + s0) * stride_x
                                        + (j * 2 + s1) * stride_y
                                        + (k * 2 + s2) * stride_z;
                                    if values[index] > 0.0 {
                                        any = true;
                                    }
                                }
                            }
                        }
                        if any {
                            coords.push([b as i32, i as i32, j as i32, k as i32]);
                        }
                    }
                }
            }
        }
        return Ok(coords);
    }
    candle_core::bail!(
        "occupancy target_res {target_res} must be the logits resolution {dx} or half of it"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::collections::HashMap;

    /// `pixel_shuffle_3d` channel order: with C_=1, scale 2, the 8 input
    /// channels must scatter to (s0, s1, s2) = (dx, dy, dz) offsets as
    /// `c = 4*s0 + 2*s1 + s2` (PyTorch row-major channel split).
    #[test]
    fn pixel_shuffle_3d_scatters_channels_as_reference() {
        // Input [1, 8, 1, 1, 1]: channel c holds value c+1.
        let values: Vec<f32> = (0..8).map(|c| (c + 1) as f32).collect();
        let xs = Tensor::from_vec(values, (1, 8, 1, 1, 1), &Device::Cpu).expect("xs");
        let out = pixel_shuffle_3d(&xs).expect("shuffle");
        assert_eq!(out.dims(), &[1, 1, 2, 2, 2]);
        let out = out
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .expect("vec");
        // Output voxel (dx, dy, dz) reads channel 4*dx + 2*dy + dz.
        for dx in 0..2usize {
            for dy in 0..2usize {
                for dz in 0..2usize {
                    let expected = (4 * dx + 2 * dy + dz + 1) as f32;
                    let index = dx * 4 + dy * 2 + dz;
                    assert_eq!(out[index], expected, "voxel ({dx},{dy},{dz})");
                }
            }
        }
    }

    /// Dense 3D convolution matches a naive cross-correlation over the same
    /// `[O, I, kD, kH, kW]` weight, which pins the `weight27` offset order.
    #[test]
    fn conv3d_matches_naive_cross_correlation() {
        let device = Device::Cpu;
        let (batch, in_ch, out_ch, side) = (1usize, 2usize, 3usize, 4usize);
        let xs: Vec<f32> = (0..batch * in_ch * side * side * side)
            .map(|i| (i as f32 * 0.37).sin())
            .collect();
        let xs = Tensor::from_vec(xs, (batch, in_ch, side, side, side), &device).expect("xs");
        let w: Vec<f32> = (0..out_ch * in_ch * 27)
            .map(|i| ((i as f32) * 1.13).cos())
            .collect();
        let weight = Tensor::from_vec(w.clone(), (out_ch, in_ch, 3, 3, 3), &device).expect("w");
        let bias = Tensor::from_vec(vec![0.25_f32, -0.5, 1.0], out_ch, &device).expect("b");
        let weight27 = weight
            .reshape((out_ch, in_ch, 27))
            .and_then(|t| t.permute((2, 1, 0)))
            .and_then(|t| t.contiguous())
            .expect("weight27");
        let kernels = (0..27)
            .map(|k| {
                weight27
                    .narrow(0, k, 1)
                    .and_then(|t| t.squeeze(0))
                    .and_then(|t| t.affine(1.0, 0.0))
            })
            .collect::<candle_core::Result<Vec<_>>>()
            .expect("kernels");
        let conv = Conv3d {
            kernels,
            in_channels: in_ch,
            out_channels: out_ch,
            bias: Some(bias.clone()),
            padding: 1,
        };
        let out = conv.forward(&xs).expect("forward");
        assert_eq!(out.dims(), &[batch, out_ch, side, side, side]);
        let out = out
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("vec");
        let xs = xs
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("vec");
        let w = w;
        let bias = [0.25_f32, -0.5, 1.0];
        let idx = |b: usize, c: usize, x: usize, y: usize, z: usize| {
            ((b * in_ch + c) * side + x) * side * side + y * side + z
        };
        let mut expected = vec![0.0_f32; batch * out_ch * side * side * side];
        for b in 0..batch {
            for o in 0..out_ch {
                for x in 0..side {
                    for y in 0..side {
                        for z in 0..side {
                            let mut acc = bias[o];
                            for c in 0..in_ch {
                                for kd in 0..3 {
                                    for kh in 0..3 {
                                        for kw in 0..3 {
                                            let (sx, sy, sz) = (
                                                x as isize + kd as isize - 1,
                                                y as isize + kh as isize - 1,
                                                z as isize + kw as isize - 1,
                                            );
                                            if sx < 0
                                                || sy < 0
                                                || sz < 0
                                                || sx >= side as isize
                                                || sy >= side as isize
                                                || sz >= side as isize
                                            {
                                                continue;
                                            }
                                            let wv =
                                                w[((o * in_ch + c) * 3 + kd) * 9 + kh * 3 + kw];
                                            acc += wv
                                                * xs[idx(
                                                    b,
                                                    c,
                                                    sx as usize,
                                                    sy as usize,
                                                    sz as usize,
                                                )];
                                        }
                                    }
                                }
                            }
                            let oi = ((b * out_ch + o) * side + x) * side * side + y * side + z;
                            expected[oi] = acc;
                        }
                    }
                }
            }
        }
        for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 2e-5,
                "conv3d mismatch at {i}: got {got}, want {want}"
            );
        }
    }

    /// `occupancy_coords` threshold + 2x max-pool on a synthetic grid.
    #[test]
    fn occupancy_coords_threshold_and_maxpool() {
        // 2^3 grid, single positive voxel at (5, 3, 2) plus a positive
        // voxel at (0, 0, 0).
        let mut values = vec![0.0_f32; 8 * 8 * 8];
        let idx = |x: usize, y: usize, z: usize| x * 64 + y * 8 + z;
        values[idx(5, 3, 2)] = 12.0;
        values[idx(0, 0, 0)] = 0.5;
        values[idx(7, 7, 7)] = -3.0; // negative: not occupied
        let logits = Tensor::from_vec(values, (1, 1, 8, 8, 8), &Device::Cpu).expect("logits");

        let coords_8 = occupancy_coords(&logits, 8).expect("coords8");
        assert_eq!(coords_8, vec![[0, 0, 0, 0], [0, 5, 3, 2]]);

        // Half resolution: (5,3,2) falls into 2x2x2 block (2,1,1);
        // (0,0,0) into block (0,0,0); (7,7,7) is not occupied.
        let coords_4 = occupancy_coords(&logits, 4).expect("coords4");
        assert_eq!(coords_4, vec![[0, 0, 0, 0], [0, 2, 1, 1]]);
    }

    /// Tiny synthetic decoder forward: channels [8, 4] (two levels, one
    /// upsample) on a [1, 2, 4, 4, 4] latent -> [1, 1, 8, 8, 8] logits.
    #[test]
    fn tiny_decoder_forward_shape() {
        let device = Device::Cpu;
        let config = Trellis2SsDecoderConfig {
            out_channels: 1,
            latent_channels: 2,
            num_res_blocks: 1,
            num_res_blocks_middle: 1,
            channels: vec![8, 4],
        };
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        let add = |name: &str, shape: &[usize], device: &Device| -> Tensor {
            let len: usize = shape.iter().product();
            Tensor::from_vec(
                (0..len).map(|i| (i % 13) as f32 * 0.05 - 0.3).collect(),
                shape.to_vec(),
                device,
            )
            .expect(name)
        };
        // input_layer conv [8, 2, 3, 3, 3]
        tensors.insert(
            "input_layer.weight".into(),
            add("w", &[8, 2, 3, 3, 3], &device),
        );
        tensors.insert("input_layer.bias".into(), add("b", &[8], &device));
        for index in 0..1usize {
            for block in [format!("middle_block.{index}")].iter() {
                for norm in ["norm1", "norm2"] {
                    tensors.insert(format!("{block}.{norm}.weight"), add("nw", &[8], &device));
                    tensors.insert(format!("{block}.{norm}.bias"), add("nb", &[8], &device));
                }
                tensors.insert(
                    format!("{block}.conv1.weight"),
                    add("c1w", &[8, 8, 3, 3, 3], &device),
                );
                tensors.insert(format!("{block}.conv1.bias"), add("c1b", &[8], &device));
                tensors.insert(
                    format!("{block}.conv2.weight"),
                    add("c2w", &[8, 8, 3, 3, 3], &device),
                );
                tensors.insert(format!("{block}.conv2.bias"), add("c2b", &[8], &device));
            }
        }
        // blocks.0 = ResBlock(8), blocks.1 = Up(8 -> 4*8), blocks.2 = ResBlock(4)
        for (name, channels) in [("blocks.0", 8usize), ("blocks.2", 4usize)] {
            for norm in ["norm1", "norm2"] {
                tensors.insert(
                    format!("{name}.{norm}.weight"),
                    add("nw", &[channels], &device),
                );
                tensors.insert(
                    format!("{name}.{norm}.bias"),
                    add("nb", &[channels], &device),
                );
            }
            tensors.insert(
                format!("{name}.conv1.weight"),
                add("c1w", &[channels, channels, 3, 3, 3], &device),
            );
            tensors.insert(
                format!("{name}.conv1.bias"),
                add("c1b", &[channels], &device),
            );
            tensors.insert(
                format!("{name}.conv2.weight"),
                add("c2w", &[channels, channels, 3, 3, 3], &device),
            );
            tensors.insert(
                format!("{name}.conv2.bias"),
                add("c2b", &[channels], &device),
            );
        }
        tensors.insert(
            "blocks.1.conv.weight".into(),
            add("uw", &[32, 8, 3, 3, 3], &device),
        );
        tensors.insert("blocks.1.conv.bias".into(), add("ub", &[32], &device));
        tensors.insert("out_layer.0.weight".into(), add("onw", &[4], &device));
        tensors.insert("out_layer.0.bias".into(), add("onb", &[4], &device));
        tensors.insert(
            "out_layer.2.weight".into(),
            add("ocw", &[1, 4, 3, 3, 3], &device),
        );
        tensors.insert("out_layer.2.bias".into(), add("ocb", &[1], &device));

        let vb = candle_nn::VarBuilder::from_tensors(tensors, DType::F32, &device);
        let decoder = Trellis2SsDecoder::load(vb, &config).expect("decoder");
        let xs = Tensor::randn(0f32, 1f32, (1, 2, 4, 4, 4), &device).expect("latent");
        let logits = decoder.forward(&xs).expect("forward");
        assert_eq!(logits.dims(), &[1, 1, 8, 8, 8]);
    }
}
