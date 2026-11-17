//! ALIKED-n16 detector/descriptor (LightGlue port) for VGGT tracking stage A.
//!
//! Reference: `lightglue/aliked.py` (`ALIKED`, `DKD`, `SDDH`) at
//! `G:/jobs/candle-based-projects/candle_refs/LightGlue`, model config
//! `aliked-n16` (`c1=16,c2=32,c3=64,c4=128,dim=128,K=3,M=16`), official
//! checkpoint `aliked-n16.pth` converted to safetensors with unchanged names.
//!
//! The port follows the reference line by line:
//!
//! * input is replicate-padded to a multiple of 32 (`InputPadder`);
//! * `block1` is a 3x3 `ConvBlock` (SELU gate), `block2` is a 1x1-downsampled
//!   `ResBlock`, `block3`/`block4` are DCN `ResBlock`s (offset conv 3x3 ->
//!   18 channels, clamp `+-max(h,w)/4`, torchvision `deform_conv2d` bilinear
//!   sampling, no mask);
//! * feature aggregation is four 1x1 convs to 32 channels, bilinear
//!   `align_corners=True` upsamples by 2/8/32, concatenation, and a SELU
//!   score head; the 128-channel feature map is L2-normalized and both maps
//!   are unpadded;
//! * `DKD` runs `simple_nms` (radius 2), zeroes a 2-pixel border, applies the
//!   score threshold with the `n_limit` score sort, and refines every maximum
//!   with the 5x5 soft-argmax (temperature 0.1);
//! * `SDDH` samples a 3x3 patch per keypoint, predicts 16 (dx,dy) offsets,
//!   bilinearly samples the feature map at those 16 positions, applies
//!   `sf_conv` + SELU, contracts with `agg_weights`, and L2-normalizes.
//!
//! Numerical notes / deviations:
//!
//! * `DKD` selection and soft-argmax are evaluated on the host in f32 after a
//!   device->CPU copy of the single-channel score map. This matches the
//!   reference CPU arithmetic and keeps selection deterministic; the
//!   descriptor path stays on the model device.
//! * torchvision's `deform_conv2d` and `grid_sample` zero-padding semantics
//!   are reproduced by the local bilinear gather helpers (`in_range` masks
//!   the weights instead of clamping the value), including the
//!   `coord <= -1 || coord >= len` zero window.
//! * the optional long-side resize uses bilinear `align_corners=False` like
//!   `kornia.geometry.transform.resize`, but does not apply kornia's
//!   antialias Gaussian prefilter (reference `ImagePreprocessor.antialias`).
//!   The official golden test calls `extract(..., resize=None)`, so the
//!   parity contract is exact for the detector itself; the resize path is
//!   covered by a weights-gated smoke test.

use std::path::Path;

use candle_core::{DType, Device, Module, Result as CandleResult, Tensor};
use candle_nn::{
    BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, ModuleT, VarBuilder, batch_norm, conv2d,
    conv2d_no_bias, ops::sigmoid,
};

use super::keypoints::{KeypointOptions, KeypointSet};
use crate::error::Lux3dError;

const C1: usize = 16;
const C2: usize = 32;
const C3: usize = 64;
const C4: usize = 128;
const DIM: usize = 128;
const PATCH_KERNEL: usize = 3;
const N_POS: usize = 16;
const PAD_DIVISOR: usize = 32;
const NMS_RADIUS: usize = 2;
const DKD_TEMPERATURE: f32 = 0.1;
const N_LIMIT_MAX: usize = 20_000;
/// PyTorch `F.normalize` default epsilon.
const NORMALIZE_EPS: f64 = 1e-12;
const SELU_ALPHA: f64 = 1.673_263_242_354_377_2;
const SELU_GAMMA: f64 = 1.050_700_987_355_480_5;

/// ALIKED-n16 detector and descriptor extractor.
#[derive(Debug)]
pub struct AlikedDetector {
    block1: ConvBlock,
    block2: ResBlock,
    block3: ResBlock,
    block4: ResBlock,
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv4: Conv2d,
    score_head: ScoreHead,
    desc_head: Sddh,
    device: Device,
    dtype: DType,
}

impl AlikedDetector {
    /// Load the converted official `aliked-n16` safetensors checkpoint.
    ///
    /// Tensor names are the LightGlue state-dict names; BatchNorm
    /// `num_batches_tracked` counters are not read.
    pub fn load(weights: &Path, device: &Device, dtype: DType) -> crate::Result<Self> {
        if !weights.is_file() {
            return Err(Lux3dError::MissingWeightFile {
                path: weights.to_path_buf(),
            });
        }
        // SAFETY: the mmap-backed builder's tensors are owned by the returned
        // model (candle tensors keep the mapping alive); the file is read-only.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device) }
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "failed to open ALIKED-n16 weights `{}`: {source}",
                    weights.display()
                ),
            })?;
        Self::load_vb(vb, device.clone(), dtype).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "failed to load ALIKED-n16 ({}): {source}",
                    weights.display()
                ),
            }
        })
    }

    fn load_vb(vb: VarBuilder, device: Device, dtype: DType) -> CandleResult<Self> {
        Ok(Self {
            block1: ConvBlock::load(3, C1, vb.pp("block1"))?,
            block2: ResBlock::load(C1, C2, ResConvKind::Plain, vb.pp("block2"))?,
            block3: ResBlock::load(C2, C3, ResConvKind::Deform, vb.pp("block3"))?,
            block4: ResBlock::load(C3, C4, ResConvKind::Deform, vb.pp("block4"))?,
            conv1: conv2d_no_bias(C1, DIM / 4, 1, Conv2dConfig::default(), vb.pp("conv1"))?,
            conv2: conv2d_no_bias(C2, DIM / 4, 1, Conv2dConfig::default(), vb.pp("conv2"))?,
            conv3: conv2d_no_bias(C3, DIM / 4, 1, Conv2dConfig::default(), vb.pp("conv3"))?,
            conv4: conv2d_no_bias(C4, DIM / 4, 1, Conv2dConfig::default(), vb.pp("conv4"))?,
            score_head: ScoreHead::load(vb.pp("score_head"))?,
            desc_head: Sddh::load(DIM, PATCH_KERNEL, N_POS, vb.pp("desc_head"))?,
            device,
            dtype,
        })
    }

    /// Detect keypoints with scores and descriptors on `rgb`.
    ///
    /// `rgb` is `[3, H, W]` or `[1, 3, H, W]` in `[0, 1]`. Returned keypoints
    /// are pixel-center coordinates in the original input image (the official
    /// `origin="center"` convention), matching `Extractor.scale_keypoints`
    /// after the optional `KeypointOptions::resize_long_side` resize.
    pub fn detect(&self, rgb: &Tensor, options: &KeypointOptions) -> crate::Result<KeypointSet> {
        self.detect_impl(rgb, options)
            .map_err(|source| Lux3dError::ContractValidation {
                message: format!("ALIKED-n16 detection failed: {source}"),
            })
    }

    fn detect_impl(&self, rgb: &Tensor, options: &KeypointOptions) -> CandleResult<KeypointSet> {
        let image = normalize_input(rgb, &self.device, self.dtype)?;
        let (image, scales) = resize_long_side(&image, options.resize_long_side)?;
        let (feature_map, score_map) = self.forward_dense(&image)?;
        let keypoints = dkd(
            &score_map,
            options.detection_threshold,
            if options.max_keypoints > 0 {
                options.max_keypoints
            } else {
                N_LIMIT_MAX
            },
            options.detection_threshold <= 0.0 && options.max_keypoints > 0,
        )?;
        if keypoints.is_empty() {
            return Ok(KeypointSet::default());
        }
        let descriptors = self.describe(&feature_map, &keypoints)?;
        let mut set = KeypointSet {
            keypoints: Vec::with_capacity(keypoints.len()),
            scores: Vec::with_capacity(keypoints.len()),
            descriptors,
            descriptor_dim: DIM,
        };
        // `Extractor.scale_keypoints` with `origin="center"`; the reference
        // applies it even when the scale is 1 (resize disabled).
        let (sx, sy) = scales.unwrap_or((1.0, 1.0));
        for kp in &keypoints {
            let (x, y) = ((kp.px + 0.5) / sx - 0.5, (kp.py + 0.5) / sy - 0.5);
            set.keypoints.push([x, y]);
            set.scores.push(kp.score);
        }
        Ok(set)
    }

    /// `ALIKED.extract_dense_map`: returns the unpadded `[1, 128, H, W]`
    /// L2-normalized feature map and `[1, 1, H, W]` sigmoid score map.
    fn forward_dense(&self, image: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let (_, _, height, width) = image.dims4()?;
        let pad_h = (PAD_DIVISOR - height % PAD_DIVISOR) % PAD_DIVISOR;
        let pad_w = (PAD_DIVISOR - width % PAD_DIVISOR) % PAD_DIVISOR;
        let pad_top = pad_h / 2;
        let pad_left = pad_w / 2;
        let padded = image
            .pad_with_same(2, pad_top, pad_h - pad_top)?
            .pad_with_same(3, pad_left, pad_w - pad_left)?;
        let (_, _, padded_h, padded_w) = padded.dims4()?;

        let x1 = self.block1.forward(&padded)?;
        let x2 = self.block2.forward(&x1.avg_pool2d_with_stride(2, 2)?)?;
        let x3 = self.block3.forward(&x2.avg_pool2d_with_stride(4, 4)?)?;
        let x4 = self.block4.forward(&x3.avg_pool2d_with_stride(4, 4)?)?;

        let a1 = selu(&self.conv1.forward(&x1)?)?;
        let a2 = selu(&self.conv2.forward(&x2)?)?;
        let a3 = selu(&self.conv3.forward(&x3)?)?;
        let a4 = selu(&self.conv4.forward(&x4)?)?;
        let x2_up = a2.upsample_bilinear2d(padded_h, padded_w, true)?;
        let x3_up = a3.upsample_bilinear2d(padded_h, padded_w, true)?;
        let x4_up = a4.upsample_bilinear2d(padded_h, padded_w, true)?;
        let x1234 = Tensor::cat(&[&a1, &x2_up, &x3_up, &x4_up], 1)?;

        let score_map = sigmoid(&self.score_head.forward(&x1234)?)?;
        let feature_map = l2_normalize_channels(&x1234)?;
        let feature_map = feature_map
            .narrow(2, pad_top, height)?
            .narrow(3, pad_left, width)?
            .contiguous()?;
        let score_map = score_map
            .narrow(2, pad_top, height)?
            .narrow(3, pad_left, width)?
            .contiguous()?;
        Ok((feature_map, score_map))
    }

    /// `SDDH.forward`: descriptors for normalized keypoints.
    fn describe(
        &self,
        feature_map: &Tensor,
        keypoints: &[DetectedKeypoint],
    ) -> CandleResult<Vec<f32>> {
        let (_, channels, height, width) = feature_map.dims4()?;
        if height < PATCH_KERNEL || width < PATCH_KERNEL {
            candle_core::bail!(
                "ALIKED descriptor head needs a feature map of at least {PATCH_KERNEL}x{PATCH_KERNEL}, got {height}x{width}"
            );
        }
        let n = keypoints.len();
        let device = feature_map.device();
        let dtype = feature_map.dtype();

        // kptsi_wh = (kp / 2 + 0.5) * wh, in pixel-center coordinates.
        let mut kp_x = Vec::with_capacity(n);
        let mut kp_y = Vec::with_capacity(n);
        for kp in keypoints {
            kp_x.push(kp.px_norm / 2.0 + 0.5);
            kp_y.push(kp.py_norm / 2.0 + 0.5);
        }
        let kp_x: Vec<f32> = kp_x.iter().map(|v| v * (width - 1) as f32).collect();
        let kp_y: Vec<f32> = kp_y.iter().map(|v| v * (height - 1) as f32).collect();

        // get_patches: corner = (kptsi_wh.long() - ps/2 + 1).long(), clamped.
        let max_corner_x = (width - PATCH_KERNEL) as i64;
        let max_corner_y = (height - PATCH_KERNEL) as i64;
        let mut corner_x = Vec::with_capacity(n);
        let mut corner_y = Vec::with_capacity(n);
        for (&x, &y) in kp_x.iter().zip(kp_y.iter()) {
            let ix = x as i64;
            let iy = y as i64;
            let cx = ((ix as f32) - PATCH_KERNEL as f32 / 2.0 + 1.0) as i64;
            let cy = ((iy as f32) - PATCH_KERNEL as f32 / 2.0 + 1.0) as i64;
            corner_x.push(cx.clamp(0, max_corner_x) as u32);
            corner_y.push(cy.clamp(0, max_corner_y) as u32);
        }

        let feature_flat = feature_map.reshape((1, channels, height * width))?;
        // [N, C, 3, 3] patches, row-major over (ky, kx) like `get_patches`.
        let mut patch_values = Vec::with_capacity(PATCH_KERNEL * PATCH_KERNEL);
        for ky in 0..PATCH_KERNEL as u32 {
            for kx in 0..PATCH_KERNEL as u32 {
                let mut indices = Vec::with_capacity(n);
                for i in 0..n {
                    let y = corner_y[i] + ky;
                    let x = corner_x[i] + kx;
                    indices.push(y * width as u32 + x);
                }
                let gathered = gather_pixels(&feature_flat, &indices, channels, device)?;
                patch_values.push(gathered.reshape((channels, n))?);
            }
        }
        let patch = Tensor::stack(&patch_values, 2)?
            .permute((1, 0, 2))?
            .reshape((n, channels, PATCH_KERNEL, PATCH_KERNEL))?;

        let max_offset = height.max(width) as f64 / 4.0;
        let raw = self.desc_head.offset(&patch, max_offset)?;
        let raw = raw.reshape((n, 2 * N_POS))?;

        let kp_x_t = Tensor::from_vec(kp_x, n, device)?.to_dtype(dtype)?;
        let kp_y_t = Tensor::from_vec(kp_y, n, device)?.to_dtype(dtype)?;
        let mut samples = Vec::with_capacity(N_POS);
        for pos in 0..N_POS {
            let off_x = raw.narrow(1, pos, 1)?.squeeze(1)?;
            let off_y = raw.narrow(1, N_POS + pos, 1)?.squeeze(1)?;
            let px = kp_x_t.add(&off_x)?;
            let py = kp_y_t.add(&off_y)?;
            samples.push(bilinear_sample(&feature_flat, &py, &px, height, width)?);
        }
        let features = Tensor::stack(&samples, 3)?
            .squeeze(0)?
            .permute((1, 0, 2))?
            .unsqueeze(3)?
            .to_dtype(dtype)?;
        let features = selu(&self.desc_head.sf_conv.forward(&features)?)?.squeeze(3)?;
        // einsum("ncp,pcd->nd").
        let features = features
            .permute((0, 2, 1))?
            .reshape((n, N_POS * channels))?;
        let agg = self
            .desc_head
            .agg_weights
            .reshape((N_POS * channels, DIM))?;
        let descs = features.matmul(&agg)?;
        let norm = descs
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(NORMALIZE_EPS, f64::MAX)?;
        let descs = descs.broadcast_div(&norm)?;
        let descs = descs
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .contiguous()?;
        descs.flatten_all()?.to_vec1::<f32>()
    }
}

fn normalize_input(rgb: &Tensor, device: &Device, dtype: DType) -> CandleResult<Tensor> {
    let image = match rgb.rank() {
        3 => rgb.unsqueeze(0)?,
        4 => rgb.clone(),
        rank => candle_core::bail!("ALIKED input must be [3,H,W] or [1,3,H,W], got rank {rank}"),
    };
    let (batch, channels, _, _) = image.dims4()?;
    if batch != 1 {
        candle_core::bail!("ALIKED input batch must be 1, got {batch}");
    }
    if channels != 3 {
        candle_core::bail!("ALIKED input must have 3 channels, got {channels}");
    }
    let image = image.to_device(device)?;
    image.to_dtype(dtype)
}

/// `ImagePreprocessor` long-side resize. Returns the resized image and the
/// `(scale_x, scale_y)` used by `Extractor.scale_keypoints`.
fn resize_long_side(
    image: &Tensor,
    target: Option<usize>,
) -> CandleResult<(Tensor, Option<(f32, f32)>)> {
    let Some(target) = target else {
        return Ok((image.clone(), None));
    };
    let (_, _, height, width) = image.dims4()?;
    let longest = height.max(width);
    if target == 0 {
        candle_core::bail!("ALIKED resize_long_side must be positive");
    }
    if longest == target {
        return Ok((image.clone(), None));
    }
    // kornia `side="long"` truncates the short side to an integer.
    let scale = target as f64 / longest as f64;
    let new_height = ((height as f64 * scale).floor() as usize).max(1);
    let new_width = ((width as f64 * scale).floor() as usize).max(1);
    let resized = image.upsample_bilinear2d(new_height, new_width, false)?;
    let scales = (
        new_width as f32 / width as f32,
        new_height as f32 / height as f32,
    );
    Ok((resized, Some(scales)))
}

// ---------------------------------------------------------------------------
// Backbone
// ---------------------------------------------------------------------------

/// Reference `ConvBlock`: conv3x3 -> BN -> SELU, twice.
#[derive(Debug)]
struct ConvBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
}

impl ConvBlock {
    fn load(in_channels: usize, out_channels: usize, vb: VarBuilder) -> CandleResult<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            conv1: conv2d_no_bias(in_channels, out_channels, 3, cfg, vb.pp("conv1"))?,
            bn1: batch_norm(out_channels, BatchNormConfig::default(), vb.pp("bn1"))?,
            conv2: conv2d_no_bias(out_channels, out_channels, 3, cfg, vb.pp("conv2"))?,
            bn2: batch_norm(out_channels, BatchNormConfig::default(), vb.pp("bn2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x = selu(&self.bn1.forward_t(&self.conv1.forward(x)?, false)?)?;
        selu(&self.bn2.forward_t(&self.conv2.forward(&x)?, false)?)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResConvKind {
    Plain,
    Deform,
}

/// Reference `ResBlock`: conv/BN/SELU, conv/BN, identity (1x1 conv), add,
/// SELU.
#[derive(Debug)]
struct ResBlock {
    conv1: ResConv,
    bn1: BatchNorm,
    conv2: ResConv,
    bn2: BatchNorm,
    downsample: Conv2d,
}

impl ResBlock {
    fn load(
        in_channels: usize,
        out_channels: usize,
        kind: ResConvKind,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        Ok(Self {
            conv1: ResConv::load(in_channels, out_channels, kind, vb.pp("conv1"))?,
            bn1: batch_norm(out_channels, BatchNormConfig::default(), vb.pp("bn1"))?,
            conv2: ResConv::load(out_channels, out_channels, kind, vb.pp("conv2"))?,
            bn2: batch_norm(out_channels, BatchNormConfig::default(), vb.pp("bn2"))?,
            downsample: conv2d(
                in_channels,
                out_channels,
                1,
                Conv2dConfig::default(),
                vb.pp("downsample"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let out = self.conv1.forward(x)?;
        let out = selu(&self.bn1.forward_t(&out, false)?)?;
        let out = self.conv2.forward(&out)?;
        let out = self.bn2.forward_t(&out, false)?;
        let identity = self.downsample.forward(x)?;
        selu(&out.add(&identity)?)
    }
}

#[derive(Debug)]
enum ResConv {
    Plain(Conv2d),
    Deform(DeformConv),
}

impl ResConv {
    fn load(
        in_channels: usize,
        out_channels: usize,
        kind: ResConvKind,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        match kind {
            ResConvKind::Plain => Ok(Self::Plain(conv2d_no_bias(
                in_channels,
                out_channels,
                3,
                Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vb,
            )?)),
            ResConvKind::Deform => Ok(Self::Deform(DeformConv::load(
                in_channels,
                out_channels,
                3,
                1,
                false,
                vb,
            )?)),
        }
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Plain(conv) => conv.forward(x),
            Self::Deform(conv) => conv.forward(x),
        }
    }
}

// ---------------------------------------------------------------------------
// Deformable convolution (torchvision `deform_conv2d`, no mask)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct DeformConv {
    offset_conv: Conv2d,
    weight: Tensor,
    bias: Option<Tensor>,
    kernel: usize,
    padding: usize,
    stride: usize,
}

impl DeformConv {
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        padding: usize,
        bias: bool,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let cfg = Conv2dConfig {
            padding,
            stride: 1,
            ..Default::default()
        };
        let offset_conv = conv2d(
            in_channels,
            2 * kernel * kernel,
            kernel,
            cfg,
            vb.pp("offset_conv"),
        )?;
        let reg = vb.pp("regular_conv");
        let weight = reg.get((out_channels, in_channels, kernel, kernel), "weight")?;
        let bias = if bias {
            Some(reg.get(out_channels, "bias")?)
        } else {
            None
        };
        Ok(Self {
            offset_conv,
            weight,
            bias,
            kernel,
            padding,
            stride: 1,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (b, c, ih, iw) = x.dims4()?;
        let max_offset = ih.max(iw) as f64 / 4.0;
        let offset = self
            .offset_conv
            .forward(x)?
            .clamp(-max_offset, max_offset)?;
        let (_, _, oh, ow) = offset.dims4()?;
        let n = oh * ow;
        let k2 = self.kernel * self.kernel;

        let x_flat = x.reshape((b, c, ih * iw))?;
        let mut out: Option<Tensor> = None;
        for ki in 0..k2 {
            let kh = ki / self.kernel;
            let kw = ki % self.kernel;
            let y_base = base_coords(
                oh,
                ow,
                self.stride,
                self.padding,
                kh,
                true,
                x.device(),
                x.dtype(),
            )?;
            let x_base = base_coords(
                oh,
                ow,
                self.stride,
                self.padding,
                kw,
                false,
                x.device(),
                x.dtype(),
            )?;
            let dy = offset
                .narrow(1, 2 * ki, 1)?
                .narrow(2, 0, oh)?
                .narrow(3, 0, ow)?
                .reshape((b, n))?;
            let dx = offset
                .narrow(1, 2 * ki + 1, 1)?
                .narrow(2, 0, oh)?
                .narrow(3, 0, ow)?
                .reshape((b, n))?;
            let y = dy.broadcast_add(&y_base)?;
            let xc = dx.broadcast_add(&x_base)?;
            let (iy0, iy1, wy0, wy1) = bilinear_axis(y.as_ref(), ih)?;
            let (ix0, ix1, wx0, wx1) = bilinear_axis(xc.as_ref(), iw)?;

            let i00 = flat_index(&iy0, &ix0, iw)?;
            let i01 = flat_index(&iy0, &ix1, iw)?;
            let i10 = flat_index(&iy1, &ix0, iw)?;
            let i11 = flat_index(&iy1, &ix1, iw)?;

            let v00 = gather_flat(&x_flat, &i00, b, c, n)?.reshape((b, c, oh, ow))?;
            let v01 = gather_flat(&x_flat, &i01, b, c, n)?.reshape((b, c, oh, ow))?;
            let v10 = gather_flat(&x_flat, &i10, b, c, n)?.reshape((b, c, oh, ow))?;
            let v11 = gather_flat(&x_flat, &i11, b, c, n)?.reshape((b, c, oh, ow))?;

            let weight_shape = (b, 1, oh, ow);
            let w00 = wy0.broadcast_mul(&wx0)?.reshape(weight_shape)?;
            let w01 = wy0.broadcast_mul(&wx1)?.reshape(weight_shape)?;
            let w10 = wy1.broadcast_mul(&wx0)?.reshape(weight_shape)?;
            let w11 = wy1.broadcast_mul(&wx1)?.reshape(weight_shape)?;
            let sampled = v00
                .broadcast_mul(&w00)?
                .add(&v01.broadcast_mul(&w01)?)?
                .add(&v10.broadcast_mul(&w10)?)?
                .add(&v11.broadcast_mul(&w11)?)?;

            let weight_k = self
                .weight
                .narrow(2, kh, 1)?
                .narrow(3, kw, 1)?
                .contiguous()?;
            let term = sampled.conv2d(&weight_k, 0, 1, 1, 1)?;
            out = Some(match out {
                Some(acc) => acc.add(&term)?,
                None => term,
            });
        }
        let mut out = out.expect("deformable convolution kernel is non-empty");
        if let Some(bias) = &self.bias {
            out = out.broadcast_add(&bias.reshape((1, bias.dim(0)?, 1, 1))?)?;
        }
        Ok(out)
    }
}

#[allow(clippy::too_many_arguments)]
fn base_coords(
    oh: usize,
    ow: usize,
    stride: usize,
    padding: usize,
    kernel_pos: usize,
    is_y: bool,
    device: &Device,
    dtype: DType,
) -> CandleResult<Tensor> {
    let mut values = Vec::with_capacity(oh * ow);
    for oy in 0..oh {
        for ox in 0..ow {
            let base = if is_y { oy * stride } else { ox * stride };
            values.push(base as f32 - padding as f32 + kernel_pos as f32);
        }
    }
    Tensor::from_vec(values, (1, oh * ow), device)?.to_dtype(dtype)
}

/// Integer low/high indices and bilinear weights along one axis following
/// torchvision `deform_conv2d` / `grid_sample` zero padding: a sample outside
/// `[0, len)` contributes a zero weight, while `(-1, 0)` and `(len-1, len)`
/// interpolate against a virtual zero neighbour.
fn bilinear_axis(coord: &Tensor, len: usize) -> CandleResult<(Tensor, Tensor, Tensor, Tensor)> {
    let low = coord.floor()?;
    let frac = coord.sub(&low)?;
    let max = len.saturating_sub(1) as f64;
    let low_idx = low.clamp(0.0, max)?.to_dtype(DType::U32)?;
    let high_idx = low
        .affine(1.0, 1.0)?
        .clamp(0.0, max)?
        .to_dtype(DType::U32)?;
    let low_valid = in_range(&low, len)?;
    let high_valid = in_range(&low.affine(1.0, 1.0)?, len)?;
    let w_low = frac.affine(-1.0, 1.0)?.broadcast_mul(&low_valid)?;
    let w_high = frac.broadcast_mul(&high_valid)?;
    Ok((low_idx, high_idx, w_low, w_high))
}

fn flat_index(iy: &Tensor, ix: &Tensor, width: usize) -> CandleResult<Tensor> {
    let row = iy.to_dtype(DType::F32)?.affine(width as f64, 0.0)?;
    row.add(&ix.to_dtype(DType::F32)?)?.to_dtype(DType::U32)
}

fn gather_flat(
    x_flat: &Tensor,
    index: &Tensor,
    b: usize,
    c: usize,
    n: usize,
) -> CandleResult<Tensor> {
    let index = index
        .reshape((b, 1, n))?
        .broadcast_as((b, c, n))?
        .contiguous()?;
    x_flat.gather(&index, 2)
}

fn in_range(values: &Tensor, len: usize) -> CandleResult<Tensor> {
    let dtype = values.dtype();
    values
        .ge(0.0)?
        .to_dtype(dtype)?
        .broadcast_mul(&values.lt(len as f64)?.to_dtype(dtype)?)
}

/// Bilinearly samples `feature_flat` (`[1, C, H*W]`) at pixel coordinates
/// `(px, py)` with `align_corners=True` / zero padding semantics.
fn bilinear_sample(
    feature_flat: &Tensor,
    py: &Tensor,
    px: &Tensor,
    height: usize,
    width: usize,
) -> CandleResult<Tensor> {
    let channels = feature_flat.dim(1)?;
    let n = py.elem_count();
    let (iy0, iy1, wy0, wy1) = bilinear_axis(py, height)?;
    let (ix0, ix1, wx0, wx1) = bilinear_axis(px, width)?;
    let i00 = flat_index(&iy0, &ix0, width)?;
    let i01 = flat_index(&iy0, &ix1, width)?;
    let i10 = flat_index(&iy1, &ix0, width)?;
    let i11 = flat_index(&iy1, &ix1, width)?;
    let v00 = gather_flat(feature_flat, &i00, 1, channels, n)?;
    let v01 = gather_flat(feature_flat, &i01, 1, channels, n)?;
    let v10 = gather_flat(feature_flat, &i10, 1, channels, n)?;
    let v11 = gather_flat(feature_flat, &i11, 1, channels, n)?;
    let shape = (1, 1, n);
    let w00 = wy0.broadcast_mul(&wx0)?.reshape(shape)?;
    let w01 = wy0.broadcast_mul(&wx1)?.reshape(shape)?;
    let w10 = wy1.broadcast_mul(&wx0)?.reshape(shape)?;
    let w11 = wy1.broadcast_mul(&wx1)?.reshape(shape)?;
    v00.broadcast_mul(&w00)?
        .add(&v01.broadcast_mul(&w01)?)?
        .add(&v10.broadcast_mul(&w10)?)?
        .add(&v11.broadcast_mul(&w11)?)
}

/// Integer-pixel gather for the `get_patches` corner windows.
fn gather_pixels(
    feature_flat: &Tensor,
    indices: &[u32],
    channels: usize,
    device: &Device,
) -> CandleResult<Tensor> {
    let n = indices.len();
    let index = Tensor::from_vec(indices.to_vec(), (1, n), device)?.to_dtype(DType::U32)?;
    gather_flat(feature_flat, &index, 1, channels, n)
}

// ---------------------------------------------------------------------------
// Heads
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct ScoreHead {
    conv0: Conv2d,
    conv2: Conv2d,
    conv4: Conv2d,
    conv6: Conv2d,
}

impl ScoreHead {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        let cfg3 = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            conv0: conv2d_no_bias(DIM, 8, 1, Conv2dConfig::default(), vb.pp("0"))?,
            conv2: conv2d_no_bias(8, 4, 3, cfg3, vb.pp("2"))?,
            conv4: conv2d_no_bias(4, 4, 3, cfg3, vb.pp("4"))?,
            conv6: conv2d_no_bias(4, 1, 3, cfg3, vb.pp("6"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x = selu(&self.conv0.forward(x)?)?;
        let x = selu(&self.conv2.forward(&x)?)?;
        let x = selu(&self.conv4.forward(&x)?)?;
        self.conv6.forward(&x)
    }
}

#[derive(Debug)]
struct Sddh {
    offset0: Conv2d,
    offset2: Conv2d,
    sf_conv: Conv2d,
    agg_weights: Tensor,
}

impl Sddh {
    fn load(dims: usize, kernel: usize, n_pos: usize, vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            offset0: conv2d(
                dims,
                2 * n_pos,
                kernel,
                Conv2dConfig::default(),
                vb.pp("offset_conv").pp("0"),
            )?,
            offset2: conv2d(
                2 * n_pos,
                2 * n_pos,
                1,
                Conv2dConfig::default(),
                vb.pp("offset_conv").pp("2"),
            )?,
            sf_conv: conv2d_no_bias(dims, dims, 1, Conv2dConfig::default(), vb.pp("sf_conv"))?,
            agg_weights: vb.get((n_pos, dims, dims), "agg_weights")?,
        })
    }

    /// `offset_conv` Sequential (conv3x3 -> SELU -> conv1x1) plus the
    /// `max(h,w)/4` clamp; returns `[N, 2*n_pos]`.
    fn offset(&self, patch: &Tensor, max_offset: f64) -> CandleResult<Tensor> {
        let x = selu(&self.offset0.forward(patch)?)?;
        self.offset2
            .forward(&x)?
            .clamp(-max_offset, max_offset)?
            .reshape((patch.dim(0)?, 2 * N_POS))
    }
}

// ---------------------------------------------------------------------------
// Detection (DKD)
// ---------------------------------------------------------------------------

struct DetectedKeypoint {
    /// Normalized `[-1, 1]` coordinates (DKD output).
    px_norm: f32,
    py_norm: f32,
    /// Pixel-center coordinates in the resized detector image.
    px: f32,
    py: f32,
    score: f32,
}

/// Reference `DKD.forward` with `sub_pixel=True` and `image_size=None`
/// (LightGlue `Extractor.extract` never passes `image_size`).
fn dkd(
    score_map: &Tensor,
    detection_threshold: f32,
    n_limit: usize,
    top_k_mode: bool,
) -> CandleResult<Vec<DetectedKeypoint>> {
    let (_, _, height, width) = score_map.dims4()?;
    let scores = score_map
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut nms = simple_nms(&scores, height, width, NMS_RADIUS);
    zero_border(&mut nms, height, width, NMS_RADIUS);

    let indices: Vec<usize> = if top_k_mode && n_limit > 0 {
        let mut indices: Vec<usize> = (0..nms.len()).collect();
        indices.sort_by(|&a, &b| nms[b].total_cmp(&nms[a]));
        indices.truncate(n_limit.min(nms.len()));
        indices
    } else {
        let threshold = if detection_threshold > 0.0 {
            detection_threshold
        } else {
            scores.iter().copied().sum::<f32>() / scores.len().max(1) as f32
        };
        let mut indices: Vec<usize> = nms
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| (value > threshold).then_some(index))
            .collect();
        if indices.is_empty() && detection_threshold > 0.0 {
            // Reference fallback when nothing passes the absolute threshold.
            let mean = scores.iter().copied().sum::<f32>() / scores.len().max(1) as f32;
            indices = nms
                .iter()
                .enumerate()
                .filter_map(|(index, &value)| (value > mean).then_some(index))
                .collect();
        }
        if indices.len() > n_limit {
            indices.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
            indices.truncate(n_limit);
        }
        indices
    };

    let mut keypoints = Vec::with_capacity(indices.len());
    for index in indices {
        let x = (index % width) as f32;
        let y = (index / width) as f32;
        let mut patch = [0.0_f32; 25];
        let mut max_value = f32::NEG_INFINITY;
        for ky in 0..5 {
            for kx in 0..5 {
                let yy = y as i64 + ky as i64 - 2;
                let xx = x as i64 + kx as i64 - 2;
                if yy >= 0 && yy < height as i64 && xx >= 0 && xx < width as i64 {
                    let value = scores[yy as usize * width + xx as usize];
                    patch[ky * 5 + kx] = value;
                    max_value = max_value.max(value);
                }
            }
        }
        let mut sum = 0.0_f32;
        let mut res_x = 0.0_f32;
        let mut res_y = 0.0_f32;
        for ky in 0..5 {
            for kx in 0..5 {
                let weight = ((patch[ky * 5 + kx] - max_value) / DKD_TEMPERATURE).exp();
                sum += weight;
                res_x += weight * (kx as f32 - 2.0);
                res_y += weight * (ky as f32 - 2.0);
            }
        }
        let (res_x, res_y) = if sum > 0.0 {
            (res_x / sum, res_y / sum)
        } else {
            (0.0, 0.0)
        };
        let px = x + res_x;
        let py = y + res_y;
        let px_norm = px / (width - 1) as f32 * 2.0 - 1.0;
        let py_norm = py / (height - 1) as f32 * 2.0 - 1.0;
        let score = sample_score(&scores, height, width, px_norm, py_norm);
        keypoints.push(DetectedKeypoint {
            px_norm,
            py_norm,
            px,
            py,
            score,
        });
    }
    Ok(keypoints)
}

/// `simple_nms` with the two suppression rounds of the reference.
fn simple_nms(scores: &[f32], height: usize, width: usize, radius: usize) -> Vec<f32> {
    let pooled = max_pool_same(scores, height, width, radius);
    let mut max_mask: Vec<bool> = scores
        .iter()
        .zip(pooled.iter())
        .map(|(&score, &pool)| score == pool)
        .collect();
    for _ in 0..2 {
        let supp_mask: Vec<bool> = max_mask
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let y = index / width;
                let x = index % width;
                any_in_window(&max_mask, height, width, y, x, radius, value)
            })
            .collect();
        let supp_scores: Vec<f32> = scores
            .iter()
            .zip(supp_mask.iter())
            .map(|(&score, &supp)| if supp { 0.0 } else { score })
            .collect();
        let pooled_supp = max_pool_same(&supp_scores, height, width, radius);
        let new_max_mask: Vec<bool> = supp_scores
            .iter()
            .zip(pooled_supp.iter())
            .map(|(&score, &pool)| score == pool)
            .collect();
        for index in 0..max_mask.len() {
            max_mask[index] |= new_max_mask[index] && !supp_mask[index];
        }
    }
    scores
        .iter()
        .zip(max_mask.iter())
        .map(|(&score, &keep)| if keep { score } else { 0.0 })
        .collect()
}

/// `F.max_pool2d(kernel=2r+1, stride=1, padding=r)` with `-inf` padding.
fn max_pool_same(values: &[f32], height: usize, width: usize, radius: usize) -> Vec<f32> {
    let mut out = vec![f32::NEG_INFINITY; height * width];
    for y in 0..height {
        for x in 0..width {
            let mut best = f32::NEG_INFINITY;
            let y0 = y.saturating_sub(radius);
            let y1 = (y + radius).min(height - 1);
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius).min(width - 1);
            for yy in y0..=y1 {
                for xx in x0..=x1 {
                    best = best.max(values[yy * width + xx]);
                }
            }
            out[y * width + x] = best;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn any_in_window(
    mask: &[bool],
    height: usize,
    width: usize,
    y: usize,
    x: usize,
    radius: usize,
    current: bool,
) -> bool {
    if current {
        return true;
    }
    let y0 = y.saturating_sub(radius);
    let y1 = (y + radius).min(height - 1);
    let x0 = x.saturating_sub(radius);
    let x1 = (x + radius).min(width - 1);
    for yy in y0..=y1 {
        for xx in x0..=x1 {
            if mask[yy * width + xx] {
                return true;
            }
        }
    }
    false
}

fn zero_border(values: &mut [f32], height: usize, width: usize, radius: usize) {
    for y in 0..height {
        if y < radius || y >= height.saturating_sub(radius) {
            values[y * width..(y + 1) * width].fill(0.0);
        } else {
            for x in 0..width {
                if x < radius || x >= width.saturating_sub(radius) {
                    values[y * width + x] = 0.0;
                }
            }
        }
    }
}

/// `grid_sample` of the score map at normalized keypoints
/// (`align_corners=True`, zero padding).
fn sample_score(scores: &[f32], height: usize, width: usize, px_norm: f32, py_norm: f32) -> f32 {
    // torch grid_sampler_compute_source_index, align_corners=True.
    let px = (px_norm + 1.0) / 2.0 * (width - 1) as f32;
    let py = (py_norm + 1.0) / 2.0 * (height - 1) as f32;
    let x0 = px.floor();
    let y0 = py.floor();
    let fx = px - x0;
    let fy = py - y0;
    let mut value = 0.0_f32;
    for (dy, wy) in [(0, 1.0 - fy), (1, fy)] {
        for (dx, wx) in [(0, 1.0 - fx), (1, fx)] {
            let yy = y0 + dy as f32;
            let xx = x0 + dx as f32;
            if yy >= 0.0 && yy < height as f32 && xx >= 0.0 && xx < width as f32 {
                value += scores[yy as usize * width + xx as usize] * wy * wx;
            }
        }
    }
    value
}

// ---------------------------------------------------------------------------
// Activation / normalization helpers
// ---------------------------------------------------------------------------

/// `nn.SELU`: `gamma * (max(0, x) + min(0, alpha * (exp(x) - 1)))`.
fn selu(x: &Tensor) -> CandleResult<Tensor> {
    let zeros = x.zeros_like()?;
    let positive = x.maximum(&zeros)?;
    let negative = x.exp()?.affine(SELU_ALPHA, -SELU_ALPHA)?.minimum(&zeros)?;
    positive.add(&negative)?.affine(SELU_GAMMA, 0.0)
}

/// `F.normalize(x, p=2, dim=1)` with the reference epsilon.
fn l2_normalize_channels(x: &Tensor) -> CandleResult<Tensor> {
    let norm = x
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .clamp(NORMALIZE_EPS, f64::MAX)?;
    x.broadcast_div(&norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    const FIXTURE: &str = "tests/fixtures/keypoints/aliked_n16_256.npz";
    const WEIGHTS: &str = "../../models/aliked-superpoint/aliked-n16.safetensors";

    fn read_f32(
        npz: &mut ndarray_npy::NpzReader<std::fs::File>,
        name: &str,
    ) -> ndarray::ArrayD<f32> {
        npz.by_name(name)
            .unwrap_or_else(|err| panic!("missing golden tensor {name}: {err}"))
    }

    fn f32_vec(array: ndarray::ArrayD<f32>) -> Vec<f32> {
        let (data, _) = array.into_raw_vec_and_offset();
        data
    }

    fn official_options() -> KeypointOptions {
        KeypointOptions {
            max_keypoints: 4096,
            detection_threshold: 0.005,
            resize_long_side: None,
        }
    }

    /// Official LightGlue CPU golden produced by
    /// `scripts/gen_aliked_golden.py`:
    /// `ALIKED(model_name="aliked-n16", max_num_keypoints=4096,
    /// detection_threshold=0.005).eval().extract(image, resize=None)`.
    #[test]
    fn aliked_n16_matches_official_golden() {
        let fixture = Path::new(FIXTURE);
        let weights = Path::new(WEIGHTS);
        if !fixture.is_file() || !weights.is_file() {
            eprintln!(
                "skipping ALIKED official golden: fixture ({}) or weights ({}) missing",
                fixture.display(),
                weights.display()
            );
            return;
        }
        let mut npz = ndarray_npy::NpzReader::new(std::fs::File::open(fixture).unwrap()).unwrap();
        let image: ndarray::Array3<u8> = npz.by_name("image").expect("golden image");
        let golden_keypoints = f32_vec(read_f32(&mut npz, "keypoints"));
        let golden_scores = f32_vec(read_f32(&mut npz, "keypoint_scores"));
        let golden_descriptors = f32_vec(read_f32(&mut npz, "descriptors"));
        let n = golden_scores.len();
        assert_eq!(golden_keypoints.len(), 2 * n);
        assert_eq!(golden_descriptors.len(), n * DIM);

        let shape = image.shape().to_vec();
        let (channels, height, width) = (shape[0], shape[1], shape[2]);
        let values: Vec<f32> = image
            .into_raw_vec_and_offset()
            .0
            .into_iter()
            .map(|value| f32::from(value) / 255.0)
            .collect();
        let input = Tensor::from_vec(values, (1, channels, height, width), &Device::Cpu).unwrap();

        let detector = AlikedDetector::load(weights, &Device::Cpu, DType::F32).unwrap();
        let got = detector.detect(&input, &official_options()).unwrap();
        assert_eq!(
            got.len(),
            n,
            "keypoint count differs from the official golden"
        );
        assert_eq!(got.descriptor_dim, DIM);
        assert_eq!(got.descriptors.len(), n * DIM);

        // Compare in the official order first; fall back to the sorted set when
        // the ordering differs, as required by the parity brief.
        let same_order = (0..n).all(|i| {
            let dx = got.keypoints[i][0] - golden_keypoints[2 * i];
            let dy = got.keypoints[i][1] - golden_keypoints[2 * i + 1];
            dx.abs() <= 1e-3 && dy.abs() <= 1e-3
        });
        if same_order {
            let mut worst_kp = 0.0_f32;
            let mut worst_score = 0.0_f32;
            let mut worst_cosine = 1.0_f32;
            for i in 0..n {
                worst_kp = worst_kp.max(
                    (got.keypoints[i][0] - golden_keypoints[2 * i])
                        .abs()
                        .max((got.keypoints[i][1] - golden_keypoints[2 * i + 1]).abs()),
                );
                worst_score = worst_score.max((got.scores[i] - golden_scores[i]).abs());
                let cosine = cosine(
                    &got.descriptors[i * DIM..(i + 1) * DIM],
                    &golden_descriptors[i * DIM..(i + 1) * DIM],
                );
                worst_cosine = worst_cosine.min(cosine);
                assert!(
                    (got.scores[i] - golden_scores[i]).abs() <= 1e-4,
                    "score mismatch at {i}: {} vs {}",
                    got.scores[i],
                    golden_scores[i]
                );
                assert!(
                    1.0 - cosine <= 1e-3,
                    "descriptor cosine mismatch at {i}: {cosine}"
                );
            }
            eprintln!(
                "ALIKED golden: order and all {n} points match; worst kp={worst_kp:e}, score={worst_score:e}, cosine={worst_cosine:e}"
            );
        } else {
            let mut got_order: Vec<usize> = (0..n).collect();
            let mut want_order: Vec<usize> = (0..n).collect();
            got_order.sort_by(|&a, &b| {
                got.keypoints[a][0]
                    .total_cmp(&got.keypoints[b][0])
                    .then(got.keypoints[a][1].total_cmp(&got.keypoints[b][1]))
            });
            want_order.sort_by(|&a, &b| {
                golden_keypoints[2 * a]
                    .total_cmp(&golden_keypoints[2 * b])
                    .then(golden_keypoints[2 * a + 1].total_cmp(&golden_keypoints[2 * b + 1]))
            });
            let mut worst_kp = 0.0_f32;
            let mut worst_score = 0.0_f32;
            let mut worst_cosine = 1.0_f32;
            for (&want, &got_index) in want_order.iter().zip(got_order.iter()) {
                worst_kp = worst_kp.max(
                    (got.keypoints[got_index][0] - golden_keypoints[2 * want])
                        .abs()
                        .max((got.keypoints[got_index][1] - golden_keypoints[2 * want + 1]).abs()),
                );
                worst_score = worst_score.max((got.scores[got_index] - golden_scores[want]).abs());
                worst_cosine = worst_cosine.min(cosine(
                    &got.descriptors[got_index * DIM..(got_index + 1) * DIM],
                    &golden_descriptors[want * DIM..(want + 1) * DIM],
                ));
            }
            eprintln!(
                "ALIKED golden sorted-set worst: kp={worst_kp:e}, score={worst_score:e}, cosine={worst_cosine:e}"
            );
            assert!(worst_kp <= 1e-3, "sorted keypoint drift {worst_kp}");
            assert!(worst_score <= 1e-4, "sorted score drift {worst_score}");
            assert!(
                1.0 - worst_cosine <= 1e-3,
                "sorted cosine drift {worst_cosine}"
            );
        }
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    /// Real-weights smoke test; skipped when the converted checkpoint is
    /// absent. Covers the resize path at an odd aspect ratio.
    #[test]
    fn aliked_n16_real_weights_128() {
        let weights = Path::new(WEIGHTS);
        if !weights.is_file() {
            eprintln!(
                "skipping ALIKED real-weights smoke test: {} missing",
                weights.display()
            );
            return;
        }
        let detector = AlikedDetector::load(weights, &Device::Cpu, DType::F32).unwrap();
        let (height, width) = (128_usize, 128_usize);
        let mut values = Vec::with_capacity(3 * height * width);
        for channel in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    // Deterministic in [0, 1] with hard edges for the score head.
                    let base = ((x * 13 + y * 7 + channel * 29) % 251) as f32 / 250.0;
                    let block = if ((x / 16) + (y / 16)) % 2 == 0 {
                        0.8
                    } else {
                        0.2
                    };
                    values.push(0.5 * base + 0.5 * block);
                }
            }
        }
        let input = Tensor::from_vec(values, (3, height, width), &Device::Cpu).unwrap();
        let got = detector.detect(&input, &official_options()).unwrap();
        assert!(
            !got.is_empty(),
            "ALIKED found no keypoints on the smoke image"
        );
        assert_eq!(got.descriptor_dim, DIM);
        assert_eq!(got.descriptors.len(), got.len() * DIM);
        for (index, keypoint) in got.keypoints.iter().enumerate() {
            assert!(keypoint[0].is_finite() && keypoint[1].is_finite());
            assert!(
                keypoint[0] >= -1e-3
                    && keypoint[0] <= width as f32 - 1.0 + 1e-3
                    && keypoint[1] >= -1e-3
                    && keypoint[1] <= height as f32 - 1.0 + 1e-3,
                "keypoint {index} out of bounds: {keypoint:?}"
            );
            assert!(got.scores[index].is_finite());
        }
        for row in got.descriptors.chunks_exact(DIM) {
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() <= 1e-4, "descriptor norm {norm}");
        }

        // Resize path: 160x96 input resized to long side 128 -> 128x76.
        let (height, width) = (96_usize, 160_usize);
        let mut values = Vec::with_capacity(3 * height * width);
        for channel in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    values.push(((x * 5 + y * 11 + channel * 3) % 97) as f32 / 96.0);
                }
            }
        }
        let input = Tensor::from_vec(values, (3, height, width), &Device::Cpu).unwrap();
        let options = KeypointOptions {
            resize_long_side: Some(128),
            ..official_options()
        };
        let got = detector.detect(&input, &options).unwrap();
        assert_eq!(got.descriptor_dim, DIM);
        for keypoint in &got.keypoints {
            assert!(
                keypoint[0] >= -0.5 - 1e-3
                    && keypoint[0] <= width as f32 - 1.0 + 1e-3
                    && keypoint[1] >= -0.5 - 1e-3
                    && keypoint[1] <= height as f32 - 1.0 + 1e-3,
                "resized keypoint out of original-image bounds: {keypoint:?}"
            );
        }
    }
}
