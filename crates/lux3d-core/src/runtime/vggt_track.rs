//! VGGT track head — candle port of `vggt.heads.track_head.TrackHead`.
//!
//! Two components, mirroring the upstream modules:
//!
//! 1. [`VggtTrackFeatureExtractor`] — `DPTHead(dim_in=2048, patch_size=14,
//!    features=128, feature_only=True, down_ratio=2, pos_embed=False)`.
//!    Consumes the four cached aggregator layers (4/11/17/23) and produces a
//!    `[B, S, 128, H/2, W/2]` feature map per frame. It reuses the exact
//!    point/depth DPT building blocks (`ResidualConvUnit`,
//!    `FeatureFusionBlock`, `ResizeLayer`) from `vggt_dpt`, which are
//!    unchanged, so the point/depth heads stay bit-identical. Differences
//!    from those heads: 128 scratch features, no positional embedding, no
//!    `output_conv2`/activation head (feature-only path), and a final
//!    bilinear upsample to `patch_grid * 14 / 2`.
//!
//! 2. [`VggtTrackerPredictor`] — `BaseTrackerPredictor(latent_dim=128,
//!    predict_conf=true, stride=2, corr_levels=7, corr_radius=4,
//!    hidden_size=384, use_spaceatt=True, depth=6, max_scale=518)`: fmap
//!    LayerNorm, correlation pyramid (7 avg-pool levels), local correlation
//!    sampling over a `(-4..4)` delta grid, iterative refinement through an
//!    `EfficientUpdateFormer` (6 time self-attention + 6 space
//!    virtual/point cross-attention blocks over 64 virtual tracks), linear
//!    visibility/confidence heads with sigmoid, and the per-iteration track
//!    feature update (`ffeat_norm` GroupNorm + `ffeat_updater`).
//!
//! # Coordinate conventions (must match upstream exactly)
//!
//! * Queries ([`VggtTrackQuery::xy`]) are **pixel coordinates in the model
//!   input resolution** (e.g. 518x518), `(x, y)` order, with `x` in
//!   `[0, W-1]` and `y` in `[0, H-1]`. The sampling primitives use
//!   `align_corners=True` semantics, so `0` is the center of the
//!   leftmost/topmost pixel and `W-1`/`H-1` the center of the
//!   rightmost/bottom-most pixel — identical to what
//!   `VGGT.forward(query_points=...)` expects (queries are NOT normalized to
//!   `[-1, 1]`).
//! * Internally the tracker divides coordinates by `stride=2`, so they live
//!   in the feature-map grid `[0, W/2-1] x [0, H/2-1]` (feature maps are
//!   `[B, S, 128, H/2, W/2]`).
//! * Returned `coords` are scaled back to model-resolution pixels
//!   (`coords * stride`), `[S, Q, 2]` for batch size 1. Frame 0 is forced to
//!   the query points every iteration (upstream behavior), so
//!   `coords[0, q] == queries[q].xy` exactly.
//!
//! # Differences from the full upstream track stack
//!
//! * Upstream `vggsfm_tracker.predict_tracks` selects queries automatically
//!   (ALIKED+SP keypoints, DINO-based query-frame ranking, non-visible track
//!   completion) and runs the tracker once per query frame after reordering
//!   frames. That needs external keypoint extractors and is NOT included
//!   here; queries are explicit and the query frame is always frame 0
//!   (`VggtTrackQuery.frame` must be 0, matching `VGGT.forward`). To track
//!   from another frame, rotate the clip so that frame comes first.
//! * `query_points=None` auto initialization (`get_query_points` in
//!   `dependency/track_predict.py`) is likewise not included.
//! * Correlation/coordinate sampling is implemented with explicit gather +
//!   weight arithmetic (candle has no `grid_sample`), reproducing
//!   `F.grid_sample(align_corners=True)` for both `zeros` (correlation
//!   pyramid) and `border` (`sample_features4d`) padding.
//!
//! # Memory
//!
//! Peak transient memory is the level-0 correlation volume
//! `[B, S, chunk_queries, H/2 * W/2]` f32 (~200 MB at 518x518 with 256
//! queries); query chunking bounds it regardless of query count. The
//! sincos position-embedding tables are kept separable (per-axis strips)
//! instead of a full `[1, 388, H/2, W/2]` grid (~100 MB), and are sampled
//! per axis — exactly equivalent because each half of the embedding varies
//! along one axis only. The head itself is ~280 MB of F32 weights and is
//! lazily loaded per device (first track call), so plain camera/point/depth
//! runs do not pay for it.

use candle_core::{D, DType, Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, LayerNorm, Linear as DenseLinear, Module, conv2d, layer_norm,
};

use super::nn_blocks::linear;
use super::vggt_dpt::{DPT_INTERMEDIATE_LAYERS, DPT_PATCH_SIZE, FeatureFusionBlock, ResizeLayer};
use crate::CanonicalWeightSetPaths;
use crate::Result;
use crate::error::Lux3dError;
use crate::weights::VarBuilder;

const TRACK_FEATURES: usize = 128;
const TRACK_DOWN_RATIO: usize = 2;
const TRACK_OUT_CHANNELS: [usize; 4] = [256, 512, 1024, 1024];
/// `BaseTrackerPredictor.stride`: query pixels are divided by this to land in
/// the `[B, S, 128, H/2, W/2]` feature-map grid.
const TRACKER_STRIDE: usize = 2;
const TRACKER_CORR_LEVELS: usize = 7;
const TRACKER_CORR_RADIUS: usize = 4;
const TRACKER_HIDDEN: usize = 384;
const TRACKER_LATENT_DIM: usize = 128;
const TRACKER_MAX_SCALE: f64 = 518.0;
const UPDATEFORMER_HEADS: usize = 8;
const UPDATEFORMER_VIRTUAL_TRACKS: usize = 64;
const UPDATEFORMER_DEPTH: usize = 6;
/// `latent_dim * 3 + 4` (flow embedding + correlation MLP output + track
/// features + 4 raw flow channels).
const TRANSFORMER_DIM: usize = 388;
/// Half of the flow embedding width; `get_2d_embedding` returns `2 * C`.
const FLOWS_EMB_DIM: usize = 64;
/// Default refinement iterations (`TrackHead.iters`).
pub const TRACK_DEFAULT_ITERS: usize = 4;

/// Peak f32 elements for one correlation-volume chunk (~256 MB).
const CORR_CHUNK_ELEMENTS: usize = 64 * 1024 * 1024;

/// One explicit query point: the frame it lives on (must be 0, see module
/// docs) and its `(x, y)` pixel position in the model input resolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VggtTrackQuery {
    pub frame: usize,
    pub xy: [f32; 2],
}

/// Tracks for a batch of queries: pixel coordinates `[S, Q, 2]`, visibility
/// `[S, Q]` (sigmoid) and optional confidence `[S, Q]` (sigmoid), all F32.
#[derive(Debug, Clone)]
pub struct VggtTrackResult {
    pub coords: Tensor,
    pub visibility: Tensor,
    pub confidence: Option<Tensor>,
}

// ---------------------------------------------------------------------------
// Bilinear sampling (F.grid_sample(align_corners=True) equivalents)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PaddingMode {
    /// Out-of-range samples contribute zero (correlation pyramid).
    Zeros,
    /// Coordinates clamp to the image border (`sample_features4d`).
    Border,
}

/// Bilinear sample `input [N, C, L]` (an `[N, C, H, W]` map flattened) at
/// `coords [N, P, 2]`, given as `(x, y)` **pixel coordinates** in
/// `[0, W-1] x [0, H-1]` — the upstream `bilinear_sampler` convention
/// (`align_corners=True`; upstream converts normalized grid coordinates to
/// this space before calling `grid_sample`). Reproduces
/// `F.grid_sample(align_corners=True)` with `zeros` and `border` padding.
///
/// Computation happens in F32; the result is cast back to the input dtype.
fn bilinear_sample_flat(
    input: &Tensor,
    coords: &Tensor,
    height: usize,
    width: usize,
    padding: PaddingMode,
) -> CandleResult<Tensor> {
    let (n, c, _l) = input.dims3()?;
    let p = coords.dim(1)?;
    if coords.dims() != [n, p, 2] {
        candle_core::bail!(
            "bilinear coords shape {:?} does not match [{n}, P, 2]",
            coords.dims()
        );
    }
    let out_dtype = input.dtype();
    let input = input.to_dtype(DType::F32)?.contiguous()?;
    let xs = coords.i((.., .., 0))?.to_dtype(DType::F32)?.contiguous()?;
    let ys = coords.i((.., .., 1))?.to_dtype(DType::F32)?.contiguous()?;

    let x0f = xs.floor()?;
    let y0f = ys.floor()?;
    let tx = xs.sub(&x0f)?;
    let ty = ys.sub(&y0f)?;

    let x_max = (width - 1) as f32;
    let y_max = (height - 1) as f32;
    // Corner indices, clamped so the gather is always in range. For Border
    // this reproduces the reference's coordinate clip exactly (clamped
    // corners coincide and their weights sum to the same value); for Zeros
    // the per-corner validity masks below kill out-of-range contributions.
    let x0c = x0f.clamp(0.0, x_max)?;
    let x1c = x0f.affine(1.0, 1.0)?.clamp(0.0, x_max)?;
    let y0c = y0f.clamp(0.0, y_max)?;
    let y1c = y0f.affine(1.0, 1.0)?.clamp(0.0, y_max)?;
    let x0i = x0c.to_dtype(DType::U32)?;
    let x1i = x1c.to_dtype(DType::U32)?;
    let y0i = y0c.to_dtype(DType::U32)?;
    let y1i = y1c.to_dtype(DType::U32)?;

    let (v00, v01, v10, v11) = match padding {
        PaddingMode::Border => {
            let ones = Tensor::ones((n, p), DType::F32, xs.device())?;
            (ones.clone(), ones.clone(), ones.clone(), ones)
        }
        PaddingMode::Zeros => {
            let mask = |xv: &Tensor, yv: &Tensor| -> CandleResult<Tensor> {
                Ok(xv
                    .ge(0.0f32)?
                    .to_dtype(DType::F32)?
                    .mul(&xv.le(x_max)?.to_dtype(DType::F32)?)?
                    .mul(&yv.ge(0.0f32)?.to_dtype(DType::F32)?)?
                    .mul(&yv.le(y_max)?.to_dtype(DType::F32)?)?)
            };
            (
                mask(&x0f, &y0f)?,
                mask(&x0f.affine(1.0, 1.0)?, &y0f)?,
                mask(&x0f, &y0f.affine(1.0, 1.0)?)?,
                mask(&x0f.affine(1.0, 1.0)?, &y0f.affine(1.0, 1.0)?)?,
            )
        }
    };

    let one_minus_tx = tx.affine(-1.0, 1.0)?;
    let one_minus_ty = ty.affine(-1.0, 1.0)?;
    let w00 = one_minus_tx.mul(&one_minus_ty)?.mul(&v00)?;
    let w01 = tx.mul(&one_minus_ty)?.mul(&v01)?;
    let w10 = one_minus_tx.mul(&ty)?.mul(&v10)?;
    let w11 = tx.mul(&ty)?.mul(&v11)?;

    // Flat gather index y * W + x (U32 arithmetic, exact for H*W < 2^24).
    let index = |yi: &Tensor, xi: &Tensor| -> CandleResult<Tensor> {
        let flat = yi
            .to_dtype(DType::F32)?
            .affine(width as f64, 0.0)?
            .add(&xi.to_dtype(DType::F32)?)?
            .to_dtype(DType::U32)?;
        // gather requires a contiguous index tensor, so materialize the
        // broadcast (cheap: only the P index rows are real data).
        Ok(flat.unsqueeze(1)?.broadcast_as((n, c, p))?.contiguous()?)
    };
    let input_flat = input.reshape((n, c, height * width))?;
    let g00 = input_flat.gather(&index(&y0i, &x0i)?, 2)?;
    let g01 = input_flat.gather(&index(&y0i, &x1i)?, 2)?;
    let g10 = input_flat.gather(&index(&y1i, &x0i)?, 2)?;
    let g11 = input_flat.gather(&index(&y1i, &x1i)?, 2)?;

    let out = w00
        .unsqueeze(1)?
        .broadcast_mul(&g00)?
        .add(&w01.unsqueeze(1)?.broadcast_mul(&g01)?)?
        .add(&w10.unsqueeze(1)?.broadcast_mul(&g10)?)?
        .add(&w11.unsqueeze(1)?.broadcast_mul(&g11)?)?;
    out.to_dtype(out_dtype)
}

/// `track_modules.utils.sample_features4d`: sample `[B, C, H, W]` features at
/// `[B, R, 2]` pixel coordinates (border padding), returning `[B, R, C]`.
fn sample_features4d(input: &Tensor, coords: &Tensor) -> CandleResult<Tensor> {
    let (b, c, height, width) = input.dims4()?;
    let input = input.reshape((b, c, height * width))?;
    let sampled = bilinear_sample_flat(&input, coords, height, width, PaddingMode::Border)?;
    // [B, C, R] -> [B, R, C]
    sampled.permute((0, 2, 1))?.contiguous()
}

/// Public bilinear sampling primitive: sample `[B, C, H, W]` maps at
/// `[B, R, 2]` (x, y) pixel coordinates using `F.grid_sample` semantics with
/// `align_corners=True` and border padding (the upstream
/// `track_modules.utils.sample_features4d` convention). Returns `[B, R, C]`.
/// Coordinates outside `[0, W-1] x [0, H-1]` clamp to the border.
pub fn bilinear_sample_features(input: &Tensor, coords: &Tensor) -> CandleResult<Tensor> {
    sample_features4d(input, coords)
}

// ---------------------------------------------------------------------------
// Feature extractor: DPTHead(feature_only=True, down_ratio=2, pos_embed=False)
// ---------------------------------------------------------------------------

/// DPT feature extractor for the track head. Weight keys under
/// `track_head.feature_extractor.*`.
#[derive(Debug)]
pub struct VggtTrackFeatureExtractor {
    norm: LayerNorm,
    projects: Vec<Conv2d>,
    resize_layers: Vec<ResizeLayer>,
    layer_rn: Vec<Conv2d>,
    refinenet1: FeatureFusionBlock,
    refinenet2: FeatureFusionBlock,
    refinenet3: FeatureFusionBlock,
    refinenet4: FeatureFusionBlock,
    output_conv1: Conv2d,
}

impl VggtTrackFeatureExtractor {
    pub(crate) fn load(vb: VarBuilder, dim_in: usize) -> CandleResult<Self> {
        let norm = layer_norm(dim_in, 1e-5, vb.pp("norm"))?;
        let projects = TRACK_OUT_CHANNELS
            .iter()
            .enumerate()
            .map(|(idx, channels)| {
                conv2d(
                    dim_in,
                    *channels,
                    1,
                    Conv2dConfig::default(),
                    vb.pp("projects").pp(idx.to_string()),
                )
            })
            .collect::<CandleResult<Vec<_>>>()?;
        let resize_layers = vec![
            ResizeLayer::Transpose(candle_nn::conv_transpose2d(
                TRACK_OUT_CHANNELS[0],
                TRACK_OUT_CHANNELS[0],
                4,
                candle_nn::ConvTranspose2dConfig {
                    stride: 4,
                    ..Default::default()
                },
                vb.pp("resize_layers").pp("0"),
            )?),
            ResizeLayer::Transpose(candle_nn::conv_transpose2d(
                TRACK_OUT_CHANNELS[1],
                TRACK_OUT_CHANNELS[1],
                2,
                candle_nn::ConvTranspose2dConfig {
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("resize_layers").pp("1"),
            )?),
            ResizeLayer::Identity,
            ResizeLayer::Conv(conv2d(
                TRACK_OUT_CHANNELS[3],
                TRACK_OUT_CHANNELS[3],
                3,
                Conv2dConfig {
                    padding: 1,
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("resize_layers").pp("3"),
            )?),
        ];

        let rn_config = Conv2dConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        };
        let scratch = vb.pp("scratch");
        let layer_rn = TRACK_OUT_CHANNELS
            .iter()
            .enumerate()
            .map(|(idx, channels)| {
                candle_nn::conv2d_no_bias(
                    *channels,
                    TRACK_FEATURES,
                    3,
                    rn_config.clone(),
                    scratch.pp(format!("layer{}_rn", idx + 1)),
                )
            })
            .collect::<CandleResult<Vec<_>>>()?;

        let refinenet1 = FeatureFusionBlock::load(scratch.pp("refinenet1"), TRACK_FEATURES, true)?;
        let refinenet2 = FeatureFusionBlock::load(scratch.pp("refinenet2"), TRACK_FEATURES, true)?;
        let refinenet3 = FeatureFusionBlock::load(scratch.pp("refinenet3"), TRACK_FEATURES, true)?;
        let refinenet4 = FeatureFusionBlock::load(scratch.pp("refinenet4"), TRACK_FEATURES, false)?;

        // feature_only=True: a single full-width 3x3 conv, no activation head.
        let output_conv1 = conv2d(
            TRACK_FEATURES,
            TRACK_FEATURES,
            3,
            Conv2dConfig {
                padding: 1,
                stride: 1,
                ..Default::default()
            },
            scratch.pp("output_conv1"),
        )?;

        super::drain_upload_staging(vb.device())?;
        Ok(Self {
            norm,
            projects,
            resize_layers,
            layer_rn,
            refinenet1,
            refinenet2,
            refinenet3,
            refinenet4,
            output_conv1,
        })
    }

    /// Run the extractor over frames `[frame_start, frame_end)`.
    ///
    /// `cached_layers` holds the four aggregator outputs in DPT order
    /// (layers 4/11/17/23), each `[B, S, P, 2048]`. Returns
    /// `[B * (frame_end - frame_start), 128, H/2, W/2]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_chunk(
        &self,
        cached_layers: &[&Tensor],
        patch_start_idx: usize,
        frame_start: usize,
        frame_end: usize,
        image_height: usize,
        image_width: usize,
    ) -> CandleResult<Tensor> {
        if cached_layers.len() != DPT_INTERMEDIATE_LAYERS.len() {
            candle_core::bail!(
                "track feature extractor needs {} cached aggregator layers, got {}",
                DPT_INTERMEDIATE_LAYERS.len(),
                cached_layers.len()
            );
        }
        let patch_h = image_height / DPT_PATCH_SIZE;
        let patch_w = image_width / DPT_PATCH_SIZE;

        let mut features = Vec::with_capacity(DPT_INTERMEDIATE_LAYERS.len());
        for (dpt_idx, layer) in cached_layers.iter().enumerate() {
            // Drop the special tokens (camera + registers).
            let chunk = layer.i((.., frame_start..frame_end, patch_start_idx..))?;
            let (batch, frames, patches, channels) = chunk.dims4()?;
            if patches != patch_h * patch_w {
                candle_core::bail!(
                    "track feature extractor patch token count {patches} does not match {patch_h}x{patch_w}"
                );
            }
            let x = chunk.reshape((batch * frames, patches, channels))?;
            let x = self.norm.forward(&x)?;
            let x = x
                .permute((0, 2, 1))?
                .reshape((batch * frames, channels, patch_h, patch_w))?;
            let x = self.projects[dpt_idx].forward(&x)?;
            // pos_embed=False for the track head: no embedding here.
            let x = self.resize_layers[dpt_idx].forward(&x)?;
            features.push(x);
        }

        let layer1_rn = self.layer_rn[0].forward(&features[0])?;
        let layer2_rn = self.layer_rn[1].forward(&features[1])?;
        let layer3_rn = self.layer_rn[2].forward(&features[2])?;
        let layer4_rn = self.layer_rn[3].forward(&features[3])?;

        let (_, _, layer3_h, layer3_w) = layer3_rn.dims4()?;
        let mut out = self
            .refinenet4
            .forward(&layer4_rn, None, Some((layer3_h, layer3_w)))?;
        let (_, _, layer2_h, layer2_w) = layer2_rn.dims4()?;
        out = self
            .refinenet3
            .forward(&out, Some(&layer3_rn), Some((layer2_h, layer2_w)))?;
        let (_, _, layer1_h, layer1_w) = layer1_rn.dims4()?;
        out = self
            .refinenet2
            .forward(&out, Some(&layer2_rn), Some((layer1_h, layer1_w)))?;
        out = self.refinenet1.forward(&out, Some(&layer1_rn), None)?;
        out = self.output_conv1.forward(&out)?;

        // custom_interpolate to (patch_h * patch_size / down_ratio, ...).
        let target_h = patch_h * DPT_PATCH_SIZE / TRACK_DOWN_RATIO;
        let target_w = patch_w * DPT_PATCH_SIZE / TRACK_DOWN_RATIO;
        out.upsample_bilinear2d(target_h, target_w, true)
    }

    /// Run over every frame in chunks, returning `[B, S, 128, H/2, W/2]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        cached_layers: &[&Tensor],
        patch_start_idx: usize,
        frames: usize,
        chunk_size: usize,
        image_height: usize,
        image_width: usize,
    ) -> CandleResult<Tensor> {
        let chunk_size = chunk_size.max(1);
        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < frames {
            let end = (start + chunk_size).min(frames);
            chunks.push(self.forward_chunk(
                cached_layers,
                patch_start_idx,
                start,
                end,
                image_height,
                image_width,
            )?);
            start = end;
        }
        let out = Tensor::cat(&chunks, 0)?;
        let (bs, channels, height, width) = out.dims4()?;
        let batch = bs / frames;
        out.reshape((batch, frames, channels, height, width))
    }
}

// ---------------------------------------------------------------------------
// Correlation pyramid (`track_modules.blocks.CorrBlock`)
// ---------------------------------------------------------------------------

/// Correlation pyramid over the (normalized) feature maps. Level 0 is the
/// full-resolution fmap; each further level is a 2x2 avg pool of the previous
/// (`floor` output size, matching `F.avg_pool2d` defaults).
struct CorrBlock {
    /// Level fmaps as `[B, S, C, H*W]` contiguous tensors.
    pyramid: Vec<Tensor>,
    /// Per-level spatial dims.
    dims: Vec<(usize, usize)>,
    /// `(2r+1)^2` delta grid, flattened row-major, `[1, P, 2]`. Reproduces
    /// `delta = stack(meshgrid(dy, dx, indexing="ij"), -1)`: entry `(i, j)`
    /// is `(dy[i], dx[j])` and is ADDED to `(x, y)` respectively — the
    /// (transposed-grid) upstream quirk the model was trained with.
    delta: Tensor,
    grid_points: usize,
}

impl CorrBlock {
    fn new(fmaps: &Tensor, num_levels: usize, radius: usize) -> CandleResult<Self> {
        let (b, s, c, h, w) = fmaps.dims5()?;
        let mut pyramid = Vec::with_capacity(num_levels);
        let mut dims = Vec::with_capacity(num_levels);
        let mut current = fmaps.reshape((b * s, c, h, w))?.contiguous()?;
        let (mut ch, mut cw) = (h, w);
        pyramid.push(current.reshape((b, s, c, ch * cw))?.contiguous()?);
        dims.push((ch, cw));
        for _ in 1..num_levels {
            current = current.avg_pool2d((2, 2))?;
            let (_, _, nh, nw) = current.dims4()?;
            (ch, cw) = (nh, nw);
            if ch == 0 || cw == 0 {
                candle_core::bail!(
                    "correlation pyramid level collapses to empty ({ch}x{cw}); \
                     the feature map is too small for {num_levels} levels"
                );
            }
            pyramid.push(current.reshape((b, s, c, ch * cw))?.contiguous()?);
            dims.push((ch, cw));
        }

        let size = 2 * radius + 1;
        let r = radius as f64;
        let mut delta = Vec::with_capacity(size * size * 2);
        for i in 0..size {
            let dy = (-r + i as f64) as f32;
            for j in 0..size {
                let dx = (-r + j as f64) as f32;
                delta.push(dy);
                delta.push(dx);
            }
        }
        let delta = Tensor::from_vec(delta, (1, size * size, 2), fmaps.device())?;
        Ok(Self {
            pyramid,
            dims,
            delta,
            grid_points: size * size,
        })
    }

    /// `corr_sample`: for each pyramid level, compute the full correlation
    /// volume against `targets [B, S, N, C]`, sample the `(2r+1)^2` delta
    /// neighborhood at `coords [B, S, N, 2]` and discard the volume.
    /// Returns `[B, S, N, levels * (2r+1)^2]`.
    ///
    /// Query chunks bound the transient `[B, S, chunk, H*W]` volume; rows are
    /// independent, so chunking is numerically exact.
    fn corr_sample(&self, targets: &Tensor, coords: &Tensor) -> CandleResult<Tensor> {
        let (b, s, n, c) = targets.dims4()?;
        if coords.dims() != [b, s, n, 2] {
            candle_core::bail!(
                "corr coords {:?} does not match targets {:?}",
                coords.dims(),
                targets.dims()
            );
        }
        let scale = 1.0 / (c as f64).sqrt();
        let delta = self.delta.to_dtype(coords.dtype())?;

        let (level0_h, level0_w) = self.dims[0];
        let per_query = level0_h * level0_w;
        let chunk_n = (CORR_CHUNK_ELEMENTS / (s * per_query).max(1)).clamp(1, n);

        let mut out_chunks = Vec::new();
        let mut start = 0usize;
        while start < n {
            let len = (n - start).min(chunk_n);
            let tgt = targets.narrow(2, start, len)?.contiguous()?;
            let crd = coords.narrow(2, start, len)?;
            let mut levels = Vec::with_capacity(self.pyramid.len());
            for (level, fmap) in self.pyramid.iter().enumerate() {
                let (hi, wi) = self.dims[level];
                let fmap = fmap.reshape((b, s, c, hi * wi))?;
                let corrs = tgt.matmul(&fmap)?.affine(scale, 0.0)?;
                let corrs = corrs.reshape((b * s * len, 1, hi * wi))?.contiguous()?;
                let grid = crd
                    .reshape((b * s * len, 1, 2))?
                    .broadcast_add(&delta)?
                    .contiguous()?;
                let sampled = bilinear_sample_flat(&corrs, &grid, hi, wi, PaddingMode::Zeros)?;
                levels.push(sampled.reshape((b, s, len, self.grid_points))?);
            }
            out_chunks.push(Tensor::cat(&levels, D::Minus1)?);
            start += len;
        }
        if out_chunks.len() == 1 {
            Ok(out_chunks.remove(0))
        } else {
            Tensor::cat(&out_chunks, 2)
        }
    }
}

// ---------------------------------------------------------------------------
// Embeddings (`track_modules.utils`)
// ---------------------------------------------------------------------------

/// `get_2d_embedding(xy, C, cat_coords=False)`: sine/cosine embedding of the
/// `(x, y)` flow values, `[K, M, 2] -> [K, M, 2C]`. Computed in F32 like the
/// reference (`div_term` is an f32 arange times `1000.0 / C`), with even
/// channels sin and odd channels cos, then `[pe_x, pe_y]` concatenated.
fn get_2d_embedding(xy: &Tensor, c: usize) -> CandleResult<Tensor> {
    let (k, m, d) = xy.dims3()?;
    if d != 2 {
        candle_core::bail!("get_2d_embedding expects [K, M, 2], got trailing dim {d}");
    }
    let half = c / 2;
    let div_term: Vec<f32> = (0..half)
        .map(|idx| (2 * idx) as f32 * (1000.0 / c as f32))
        .collect();
    let div_term = Tensor::from_vec(div_term, (1, 1, half), xy.device())?;
    let x = xy.i((.., .., 0))?.unsqueeze(2)?;
    let y = xy.i((.., .., 1))?.unsqueeze(2)?;
    let axis_embed = |v: &Tensor| -> CandleResult<Tensor> {
        let scaled = v.broadcast_mul(&div_term)?;
        let sin = scaled.sin()?;
        let cos = scaled.cos()?;
        // interleave: even channels sin, odd channels cos
        Tensor::stack(&[&sin, &cos], D::Minus1)?.reshape((k, m, c))
    };
    let pe_x = axis_embed(&x)?;
    let pe_y = axis_embed(&y)?;
    Tensor::cat(&[&pe_x, &pe_y], D::Minus1)
}

/// Separable sincos position-embedding tables for the update-former input,
/// reproducing `get_2d_sincos_pos_embed(transformer_dim, (HH, WW))`.
///
/// The full grid `[1, D, HH, WW]` concatenates a per-x embedding (channels
/// `0..D/2`, from `meshgrid(grid_w, grid_h, indexing="xy")[0]`) and a per-y
/// embedding (channels `D/2..D`). Each half varies along one axis only, so
/// storing and sampling per-axis strips is exactly equivalent to sampling the
/// full grid with `sample_features4d`: 2D bilinear interpolation of an
/// axis-constant function degenerates to 1D interpolation.
struct SincosPosTables {
    /// `[1, D/2, 1, WW]`, value `[c, 0, x] = emb_x(x)[c]`.
    u: Tensor,
    /// `[1, D/2, HH, 1]`, value `[c, y, 0] = emb_y(y)[c]`.
    v: Tensor,
    dim: usize,
}

impl SincosPosTables {
    fn new(height: usize, width: usize, embed_dim: usize, device: &Device) -> CandleResult<Self> {
        if embed_dim % 4 != 0 {
            candle_core::bail!("sincos pos embed requires embed_dim % 4 == 0, got {embed_dim}");
        }
        let axis_dim = embed_dim / 2;
        let n_freq = axis_dim / 2;
        // get_1d_sincos_pos_embed_from_grid: omega = 1/10000^(i / n_freq),
        // computed in f64; angles are f64 (f32 positions promoted) and cast
        // to f32 at the end, matching the reference exactly.
        let omega: Vec<f64> = (0..n_freq)
            .map(|idx| 1.0 / 10000f64.powf(idx as f64 / n_freq as f64))
            .collect();
        let axis_embed = |coords: &[f32]| -> CandleResult<Tensor> {
            let mut emb = Vec::with_capacity(coords.len() * axis_dim);
            for pos in coords {
                let pos = *pos as f64;
                for freq in &omega {
                    emb.push((pos * freq).sin() as f32);
                }
                for freq in &omega {
                    emb.push((pos * freq).cos() as f32);
                }
            }
            // [axis, coords] -> [1, axis, 1, coords] / [1, axis, coords, 1]
            // is applied by the caller via transposes.
            Ok(Tensor::from_vec(emb, (coords.len(), axis_dim), device)?)
        };
        let x_coords: Vec<f32> = (0..width).map(|v| v as f32).collect();
        let y_coords: Vec<f32> = (0..height).map(|v| v as f32).collect();
        let u = axis_embed(&x_coords)?
            .t()?
            .unsqueeze(0)?
            .unsqueeze(2)?
            .contiguous()?;
        let v = axis_embed(&y_coords)?
            .t()?
            .unsqueeze(0)?
            .unsqueeze(3)?
            .contiguous()?;
        debug_assert_eq!(u.dims(), &[1, axis_dim, 1, width]);
        debug_assert_eq!(v.dims(), &[1, axis_dim, height, 1]);
        Ok(Self {
            u,
            v,
            dim: embed_dim,
        })
    }

    /// `sample_features4d(pos_embed.expand(B, -1, -1, -1), coords)`:
    /// sample the embedding at `[B, N, 2]` feature-map coordinates,
    /// returning `[B, N, D]` (F32, like the reference tables).
    fn sample(&self, coords: &Tensor, batch: usize) -> CandleResult<Tensor> {
        let (b, _n, _) = coords.dims3()?;
        if b != batch {
            candle_core::bail!("coords batch {b} does not match expected {batch}");
        }
        let xs = coords.i((.., .., 0))?.unsqueeze(2)?.contiguous()?;
        let ys = coords.i((.., .., 1))?.unsqueeze(2)?.contiguous()?;
        let zeros = xs.zeros_like()?;
        let coords_u = Tensor::cat(&[&xs, &zeros], D::Minus1)?;
        let coords_v = Tensor::cat(&[&zeros, &ys], D::Minus1)?;
        debug_assert_eq!(self.u.dim(1)? * 2, self.dim);
        let (u_c, _ur, _uh, u_w) = self.u.dims4()?;
        let (v_c, _vr, v_h, _vw) = self.v.dims4()?;
        let u = if b == 1 {
            self.u.clone()
        } else {
            self.u.broadcast_as((b, u_c, _ur, u_w))?.contiguous()?
        };
        let v = if b == 1 {
            self.v.clone()
        } else {
            self.v.broadcast_as((b, v_c, v_h, _vw))?.contiguous()?
        };
        let u = sample_features4d(&u, &coords_u)?; // [B, N, D/2]
        let v = sample_features4d(&v, &coords_v)?; // [B, N, D/2]
        Tensor::cat(&[&u, &v], D::Minus1)
    }
}

// ---------------------------------------------------------------------------
// EfficientUpdateFormer (`track_modules.blocks`)
// ---------------------------------------------------------------------------

/// MLP as used by the tracker (`track_modules.modules.Mlp`): fc1 -> exact-erf
/// GELU -> fc2, all with bias.
#[derive(Debug)]
struct TrackMlp {
    fc1: super::nn_blocks::Linear,
    fc2: super::nn_blocks::Linear,
}

impl TrackMlp {
    fn load(
        vb: VarBuilder,
        in_features: usize,
        hidden: usize,
        out_features: usize,
    ) -> CandleResult<Self> {
        Ok(Self {
            fc1: linear(vb.pp("fc1"), in_features, hidden, true)?,
            fc2: linear(vb.pp("fc2"), hidden, out_features, true)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let hidden = self.fc1.forward(xs)?.gelu_erf()?;
        self.fc2.forward(&hidden)
    }
}

/// Project QKV weights out of a PyTorch `nn.MultiheadAttention` fused
/// `in_proj_weight [3E, E]` / `in_proj_bias [3E]`.
fn proj_from_in_proj(
    weight: &Tensor,
    bias: &Tensor,
    row: usize,
    embed_dim: usize,
) -> CandleResult<DenseLinear> {
    // `narrow` yields a contiguous 1D *view* with the original (3E) storage
    // count for row 0. Candle's Vulkan fused Linear checks the bias view shape
    // but the kernel checks the underlying storage count, so materialize an
    // exactly-sized buffer for every row. CUDA/WGPU take the unfused path and
    // are unaffected.
    let w = weight
        .narrow(0, row * embed_dim, embed_dim)?
        .force_contiguous()?;
    let b = bias
        .narrow(0, row * embed_dim, embed_dim)?
        .force_contiguous()?;
    Ok(DenseLinear::new(w, Some(b)))
}

/// Attention over `[K, T, E]` with PyTorch `nn.MultiheadAttention` semantics
/// (batch_first, no mask, q scaled by `1/sqrt(head_dim)` before the matmul).
#[derive(Debug)]
struct Mha {
    q: DenseLinear,
    k: DenseLinear,
    v: DenseLinear,
    out: DenseLinear,
    num_heads: usize,
    head_dim: usize,
}

impl Mha {
    fn from_in_proj(vb: VarBuilder, embed_dim: usize, num_heads: usize) -> CandleResult<Self> {
        let weight = vb.get((3 * embed_dim, embed_dim), "in_proj_weight")?;
        let bias = vb.get(3 * embed_dim, "in_proj_bias")?;
        Ok(Self {
            q: proj_from_in_proj(&weight, &bias, 0, embed_dim)?,
            k: proj_from_in_proj(&weight, &bias, 1, embed_dim)?,
            v: proj_from_in_proj(&weight, &bias, 2, embed_dim)?,
            out: DenseLinear::new(
                vb.get((embed_dim, embed_dim), "out_proj.weight")?,
                Some(vb.get(embed_dim, "out_proj.bias")?),
            ),
            num_heads,
            head_dim: embed_dim / num_heads,
        })
    }

    fn attend(&self, q_in: &Tensor, k_in: &Tensor, v_in: &Tensor) -> CandleResult<Tensor> {
        let (rows, t, e) = q_in.dims3()?;
        // Cross-attention keys/values may have a different token count than
        // the query; project each input against its own dims.
        let project = |l: &DenseLinear, x: &Tensor| -> CandleResult<Tensor> {
            let (r, tk, _e) = x.dims3()?;
            Ok(l.forward(x)?
                .reshape((r, tk, self.num_heads, self.head_dim))?
                .permute((0, 2, 1, 3))?
                .contiguous()?)
        };
        let q = project(&self.q, q_in)?;
        let k = project(&self.k, k_in)?;
        let v = project(&self.v, v_in)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let out =
            super::attention_math::exact_query_chunked_sdpa(&q, &k, &v, scale as f32, usize::MAX)?;
        let out = out
            .permute((0, 2, 1, 3))?
            .reshape((rows, t, e))?
            .contiguous()?;
        self.out.forward(&out)
    }
}

/// Self-attention block (`AttnBlock`): `x += attn(norm1(x)); x += mlp(norm2(x))`.
#[derive(Debug)]
struct AttnBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    mha: Mha,
    mlp: TrackMlp,
}

impl AttnBlock {
    fn load(vb: VarBuilder, hidden: usize, num_heads: usize) -> CandleResult<Self> {
        Ok(Self {
            norm1: layer_norm(hidden, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(hidden, 1e-5, vb.pp("norm2"))?,
            mha: Mha::from_in_proj(vb.pp("attn"), hidden, num_heads)?,
            mlp: TrackMlp::load(vb.pp("mlp"), hidden, hidden * 4, hidden)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let xs = self.norm1.forward(xs)?;
        let attn = self.mha.attend(&xs, &xs, &xs)?;
        let xs = xs.add(&attn)?;
        let mlp = self.mlp.forward(&self.norm2.forward(&xs)?)?;
        xs.add(&mlp)
    }
}

/// Cross-attention block (`CrossAttnBlock`): query from `x`, keys/values from
/// `context`, both pre-normed.
#[derive(Debug)]
struct CrossAttnBlock {
    norm1: LayerNorm,
    norm_context: LayerNorm,
    norm2: LayerNorm,
    mha: Mha,
    mlp: TrackMlp,
}

impl CrossAttnBlock {
    fn load(vb: VarBuilder, hidden: usize, num_heads: usize) -> CandleResult<Self> {
        Ok(Self {
            norm1: layer_norm(hidden, 1e-5, vb.pp("norm1"))?,
            norm_context: layer_norm(hidden, 1e-5, vb.pp("norm_context"))?,
            norm2: layer_norm(hidden, 1e-5, vb.pp("norm2"))?,
            mha: Mha::from_in_proj(vb.pp("cross_attn"), hidden, num_heads)?,
            mlp: TrackMlp::load(vb.pp("mlp"), hidden, hidden * 4, hidden)?,
        })
    }

    fn forward(&self, xs: &Tensor, context: &Tensor) -> CandleResult<Tensor> {
        let xs = self.norm1.forward(xs)?;
        let context = self.norm_context.forward(context)?;
        let attn = self.mha.attend(&xs, &context, &context)?;
        let xs = xs.add(&attn)?;
        let mlp = self.mlp.forward(&self.norm2.forward(&xs)?)?;
        xs.add(&mlp)
    }
}

/// `EfficientUpdateFormer(space_depth=6, time_depth=6, input_dim=388,
/// hidden_size=384, output_dim=130, num_heads=8, mlp_ratio=4.0,
/// add_space_attn=True, num_virtual_tracks=64)`.
#[derive(Debug)]
struct EfficientUpdateFormer {
    input_norm: LayerNorm,
    input_transform: super::nn_blocks::Linear,
    output_norm: LayerNorm,
    flow_head: super::nn_blocks::Linear,
    virtual_tracks: Tensor,
    time_blocks: Vec<AttnBlock>,
    space_virtual_blocks: Vec<AttnBlock>,
    space_point2virtual_blocks: Vec<CrossAttnBlock>,
    space_virtual2point_blocks: Vec<CrossAttnBlock>,
}

impl EfficientUpdateFormer {
    fn load(
        vb: VarBuilder,
        input_dim: usize,
        hidden: usize,
        output_dim: usize,
    ) -> CandleResult<Self> {
        let time_blocks = (0..UPDATEFORMER_DEPTH)
            .map(|idx| {
                AttnBlock::load(
                    vb.pp("time_blocks").pp(idx.to_string()),
                    hidden,
                    UPDATEFORMER_HEADS,
                )
            })
            .collect::<CandleResult<Vec<_>>>()?;
        let space_virtual_blocks = (0..UPDATEFORMER_DEPTH)
            .map(|idx| {
                AttnBlock::load(
                    vb.pp("space_virtual_blocks").pp(idx.to_string()),
                    hidden,
                    UPDATEFORMER_HEADS,
                )
            })
            .collect::<CandleResult<Vec<_>>>()?;
        let space_point2virtual_blocks = (0..UPDATEFORMER_DEPTH)
            .map(|idx| {
                CrossAttnBlock::load(
                    vb.pp("space_point2virtual_blocks").pp(idx.to_string()),
                    hidden,
                    UPDATEFORMER_HEADS,
                )
            })
            .collect::<CandleResult<Vec<_>>>()?;
        let space_virtual2point_blocks = (0..UPDATEFORMER_DEPTH)
            .map(|idx| {
                CrossAttnBlock::load(
                    vb.pp("space_virtual2point_blocks").pp(idx.to_string()),
                    hidden,
                    UPDATEFORMER_HEADS,
                )
            })
            .collect::<CandleResult<Vec<_>>>()?;
        super::drain_upload_staging(vb.device())?;
        Ok(Self {
            input_norm: layer_norm(input_dim, 1e-5, vb.pp("input_norm"))?,
            input_transform: linear(vb.pp("input_transform"), input_dim, hidden, true)?,
            output_norm: layer_norm(hidden, 1e-5, vb.pp("output_norm"))?,
            flow_head: linear(vb.pp("flow_head"), hidden, output_dim, true)?,
            // The upstream attribute name has a typo; keep it for the weights.
            virtual_tracks: vb.get((1, UPDATEFORMER_VIRTUAL_TRACKS, 1, hidden), "virual_tracks")?,
            time_blocks,
            space_virtual_blocks,
            space_point2virtual_blocks,
            space_virtual2point_blocks,
        })
    }

    /// `input [B, N, T, input_dim]` -> `flow [B, N, T, output_dim]`.
    /// `N` is the number of tracks, `T` the number of frames.
    fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        let (b, n, t, _d) = input.dims4()?;
        let x = self.input_norm.forward(input)?;
        let tokens = self.input_transform.forward(&x)?;
        let e = tokens.dim(D::Minus1)?;
        let init_tokens = tokens.clone();
        let n_total = n + UPDATEFORMER_VIRTUAL_TRACKS;
        let virtual_t = self
            .virtual_tracks
            .broadcast_as((b, UPDATEFORMER_VIRTUAL_TRACKS, t, e))?
            .contiguous()?;
        let mut tokens = Tensor::cat(&[&tokens, &virtual_t], 1)?;

        let space_every = self.time_blocks.len() / self.space_virtual_blocks.len();
        let mut j = 0usize;
        for (i, block) in self.time_blocks.iter().enumerate() {
            let time_tokens = tokens.reshape((b * n_total, t, e))?.contiguous()?;
            let time_tokens = block.forward(&time_tokens)?;
            tokens = time_tokens.reshape((b, n_total, t, e))?;
            if i % space_every == 0 {
                let space =
                    tokens
                        .permute((0, 2, 1, 3))?
                        .contiguous()?
                        .reshape((b * t, n_total, e))?;
                let point = space.narrow(1, 0, n)?;
                let virtual_t = space.narrow(1, n, UPDATEFORMER_VIRTUAL_TRACKS)?;
                let virtual_t = self.space_virtual2point_blocks[j].forward(&virtual_t, &point)?;
                let virtual_t = self.space_virtual_blocks[j].forward(&virtual_t)?;
                let point = self.space_point2virtual_blocks[j].forward(&point, &virtual_t)?;
                let space = Tensor::cat(&[&point, &virtual_t], 1)?;
                tokens = space
                    .reshape((b, t, n_total, e))?
                    .permute((0, 2, 1, 3))?
                    .contiguous()?;
                j += 1;
            }
        }

        let tokens = tokens.narrow(1, 0, n)?;
        let tokens = tokens.add(&init_tokens)?;
        let tokens = self.output_norm.forward(&tokens)?;
        self.flow_head.forward(&tokens)
    }
}

// ---------------------------------------------------------------------------
// BaseTrackerPredictor (`track_modules.base_track_predictor`)
// ---------------------------------------------------------------------------

/// GroupNorm(1, C) applied to `[K, C]`: per-row normalization over all
/// channels, then per-channel affine. `eps` matches PyTorch's default 1e-5.
fn group_norm_rows(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> CandleResult<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = centered.broadcast_div(&variance.affine(1.0, eps)?.sqrt()?)?;
    normed.broadcast_mul(&weight)?.broadcast_add(&bias)
}

/// `BaseTrackerPredictor` with the TrackHead configuration. Weight keys under
/// `track_head.tracker.*`.
#[derive(Debug)]
pub struct VggtTrackerPredictor {
    corr_mlp: TrackMlp,
    updateformer: EfficientUpdateFormer,
    fmap_norm: LayerNorm,
    ffeat_norm_weight: Tensor,
    ffeat_norm_bias: Tensor,
    ffeat_updater: super::nn_blocks::Linear,
    vis_predictor: super::nn_blocks::Linear,
    conf_predictor: Option<super::nn_blocks::Linear>,
    query_ref_token: Tensor,
    stride: usize,
    corr_levels: usize,
    corr_radius: usize,
    max_scale: f64,
}

impl VggtTrackerPredictor {
    pub(crate) fn load(
        vb: VarBuilder,
        latent_dim: usize,
        predict_conf: bool,
    ) -> CandleResult<Self> {
        let corr_in = TRACKER_CORR_LEVELS * (TRACKER_CORR_RADIUS * 2 + 1).pow(2);
        let transformer_dim = latent_dim * 3 + 4;
        let updateformer = EfficientUpdateFormer::load(
            vb.pp("updateformer"),
            transformer_dim,
            TRACKER_HIDDEN,
            latent_dim + 2,
        )?;
        Ok(Self {
            corr_mlp: TrackMlp::load(vb.pp("corr_mlp"), corr_in, TRACKER_HIDDEN, latent_dim)?,
            updateformer,
            fmap_norm: layer_norm(latent_dim, 1e-5, vb.pp("fmap_norm"))?,
            ffeat_norm_weight: vb.get(latent_dim, "ffeat_norm.weight")?,
            ffeat_norm_bias: vb.get(latent_dim, "ffeat_norm.bias")?,
            ffeat_updater: linear(vb.pp("ffeat_updater").pp("0"), latent_dim, latent_dim, true)?,
            vis_predictor: linear(vb.pp("vis_predictor").pp("0"), latent_dim, 1, true)?,
            conf_predictor: if predict_conf {
                Some(linear(
                    vb.pp("conf_predictor").pp("0"),
                    latent_dim,
                    1,
                    true,
                )?)
            } else {
                None
            },
            query_ref_token: vb.get((1, 2, transformer_dim), "query_ref_token")?,
            stride: TRACKER_STRIDE,
            corr_levels: TRACKER_CORR_LEVELS,
            corr_radius: TRACKER_CORR_RADIUS,
            max_scale: TRACKER_MAX_SCALE,
        })
    }

    /// Track `query_points [B, N, 2]` (model-resolution pixel coordinates)
    /// across `fmaps [B, S, C, HH, WW]`.
    ///
    /// Returns `(coord_preds, vis, conf)`: one `[B, S, N, 2]` pixel-coordinate
    /// tensor per iteration (in order), plus `[B, S, N]` sigmoid visibility
    /// and optional confidence.
    pub(crate) fn forward(
        &self,
        query_points: &Tensor,
        fmaps: &Tensor,
        iters: usize,
    ) -> CandleResult<(Vec<Tensor>, Tensor, Option<Tensor>)> {
        if iters == 0 {
            candle_core::bail!("tracker requires at least one refinement iteration");
        }
        let (b, s, c, hh, ww) = fmaps.dims5()?;
        let (qb, n, d) = query_points.dims3()?;
        if d != 2 {
            candle_core::bail!("query points must be [B, N, 2], got trailing dim {d}");
        }
        if qb != b {
            candle_core::bail!("query batch {qb} does not match feature-map batch {b}");
        }
        let dtype = fmaps.dtype();
        let device = fmaps.device();
        let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();

        // fmap LayerNorm over channels.
        let t = std::time::Instant::now();
        let fmaps = if device.is_vulkan() {
            // Avoid the NHWC transpose: see `channel_layer_norm_nchw`.
            channel_layer_norm_nchw(
                &fmaps,
                self.fmap_norm.weight(),
                self.fmap_norm
                    .bias()
                    .ok_or_else(|| candle_core::Error::Msg("fmap_norm bias missing".into()))?,
                self.fmap_norm.eps(),
            )?
        } else {
            self.fmap_norm
                .forward(&fmaps.permute((0, 1, 3, 4, 2))?)?
                .permute((0, 1, 4, 2, 3))?
                .contiguous()?
        };
        if stage_time {
            eprintln!(
                "[stage]       track.fmap_norm: {:.2}s",
                t.elapsed().as_secs_f64()
            );
        }

        // Query pixels -> feature-map pixels (stride), broadcast over frames.
        let qp = query_points
            .to_dtype(dtype)?
            .affine(1.0 / self.stride as f64, 0.0)?;
        let coords0 = qp.unsqueeze(1)?.broadcast_as((b, s, n, 2))?.contiguous()?;
        let mut coords = coords0.clone();

        // Query-frame features.
        let fmaps0 = fmaps.i((.., 0))?.contiguous()?;
        let coords0_q = coords.i((.., 0))?.contiguous()?;
        let query_feat = sample_features4d(&fmaps0, &coords0_q)?; // [B, N, C]
        let mut track_feats = query_feat
            .unsqueeze(1)?
            .broadcast_as((b, s, n, c))?
            .contiguous()?;
        let coords_backup = coords.clone();

        let t = std::time::Instant::now();
        let corr = CorrBlock::new(&fmaps, self.corr_levels, self.corr_radius)?;
        if stage_time {
            eprintln!(
                "[stage]       track.corr_pyramid: {:.2}s",
                t.elapsed().as_secs_f64()
            );
        }
        let t = std::time::Instant::now();
        let pos_tables = SincosPosTables::new(hh, ww, TRANSFORMER_DIM, &device)?;
        if stage_time {
            eprintln!(
                "[stage]       track.pos_tables: {:.2}s",
                t.elapsed().as_secs_f64()
            );
        }

        // query_ref_token [1, S, D]: slot 0 for the query frame, slot 1 for
        // every other frame.
        let token0 = self.query_ref_token.i((.., 0..1, ..))?;
        let query_ref_token = if s == 1 {
            token0
        } else {
            let token1 = self.query_ref_token.i((.., 1..2, ..))?;
            let rest = token1.broadcast_as((1, s - 1, TRANSFORMER_DIM))?;
            Tensor::cat(&[&token0, &rest], 1)?
        };

        let mut coord_preds = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t_corr = std::time::Instant::now();
            let fcorrs = corr.corr_sample(&track_feats, &coords)?; // [B, S, N, corr_dim]
            if stage_time {
                eprintln!(
                    "[stage]       track.corr: {:.2}s",
                    t_corr.elapsed().as_secs_f64()
                );
            }
            let corr_dim = fcorrs.dim(D::Minus1)?;
            let t_mlp = std::time::Instant::now();
            let fcorrs_ = self.corr_mlp.forward(
                &fcorrs
                    .permute((0, 2, 1, 3))?
                    .reshape((b * n, s, corr_dim))?
                    .contiguous()?,
            )?; // [B*N, S, C]
            if stage_time {
                eprintln!(
                    "[stage]       track.corr_mlp: {:.2}s",
                    t_mlp.elapsed().as_secs_f64()
                );
            }

            // Flow from the query frame + its 2D sine/cosine embedding.
            let flows = coords
                .broadcast_sub(&coords.narrow(1, 0, 1)?)?
                .permute((0, 2, 1, 3))?
                .reshape((b * n, s, 2))?
                .contiguous()?; // [B*N, S, 2]
            let flows_emb = get_2d_embedding(&flows, FLOWS_EMB_DIM)?; // [B*N, S, 2C]
            let scaled = flows.affine(1.0 / self.max_scale, 0.0)?;
            let flows_emb = Tensor::cat(&[&flows_emb, &scaled, &scaled], D::Minus1)?; // [B*N, S, 2C+4]

            let track_feats_ = track_feats
                .permute((0, 2, 1, 3))?
                .reshape((b * n, s, c))?
                .contiguous()?;
            let transformer_input = Tensor::cat(&[&flows_emb, &fcorrs_, &track_feats_], D::Minus1)?; // [B*N, S, D]

            // Positional embedding sampled at the (fixed) query-frame coords.
            let sampled_pos = pos_tables
                .sample(&coords.i((.., 0))?.contiguous()?, b)?
                .to_dtype(dtype)?
                .reshape((b * n, 1, TRANSFORMER_DIM))?;
            let mut x = transformer_input.broadcast_add(&sampled_pos)?;
            x = x.broadcast_add(&query_ref_token.to_dtype(dtype)?)?;
            let x = x.reshape((b, n, s, TRANSFORMER_DIM))?.contiguous()?;

            let t_update = std::time::Instant::now();
            let delta = self.updateformer.forward(&x)?; // [B, N, S, latent+2]
            if stage_time {
                eprintln!(
                    "[stage]       track.updateformer: {:.2}s",
                    t_update.elapsed().as_secs_f64()
                );
            }
            let delta = delta.reshape((b * n, s, c + 2))?;
            let delta_coords = delta.narrow(D::Minus1, 0, 2)?.contiguous()?;
            let delta_feats = delta.narrow(D::Minus1, 2, c)?.contiguous()?;

            let feats_update = self.ffeat_updater.forward(&group_norm_rows(
                &delta_feats,
                &self.ffeat_norm_weight,
                &self.ffeat_norm_bias,
                1e-5,
            )?)?;
            let track_feats_ = feats_update.add(&track_feats_)?; // [B*N, S, C]
            track_feats = track_feats_
                .reshape((b, n, s, c))?
                .permute((0, 2, 1, 3))?
                .contiguous()?; // [B, S, N, C]

            let delta_coords = delta_coords
                .reshape((b, n, s, 2))?
                .permute((0, 2, 1, 3))?
                .contiguous()?; // [B, S, N, 2]
            coords = coords.add(&delta_coords)?;
            // Force frame 0 back to the query points.
            coords = if s == 1 {
                coords_backup.clone()
            } else {
                Tensor::cat(
                    &[
                        &coords_backup.narrow(1, 0, 1)?,
                        &coords.narrow(1, 1, s - 1)?,
                    ],
                    1,
                )?
            };
            coord_preds.push(coords.affine(self.stride as f64, 0.0)?);
        }

        let feats_flat = track_feats.reshape((b * s * n, c))?.contiguous()?;
        let vis = self
            .vis_predictor
            .forward(&feats_flat)?
            .reshape((b, s, n))?;
        let vis = candle_nn::ops::sigmoid(&vis)?;
        let conf = match &self.conf_predictor {
            Some(head) => {
                let conf = head.forward(&feats_flat)?.reshape((b, s, n))?;
                Some(candle_nn::ops::sigmoid(&conf)?)
            }
            None => None,
        };
        Ok((coord_preds, vis, conf))
    }
}

// ---------------------------------------------------------------------------
// TrackHead wrapper + public entry points
// ---------------------------------------------------------------------------

/// LayerNorm over the channel axis of a contiguous `[B, S, C, H, W]` tensor,
/// without the channel-last round trip.
///
/// Upstream normalizes `fmaps.permute(0, 1, 3, 4, 2)` (NHWC). Candle's
/// `layer_norm` only spans the last axis, so that permute forces a
/// `copy_strided_src` transpose; the Vulkan transpose shader is
/// uncoalesced and ran at ~16 MB/s, so the `[1, 3, 128, 259, 259]` feature
/// map alone took 6.5 s. Reducing over C with a ones-vector GEMM keeps every
/// operand contiguous and turns the norm into memory-bound elementwise
/// passes (~50 ms on the same feature map).
fn channel_layer_norm_nchw(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
) -> CandleResult<Tensor> {
    let (b, s, c, h, w) = x.dims5()?;
    let rows = b * s;
    let hw = h * w;
    let out_dtype = x.dtype();
    // The native Vulkan kernel accumulates in F32; match that for F16/BF16.
    let x = if out_dtype == DType::F32 {
        x.clone()
    } else {
        x.to_dtype(DType::F32)?
    };
    let x = x.reshape((rows, c, hw))?;
    let ones = Tensor::ones((rows, 1, c), DType::F32, x.device())?;
    let mean = ones.matmul(&x)?.affine(1.0 / c as f64, 0.0)?; // [rows, 1, hw]
    let centered = x.broadcast_sub(&mean)?;
    // Variance of the centered values (not E[x^2] - E[x]^2): no cancellation.
    let var = ones.matmul(&centered.sqr()?)?.affine(1.0 / c as f64, 0.0)?;
    let inv_std = (var + eps)?.sqrt()?.recip()?;
    let normed = centered.broadcast_mul(&inv_std)?; // [rows, C, hw]
    let weight = weight.to_dtype(DType::F32)?.reshape((1, c, 1))?;
    let bias = bias.to_dtype(DType::F32)?.reshape((1, c, 1))?;
    let out = normed
        .broadcast_mul(&weight)?
        .broadcast_add(&bias)?
        .reshape((b, s, c, h, w))?;
    if out_dtype == DType::F32 {
        Ok(out)
    } else {
        out.to_dtype(out_dtype)
    }
}

/// `TrackHead`: DPT feature extractor + iterative tracker.
#[derive(Debug)]
pub struct VggtTrackHead {
    feature_extractor: VggtTrackFeatureExtractor,
    tracker: VggtTrackerPredictor,
}

impl VggtTrackHead {
    /// Load `track_head.*` from a VarBuilder rooted at the `track_head`
    /// prefix (`dim_in` = 2 * aggregator embed dim = 2048).
    pub(crate) fn load(vb: VarBuilder, dim_in: usize) -> CandleResult<Self> {
        Ok(Self {
            feature_extractor: VggtTrackFeatureExtractor::load(vb.pp("feature_extractor"), dim_in)?,
            tracker: VggtTrackerPredictor::load(vb.pp("tracker"), TRACKER_LATENT_DIM, true)?,
        })
    }

    /// Load the track head alone from a canonical weight set (e.g. for
    /// track-only workloads that never need the aggregator weights).
    ///
    /// # Safety
    /// See [`CanonicalWeightSetPaths::var_builder`]: the caller must keep the
    /// mapped weight file alive and unmutated for the lifetime of the tensors.
    pub unsafe fn load_from_canonical(
        weights: &CanonicalWeightSetPaths,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = unsafe { weights.var_builder(dtype, device) }.map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to open VGGT canonical weights for track head: {source}"),
            }
        })?;
        Self::load(vb.pp("track_head"), 2 * 1024).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to load VGGT track head: {source}"),
            }
        })
    }

    /// Feature-map spatial size for an input image size (`down_ratio=2`).
    pub fn feature_map_size(&self, image_height: usize, image_width: usize) -> (usize, usize) {
        (
            image_height / DPT_PATCH_SIZE * DPT_PATCH_SIZE / TRACK_DOWN_RATIO,
            image_width / DPT_PATCH_SIZE * DPT_PATCH_SIZE / TRACK_DOWN_RATIO,
        )
    }

    /// Full track-head forward: feature extraction + iterative tracking.
    ///
    /// * `cached_layers`: the four cached aggregator outputs in DPT order
    ///   (layers 4/11/17/23), each `[B, S, P, 2048]`.
    /// * `images`: `[B, S, 3, H, W]` model-resolution frames (only the shape
    ///   is used, like upstream).
    /// * `query_points`: `[B, N, 2]` pixel coordinates in the
    ///   model-resolution image (same dtype as the head weights, or F32).
    ///
    /// Returns one `[B, S, N, 2]` pixel-coordinate tensor per iteration plus
    /// `[B, S, N]` visibility and confidence. Output dtype follows the head
    /// weights; convert to F32 at the API boundary.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        cached_layers: &[&Tensor],
        images: &Tensor,
        patch_start_idx: usize,
        query_points: &Tensor,
        iters: usize,
        chunk_size: usize,
    ) -> CandleResult<(Vec<Tensor>, Tensor, Option<Tensor>)> {
        let (_b, s, _channels, height, width) = images.dims5()?;
        let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
        let t = std::time::Instant::now();
        let fmaps = self.feature_extractor.forward(
            cached_layers,
            patch_start_idx,
            s,
            chunk_size,
            height,
            width,
        )?;
        if stage_time {
            eprintln!(
                "[stage]       track.fmaps: {:.2}s {:?}",
                t.elapsed().as_secs_f64(),
                fmaps.dims()
            );
        }
        let t = std::time::Instant::now();
        let out = self.tracker.forward(query_points, &fmaps, iters)?;
        if stage_time {
            eprintln!(
                "[stage]       track.predictor: {:.2}s",
                t.elapsed().as_secs_f64()
            );
        }
        Ok(out)
    }

    /// Public track entry: explicit queries -> [`VggtTrackResult`] (F32,
    /// batch 1, `[S, Q, 2]` / `[S, Q]`).
    ///
    /// `cached_layers` are the four cached aggregator tensors in DPT order
    /// (layers 4/11/17/23); `images` is `[1, S, 3, H, W]` in the model input
    /// resolution (only its shape is used, as upstream).
    pub fn track(
        &self,
        cached_layers: &[&Tensor],
        images: &Tensor,
        patch_start_idx: usize,
        queries: &[VggtTrackQuery],
        iters: usize,
        chunk_size: usize,
    ) -> Result<VggtTrackResult> {
        let (b, _s, _c, height, width) = images
            .dims5()
            .map_err(|_| Lux3dError::InvalidInput("track images must be [B, S, 3, H, W]"))?;
        if b != 1 {
            return Err(Lux3dError::InvalidInput(
                "VggtTrackHead::track supports batch size 1 (one clip per call)",
            ));
        }
        if queries.is_empty() {
            return Err(Lux3dError::InvalidInput(
                "track requires at least one query point",
            ));
        }
        if iters == 0 {
            return Err(Lux3dError::InvalidInput(
                "track requires at least one refinement iteration",
            ));
        }
        let device = images.device();
        let mut flat = Vec::with_capacity(queries.len() * 2);
        for query in queries {
            if query.frame != 0 {
                return Err(Lux3dError::InvalidInput(
                    "track queries must reference frame 0 (the tracker's query frame); \
                     to track from another frame, rotate the clip so that frame is first",
                ));
            }
            let [x, y] = query.xy;
            let x_max = (width - 1) as f32;
            let y_max = (height - 1) as f32;
            if !(0.0..=x_max).contains(&x) || !(0.0..=y_max).contains(&y) {
                return Err(Lux3dError::CanonicalWeightsValidation {
                    message: format!(
                        "track query ({x}, {y}) outside the model-resolution pixel range \
                         [0, {x_max}] x [0, {y_max}]"
                    ),
                });
            }
            flat.push(x);
            flat.push(y);
        }
        let query_points = Tensor::from_vec(flat, (1, queries.len(), 2), device).map_err(|_| {
            Lux3dError::InvalidInput("failed to materialize the query-point tensor")
        })?;

        let (coord_preds, vis, conf) = self
            .forward(
                cached_layers,
                images,
                patch_start_idx,
                &query_points,
                iters,
                chunk_size,
            )
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT track head forward failed: {source}"),
            })?;

        let last = coord_preds.last().ok_or_else(|| {
            Lux3dError::InvalidInput("track head produced no coordinate predictions")
        })?;
        let to_f32 = |tensor: &Tensor| -> Result<Tensor> {
            tensor
                .to_dtype(DType::F32)
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to convert track output to F32: {source}"),
                })
        };
        // [B, S, N, ...] -> [S, N, ...]
        let squeeze = |tensor: &Tensor| -> Result<Tensor> {
            tensor
                .squeeze(0)
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to squeeze track batch dim: {source}"),
                })
        };
        Ok(VggtTrackResult {
            coords: squeeze(&to_f32(last)?)?,
            visibility: squeeze(&to_f32(&vis)?)?,
            confidence: conf.as_ref().map(|c| squeeze(&to_f32(c)?)).transpose()?,
        })
    }
}

/// The exact tensor names the track head consumes under the canonical
/// `track_head.` prefix (394 tensors: 58 feature-extractor + 336 tracker).
pub fn expected_track_head_tensor_names() -> Vec<String> {
    let mut names = Vec::new();
    // feature_extractor
    let fe = "track_head.feature_extractor.";
    names.push(format!("{fe}norm.weight"));
    names.push(format!("{fe}norm.bias"));
    for idx in 0..TRACK_OUT_CHANNELS.len() {
        names.push(format!("{fe}projects.{idx}.weight"));
        names.push(format!("{fe}projects.{idx}.bias"));
    }
    for idx in [0usize, 1, 3] {
        names.push(format!("{fe}resize_layers.{idx}.weight"));
        names.push(format!("{fe}resize_layers.{idx}.bias"));
    }
    for idx in 1..=4usize {
        names.push(format!("{fe}scratch.layer{idx}_rn.weight"));
    }
    for idx in 1..=3usize {
        let rn = format!("{fe}scratch.refinenet{idx}.");
        names.push(format!("{rn}out_conv.weight"));
        names.push(format!("{rn}out_conv.bias"));
        for unit in ["resConfUnit1", "resConfUnit2"] {
            for conv in ["conv1", "conv2"] {
                names.push(format!("{rn}{unit}.{conv}.weight"));
                names.push(format!("{rn}{unit}.{conv}.bias"));
            }
        }
    }
    let rn = format!("{fe}scratch.refinenet4.");
    names.push(format!("{rn}out_conv.weight"));
    names.push(format!("{rn}out_conv.bias"));
    for conv in ["conv1", "conv2"] {
        names.push(format!("{rn}resConfUnit2.{conv}.weight"));
        names.push(format!("{rn}resConfUnit2.{conv}.bias"));
    }
    names.push(format!("{fe}scratch.output_conv1.weight"));
    names.push(format!("{fe}scratch.output_conv1.bias"));

    // tracker
    let tr = "track_head.tracker.";
    names.push(format!("{tr}fmap_norm.weight"));
    names.push(format!("{tr}fmap_norm.bias"));
    for fc in ["fc1", "fc2"] {
        names.push(format!("{tr}corr_mlp.{fc}.weight"));
        names.push(format!("{tr}corr_mlp.{fc}.bias"));
    }
    for name in [
        "input_norm.weight",
        "input_norm.bias",
        "input_transform.weight",
        "input_transform.bias",
        "output_norm.weight",
        "output_norm.bias",
        "flow_head.weight",
        "flow_head.bias",
        "virual_tracks",
    ] {
        names.push(format!("{tr}updateformer.{name}"));
    }
    let attn_block_names = [
        "norm1.weight",
        "norm1.bias",
        "attn.in_proj_weight",
        "attn.in_proj_bias",
        "attn.out_proj.weight",
        "attn.out_proj.bias",
        "norm2.weight",
        "norm2.bias",
        "mlp.fc1.weight",
        "mlp.fc1.bias",
        "mlp.fc2.weight",
        "mlp.fc2.bias",
    ];
    let cross_block_names = [
        "norm1.weight",
        "norm1.bias",
        "norm_context.weight",
        "norm_context.bias",
        "cross_attn.in_proj_weight",
        "cross_attn.in_proj_bias",
        "cross_attn.out_proj.weight",
        "cross_attn.out_proj.bias",
        "norm2.weight",
        "norm2.bias",
        "mlp.fc1.weight",
        "mlp.fc1.bias",
        "mlp.fc2.weight",
        "mlp.fc2.bias",
    ];
    for group in ["time_blocks", "space_virtual_blocks"] {
        for idx in 0..UPDATEFORMER_DEPTH {
            for name in attn_block_names {
                names.push(format!("{tr}updateformer.{group}.{idx}.{name}"));
            }
        }
    }
    for group in ["space_point2virtual_blocks", "space_virtual2point_blocks"] {
        for idx in 0..UPDATEFORMER_DEPTH {
            for name in cross_block_names {
                names.push(format!("{tr}updateformer.{group}.{idx}.{name}"));
            }
        }
    }
    for name in [
        "ffeat_norm.weight",
        "ffeat_norm.bias",
        "ffeat_updater.0.weight",
        "ffeat_updater.0.bias",
        "vis_predictor.0.weight",
        "vis_predictor.0.bias",
        "conf_predictor.0.weight",
        "conf_predictor.0.bias",
        "query_ref_token",
    ] {
        names.push(format!("{tr}{name}"));
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Vulkan-specific channel norm must reproduce candle's native
    /// last-axis LayerNorm on the upstream NHWC view.
    #[test]
    fn channel_layer_norm_matches_native() {
        let device = Device::Cpu;
        let (b, s, c, h, w) = (2, 3, 16, 5, 7);
        let elems: Vec<f32> = (0..b * s * c * h * w)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 17.0)
            .collect();
        let x = Tensor::from_vec(elems, (b, s, c, h, w), &device).expect("x");
        let gamma: Vec<f32> = (0..c).map(|i| 0.5 + i as f32 * 0.01).collect();
        let beta: Vec<f32> = (0..c).map(|i| -0.2 + i as f32 * 0.005).collect();
        let gamma = Tensor::from_vec(gamma, c, &device).expect("gamma");
        let beta = Tensor::from_vec(beta, c, &device).expect("beta");

        let native = candle_nn::ops::layer_norm(
            &x.permute((0, 1, 3, 4, 2))
                .expect("permute")
                .contiguous()
                .expect("contiguous NHWC"),
            &gamma,
            &beta,
            1e-5,
        )
        .expect("native layer_norm")
        .permute((0, 1, 4, 2, 3))
        .expect("permute back")
        .contiguous()
        .expect("contiguous");
        let ours = channel_layer_norm_nchw(&x, &gamma, &beta, 1e-5).expect("channel norm");

        let diff = (ours - native)
            .expect("sub")
            .abs()
            .expect("abs")
            .flatten_all()
            .expect("flatten")
            .max(0)
            .expect("max")
            .to_scalar::<f32>()
            .expect("scalar");
        assert!(diff < 1e-5, "channel norm differs from native: {diff}");
    }
}
