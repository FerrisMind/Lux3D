//! VGGT runtime pipeline.
//!
//! # Run options ([`VggtRunOptions`])
//!
//! [`VggtPipeline::load_with_options`] and
//! [`VggtPipeline::infer_from_path_with_options`] are the single high-level
//! entry points over the historical split APIs:
//!
//! * `dtype` — consumed at load time (`None` = F32, the reference mode). F16
//!   halves weight memory and runs the network in half precision; all public
//!   outputs (inference tensors, tracks, exports) stay F32 exactly as in the
//!   F32 pipeline, so no caller-side dtype handling is needed.
//! * `preprocess_mode` — [`VggtPreprocessMode::Crop`] (upstream default,
//!   width-518 + center crop), [`VggtPreprocessMode::Pad`] (largest side
//!   518, center-pad to 518x518 with white; no pixels discarded) or
//!   [`VggtPreprocessMode::Square`] (official COLMAP demo path: black-pad to
//!   a 1024 square, then bilinear downsample to the 518x518 model input; the
//!   official `[x1, y1, x2, y2, width, height]` tuple is carried per frame).
//! * `interval` — frame sampling interval (`None` = source default: 10 for
//!   videos, 1 otherwise).
//! * `batch_size` — sequential frame chunking (see below).
//!
//! # Batching semantics
//!
//! The aggregator and heads are *not* hardcoded to a batch of 1 (all tensor
//! reshapes carry the `[B, S, ...]` batch dim), but a single scene is always
//! run as `B=1`. `batch_size = Some(n)` splits the frame list into chunks of
//! `n` frames and runs each chunk as its own `B=1, S=n` forward pass, then
//! concatenates the outputs along the frame dimension
//! ([`merge_vggt_inference_outputs`]). This bounds peak memory; it is **not**
//! a single attention pass over all frames: the global (cross-frame)
//! attention blocks only see frames inside their chunk, so chunked results
//! differ from a full-scene pass whenever the scene spans more than one chunk
//! (frames are otherwise processed in order, and a chunk's output equals a
//! run over exactly that chunk's frames). `batch_size` of 0 is rejected;
//! `batch_size >= frame count` is a plain single pass. `pose_enc_list` is
//! concatenated per refinement iteration so entry `k` remains the k-th
//! iteration over all frames.

use std::{path::Path, sync::Arc};

use super::nn_blocks::Linear;
use crate::weights::VarBuilder;
use candle_core::{DType, Device, IndexOp, Module, Result as CandleResult, Tensor, D};
use candle_nn::{layer_norm, LayerNorm};

use super::{
    attention_math::{
        apply_layernorm_rope, apply_rope, exact_query_chunked_sdpa, position_getter, Rope2d,
        RopeEmbeddings,
    },
    nn_blocks::{linear, linear_fwd, LayerScale, Mlp},
    vggt_dpt::{VggtDptActivation, VggtDptHead},
    vggt_track::{VggtTrackHead, VggtTrackQuery, VggtTrackResult, TRACK_DEFAULT_ITERS},
};
use crate::{
    error::Lux3dError,
    export::{PointCloudFormat, VggtExportStage},
    geometry::VggtGeometryStage,
    load_canonical_weights,
    neural::VggtNeuralStage,
    preprocess::VggtPreprocessStage,
    runtime::{pi3_encoder::load_pi3_encoder, pi3_encoder::Pi3DinoEncoder, DeviceLocalCache},
    CanonicalWeightSetPaths, ModelAssetOptions, ModelFamily, Result,
};

pub use crate::preprocess::VggtPreparedInputs;
pub use crate::preprocess::VggtPreprocessMode;

const VGGT_EMBED_DIM: usize = 1024;
const VGGT_CAMERA_DIM: usize = 2048;
const VGGT_NUM_HEADS: usize = 16;
const VGGT_DEPTH: usize = 24;
const VGGT_PATCH_START_IDX: usize = 5;
const VGGT_HEAD_DIM: usize = VGGT_EMBED_DIM / VGGT_NUM_HEADS;
/// Aggregator/camera block LayerNorm eps (VGGT blocks use 1e-5, unlike the
/// DINOv2 backbone's 1e-6).
const BLOCK_LAYER_NORM_EPS: f64 = 1e-5;
const CAMERA_LAYER_NORM_EPS: f64 = 1e-5;
const ADALN_EPS: f64 = 1e-6;
const CACHED_LAYER_INDICES: [usize; 4] = [4, 11, 17, 23];

/// High-level VGGT run-surface options consumed by
/// [`VggtPipeline::load_with_options`] (the `dtype` field) and
/// [`VggtPipeline::infer_from_path_with_options`] /
/// [`VggtPipeline::infer_prepared_with_options`] (the remaining fields).
///
/// Defaults reproduce the historical behavior: F32 weights, crop-mode
/// preprocessing, source-default interval, one forward pass over all frames.
#[derive(Debug, Clone, Default)]
pub struct VggtRunOptions {
    /// Neural-network weight/activation dtype. `None` (or F32) is the
    /// reference mode; F16 halves weight memory. Public outputs remain F32
    /// either way. Ignored after load.
    pub dtype: Option<DType>,
    /// Preprocess mode for [`VggtPreprocessStage::prepare_inputs_from_path_with_mode`]
    /// (crop/pad/square; square carries the official 1024-loaded coords).
    pub preprocess_mode: VggtPreprocessMode,
    /// Frame sampling interval; `None` uses the source default (10 for
    /// videos, 1 for directories and single images).
    pub interval: Option<usize>,
    /// Frames per forward pass for sequential chunking; `None` runs all
    /// frames in one pass. Must be at least 1.
    pub batch_size: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct VggtInferenceOutput {
    pub world_points: Tensor,
    pub world_points_conf: Tensor,
    pub depth: Tensor,
    pub depth_conf: Tensor,
    pub pose_enc: Tensor,
    /// All camera-head pose encodings (one per refinement iteration, in
    /// order); `pose_enc` equals the last element. Public tensors are F32.
    pub pose_enc_list: Vec<Tensor>,
    pub extrinsics: Tensor,
    pub intrinsics: Tensor,
    pub rgb_frames: Tensor,
    pub export_mask: Tensor,
}

impl VggtInferenceOutput {
    /// Depth-based world points, matching the reference
    /// `unproject_depth_map_to_point_map` (used by the upstream COLMAP demo).
    /// Computed on demand from this output's `depth`, `extrinsics` and
    /// `intrinsics`; never stored on the struct.
    pub fn world_points_from_depth(&self) -> Result<Tensor> {
        let depth = squeeze_batch_dim(&self.depth, "VGGT depth")?;
        let extrinsics = squeeze_batch_dim(&self.extrinsics, "VGGT extrinsics")?;
        let intrinsics = squeeze_batch_dim(&self.intrinsics, "VGGT intrinsics")?;
        crate::geometry::unproject_depth_to_world(&depth, &extrinsics, &intrinsics)
    }
}

/// Drop a leading singleton batch dim (`[1, S, ...]` -> `[S, ...]`).
fn squeeze_batch_dim(tensor: &Tensor, name: &str) -> Result<Tensor> {
    let dims = tensor.dims();
    if dims.len() >= 4 && dims[0] == 1 {
        tensor
            .squeeze(0)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to squeeze VGGT {name} batch dim: {source}"),
            })
    } else {
        Ok(tensor.clone())
    }
}

#[derive(Debug, Clone)]
pub struct VggtDebugTaps {
    pub patch_tokens: Tensor,
    pub aggregator_layer4: Tensor,
    pub aggregator_layer11: Tensor,
    pub aggregator_layer17: Tensor,
    pub aggregator_layer23: Tensor,
    pub pose_enc_iterations: Vec<Tensor>,
    /// The four cached aggregator tensors (layers 4/11/17/23, `[B, S, P,
    /// 2048]`) in DPT order — exactly what `VggtPipeline::track_points`
    /// consumes, so callers can run the track head without rerunning the
    /// aggregator.
    pub aggregated_tokens_list: Vec<Tensor>,
}

#[derive(Debug, Clone)]
pub struct VggtAggregatorOutput {
    pub layers: Vec<Option<Tensor>>,
    pub final_tokens: Tensor,
    pub patch_start_idx: usize,
}

#[derive(Debug, Clone)]
pub struct VggtPipeline {
    pub weights: CanonicalWeightSetPaths,
    pub preprocess: VggtPreprocessStage,
    pub neural: VggtNeuralStage,
    pub geometry: VggtGeometryStage,
    pub export: VggtExportStage,
    /// Weight/activation dtype for the neural network. Public API outputs are
    /// always converted back to F32; F16 is an inference-time storage/compute
    /// mode.
    dtype: DType,
    bundle_cache: Arc<DeviceLocalCache<VggtModelBundle>>,
    /// Track head, lazily loaded per device on the first track call so
    /// camera/point/depth runs never pay its ~280 MB of weights.
    track_head_cache: Arc<DeviceLocalCache<VggtTrackHead>>,
}

/// Alternating-attention transformer block: `x += ls1(attn(norm1(x)))`,
/// `x += ls2(mlp(norm2(x)))` with LayerNorm eps 1e-5, GELU-erf MLP.
///
/// Used both by the aggregator (dim 1024, QK-norm + 2D RoPE, MLP hidden 4096)
/// and by the camera-head trunk (dim 2048, no QK-norm, no RoPE, hidden 8192).
#[derive(Debug)]
struct VggtBlock {
    norm1: LayerNorm,
    attn: VggtAttention,
    ls1: LayerScale,
    norm2: LayerNorm,
    mlp: Mlp,
    ls2: LayerScale,
}

impl VggtBlock {
    fn load(vb: VarBuilder, dim: usize, num_heads: usize, qk_norm: bool) -> CandleResult<Self> {
        Ok(Self {
            norm1: layer_norm(dim, BLOCK_LAYER_NORM_EPS, vb.pp("norm1"))?,
            attn: VggtAttention::load(vb.pp("attn"), dim, num_heads, qk_norm)?,
            ls1: LayerScale::new(vb.pp("ls1"), dim)?,
            norm2: layer_norm(dim, BLOCK_LAYER_NORM_EPS, vb.pp("norm2"))?,
            mlp: Mlp::new(vb.pp("mlp"), dim)?,
            ls2: LayerScale::new(vb.pp("ls2"), dim)?,
        })
    }

    fn forward(&self, xs: &Tensor, rope: Option<&RopeEmbeddings>) -> CandleResult<Tensor> {
        let residual = xs;
        let xs = self
            .ls1
            .forward(&self.attn.forward(&self.norm1.forward(xs)?, rope)?)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .ls2
            .forward(&self.mlp.forward(&self.norm2.forward(&xs)?)?)?;
        xs + residual
    }
}

#[derive(Debug)]
struct VggtAttention {
    qkv: Linear,
    proj: Linear,
    q_norm: Option<LayerNorm>,
    k_norm: Option<LayerNorm>,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl VggtAttention {
    fn load(vb: VarBuilder, dim: usize, num_heads: usize, qk_norm: bool) -> CandleResult<Self> {
        let head_dim = dim / num_heads;
        let (q_norm, k_norm) = if qk_norm {
            (
                Some(layer_norm(head_dim, BLOCK_LAYER_NORM_EPS, vb.pp("q_norm"))?),
                Some(layer_norm(head_dim, BLOCK_LAYER_NORM_EPS, vb.pp("k_norm"))?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            qkv: linear(vb.pp("qkv"), dim, dim * 3, true)?,
            proj: linear(vb.pp("proj"), dim, dim, true)?,
            q_norm,
            k_norm,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    fn forward(&self, xs: &Tensor, rope: Option<&RopeEmbeddings>) -> CandleResult<Tensor> {
        let (batch, tokens, channels) = xs.dims3()?;
        // [B, N, 3, H, D] -> [3, B, H, N, D]
        let qkv = linear_fwd(&self.qkv, xs)?
            .reshape((batch, tokens, 3, self.num_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?.contiguous()?;

        let q = match (&self.q_norm, rope) {
            (Some(norm), Some(embeddings)) => apply_layernorm_rope(&q, norm, embeddings)?,
            (Some(norm), None) => norm.forward(&q)?.contiguous()?,
            (None, Some(embeddings)) => apply_rope(&q, embeddings)?,
            (None, None) => q.contiguous()?,
        };
        let k = match (&self.k_norm, rope) {
            (Some(norm), Some(embeddings)) => apply_layernorm_rope(&k, norm, embeddings)?,
            (Some(norm), None) => norm.forward(&k)?.contiguous()?,
            (None, Some(embeddings)) => apply_rope(&k, embeddings)?,
            (None, None) => k.contiguous()?,
        };

        let out = exact_query_chunked_sdpa(&q, &k, &v, self.scale as f32, usize::MAX)?;
        let out = out.transpose(1, 2)?.reshape((batch, tokens, channels))?;
        linear_fwd(&self.proj, &out)
    }
}

/// Aggregator: DINOv2 ViT-L/14 patch backbone (reused from Pi3) plus 24
/// alternating frame/global blocks and the four cached layer outputs.
#[derive(Debug)]
struct VggtAggregator {
    encoder: Pi3DinoEncoder,
    camera_token: Tensor,
    register_token: Tensor,
    frame_blocks: Vec<VggtBlock>,
    global_blocks: Vec<VggtBlock>,
}

impl VggtAggregator {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        let encoder = load_pi3_encoder(vb.pp("patch_embed"))?;
        let camera_token = vb.get((1, 2, 1, VGGT_EMBED_DIM), "camera_token")?;
        let register_token = vb.get((1, 2, 4, VGGT_EMBED_DIM), "register_token")?;
        let frame_blocks = (0..VGGT_DEPTH)
            .map(|idx| {
                let block = VggtBlock::load(
                    vb.pp("frame_blocks").pp(idx.to_string()),
                    VGGT_EMBED_DIM,
                    VGGT_NUM_HEADS,
                    true,
                )?;
                super::drain_upload_staging(vb.device())?;
                Ok(block)
            })
            .collect::<CandleResult<Vec<_>>>()?;
        let global_blocks = (0..VGGT_DEPTH)
            .map(|idx| {
                let block = VggtBlock::load(
                    vb.pp("global_blocks").pp(idx.to_string()),
                    VGGT_EMBED_DIM,
                    VGGT_NUM_HEADS,
                    true,
                )?;
                super::drain_upload_staging(vb.device())?;
                Ok(block)
            })
            .collect::<CandleResult<Vec<_>>>()?;
        Ok(Self {
            encoder,
            camera_token,
            register_token,
            frame_blocks,
            global_blocks,
        })
    }

    /// `[B, S, 3, H, W]` normalized frames -> post-LayerNorm patch tokens
    /// `[B*S, H*W/patch^2, 1024]`.
    fn encode_patch_tokens(&self, normalized_frames: &Tensor) -> CandleResult<Tensor> {
        let (batch, frames, channels, height, width) = normalized_frames.dims5()?;
        let inputs = normalized_frames.reshape((batch * frames, channels, height, width))?;
        self.encoder.forward_patch_tokens(&inputs)
    }

    /// Expand `[1, 2, X, C]` special tokens to `[B*S, X, C]`: slot 0 for the
    /// first frame, slot 1 for every other frame (`slice_expand_and_flatten`).
    fn expand_special_tokens(token: &Tensor, batch: usize, frames: usize) -> CandleResult<Tensor> {
        let (_, slots, x, channels) = token.dims4()?;
        debug_assert_eq!(slots, 2);
        let mut indices = Vec::with_capacity(frames);
        indices.push(0u32);
        for _ in 1..frames {
            indices.push(1);
        }
        let indices = Tensor::from_vec(indices, (frames,), token.device())?;
        let selected = token.index_select(&indices, 1)?; // [1, S, X, C]
        selected
            .broadcast_as((batch, frames, x, channels))?
            .reshape((batch * frames, x, channels))?
            .contiguous()
    }

    fn forward(
        &self,
        patch_tokens: &Tensor,
        batch: usize,
        frames: usize,
        height: usize,
        width: usize,
    ) -> CandleResult<Vec<Option<Tensor>>> {
        let device = patch_tokens.device();
        let patch_h = height / 14;
        let patch_w = width / 14;
        let patch_count = patch_h * patch_w;
        if patch_tokens.dim(1)? != patch_count {
            candle_core::bail!(
                "VGGT patch token count {} does not match {patch_h}x{patch_w}",
                patch_tokens.dim(1)?
            );
        }

        let camera = Self::expand_special_tokens(&self.camera_token, batch, frames)?;
        let register = Self::expand_special_tokens(&self.register_token, batch, frames)?;
        let mut tokens = Tensor::cat(&[&camera, &register, patch_tokens], 1)?;
        let (bs, token_count, channels) = tokens.dims3()?;

        // Positions: 0 for the special tokens, (y + 1, x + 1) for patches.
        let positions = position_getter(bs, patch_h, patch_w, device)?;
        let special = Tensor::zeros((bs, VGGT_PATCH_START_IDX, 2), DType::I64, device)?;
        let positions = Tensor::cat(&[&special, &positions], 1)?;

        let rope = Rope2d::new(100.0);
        let frame_embeddings = rope.embeddings(&positions, VGGT_HEAD_DIM)?;
        let global_positions = positions.reshape((batch, frames * token_count, 2))?;
        let global_embeddings = rope.embeddings(&global_positions, VGGT_HEAD_DIM)?;

        let mut layers: Vec<Option<Tensor>> = (0..VGGT_DEPTH).map(|_| None).collect();
        for round in 0..VGGT_DEPTH {
            let frame_tokens = self.frame_blocks[round]
                .forward(&tokens, Some(&frame_embeddings))?
                .reshape((batch, frames, token_count, channels))?;
            let global_input = frame_tokens.reshape((batch, frames * token_count, channels))?;
            let global_tokens = self.global_blocks[round]
                .forward(&global_input, Some(&global_embeddings))?
                .reshape((batch, frames, token_count, channels))?;
            if CACHED_LAYER_INDICES.contains(&round) {
                layers[round] = Some(Tensor::cat(&[&frame_tokens, &global_tokens], D::Minus1)?);
            }
            tokens = global_tokens.reshape((bs, token_count, channels))?;
        }
        Ok(layers)
    }
}

#[derive(Debug)]
struct VggtCameraHead {
    token_norm: LayerNorm,
    empty_pose_tokens: Tensor,
    embed_pose: Linear,
    pose_ln_modulation: Linear,
    trunk: Vec<VggtBlock>,
    trunk_norm: LayerNorm,
    pose_branch_fc1: Linear,
    pose_branch_fc2: Linear,
}

impl VggtCameraHead {
    const TARGET_DIM: usize = 9;
    const ITERATIONS: usize = 4;

    fn load(vb: VarBuilder) -> CandleResult<Self> {
        let trunk = (0..4)
            .map(|idx| {
                let block = VggtBlock::load(
                    vb.pp("trunk").pp(idx.to_string()),
                    VGGT_CAMERA_DIM,
                    VGGT_NUM_HEADS,
                    false,
                )?;
                super::drain_upload_staging(vb.device())?;
                Ok(block)
            })
            .collect::<CandleResult<Vec<_>>>()?;
        Ok(Self {
            token_norm: layer_norm(VGGT_CAMERA_DIM, CAMERA_LAYER_NORM_EPS, vb.pp("token_norm"))?,
            empty_pose_tokens: vb.get((1, 1, Self::TARGET_DIM), "empty_pose_tokens")?,
            embed_pose: linear(vb.pp("embed_pose"), Self::TARGET_DIM, VGGT_CAMERA_DIM, true)?,
            pose_ln_modulation: linear(
                vb.pp("poseLN_modulation").pp("1"),
                VGGT_CAMERA_DIM,
                VGGT_CAMERA_DIM * 3,
                true,
            )?,
            trunk,
            trunk_norm: layer_norm(VGGT_CAMERA_DIM, CAMERA_LAYER_NORM_EPS, vb.pp("trunk_norm"))?,
            pose_branch_fc1: linear(
                vb.pp("pose_branch").pp("fc1"),
                VGGT_CAMERA_DIM,
                VGGT_CAMERA_DIM / 2,
                true,
            )?,
            pose_branch_fc2: linear(
                vb.pp("pose_branch").pp("fc2"),
                VGGT_CAMERA_DIM / 2,
                Self::TARGET_DIM,
                true,
            )?,
        })
    }

    /// `[B, S, 1374, 2048]` aggregated tokens -> 4 activated `[B, S, 9]`
    /// pose encodings (one per refinement iteration).
    fn forward(&self, tokens: &Tensor) -> CandleResult<Vec<Tensor>> {
        let pose_tokens = tokens.i((.., .., 0))?;
        let pose_tokens = self.token_norm.forward(&pose_tokens)?;
        let (batch, frames, _channels) = pose_tokens.dims3()?;

        let mut pred_pose_enc: Option<Tensor> = None;
        let mut activated_poses = Vec::with_capacity(Self::ITERATIONS);
        for _ in 0..Self::ITERATIONS {
            let module_input = match &pred_pose_enc {
                None => {
                    let empty = self
                        .empty_pose_tokens
                        .broadcast_as((batch, frames, Self::TARGET_DIM))?
                        .contiguous()?;
                    linear_fwd(&self.embed_pose, &empty)?
                }
                Some(pred) => linear_fwd(&self.embed_pose, pred)?,
            };
            // poseLN_modulation = Sequential(SiLU, Linear).
            let modulation = linear_fwd(&self.pose_ln_modulation, &module_input.silu()?)?;
            let modulation = modulation.chunk(3, D::Minus1)?;
            let shift = &modulation[0];
            let scale = &modulation[1];
            let gate = &modulation[2];

            let normed = adaln_norm(&pose_tokens, ADALN_EPS)?;
            let modulated = gate.broadcast_mul(
                &normed
                    .broadcast_mul(&scale.affine(1.0, 1.0)?)?
                    .broadcast_add(shift)?,
            )?;
            let mut hidden = (&modulated + &pose_tokens)?;
            for block in &self.trunk {
                hidden = block.forward(&hidden, None)?;
            }
            let hidden = self.trunk_norm.forward(&hidden)?;
            let delta = linear_fwd(
                &self.pose_branch_fc2,
                &linear_fwd(&self.pose_branch_fc1, &hidden)?.gelu_erf()?,
            )?;

            let pred = match &pred_pose_enc {
                None => delta,
                Some(previous) => (previous + &delta)?,
            };
            // Only the field of view is ReLU-activated; T and quat stay linear.
            let translation = pred.narrow(D::Minus1, 0, 3)?;
            let quat = pred.narrow(D::Minus1, 3, 4)?;
            let fov = pred.narrow(D::Minus1, 7, 2)?.relu()?;
            let activated = Tensor::cat(&[&translation, &quat, &fov], D::Minus1)?;
            activated_poses.push(activated);
            pred_pose_enc = Some(pred);
        }
        Ok(activated_poses)
    }
}

/// LayerNorm without learnable affine (`elementwise_affine=False`).
fn adaln_norm(xs: &Tensor, eps: f64) -> CandleResult<Tensor> {
    let mean = xs.mean_keepdim(D::Minus1)?;
    let centered = xs.broadcast_sub(&mean)?;
    let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
    centered.broadcast_div(&variance.affine(1.0, eps)?.sqrt()?)
}

/// `pose_encoding_to_extri_intri` for `absT_quaR_FoV`, returning
/// `(extrinsics [B, S, 3, 4], intrinsics [B, S, 3, 3])`.
fn decode_pose(
    pose_enc: &Tensor,
    image_height: usize,
    image_width: usize,
) -> CandleResult<(Tensor, Tensor)> {
    let translation = pose_enc.narrow(D::Minus1, 0, 3)?;
    let quat = pose_enc.narrow(D::Minus1, 3, 4)?;
    let fov_h = pose_enc.i((.., .., 7))?;
    let fov_w = pose_enc.i((.., .., 8))?;

    let rotation = quat_to_mat(&quat)?;
    let extrinsics = Tensor::cat(&[&rotation, &translation.unsqueeze(3)?], 3)?;

    let half_height = image_height as f64 / 2.0;
    let half_width = image_width as f64 / 2.0;
    // fy = (H / 2) / tan(fov_h / 2); recip() is 1/x, so multiply after.
    let fy = tan_tensor(&fov_h.affine(0.5, 0.0)?)?
        .recip()?
        .affine(half_height, 0.0)?;
    let fx = tan_tensor(&fov_w.affine(0.5, 0.0)?)?
        .recip()?
        .affine(half_width, 0.0)?;
    let zero = fx.zeros_like()?;
    let zero_w = zero.affine(1.0, half_width)?;
    let zero_h = zero.affine(1.0, half_height)?;
    let one = zero.affine(1.0, 1.0)?;
    let row0 = Tensor::stack(&[&fx, &zero, &zero_w], 2)?;
    let row1 = Tensor::stack(&[&zero, &fy, &zero_h], 2)?;
    let row2 = Tensor::stack(&[&zero, &zero, &one], 2)?;
    let intrinsics = Tensor::stack(&[&row0, &row1, &row2], 2)?;
    Ok((extrinsics, intrinsics))
}

fn tan_tensor(xs: &Tensor) -> CandleResult<Tensor> {
    xs.sin()?.broadcast_div(&xs.cos()?)
}

/// Exact `vggt.utils.rotation.quat_to_mat` (quaternion order XYZW, scalar last).
fn quat_to_mat(quaternions: &Tensor) -> CandleResult<Tensor> {
    let i = quaternions.i((.., .., 0))?;
    let j = quaternions.i((.., .., 1))?;
    let k = quaternions.i((.., .., 2))?;
    let r = quaternions.i((.., .., 3))?;
    // two_s = 2 / sum(q^2)
    let two_s = quaternions
        .sqr()?
        .sum(D::Minus1)?
        .recip()?
        .affine(2.0, 0.0)?;

    let o0 = two_s
        .broadcast_mul(&(&j.sqr()? + &k.sqr()?)?)?
        .affine(-1.0, 1.0)?;
    let o1 = two_s.broadcast_mul(&(&(&i * &j)? - &(&k * &r)?)?)?;
    let o2 = two_s.broadcast_mul(&(&(&i * &k)? + &(&j * &r)?)?)?;
    let o3 = two_s.broadcast_mul(&(&(&i * &j)? + &(&k * &r)?)?)?;
    let o4 = two_s
        .broadcast_mul(&(&i.sqr()? + &k.sqr()?)?)?
        .affine(-1.0, 1.0)?;
    let o5 = two_s.broadcast_mul(&(&(&j * &k)? - &(&i * &r)?)?)?;
    let o6 = two_s.broadcast_mul(&(&(&i * &k)? - &(&j * &r)?)?)?;
    let o7 = two_s.broadcast_mul(&(&(&j * &k)? + &(&i * &r)?)?)?;
    let o8 = two_s
        .broadcast_mul(&(&i.sqr()? + &j.sqr()?)?)?
        .affine(-1.0, 1.0)?;

    let (batch, frames, _) = quaternions.dims3()?;
    Tensor::stack(&[&o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8], D::Minus1)?
        .reshape((batch, frames, 3, 3))
}

#[derive(Debug)]
struct VggtModelBundle {
    aggregator: VggtAggregator,
    camera_head: VggtCameraHead,
    point_head: VggtDptHead,
    depth_head: VggtDptHead,
}

impl VggtModelBundle {
    fn load(weights: &CanonicalWeightSetPaths, device: &Device, dtype: DType) -> Result<Self> {
        let vb = unsafe {
            weights.var_builder(dtype, device).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to open VGGT canonical weights: {source}"),
                }
            })?
        };
        let aggregator = VggtAggregator::load(vb.pp("aggregator")).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to load VGGT aggregator: {source}"),
            }
        })?;
        let camera_head = VggtCameraHead::load(vb.pp("camera_head")).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to load VGGT camera head: {source}"),
            }
        })?;
        let point_head = VggtDptHead::load(
            vb.pp("point_head"),
            VGGT_CAMERA_DIM,
            4,
            VggtDptActivation::InvLog,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to load VGGT point head: {source}"),
        })?;
        let depth_head = VggtDptHead::load(
            vb.pp("depth_head"),
            VGGT_CAMERA_DIM,
            2,
            VggtDptActivation::Exp,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to load VGGT depth head: {source}"),
        })?;
        Ok(Self {
            aggregator,
            camera_head,
            point_head,
            depth_head,
        })
    }
}

impl VggtPipeline {
    pub fn load(model_assets: ModelAssetOptions) -> Result<Self> {
        Self::load_with_dtype(model_assets, DType::F32)
    }

    /// Load VGGT with an explicit neural-network dtype. `F32` is the default
    /// reference mode; `F16` halves weight memory and runs the aggregator,
    /// camera head and DPT heads in half precision on CUDA/Vulkan/WGPU.
    pub fn load_with_dtype(model_assets: ModelAssetOptions, dtype: DType) -> Result<Self> {
        Ok(Self {
            weights: load_canonical_weights(ModelFamily::Vggt, model_assets)?,
            preprocess: VggtPreprocessStage::default(),
            neural: VggtNeuralStage,
            geometry: Default::default(),
            export: Default::default(),
            dtype,
            bundle_cache: Arc::default(),
            track_head_cache: Arc::default(),
        })
    }

    /// Load VGGT from a [`VggtRunOptions`]: only `options.dtype` is consumed
    /// here (`None` = F32); the preprocess/batch fields apply at inference
    /// time via [`Self::infer_from_path_with_options`]. Weights are not
    /// touched until the first forward pass on a device.
    pub fn load_with_options(
        model_assets: ModelAssetOptions,
        options: VggtRunOptions,
    ) -> Result<Self> {
        Self::load_with_dtype(model_assets, options.dtype.unwrap_or(DType::F32))
    }

    /// The neural-network dtype this pipeline was loaded with. Public
    /// inference outputs are F32 regardless.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    fn bundle_for(&self, device: &Device) -> Result<Arc<VggtModelBundle>> {
        self.bundle_cache.get_or_try_init(device, || {
            VggtModelBundle::load(&self.weights, device, self.dtype)
        })
    }

    pub fn prepare_inputs_from_path(
        &self,
        source: &Path,
        interval: Option<usize>,
        device: &Device,
    ) -> Result<VggtPreparedInputs> {
        self.preprocess
            .prepare_inputs_from_path(source, interval, device)
    }

    /// [`Self::prepare_inputs_from_path`] with an explicit preprocess mode
    /// (crop = upstream default, pad = lossless white-padded square,
    /// square = official 1024-loaded COLMAP preprocessing).
    pub fn prepare_inputs_from_path_with_mode(
        &self,
        source: &Path,
        interval: Option<usize>,
        mode: VggtPreprocessMode,
        device: &Device,
    ) -> Result<VggtPreparedInputs> {
        self.preprocess
            .prepare_inputs_from_path_with_mode(source, interval, mode, device)
    }

    pub fn infer_from_path(
        &self,
        source: &Path,
        interval: Option<usize>,
        device: &Device,
    ) -> Result<VggtInferenceOutput> {
        let prepared = self.prepare_inputs_from_path(source, interval, device)?;
        self.infer_tensors(&prepared)
    }

    /// Crop-mode inference with an explicit preprocess mode.
    pub fn infer_from_path_with_mode(
        &self,
        source: &Path,
        interval: Option<usize>,
        mode: VggtPreprocessMode,
        device: &Device,
    ) -> Result<VggtInferenceOutput> {
        let prepared = self.prepare_inputs_from_path_with_mode(source, interval, mode, device)?;
        self.infer_tensors(&prepared)
    }

    /// Single high-level entry point over the whole run surface: preprocess
    /// with `options.preprocess_mode` / `options.interval`, then run with
    /// sequential chunking of `options.batch_size` frames per forward pass
    /// (see the module docs for the exact semantics; `batch_size` of 0 is
    /// rejected before any work).
    pub fn infer_from_path_with_options(
        &self,
        source: &Path,
        device: &Device,
        options: &VggtRunOptions,
    ) -> Result<VggtInferenceOutput> {
        if options.batch_size == Some(0) {
            return Err(Lux3dError::InvalidInput(
                "VGGT batch_size must be at least 1",
            ));
        }
        let prepared = self.prepare_inputs_from_path_with_mode(
            source,
            options.interval,
            options.preprocess_mode,
            device,
        )?;
        self.infer_prepared_with_options(&prepared, options)
    }

    /// Run an already-prepared input with the batch-size option applied
    /// (`options.dtype` / preprocess fields were consumed earlier in the
    /// pipeline).
    pub fn infer_prepared_with_options(
        &self,
        prepared: &VggtPreparedInputs,
        options: &VggtRunOptions,
    ) -> Result<VggtInferenceOutput> {
        match options.batch_size {
            None => self.infer_tensors(prepared),
            Some(batch_size) => {
                if batch_size == 0 {
                    return Err(Lux3dError::InvalidInput(
                        "VGGT batch_size must be at least 1",
                    ));
                }
                let frame_count = prepared.normalized_frames.dim(1).map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("failed to read VGGT frame count: {source}"),
                    }
                })?;
                if batch_size >= frame_count {
                    return self.infer_tensors(prepared);
                }
                let chunks = chunk_prepared_inputs(prepared, batch_size)?;
                let outputs = chunks
                    .iter()
                    .map(|chunk| self.infer_tensors(chunk))
                    .collect::<Result<Vec<_>>>()?;
                merge_vggt_inference_outputs(outputs)
            }
        }
    }

    pub fn infer_tensors(&self, prepared: &VggtPreparedInputs) -> Result<VggtInferenceOutput> {
        self.run_forward(prepared, false).map(|(output, _)| output)
    }

    /// Export `world_points_conf > 1.0` (the VGGT demo threshold) as `.ply`
    /// (primary) or `.xyz`.
    pub fn export_point_cloud(
        &self,
        output: &VggtInferenceOutput,
        path: &Path,
    ) -> Result<PointCloudFormat> {
        self.export_point_cloud_with_threshold(output, path, 1.0)
    }

    /// Export `world_points_conf > conf_threshold` (strict) as `.ply` or `.xyz`.
    pub fn export_point_cloud_with_threshold(
        &self,
        output: &VggtInferenceOutput,
        path: &Path,
        conf_threshold: f32,
    ) -> Result<PointCloudFormat> {
        let point_cloud = self.geometry.assemble_cpu(output, conf_threshold)?;
        self.export.write_point_cloud(&point_cloud, path)
    }

    /// Export `world_points_conf > 1.0` as ASCII PLY (legacy entry point).
    pub fn export_ply(&self, output: &VggtInferenceOutput, path: &Path) -> Result<()> {
        self.export_ply_with_threshold(output, path, 1.0)
    }

    /// Export `world_points_conf > conf_threshold` (strict) as ASCII PLY.
    pub fn export_ply_with_threshold(
        &self,
        output: &VggtInferenceOutput,
        path: &Path,
        conf_threshold: f32,
    ) -> Result<()> {
        let point_cloud = self.geometry.assemble_cpu(output, conf_threshold)?;
        self.export.write_ply(&point_cloud, path)
    }

    /// Dump the raw inference tensors as `.npz` so a quantized run can be
    /// compared against the float run (Chamfer / PSNR / camera-pose AUC).
    ///
    /// Includes `pose_enc` (the last camera-refinement iteration) and
    /// `pose_enc_list`, the per-iteration pose encodings stacked as `[K, S, 9]`
    /// (a single-iteration output stays `[1, S, 9]`). The official Gradio demo
    /// saves the `predictions.npz` keys `images`, `extrinsic`, `intrinsic` and
    /// `world_points_from_depth` (`demo_gradio.py:216-218, 84-93`); those are
    /// written alongside the historical keys with the same batch-squeezed
    /// shapes the demo produces.
    pub fn export_inference_npz(&self, output: &VggtInferenceOutput, path: &Path) -> Result<()> {
        write_inference_npz(output, path)
    }

    pub fn debug_forward(
        &self,
        prepared: &VggtPreparedInputs,
    ) -> Result<(VggtInferenceOutput, VggtDebugTaps)> {
        let (output, taps) = self.run_forward(prepared, true)?;
        let taps = taps.ok_or_else(|| Lux3dError::CanonicalWeightsValidation {
            message: "VGGT debug forward did not capture taps".to_string(),
        })?;
        Ok((output, taps))
    }

    pub fn encode_patch_tokens(&self, normalized_frames: &Tensor) -> Result<Tensor> {
        let bundle = self.bundle_for(normalized_frames.device())?;
        bundle
            .aggregator
            .encode_patch_tokens(normalized_frames)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT encoder forward failed: {source}"),
            })
    }

    pub fn run_aggregator(&self, prepared: &VggtPreparedInputs) -> Result<VggtAggregatorOutput> {
        let normalized_frames =
            prepared
                .normalized_frames
                .to_dtype(self.dtype)
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!(
                        "failed to convert VGGT input to {dtype:?}: {source}",
                        dtype = self.dtype
                    ),
                })?;
        let device = normalized_frames.device();
        let bundle = self.bundle_for(device)?;
        let (batch, frames, _channels, height, width) = input_dims(&normalized_frames)?;
        let patch_tokens = bundle
            .aggregator
            .encode_patch_tokens(&normalized_frames)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT encoder forward failed: {source}"),
            })?;
        let layers = bundle
            .aggregator
            .forward(&patch_tokens, batch, frames, height, width)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT aggregator forward failed: {source}"),
            })?;
        let final_tokens = layers[CACHED_LAYER_INDICES[3]].clone().ok_or_else(|| {
            Lux3dError::CanonicalWeightsValidation {
                message: "VGGT aggregator did not produce layer 23".to_string(),
            }
        })?;
        Ok(VggtAggregatorOutput {
            layers,
            final_tokens,
            patch_start_idx: VGGT_PATCH_START_IDX,
        })
    }

    /// Load (or fetch the cached) track head for a device. Lazily initialized:
    /// plain camera/point/depth inference never loads the track head's ~280 MB
    /// of weights.
    pub fn track_head_for(&self, device: &Device) -> Result<Arc<VggtTrackHead>> {
        self.track_head_cache.get_or_try_init(device, || unsafe {
            VggtTrackHead::load_from_canonical(&self.weights, device, self.dtype)
        })
    }

    /// Track explicit query points through the track head.
    ///
    /// * `aggregated_tokens`: the four cached aggregator tensors (layers
    ///   4/11/17/23) in DPT order, each `[B, S, P, 2048]` — see
    ///   [`VggtDebugTaps::aggregated_tokens_list`] or
    ///   [`VggtPipeline::run_aggregator`].
    /// * `images`: `[B, S, 3, H, W]` model-resolution frames (only the shape
    ///   is used, as upstream).
    /// * `queries`: pixel coordinates in the model input resolution; the
    ///   query frame must be 0 (see `runtime::vggt_track` module docs).
    /// * `iters`: refinement iterations (upstream default 4).
    ///
    /// Returns `coords [S, Q, 2]` (model-resolution pixel coordinates,
    /// `coords[0] == queries` exactly), `visibility [S, Q]` and
    /// `confidence [S, Q]`, all F32 on `device`.
    pub fn track_points(
        &self,
        aggregated_tokens: &[Tensor],
        images: &Tensor,
        patch_start_idx: usize,
        queries: &[VggtTrackQuery],
        iters: usize,
        device: &Device,
    ) -> Result<VggtTrackResult> {
        if aggregated_tokens.len() != CACHED_LAYER_INDICES.len() {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "track_points expects the {} cached aggregator tensors (layers 4/11/17/23 in order), got {}",
                    CACHED_LAYER_INDICES.len(),
                    aggregated_tokens.len()
                ),
            });
        }
        let track_head = self.track_head_for(device)?;
        for token in aggregated_tokens {
            if token.device().location() != device.location() {
                return Err(Lux3dError::CanonicalWeightsValidation {
                    message: "track_points inputs must live on the requested device".to_string(),
                });
            }
        }
        let refs: Vec<&Tensor> = aggregated_tokens.iter().collect();
        track_head.track(
            &refs,
            images,
            patch_start_idx,
            queries,
            iters,
            dpt_chunk_size(),
        )
    }

    /// Track queries using a prior [`VggtPipeline::run_aggregator`] result and
    /// the prepared inputs, so the aggregator is not rerun.
    pub fn track_points_from_aggregator(
        &self,
        aggregator: &VggtAggregatorOutput,
        prepared: &VggtPreparedInputs,
        queries: &[VggtTrackQuery],
        iters: usize,
    ) -> Result<VggtTrackResult> {
        let cached = [
            cached_layer(&aggregator.layers, CACHED_LAYER_INDICES[0])?,
            cached_layer(&aggregator.layers, CACHED_LAYER_INDICES[1])?,
            cached_layer(&aggregator.layers, CACHED_LAYER_INDICES[2])?,
            cached_layer(&aggregator.layers, CACHED_LAYER_INDICES[3])?,
        ];
        self.track_points(
            &cached,
            &prepared.normalized_frames,
            aggregator.patch_start_idx,
            queries,
            iters,
            prepared.normalized_frames.device(),
        )
    }

    /// One-call inference + tracking: the aggregator runs once and its cached
    /// layers feed the camera/depth/point heads and the track head.
    /// `iters` defaults to the upstream `TrackHead.iters = 4`.
    pub fn run_with_tracks(
        &self,
        prepared: &VggtPreparedInputs,
        queries: &[VggtTrackQuery],
        iters: Option<usize>,
    ) -> Result<(VggtInferenceOutput, VggtTrackResult)> {
        let (output, _, track) = self.run_forward_inner(
            prepared,
            false,
            Some((queries, iters.unwrap_or(TRACK_DEFAULT_ITERS))),
        )?;
        let track = track.ok_or_else(|| Lux3dError::CanonicalWeightsValidation {
            message: "VGGT track run produced no tracks".to_string(),
        })?;
        Ok((output, track))
    }

    fn run_forward(
        &self,
        prepared: &VggtPreparedInputs,
        capture_taps: bool,
    ) -> Result<(VggtInferenceOutput, Option<VggtDebugTaps>)> {
        let (output, taps, _tracks) = self.run_forward_inner(prepared, capture_taps, None)?;
        Ok((output, taps))
    }

    /// Full forward. `track = Some((queries, iters))` additionally runs the
    /// track head on the same aggregator output (no second aggregator pass).
    fn run_forward_inner(
        &self,
        prepared: &VggtPreparedInputs,
        capture_taps: bool,
        track: Option<(&[VggtTrackQuery], usize)>,
    ) -> Result<(
        VggtInferenceOutput,
        Option<VggtDebugTaps>,
        Option<VggtTrackResult>,
    )> {
        let normalized_frames =
            prepared
                .normalized_frames
                .to_dtype(self.dtype)
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!(
                        "failed to convert VGGT input to {dtype:?}: {source}",
                        dtype = self.dtype
                    ),
                })?;
        let device = normalized_frames.device();
        let bundle = self.bundle_for(device)?;
        let (batch, frames, _channels, height, width) = input_dims(&normalized_frames)?;
        let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
        let stage = |name: &str, t: &std::time::Instant| {
            if stage_time {
                eprintln!("[stage]   {name}: {:.2}s", t.elapsed().as_secs_f64());
            }
        };

        let t = std::time::Instant::now();
        let patch_tokens = bundle
            .aggregator
            .encode_patch_tokens(&normalized_frames)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT encoder forward failed: {source}"),
            })?;
        stage("encode_patch_tokens", &t);

        let t = std::time::Instant::now();
        let layers = bundle
            .aggregator
            .forward(&patch_tokens, batch, frames, height, width)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT aggregator forward failed: {source}"),
            })?;
        stage("aggregator", &t);

        let t = std::time::Instant::now();
        let layer23 = layers[CACHED_LAYER_INDICES[3]].as_ref().ok_or_else(|| {
            Lux3dError::CanonicalWeightsValidation {
                message: "VGGT aggregator did not produce layer 23".to_string(),
            }
        })?;
        let pose_enc_iterations = bundle.camera_head.forward(layer23).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT camera head forward failed: {source}"),
            }
        })?;
        let pose_enc = pose_enc_iterations.last().cloned().ok_or_else(|| {
            Lux3dError::CanonicalWeightsValidation {
                message: "VGGT camera head produced no pose encodings".to_string(),
            }
        })?;
        stage("camera_head", &t);

        let t = std::time::Instant::now();
        let (extrinsics, intrinsics) = decode_pose(&pose_enc, height, width).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT pose decode failed: {source}"),
            }
        })?;
        stage("pose_decode", &t);

        let chunk_size = dpt_chunk_size();
        let t = std::time::Instant::now();
        let (depth, depth_conf) = bundle
            .depth_head
            .forward(
                &layers,
                VGGT_PATCH_START_IDX,
                batch,
                frames,
                chunk_size,
                height,
                width,
            )
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT depth head forward failed: {source}"),
            })?;
        let depth = depth
            .reshape((batch, frames, height, width, 1))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to reshape VGGT depth: {source}"),
            })?;
        let depth_conf = depth_conf
            .reshape((batch, frames, height, width))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to reshape VGGT depth_conf: {source}"),
            })?;
        stage("depth_head", &t);

        let t = std::time::Instant::now();
        let (world_points, world_points_conf) = bundle
            .point_head
            .forward(
                &layers,
                VGGT_PATCH_START_IDX,
                batch,
                frames,
                chunk_size,
                height,
                width,
            )
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT point head forward failed: {source}"),
            })?;
        let world_points = world_points
            .reshape((batch, frames, height, width, 3))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to reshape VGGT world_points: {source}"),
            })?;
        let world_points_conf = world_points_conf
            .reshape((batch, frames, height, width))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to reshape VGGT world_points_conf: {source}"),
            })?;
        stage("point_head", &t);

        let t = std::time::Instant::now();
        let export_mask = world_points_conf
            .gt(1.0f32)
            .and_then(|mask| mask.to_dtype(DType::U8))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT export mask computation failed: {source}"),
            })?;
        stage("postprocess", &t);

        let cached_layers = [
            cached_layer(&layers, CACHED_LAYER_INDICES[0])?,
            cached_layer(&layers, CACHED_LAYER_INDICES[1])?,
            cached_layer(&layers, CACHED_LAYER_INDICES[2])?,
            cached_layer(&layers, CACHED_LAYER_INDICES[3])?,
        ];

        let t = std::time::Instant::now();
        let track_result = match track {
            Some((queries, iters)) => {
                let track_head = self.track_head_for(&device)?;
                let refs: Vec<&Tensor> = cached_layers.iter().collect();
                Some(track_head.track(
                    &refs,
                    &normalized_frames,
                    VGGT_PATCH_START_IDX,
                    queries,
                    iters,
                    chunk_size,
                )?)
            }
            None => None,
        };
        stage("track_head", &t);

        let taps = if capture_taps {
            Some(VggtDebugTaps {
                patch_tokens: patch_tokens.clone(),
                aggregator_layer4: cached_layers[0].clone(),
                aggregator_layer11: cached_layers[1].clone(),
                aggregator_layer17: cached_layers[2].clone(),
                aggregator_layer23: cached_layers[3].clone(),
                pose_enc_iterations: pose_enc_iterations.clone(),
                aggregated_tokens_list: cached_layers.to_vec(),
            })
        } else {
            None
        };

        // Public tensors stay F32 regardless of the internal compute dtype so
        // geometry/export code and callers keep the historical contract.
        let to_f32 = |tensor: &Tensor| -> Result<Tensor> {
            tensor
                .to_dtype(DType::F32)
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to convert VGGT output to f32: {source}"),
                })
        };
        let pose_enc_list = pose_enc_iterations
            .iter()
            .map(to_f32)
            .collect::<Result<Vec<_>>>()?;
        Ok((
            VggtInferenceOutput {
                world_points: to_f32(&world_points)?,
                world_points_conf: to_f32(&world_points_conf)?,
                depth: to_f32(&depth)?,
                depth_conf: to_f32(&depth_conf)?,
                pose_enc: to_f32(&pose_enc)?,
                pose_enc_list,
                extrinsics: to_f32(&extrinsics)?,
                intrinsics: to_f32(&intrinsics)?,
                rgb_frames: prepared.rgb_frames.clone(),
                export_mask,
            },
            taps,
            track_result,
        ))
    }
}

fn cached_layer(layers: &[Option<Tensor>], idx: usize) -> Result<Tensor> {
    layers[idx]
        .clone()
        .ok_or_else(|| Lux3dError::CanonicalWeightsValidation {
            message: format!("VGGT aggregator did not produce layer {idx}"),
        })
}

/// Stack the per-iteration pose encodings `[1, S, 9]` into one `[K, S, 9]`
/// tensor for the `.npz` dump (a single-iteration list stays `[1, S, 9]`).
fn stack_pose_enc_list(list: &[Tensor]) -> Result<Tensor> {
    if list.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "VGGT pose_enc_list is empty; nothing to stack for the npz dump",
        ));
    }
    let squeezed = list
        .iter()
        .map(|tensor| {
            if tensor.dims().first() == Some(&1) {
                tensor
                    .squeeze(0)
                    .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                        message: format!(
                            "failed to squeeze VGGT pose_enc_list batch dim: {source}"
                        ),
                    })
            } else {
                Ok(tensor.clone())
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Tensor::stack(&squeezed, 0).map_err(|source| Lux3dError::CanonicalWeightsValidation {
        message: format!("failed to stack VGGT pose_enc_list: {source}"),
    })
}

/// Write the `.npz` prediction dump. Free function so unit tests can cover
/// the key set and shapes without loading VGGT weights.
fn write_inference_npz(output: &VggtInferenceOutput, path: &Path) -> Result<()> {
    use ndarray::{ArrayD, IxDyn};

    let pose_enc_list = stack_pose_enc_list(&output.pose_enc_list)?;
    // Official demo keys: `images` ([S,3,H,W], preprocessed [0,1]),
    // `extrinsic`/`intrinsic` ([S,3,4]/[S,3,3], batch-squeezed) and
    // `world_points_from_depth` ([S,H,W,3]).
    let world_points_from_depth = output.world_points_from_depth()?;
    let extrinsic = squeeze_batch_dim(&output.extrinsics, "VGGT extrinsics")?;
    let intrinsic = squeeze_batch_dim(&output.intrinsics, "VGGT intrinsics")?;
    let file = std::fs::File::create(path).map_err(|source| Lux3dError::ModelAssetIo {
        path: path.to_path_buf(),
        source,
    })?;
    let mut npz = ndarray_npy::NpzWriter::new(file);
    let fields: [(&str, &Tensor); 12] = [
        ("world_points", &output.world_points),
        ("world_points_conf", &output.world_points_conf),
        ("depth", &output.depth),
        ("depth_conf", &output.depth_conf),
        ("extrinsics", &output.extrinsics),
        ("intrinsics", &output.intrinsics),
        ("pose_enc", &output.pose_enc),
        ("pose_enc_list", &pose_enc_list),
        ("images", &output.rgb_frames),
        ("extrinsic", &extrinsic),
        ("intrinsic", &intrinsic),
        ("world_points_from_depth", &world_points_from_depth),
    ];
    for (name, tensor) in fields {
        let tensor = tensor
            .to_device(&Device::Cpu)
            .and_then(|tensor| tensor.to_dtype(DType::F32))
            .and_then(|tensor| tensor.contiguous())
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to stage `{name}` for npz: {source}"),
            })?;
        let shape = tensor.dims().to_vec();
        let data = tensor
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to read `{name}` for npz: {source}"),
            })?;
        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|_| Lux3dError::InvalidInput("failed to build npz tensor"))?;
        npz.add_array(name, &array)
            .map_err(|_| Lux3dError::InvalidInput("failed to write npz tensor"))?;
    }
    npz.finish()
        .map_err(|_| Lux3dError::InvalidInput("failed to finish npz"))?;
    Ok(())
}

fn input_dims(normalized_frames: &Tensor) -> Result<(usize, usize, usize, usize, usize)> {
    normalized_frames
        .dims5()
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("VGGT normalized frames must have shape [B,S,3,H,W]: {source}"),
        })
}

/// Split a prepared input into sequential chunks of `batch_size` frames along
/// the frame dimension. `batch_size` of 0 is rejected; the caller has already
/// handled the `batch_size >= frame_count` single-pass case.
fn chunk_prepared_inputs(
    prepared: &VggtPreparedInputs,
    batch_size: usize,
) -> Result<Vec<VggtPreparedInputs>> {
    if batch_size == 0 {
        return Err(Lux3dError::InvalidInput(
            "VGGT batch_size must be at least 1",
        ));
    }
    let frame_count = prepared.normalized_frames.dim(1).map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to read VGGT frame count: {source}"),
        }
    })?;

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < frame_count {
        let len = batch_size.min(frame_count - start);
        let rgb_frames = prepared
            .rgb_frames
            .narrow(0, start, len)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to chunk VGGT rgb frames: {source}"),
            })?;
        let normalized_frames =
            prepared
                .normalized_frames
                .narrow(1, start, len)
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to chunk VGGT normalized frames: {source}"),
                })?;
        chunks.push(VggtPreparedInputs {
            interval: prepared.interval,
            original_size: prepared.original_size,
            target_size: prepared.target_size,
            preprocess_mode: prepared.preprocess_mode,
            rgb_frames,
            normalized_frames,
            original_coords: prepared.original_coords[start..start + len].to_vec(),
        });
        start += len;
    }
    Ok(chunks)
}

/// Concatenate per-chunk inference outputs along the frame dimension: the
/// `[1, S, ...]` public tensors along dim 1, `rgb_frames` `[S, 3, H, W]`
/// along dim 0 and `pose_enc_list` element-wise (entry `k` is the k-th
/// refinement iteration over all frames). A single output is returned as-is;
/// an empty list is rejected.
pub fn merge_vggt_inference_outputs(
    outputs: impl IntoIterator<Item = VggtInferenceOutput>,
) -> Result<VggtInferenceOutput> {
    let mut outputs: Vec<VggtInferenceOutput> = outputs.into_iter().collect();
    if outputs.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "cannot merge an empty list of VGGT inference outputs",
        ));
    }
    if outputs.len() == 1 {
        return Ok(outputs.pop().expect("checked non-empty"));
    }

    let cat_frames = |name: &str, tensors: Vec<&Tensor>, dim: usize| -> Result<Tensor> {
        Tensor::cat(&tensors, dim).map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to merge VGGT {name} along dim {dim}: {source}"),
        })
    };
    // Every field except `rgb_frames` carries a leading singleton batch dim.
    for output in &outputs {
        let batched: [(&str, &Tensor); 8] = [
            ("world_points", &output.world_points),
            ("world_points_conf", &output.world_points_conf),
            ("depth", &output.depth),
            ("depth_conf", &output.depth_conf),
            ("pose_enc", &output.pose_enc),
            ("extrinsics", &output.extrinsics),
            ("intrinsics", &output.intrinsics),
            ("export_mask", &output.export_mask),
        ];
        for (name, tensor) in batched {
            if tensor.dims().first() != Some(&1) {
                return Err(Lux3dError::CanonicalWeightsValidation {
                    message: format!(
                        "VGGT merge expects a leading batch dim of 1 on {name}, got {:?}",
                        tensor.dims()
                    ),
                });
            }
        }
    }

    let pose_enc_list_len = outputs[0].pose_enc_list.len();
    for output in &outputs {
        if output.pose_enc_list.len() != pose_enc_list_len {
            return Err(Lux3dError::InvalidInput(
                "VGGT merge expects the same pose_enc_list length in every chunk",
            ));
        }
    }

    let world_points = cat_frames(
        "world_points",
        outputs.iter().map(|o| &o.world_points).collect(),
        1,
    )?;
    let world_points_conf = cat_frames(
        "world_points_conf",
        outputs.iter().map(|o| &o.world_points_conf).collect(),
        1,
    )?;
    let depth = cat_frames("depth", outputs.iter().map(|o| &o.depth).collect(), 1)?;
    let depth_conf = cat_frames(
        "depth_conf",
        outputs.iter().map(|o| &o.depth_conf).collect(),
        1,
    )?;
    let pose_enc = cat_frames("pose_enc", outputs.iter().map(|o| &o.pose_enc).collect(), 1)?;
    let extrinsics = cat_frames(
        "extrinsics",
        outputs.iter().map(|o| &o.extrinsics).collect(),
        1,
    )?;
    let intrinsics = cat_frames(
        "intrinsics",
        outputs.iter().map(|o| &o.intrinsics).collect(),
        1,
    )?;
    let rgb_frames = cat_frames(
        "rgb_frames",
        outputs.iter().map(|o| &o.rgb_frames).collect(),
        0,
    )?;
    let export_mask = cat_frames(
        "export_mask",
        outputs.iter().map(|o| &o.export_mask).collect(),
        1,
    )?;
    let pose_enc_list = (0..pose_enc_list_len)
        .map(|iteration| {
            cat_frames(
                "pose_enc_list",
                outputs
                    .iter()
                    .map(|output| &output.pose_enc_list[iteration])
                    .collect(),
                1,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(VggtInferenceOutput {
        world_points,
        world_points_conf,
        depth,
        depth_conf,
        pose_enc,
        pose_enc_list,
        extrinsics,
        intrinsics,
        rgb_frames,
        export_mask,
    })
}

/// Reference `DPTHead(frames_chunk_size=8)` processes frames in groups of 8;
/// frames are independent, so any chunk size is numerically identical. The
/// default is 1 to bound peak VRAM on 12 GiB cards (a 518x518 256-channel
/// activation is ~275 MB per frame).
fn dpt_chunk_size() -> usize {
    std::env::var("LUX3D_VGGT_DPT_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `world_points_from_depth` squeezes the public `[1, S, ...]` batch dim
    /// and matches the reference unprojection math.
    #[test]
    fn world_points_from_depth_unprojects_public_tensors() {
        let device = Device::Cpu;
        let zeros = |shape: (usize, usize, usize, usize, usize)| {
            Tensor::zeros(shape, DType::F32, &device).expect("zeros")
        };
        let depth = Tensor::from_vec(
            vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            (1, 2, 2, 2, 1),
            &device,
        )
        .expect("depth");
        let extrinsics = Tensor::from_vec(
            vec![
                1.0f32, 0.0, 0.0, 1.0, //
                0.0, 1.0, 0.0, -2.0, //
                0.0, 0.0, 1.0, 3.0, //
                1.0, 0.0, 0.0, 1.0, //
                0.0, 1.0, 0.0, -2.0, //
                0.0, 0.0, 1.0, 3.0,
            ],
            (1, 2, 3, 4),
            &device,
        )
        .expect("extrinsics");
        let intrinsics = Tensor::from_vec(
            vec![
                10.0f32, 0.0, 1.0, //
                0.0, 10.0, 1.0, //
                0.0, 0.0, 1.0, //
                10.0, 0.0, 1.0, //
                0.0, 10.0, 1.0, //
                0.0, 0.0, 1.0,
            ],
            (1, 2, 3, 3),
            &device,
        )
        .expect("intrinsics");
        let pose_enc = Tensor::zeros((1, 2, 9), DType::F32, &device).expect("pose enc");

        let output = VggtInferenceOutput {
            world_points: zeros((1, 2, 2, 2, 3)),
            world_points_conf: Tensor::zeros((1, 2, 2, 2), DType::F32, &device).expect("conf"),
            depth,
            depth_conf: Tensor::zeros((1, 2, 2, 2), DType::F32, &device).expect("depth conf"),
            pose_enc: pose_enc.clone(),
            pose_enc_list: vec![pose_enc],
            extrinsics,
            intrinsics,
            rgb_frames: Tensor::zeros((2, 3, 2, 2), DType::F32, &device).expect("rgb"),
            export_mask: Tensor::zeros((1, 2, 2, 2), DType::U8, &device).expect("mask"),
        };

        let world = output.world_points_from_depth().expect("unproject");
        assert_eq!(world.dims(), &[2, 2, 2, 3]);
        let values = world
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .expect("world vec");

        // Frame 0, pixel (0, 0): p_cam = ((0-1)*2/10, (0-1)*2/10, 2) =
        // (-0.2, -0.2, 2); R = I, so world = p_cam - t = (-1.2, 1.8, -1.0).
        assert!((values[0] - (-1.2)).abs() < 1e-6);
        assert!((values[1] - 1.8).abs() < 1e-6);
        assert!((values[2] - (-1.0)).abs() < 1e-6);
        // Frame 0, pixel (1, 0): z = 4 -> ((0-1)*0.4, (1-1)*0.4, 4) - t =
        // (-1.4, 2.0, 1.0); flat offset (y=1, x=0) -> value index 6.
        assert!((values[6] - (-1.4)).abs() < 1e-6);
        assert!((values[7] - 2.0).abs() < 1e-6);
        assert!((values[8] - 1.0).abs() < 1e-6);
    }

    /// The `.npz` dump carries the official demo's prediction keys with the
    /// batch-squeezed shapes (`demo_gradio.py:216-218, 84-93`) alongside the
    /// historical keys.
    #[test]
    fn export_inference_npz_includes_official_prediction_keys() {
        let output = synthetic_output(2, 10.0);
        let path = std::env::temp_dir().join(format!(
            "lux3d-vggt-npz-{}-{}.npz",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        write_inference_npz(&output, &path).expect("write npz");

        let mut npz = ndarray_npy::NpzReader::new(std::fs::File::open(&path).expect("open npz"))
            .expect("npz reader");
        let mut names = npz.names().expect("npz names");
        names.sort();
        let mut expected = vec![
            "depth",
            "depth_conf",
            "extrinsic",
            "extrinsics",
            "images",
            "intrinsic",
            "intrinsics",
            "pose_enc",
            "pose_enc_list",
            "world_points",
            "world_points_conf",
            "world_points_from_depth",
        ];
        expected.sort();
        assert_eq!(names, expected);

        let images: ndarray::ArrayD<f32> = npz.by_name("images").expect("images");
        assert_eq!(images.shape(), &[2, 3, 1, 1]);
        let rgb = output
            .rgb_frames
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .expect("rgb vec");
        assert_eq!(images.as_slice().expect("contiguous images"), rgb);

        let extrinsic: ndarray::ArrayD<f32> = npz.by_name("extrinsic").expect("extrinsic");
        assert_eq!(extrinsic.shape(), &[2, 3, 4]);
        let intrinsic: ndarray::ArrayD<f32> = npz.by_name("intrinsic").expect("intrinsic");
        assert_eq!(intrinsic.shape(), &[2, 3, 3]);

        let world_from_depth: ndarray::ArrayD<f32> = npz
            .by_name("world_points_from_depth")
            .expect("world_points_from_depth");
        assert_eq!(world_from_depth.shape(), &[2, 1, 1, 3]);
        let expected_world = output
            .world_points_from_depth()
            .expect("depth unprojection")
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .expect("world vec");
        assert_eq!(
            world_from_depth.as_slice().expect("contiguous world"),
            expected_world
        );

        let _ = std::fs::remove_file(&path);
    }

    /// Constant-per-frame tensor: element `idx` holds `(idx / k) + base`.
    fn frame_tensor<S: candle_core::shape::ShapeWithOneHole>(
        frames: usize,
        k: usize,
        shape: S,
        base: f32,
    ) -> Tensor {
        let device = Device::Cpu;
        let flat: Vec<f32> = (0..frames * k).map(|idx| (idx / k) as f32 + base).collect();
        Tensor::from_vec(flat, shape, &device).expect("synthetic tensor")
    }

    /// Synthetic `[1, S, ...]` output whose flat values repeat `frame + base`
    /// across each frame's elements (constant per frame, `k` elements each).
    fn synthetic_output(frames: usize, base: f32) -> VggtInferenceOutput {
        VggtInferenceOutput {
            world_points: frame_tensor(frames, 3, (1, frames, 1, 1, 3), base),
            world_points_conf: frame_tensor(frames, 1, (1, frames, 1, 1), base),
            depth: frame_tensor(frames, 1, (1, frames, 1, 1, 1), base),
            depth_conf: frame_tensor(frames, 1, (1, frames, 1, 1), base),
            pose_enc: frame_tensor(frames, 9, (1, frames, 9), base),
            pose_enc_list: vec![
                frame_tensor(frames, 9, (1, frames, 9), base),
                frame_tensor(frames, 9, (1, frames, 9), base + 100.0),
            ],
            extrinsics: frame_tensor(frames, 12, (1, frames, 3, 4), base),
            intrinsics: frame_tensor(frames, 9, (1, frames, 3, 3), base),
            rgb_frames: frame_tensor(frames, 3, (frames, 3, 1, 1), base),
            export_mask: Tensor::ones((1, frames, 1, 1), DType::U8, &Device::Cpu).expect("mask"),
        }
    }

    #[test]
    fn merge_vggt_inference_outputs_concatenates_frames_in_order() {
        let merged =
            merge_vggt_inference_outputs([synthetic_output(2, 10.0), synthetic_output(1, 20.0)])
                .expect("merge outputs");

        assert_eq!(merged.world_points.dims(), &[1, 3, 1, 1, 3]);
        let conf = merged
            .world_points_conf
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .expect("conf");
        assert_eq!(conf, vec![10.0, 11.0, 20.0]);
        let rgb = merged
            .rgb_frames
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .expect("rgb");
        assert_eq!(rgb.len(), 3 * 3);
        assert_eq!(rgb[0], 10.0);
        assert_eq!(rgb[3], 11.0);
        assert_eq!(rgb[6], 20.0);

        // pose_enc_list entries stay per-iteration: iteration 0 is `frame+base`,
        // iteration 1 is `frame+base+100`.
        assert_eq!(merged.pose_enc_list.len(), 2);
        let iteration1 = merged.pose_enc_list[1]
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .expect("pose enc iteration 1");
        assert_eq!(iteration1.len(), 3 * 9);
        assert_eq!(iteration1[0], 110.0);
        assert_eq!(iteration1[18], 120.0);
        assert_eq!(merged.export_mask.dims(), &[1, 3, 1, 1]);
        assert_eq!(merged.extrinsics.dims(), &[1, 3, 3, 4]);
        assert_eq!(merged.intrinsics.dims(), &[1, 3, 3, 3]);
        assert_eq!(merged.pose_enc.dims(), &[1, 3, 9]);
        assert_eq!(merged.depth.dims(), &[1, 3, 1, 1, 1]);
    }

    #[test]
    fn merge_vggt_inference_outputs_validates_inputs() {
        assert!(merge_vggt_inference_outputs(Vec::new()).is_err());
        let single = merge_vggt_inference_outputs([synthetic_output(2, 0.0)]).expect("single");
        assert_eq!(single.world_points.dims(), &[1, 2, 1, 1, 3]);

        // Mismatched leading batch dims are rejected with a clear error.
        let device = Device::Cpu;
        let mut bad = synthetic_output(1, 0.0);
        bad.world_points = Tensor::zeros((2, 1, 1, 1, 3), DType::F32, &device).expect("tensor");
        let error = merge_vggt_inference_outputs([synthetic_output(1, 0.0), bad])
            .expect_err("mismatched batch dims must be rejected")
            .to_string();
        assert!(error.contains("world_points"), "unexpected error: {error}");
    }

    #[test]
    fn chunk_prepared_inputs_splits_frames_in_order() {
        let device = Device::Cpu;
        let frames = 5usize;
        // Frame `i` filled with the value `i` so chunk contents are checkable.
        let normalized = Tensor::from_vec(
            (0..frames * 12)
                .map(|idx| (idx / 12) as f32)
                .collect::<Vec<f32>>(),
            (1, frames, 3, 2, 2),
            &device,
        )
        .expect("normalized frames");
        let rgb = Tensor::from_vec(
            (0..frames * 3)
                .map(|idx| (idx / 3) as f32)
                .collect::<Vec<f32>>(),
            (frames, 3, 1, 1),
            &device,
        )
        .expect("rgb frames");
        let prepared = VggtPreparedInputs {
            interval: 1,
            original_size: crate::contracts::SpatialSize::new(1024, 768),
            target_size: crate::contracts::SpatialSize::new(518, 518),
            preprocess_mode: VggtPreprocessMode::Pad,
            rgb_frames: rgb,
            normalized_frames: normalized,
            original_coords: (0..frames)
                .map(|frame| crate::preprocess::VggtOriginalCoords {
                    width: 1024,
                    height: 768,
                    offset_x: frame as u32,
                    offset_y: 63,
                    content_width: 518,
                    content_height: 392,
                    mode: VggtPreprocessMode::Pad,
                    square: None,
                })
                .collect(),
        };

        let chunks = chunk_prepared_inputs(&prepared, 2).expect("chunks");
        assert_eq!(chunks.len(), 3);
        let chunk_lengths = [2usize, 2, 1];
        let mut frame_offset = 0usize;
        for (chunk, &len) in chunks.iter().zip(&chunk_lengths) {
            assert_eq!(chunk.preprocess_mode, VggtPreprocessMode::Pad);
            assert_eq!(chunk.original_coords.len(), len);
            assert_eq!(
                chunk.original_coords[0].offset_x, frame_offset as u32,
                "coords must follow the frame order"
            );
            let values = chunk
                .normalized_frames
                .flatten_all()
                .and_then(|tensor| tensor.to_vec1::<f32>())
                .expect("chunk values");
            assert_eq!(values.len(), len * 12);
            for (idx, value) in values.iter().enumerate() {
                let expected = frame_offset + idx / 12;
                assert_eq!(*value, expected as f32, "chunk frame order broken");
            }
            frame_offset += len;
        }

        assert!(chunk_prepared_inputs(&prepared, 0).is_err());
        // A batch covering every frame still yields exactly one chunk.
        let single = chunk_prepared_inputs(&prepared, frames).expect("single chunk");
        assert_eq!(single.len(), 1);
    }

    /// `batch_size == 0` is rejected before any weights are touched.
    #[test]
    fn batch_size_zero_is_rejected_before_inference() {
        use std::path::PathBuf;

        use crate::weights::{CanonicalWeightSetPaths, CanonicalWeightsManifest};

        let device = Device::Cpu;
        let prepared = VggtPreparedInputs {
            interval: 1,
            original_size: crate::contracts::SpatialSize::new(64, 64),
            target_size: crate::contracts::SpatialSize::new(14, 14),
            preprocess_mode: VggtPreprocessMode::Crop,
            rgb_frames: Tensor::zeros((1, 3, 14, 14), DType::F32, &device).expect("rgb"),
            normalized_frames: Tensor::zeros((1, 1, 3, 14, 14), DType::F32, &device)
                .expect("normalized"),
            original_coords: Vec::new(),
        };
        let options = VggtRunOptions {
            batch_size: Some(0),
            ..Default::default()
        };
        // A pipeline whose weights were never loaded is fine: the option
        // check runs before `bundle_for` on every code path.
        let pipeline = VggtPipeline {
            weights: CanonicalWeightSetPaths {
                manifest: CanonicalWeightsManifest {
                    family: ModelFamily::Vggt,
                    normalizer_version: 0,
                    raw_files: Box::new([]),
                    canonical_file: String::new(),
                    resolved_config_file: String::new(),
                    tensor_count: 0,
                    dtype_histogram: Default::default(),
                    source_checksums: Default::default(),
                },
                canonical_file: PathBuf::new(),
                resolved_config_file: PathBuf::new(),
                manifest_file: PathBuf::new(),
                checksums_file: PathBuf::new(),
                embedded_resolved_config: None,
            },
            preprocess: VggtPreprocessStage::default(),
            neural: VggtNeuralStage,
            geometry: Default::default(),
            export: Default::default(),
            dtype: DType::F32,
            bundle_cache: Arc::default(),
            track_head_cache: Arc::default(),
        };
        let error = pipeline
            .infer_prepared_with_options(&prepared, &options)
            .expect_err("batch_size 0 must be rejected")
            .to_string();
        assert!(error.contains("batch_size"), "unexpected error: {error}");
        assert!(error.contains("at least 1"), "unexpected error: {error}");
    }

    /// The npz dump stacks the per-iteration pose encodings `[1, S, 9]` into
    /// one `[K, S, 9]` tensor; a single-iteration list stays `[1, S, 9]` and
    /// an empty list is rejected.
    #[test]
    fn stack_pose_enc_list_stacks_iterations() {
        let device = Device::Cpu;
        let iteration = |base: f32| {
            Tensor::from_vec(
                (0..9)
                    .map(|value| base + value as f32)
                    .collect::<Vec<f32>>(),
                (1, 1, 9),
                &device,
            )
            .expect("pose enc")
        };
        let stacked = stack_pose_enc_list(&[iteration(0.0), iteration(100.0)]).expect("stacked");
        assert_eq!(stacked.dims(), &[2, 1, 9]);
        let values = stacked
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .expect("values");
        assert_eq!(values[0], 0.0);
        assert_eq!(values[9], 100.0);

        let single = stack_pose_enc_list(&[iteration(0.0)]).expect("single");
        assert_eq!(single.dims(), &[1, 1, 9]);

        assert!(stack_pose_enc_list(&[]).is_err());
    }
}
