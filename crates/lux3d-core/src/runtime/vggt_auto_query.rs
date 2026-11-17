//! VGGT automatic query selection — tracking stage A.
//!
//! # Official reference
//!
//! Upstream `vggt/dependency/vggsfm_utils.py:51-186` and
//! `vggt/dependency/track_predict.py:12-229`:
//!
//! * [`rank_frames_by_dino`] reproduces `generate_rank_by_dino` — resize the
//!   `[0, 1]` RGB frames to 336x336 with bilinear interpolation
//!   (`align_corners=False`), ResNet normalization
//!   (`_RESNET_MEAN`/`_RESNET_STD`), DINOv2 ViT-B/14 **with 4 registers**
//!   CLS tokens, L2-normalized CLS cosine similarity, and FPS over
//!   `100 - similarity`.
//! * [`farthest_point_sampling`] reproduces `farthest_point_sampling`
//!   (`vggsfm_utils.py:117-150`), including the first-maximum tie break.
//! * [`order_query_frames`] reproduces the `[0, *ranked_without_0]` reorder
//!   in `track_predict.py:61-64`.
//! * [`calculate_index_mappings`] / [`switch_tensor_order`] reproduce
//!   `calculate_index_mappings` / `switch_tensor_order`
//!   (`vggsfm_utils.py:153-186`): swap the query frame with frame 0, run the
//!   track head, then swap the frame axis back.
//! * [`prune_query_points_by_confidence`] reproduces the confidence gate at
//!   `track_predict.py:181-200`: `conf > 1.2`, applied only when more than
//!   512 query points survive.
//!
//! # Stage A query points
//!
//! The official query points come from ALIKED + SuperPoint keypoints
//! (`track_predict.py:67-69, 172`). Those extractors are stages B/C (VG6/VG7)
//! and are **not** part of stage A. Stage A therefore selects query points
//! with a deterministic FPS spread over the model-resolution pixel grid
//! ([`fps_query_points`], per-frame cap `--max-query-pts`), then applies the
//! official confidence gate. This is the documented stage-A limitation: the
//! query points are DINO/FPS based, not keypoint-detector based.
//!
//! # DINO weights
//!
//! The ranker needs `dinov2_vitb14_reg4_pretrain.pth` converted to
//! `model.safetensors` (tensor names unchanged). The location is resolved in
//! this order: [`VggtAutoQueryOptions::dino_weights`],
//! `LUX3D_DINOV2_REG_MODEL_PATH` (or its `LUX3D_VGGT_DINO_WEIGHTS` alias),
//! `models/dinov2_reg4/model.safetensors` under the current directory, then
//! the same path under the workspace root. A set-but-invalid environment path
//! is an error (it never silently falls back to another checkpoint), and
//! [`VggtPipeline::run_with_auto_tracks`] resolves and loads the ranker before
//! the expensive VGGT forward starts.
//! The official torch.hub URL is
//! `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth`
//! (`_make_dinov2_model_name` + `_DINOV2_BASE_URL` in
//! `dinov2/hub/backbones.py`).

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(test)]
use std::collections::HashMap;

use candle_core::{D, DType, Device, IndexOp, Module, Result as CandleResult, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder, layer_norm, ops::softmax};

use super::vggt_track::{VggtTrackHead, VggtTrackQuery, VggtTrackResult};
use super::vision_preproc::{IMAGENET_MEAN, IMAGENET_STD};
use crate::{Result, error::Lux3dError};

/// Official demo default `--query_frame_num` (`demo_colmap.py:54`).
pub const DEFAULT_QUERY_FRAME_NUM: usize = 8;
/// Official demo default `--max_query_pts` (`demo_colmap.py:55`).
pub const DEFAULT_MAX_QUERY_PTS: usize = 4096;
/// `generate_rank_by_dino(image_size=336)` (`vggsfm_utils.py:52`).
const DINO_IMAGE_SIZE: usize = 336;
/// `generate_rank_by_dino(..., model_name="dinov2_vitb14_reg")`: 4 registers,
/// 14x14 patches, 518x518 pretraining grid.
const DINO_PATCH_SIZE: usize = 14;
const DINO_PRETRAIN_IMAGE_SIZE: usize = 518;
const DINO_EMBED_DIM: usize = 768;
const DINO_DEPTH: usize = 12;
const DINO_NUM_HEADS: usize = 12;
const DINO_NUM_REGISTER_TOKENS: usize = 4;
const DINO_LAYER_NORM_EPS: f64 = 1e-6;
/// `track_predict.py:195`: `valid_mask = pred_conf > 1.2`.
const QUERY_CONF_THRESHOLD: f32 = 1.2;
/// `track_predict.py:196`: the mask is applied only when more than 512
/// points survive.
const QUERY_CONF_MIN_POINTS: usize = 512;

/// Environment variable overriding the DINOv2-reg safetensors path.
pub const DINOV2_REG_WEIGHTS_ENV: &str = "LUX3D_DINOV2_REG_MODEL_PATH";
/// Accepted alias for [`DINOV2_REG_WEIGHTS_ENV`]. Both names are validated
/// strictly: a set-but-invalid path is an error, never a silent fallback.
pub const DINOV2_REG_WEIGHTS_ENV_ALIAS: &str = "LUX3D_VGGT_DINO_WEIGHTS";
/// Default location relative to the workspace root.
pub const DINOV2_REG_DEFAULT_RELATIVE_PATH: &str = "models/dinov2_reg4/model.safetensors";

/// Stage-A auto query options.
#[derive(Debug, Clone, PartialEq)]
pub struct VggtAutoQueryOptions {
    /// Number of DINO-ranked query frames (`--query-frame-num`, official
    /// default 8). Frame 0 is always added when the ranking omits it
    /// (`track_predict.py:61-64`), so the track head may run one extra query
    /// frame.
    pub query_frame_num: usize,
    /// Per-frame query-point cap (`--max-query-pts`, official default 4096).
    pub max_query_pts: usize,
    /// Track-head refinement iterations (`TrackHead.iters` default 4).
    pub track_iters: usize,
    /// Explicit DINOv2-reg safetensors path; `None` resolves the default
    /// location (see the module docs).
    pub dino_weights: Option<PathBuf>,
}

impl Default for VggtAutoQueryOptions {
    fn default() -> Self {
        Self {
            query_frame_num: DEFAULT_QUERY_FRAME_NUM,
            max_query_pts: DEFAULT_MAX_QUERY_PTS,
            track_iters: super::vggt_track::TRACK_DEFAULT_ITERS,
            dino_weights: None,
        }
    }
}

/// Result of an automatic tracking run.
#[derive(Debug, Clone)]
pub struct VggtAutoTrackResult {
    /// Tracks `[S, Q, 2]` (frames in the original source order), visibility
    /// and confidence `[S, Q]`.
    pub track: VggtTrackResult,
    /// Source frame of each query point (`Q` entries; the query frame rotated
    /// to position 0 for that track-head pass).
    pub query_frames: Vec<usize>,
    /// Query-point pixel coordinates in the model input resolution, in the
    /// same order as [`Self::query_frames`] and the track query axis.
    pub query_points: Vec<[f32; 2]>,
    /// Ranked query frames exactly as `track_predict.py:61-64` builds them
    /// (frame 0 first, then the DINO/FPS ranking without a duplicate 0).
    pub ranked_query_frames: Vec<usize>,
    /// Query points kept per query frame.
    pub queries_per_frame: Vec<usize>,
    /// Query points removed by the confidence gate per query frame.
    pub pruned_per_frame: Vec<usize>,
    /// `calculate_index_mappings(query_frame, S)` per query frame.
    pub reorder_indices: Vec<Vec<usize>>,
}

// ---------------------------------------------------------------------------
// FPS / ranking / reorder primitives (weight-free, unit tested)
// ---------------------------------------------------------------------------

/// `farthest_point_sampling` (`vggsfm_utils.py:118-150`).
///
/// The upstream loop is *not* a min-distance FPS: every step replaces the
/// working row with `distance_matrix[newly_selected]` (clamped at zero),
/// zeroes the already selected columns, and takes the first maximum of that
/// row. This is reproduced exactly, including duplicate picks when every
/// remaining distance is zero.
pub fn farthest_point_sampling(
    distance_matrix: &[Vec<f32>],
    num_samples: usize,
    start_index: usize,
) -> Vec<usize> {
    let n = distance_matrix.len();
    if n == 0 {
        return Vec::new();
    }
    debug_assert!(start_index < n);
    let clamp_row = |row: &[f32]| -> Vec<f32> { row.iter().map(|value| value.max(0.0)).collect() };
    let mut selected = Vec::with_capacity(num_samples.max(1));
    selected.push(start_index);
    if num_samples <= 1 {
        return selected;
    }
    // `check_distances = distance_matrix[selected_indices]` (advanced
    // indexing over a one-element list) is a copy, so the first zeroing of
    // the selected column does not feed back into the matrix.
    let mut check = clamp_row(&distance_matrix[start_index]);
    while selected.len() < num_samples {
        // `torch.argmax` returns the first occurrence on ties.
        let mut farthest = 0usize;
        let mut best = check[0];
        for (index, value) in check.iter().enumerate().skip(1) {
            if *value > best {
                best = *value;
                farthest = index;
            }
        }
        selected.push(farthest);
        check = clamp_row(&distance_matrix[farthest]);
        for index in &selected {
            check[*index] = 0.0;
        }
        if selected.len() == n {
            break;
        }
    }
    selected
}

/// L2-normalize each row; zero rows stay zero (upstream `F.normalize`).
fn normalize_rows(rows: &[Vec<f32>]) -> Vec<Vec<f32>> {
    rows.iter()
        .map(|row| {
            let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
            if norm > 0.0 {
                row.iter().map(|value| value / norm).collect()
            } else {
                row.clone()
            }
        })
        .collect()
}

/// `generate_rank_by_dino` ranking math given CLS embeddings.
///
/// `embeddings[i]` is the (pre-normalization) CLS token of frame `i`. The
/// function computes the cosine-similarity matrix, `distance = 100 -
/// similarity`, picks the frame with the largest similarity sum after
/// masking the diagonal (`vggsfm_utils.py:98-105`), then runs FPS from that
/// frame (`vggsfm_utils.py:107-108`).
pub fn rank_frames_by_embeddings(
    embeddings: &[Vec<f32>],
    query_frame_num: usize,
) -> Result<Vec<usize>> {
    if embeddings.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "VGGT auto query: DINO frame embeddings are empty",
        ));
    }
    let dim = embeddings[0].len();
    if dim == 0 {
        return Err(Lux3dError::InvalidInput(
            "VGGT auto query: DINO frame embeddings have zero width",
        ));
    }
    if embeddings.iter().any(|row| row.len() != dim) {
        return Err(Lux3dError::InvalidInput(
            "VGGT auto query: DINO frame embeddings have inconsistent widths",
        ));
    }
    if query_frame_num == 0 {
        return Err(Lux3dError::InvalidInput(
            "VGGT auto query: --query-frame-num must be at least 1",
        ));
    }
    let normalized = normalize_rows(embeddings);
    let frames = normalized.len();
    let mut similarity = vec![vec![0.0f32; frames]; frames];
    for row in 0..frames {
        for column in 0..frames {
            similarity[row][column] = normalized[row]
                .iter()
                .zip(normalized[column].iter())
                .map(|(a, b)| a * b)
                .sum();
        }
    }
    // `distance_matrix = 100 - similarity_matrix.clone()` before the
    // diagonal is masked for the "most common frame" vote.
    let distance_matrix: Vec<Vec<f32>> = similarity
        .iter()
        .map(|row| row.iter().map(|value| 100.0 - value).collect())
        .collect();
    // `similarity_matrix.fill_diagonal_(-100)`; `similarity_sum` is a sum
    // over the row, so the diagonal contributes -100 to its own frame.
    let mut best_frame = 0usize;
    let mut best_sum = f32::NEG_INFINITY;
    for (index, row) in similarity.iter().enumerate() {
        let sum = row
            .iter()
            .enumerate()
            .map(|(column, value)| if column == index { -100.0 } else { *value })
            .sum::<f32>();
        if sum > best_sum {
            best_sum = sum;
            best_frame = index;
        }
    }
    Ok(farthest_point_sampling(
        &distance_matrix,
        query_frame_num,
        best_frame,
    ))
}

/// `track_predict.py:61-64`: drop frame 0 from the ranking, then prepend it.
pub fn order_query_frames(ranked: &[usize]) -> Vec<usize> {
    let mut ordered = vec![0usize];
    ordered.extend(ranked.iter().copied().filter(|index| *index != 0));
    ordered
}

/// `calculate_index_mappings` (`vggsfm_utils.py:153-171`): identity order
/// with `query_index` and `0` swapped.
pub fn calculate_index_mappings(query_index: usize, frame_count: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..frame_count).collect();
    if query_index < frame_count {
        order[0] = query_index;
        order[query_index] = 0;
    }
    order
}

/// `switch_tensor_order` (`vggsfm_utils.py:174-186`): `index_select` along
/// `dim` with a precomputed order.
pub fn switch_tensor_order(tensor: &Tensor, order: &[usize], dim: usize) -> Result<Tensor> {
    let indices = Tensor::from_vec(
        order.iter().map(|index| *index as u32).collect::<Vec<_>>(),
        (order.len(),),
        tensor.device(),
    )
    .map_err(|source| Lux3dError::CanonicalWeightsValidation {
        message: format!("failed to build a tensor reorder index: {source}"),
    })?;
    tensor
        .index_select(&indices, dim)
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to reorder a tensor: {source}"),
        })
}

/// Deterministic per-frame query-point selection.
///
/// Stage A has no keypoint extractor, so query points are a farthest-point
/// sampled spread over the model-resolution pixel grid: `max_query_pts`
/// points (or every candidate when the grid is smaller), capped by
/// `max_query_pts`. Coordinates are pixel centers in `[0, W-1] x [0, H-1]`,
/// the exact convention of the track head. The first point is the pixel
/// closest to the image center; ties pick the lowest candidate index, which
/// makes the selection deterministic and platform independent.
pub fn fps_query_points(height: usize, width: usize, max_query_pts: usize) -> Vec<[f32; 2]> {
    if height == 0 || width == 0 || max_query_pts == 0 {
        return Vec::new();
    }
    // Keep at least 4x the requested points as candidates so the FPS spread
    // is not limited to a coarse lattice, while bounding the O(N * K) cost.
    let target_candidates = max_query_pts.saturating_mul(4);
    let pixels = height.saturating_mul(width);
    let stride = if pixels <= target_candidates {
        1
    } else {
        ((pixels / target_candidates) as f64)
            .sqrt()
            .floor()
            .max(1.0) as usize
    };
    let xs = axis_candidates(width, stride);
    let ys = axis_candidates(height, stride);
    let mut candidates = Vec::with_capacity(xs.len() * ys.len());
    for y in &ys {
        for x in &xs {
            candidates.push([*x as f32, *y as f32]);
        }
    }
    if candidates.len() <= max_query_pts {
        return candidates;
    }
    let start = nearest_to_center(&candidates, width, height);
    let mut selected = Vec::with_capacity(max_query_pts);
    let mut distances = vec![f32::MAX; candidates.len()];
    let mut current = start;
    for _ in 0..max_query_pts {
        selected.push(candidates[current]);
        let [cx, cy] = candidates[current];
        let mut next = 0usize;
        let mut next_distance = -1.0f32;
        for (index, point) in candidates.iter().enumerate() {
            let dx = point[0] - cx;
            let dy = point[1] - cy;
            let squared = dx * dx + dy * dy;
            if squared < distances[index] {
                distances[index] = squared;
            }
            if distances[index] > next_distance {
                next_distance = distances[index];
                next = index;
            }
        }
        current = next;
    }
    selected
}

fn axis_candidates(size: usize, stride: usize) -> Vec<usize> {
    let mut positions: Vec<usize> = (0..size).step_by(stride.max(1)).collect();
    if positions.last().copied() != Some(size - 1) {
        positions.push(size - 1);
    }
    positions
}

fn nearest_to_center(candidates: &[[f32; 2]], width: usize, height: usize) -> usize {
    let center_x = (width.saturating_sub(1)) as f32 / 2.0;
    let center_y = (height.saturating_sub(1)) as f32 / 2.0;
    let mut best = 0usize;
    let mut best_distance = f32::MAX;
    for (index, point) in candidates.iter().enumerate() {
        let dx = point[0] - center_x;
        let dy = point[1] - center_y;
        let squared = dx * dx + dy * dy;
        if squared < best_distance {
            best_distance = squared;
            best = index;
        }
    }
    best
}

/// `track_predict.py:181-200` confidence gate.
///
/// `conf` is the query frame's confidence plane at the confidence-map
/// resolution; query points are scaled by `conf_width / image_width` and
/// rounded exactly like upstream. The mask is applied only when more than
/// 512 points survive `conf > 1.2`. Returns the kept points and the number
/// pruned.
pub fn prune_query_points_by_confidence(
    points: &[[f32; 2]],
    conf: &[Vec<f32>],
    image_width: usize,
    threshold: f32,
    min_survivors: usize,
) -> (Vec<[f32; 2]>, usize) {
    if points.is_empty() || conf.is_empty() || conf[0].is_empty() || image_width == 0 {
        return (points.to_vec(), 0);
    }
    let conf_height = conf.len();
    let conf_width = conf[0].len();
    let scale = conf_width as f32 / image_width as f32;
    let mut keep = vec![false; points.len()];
    let mut survivors = 0usize;
    for (index, [x, y]) in points.iter().enumerate() {
        let sx = (*x * scale).round() as isize;
        let sy = (*y * scale).round() as isize;
        let sx = sx.clamp(0, (conf_width - 1) as isize) as usize;
        let sy = sy.clamp(0, (conf_height - 1) as isize) as usize;
        if conf[sy][sx] > threshold {
            keep[index] = true;
            survivors += 1;
        }
    }
    if survivors > min_survivors {
        let kept = points
            .iter()
            .zip(keep.iter())
            .filter_map(|(point, keep)| keep.then_some(*point))
            .collect::<Vec<_>>();
        let pruned = points.len() - kept.len();
        (kept, pruned)
    } else {
        (points.to_vec(), 0)
    }
}

// ---------------------------------------------------------------------------
// DINOv2 ViT-B/14 with 4 registers (CLS-token feature extractor)
// ---------------------------------------------------------------------------

/// DINOv2 ViT-B/14-with-registers architecture parameters. The pretrain
/// checkpoint stores `pos_embed` for the 518x518 / 37x37 patch grid.
#[derive(Debug, Clone)]
pub struct Dinov2RegConfig {
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    pub image_size: usize,
    pub num_register_tokens: usize,
}

impl Default for Dinov2RegConfig {
    fn default() -> Self {
        Self {
            embed_dim: DINO_EMBED_DIM,
            depth: DINO_DEPTH,
            num_heads: DINO_NUM_HEADS,
            patch_size: DINO_PATCH_SIZE,
            image_size: DINO_PRETRAIN_IMAGE_SIZE,
            num_register_tokens: DINO_NUM_REGISTER_TOKENS,
        }
    }
}

#[derive(Debug)]
struct Dinov2Attention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
}

impl Dinov2Attention {
    fn load(vb: VarBuilder, dim: usize, num_heads: usize) -> CandleResult<Self> {
        Ok(Self {
            qkv: candle_nn::linear(dim, dim * 3, vb.pp("qkv"))?,
            proj: candle_nn::linear(dim, dim, vb.pp("proj"))?,
            num_heads,
        })
    }

    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let (batch, tokens, channels) = xs.dims3()?;
        let head_dim = channels / self.num_heads;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((batch, tokens, 3, self.num_heads, head_dim))?;
        // [B, N, 3, H, D] -> q/k/v [B, H, N, D]
        let split = |index: usize| -> CandleResult<Tensor> {
            qkv.narrow(2, index, 1)?
                .squeeze(2)?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = split(0)?;
        let k = split(1)?;
        let v = split(2)?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let attention = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attention = softmax(&attention, D::Minus1)?;
        let output = attention
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch, tokens, channels))?
            .contiguous()?;
        self.proj.forward(&output)
    }
}

#[derive(Debug)]
struct Dinov2Block {
    norm1: LayerNorm,
    attn: Dinov2Attention,
    ls1: Tensor,
    norm2: LayerNorm,
    mlp_fc1: Linear,
    mlp_fc2: Linear,
    ls2: Tensor,
}

impl Dinov2Block {
    fn load(vb: VarBuilder, config: &Dinov2RegConfig) -> CandleResult<Self> {
        let dim = config.embed_dim;
        Ok(Self {
            norm1: layer_norm(dim, DINO_LAYER_NORM_EPS, vb.pp("norm1"))?,
            attn: Dinov2Attention::load(vb.pp("attn"), dim, config.num_heads)?,
            ls1: vb.pp("ls1").get(dim, "gamma")?,
            norm2: layer_norm(dim, DINO_LAYER_NORM_EPS, vb.pp("norm2"))?,
            // DINOv2 `Mlp`: fc1 -> GELU (erf) -> fc2, `mlp_ratio=4`.
            mlp_fc1: candle_nn::linear(dim, dim * 4, vb.pp("mlp").pp("fc1"))?,
            mlp_fc2: candle_nn::linear(dim * 4, dim, vb.pp("mlp").pp("fc2"))?,
            ls2: vb.pp("ls2").get(dim, "gamma")?,
        })
    }

    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let residual = xs;
        let attn = self.attn.forward(&self.norm1.forward(xs)?)?;
        let xs = (&attn.broadcast_mul(&self.ls1)? + residual)?;
        let residual = &xs;
        let hidden = self
            .mlp_fc1
            .forward(&self.norm2.forward(&xs)?)?
            .gelu_erf()?;
        let mlp = self.mlp_fc2.forward(&hidden)?;
        mlp.broadcast_mul(&self.ls2)? + residual
    }
}

/// DINOv2 ViT-B/14 with 4 registers. `forward` returns the final-LayerNorm
/// CLS token (`x_norm_clstoken`, `vision_transformer.py:257-274`).
#[derive(Debug)]
pub struct Dinov2RegEncoder {
    config: Dinov2RegConfig,
    cls_token: Tensor,
    register_tokens: Option<Tensor>,
    patch_embed: Conv2d,
    pos_embed: Tensor,
    blocks: Vec<Dinov2Block>,
    norm: LayerNorm,
}

impl Dinov2RegEncoder {
    /// Load from the DINOv2 pretrain tensor names (`cls_token`,
    /// `register_tokens`, `pos_embed`, `patch_embed.proj.{weight,bias}`,
    /// `blocks.{i}.*`, `norm.*`).
    pub fn load(vb: VarBuilder, config: Dinov2RegConfig) -> CandleResult<Self> {
        let dim = config.embed_dim;
        let patch_embed = candle_nn::conv2d(
            3,
            dim,
            config.patch_size,
            Conv2dConfig {
                stride: config.patch_size,
                ..Default::default()
            },
            vb.pp("patch_embed").pp("proj"),
        )?;
        let cls_token = vb.get((1, 1, dim), "cls_token")?;
        let register_tokens = if config.num_register_tokens > 0 {
            Some(vb.get((1, config.num_register_tokens, dim), "register_tokens")?)
        } else {
            None
        };
        let grid = config.patch_size;
        let pretrain_grid = config.image_size / grid;
        let pos_embed = vb.get((1, pretrain_grid * pretrain_grid + 1, dim), "pos_embed")?;
        let mut blocks = Vec::with_capacity(config.depth);
        for index in 0..config.depth {
            blocks.push(Dinov2Block::load(
                vb.pp("blocks").pp(index.to_string()),
                &config,
            )?);
        }
        let norm = layer_norm(dim, DINO_LAYER_NORM_EPS, vb.pp("norm"))?;
        Ok(Self {
            config,
            cls_token,
            register_tokens,
            patch_embed,
            pos_embed,
            blocks,
            norm,
        })
    }

    /// `pixel_values [B, 3, H, W]` f32 (ResNet-normalized by the caller) ->
    /// CLS tokens `[B, embed_dim]` f32 after the final LayerNorm.
    pub fn forward(&self, pixel_values: &Tensor) -> CandleResult<Tensor> {
        let (batch, channels, height, width) = pixel_values.dims4()?;
        if channels != 3 {
            candle_core::bail!("DINOv2-reg expects 3 input channels, got {channels}");
        }
        if height % self.config.patch_size != 0 || width % self.config.patch_size != 0 {
            candle_core::bail!(
                "DINOv2-reg input {height}x{width} is not divisible by patch size {}",
                self.config.patch_size
            );
        }
        let patches = self
            .patch_embed
            .forward(pixel_values)?
            .flatten_from(2)?
            .transpose(1, 2)?;
        let patch_count = patches.dim(1)?;
        let pos_embed = self.pos_embed_for(patch_count, height, width)?;
        let cls = self
            .cls_token
            .broadcast_as((batch, 1, self.config.embed_dim))?;
        let mut tokens = Tensor::cat(&[&cls, &patches], 1)?;
        tokens = tokens.broadcast_add(&pos_embed)?;
        if let Some(registers) = &self.register_tokens {
            let count = registers.dim(1)?;
            let register_tokens = registers.broadcast_as((batch, count, self.config.embed_dim))?;
            tokens = Tensor::cat(
                &[
                    tokens.narrow(1, 0, 1)?,
                    register_tokens,
                    tokens.narrow(1, 1, patch_count)?,
                ],
                1,
            )?;
        }
        for block in &self.blocks {
            tokens = block.forward(&tokens)?;
        }
        let tokens = self.norm.forward(&tokens)?;
        tokens.i((.., 0))?.contiguous()
    }

    /// `interpolate_pos_encoding` (`vision_transformer.py:182-214`) with the
    /// `dinov2_vitb14_reg` configuration: `interpolate_offset=0.0` (output
    /// size mode) and `interpolate_antialias=True`. Returns `[1, 1 + N,
    /// embed_dim]` in the input dtype.
    fn pos_embed_for(
        &self,
        patch_count: usize,
        height: usize,
        width: usize,
    ) -> CandleResult<Tensor> {
        let dim = self.config.embed_dim;
        let pretrain_patches = self.pos_embed.dim(1)? - 1;
        let pretrain_grid = (pretrain_patches as f64).sqrt() as usize;
        let target_h = height / self.config.patch_size;
        let target_w = width / self.config.patch_size;
        if patch_count == pretrain_patches && height == width {
            return Ok(self.pos_embed.clone());
        }
        let source = self
            .pos_embed
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec3::<f32>()?;
        let class_row = &source[0][0];
        let patches = &source[0][1..];
        let resized = if target_h == pretrain_grid && target_w == pretrain_grid {
            patches.concat()
        } else {
            bicubic_antialias_resize(
                patches,
                pretrain_grid,
                pretrain_grid,
                target_h,
                target_w,
                dim,
            )
        };
        let mut host = Vec::with_capacity((1 + target_h * target_w) * dim);
        host.extend_from_slice(class_row);
        host.extend_from_slice(&resized);
        Tensor::from_vec(
            host,
            (1, 1 + target_h * target_w, dim),
            self.pos_embed.device(),
        )
    }
}

/// PIL/PyTorch antialias bicubic filter with `a = -0.5`
/// (`ATen/native/cuda/UpSample.cuh`, `upsample_antialias::BicubicFilterFunctor`).
fn bicubic_filter(x: f32) -> f32 {
    let a = -0.5f32;
    let x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * a
    } else {
        0.0
    }
}

/// One-axis antialias weights: `(first_index, normalized_weights)`
/// (`_compute_weights_span` + `_compute_weights` in `UpSample.cuh`).
fn antialias_axis_weights(input_size: usize, output_size: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = input_size as f32 / output_size as f32;
    let support = 2.0 * scale;
    let mut axis = Vec::with_capacity(output_size);
    for index in 0..output_size {
        let center = scale * (index as f32 + 0.5);
        let start = ((center - support + 0.5) as i64).max(0) as usize;
        let end = ((center + support + 0.5) as i64).min(input_size as i64) as usize;
        let mut weights = Vec::with_capacity(end.saturating_sub(start));
        let inv_scale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
        for position in start..end {
            weights.push(bicubic_filter((position as f32 - center + 0.5) * inv_scale));
        }
        let total: f32 = weights.iter().sum();
        if total != 0.0 {
            for weight in &mut weights {
                *weight /= total;
            }
        }
        axis.push((start, weights));
    }
    axis
}

/// Separable antialias bicubic resize of a `[in_h * in_w, dim]` patch-token
/// table.
fn bicubic_antialias_resize(
    source: &[Vec<f32>],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    dim: usize,
) -> Vec<f32> {
    let x_axis = antialias_axis_weights(in_w, out_w);
    let y_axis = antialias_axis_weights(in_h, out_h);
    // Horizontal pass: [in_h, out_w, dim].
    let mut horizontal = vec![0.0f32; in_h * out_w * dim];
    for row in 0..in_h {
        for (out_x, (start, weights)) in x_axis.iter().enumerate() {
            let target = (row * out_w + out_x) * dim;
            for (offset, weight) in weights.iter().enumerate() {
                let source_row = &source[row * in_w + start + offset];
                for channel in 0..dim {
                    horizontal[target + channel] += source_row[channel] * weight;
                }
            }
        }
    }
    // Vertical pass: [out_h, out_w, dim].
    let mut output = vec![0.0f32; out_h * out_w * dim];
    for (out_y, (start, weights)) in y_axis.iter().enumerate() {
        for (offset, weight) in weights.iter().enumerate() {
            let source_row = (start + offset) * out_w * dim;
            let target_row = out_y * out_w * dim;
            for index in 0..out_w * dim {
                output[target_row + index] += horizontal[source_row + index] * weight;
            }
        }
    }
    output
}

// ---------------------------------------------------------------------------
// DINO ranker
// ---------------------------------------------------------------------------

/// Resolve the DINOv2-reg safetensors path (explicit, env, cwd, workspace).
///
/// If an environment variable is set but names no existing candidate, this
/// returns [`Lux3dError::CanonicalWeightsValidation`] instead of falling
/// through to the cwd/workspace defaults: an explicit local-weights
/// misconfiguration must be reported, not silently replaced by a different
/// checkpoint (or by "weights not found" only after the VGGT forward).
pub fn resolve_dino_weights_path(explicit: Option<&Path>) -> Result<PathBuf> {
    let candidates = |path: PathBuf| -> Vec<PathBuf> {
        if path.is_dir() {
            vec![
                path.join("model.safetensors"),
                path.join("dinov2_vitb14_reg4_pretrain.safetensors"),
            ]
        } else {
            vec![path]
        }
    };
    if let Some(path) = explicit {
        for candidate in candidates(path.to_path_buf()) {
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "VGGT auto query: DINOv2-reg weights not found at {}",
                path.display()
            ),
        });
    }
    for variable in [DINOV2_REG_WEIGHTS_ENV, DINOV2_REG_WEIGHTS_ENV_ALIAS] {
        let Some(value) = std::env::var_os(variable) else {
            continue;
        };
        let path = PathBuf::from(value);
        for candidate in candidates(path.clone()) {
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "VGGT auto query: {variable} points to {}, but it contains no readable \
                 `model.safetensors` or `dinov2_vitb14_reg4_pretrain.safetensors`; fix the \
                 path or unset {variable} (the resolver does not silently fall back)",
                path.display()
            ),
        });
    }
    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join(DINOV2_REG_DEFAULT_RELATIVE_PATH));
    }
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(DINOV2_REG_DEFAULT_RELATIVE_PATH),
    );
    for candidate in candidates {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(Lux3dError::CanonicalWeightsValidation {
        message: format!(
            "VGGT auto query: DINOv2-reg weights not found. Download \
             https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth, \
             convert it to safetensors without renaming tensors, and place it at \
             `{DINOV2_REG_DEFAULT_RELATIVE_PATH}` (or set {DINOV2_REG_WEIGHTS_ENV})"
        ),
    })
}

/// Preflight DINOv2-reg weights for `--track-auto` before the VGGT forward.
///
/// Resolves the configured path (explicit option, environment, defaults),
/// loads the ranker on `device`, and returns it for reuse by
/// [`run_auto_tracking`]. A missing, unreadable, corrupt, or wrong-tensor-name
/// checkpoint therefore fails before the aggregator/camera/depth/point forward
/// instead of after it.
pub(crate) fn preflight_dino_weights(
    options: &VggtAutoQueryOptions,
    device: &Device,
) -> Result<Dinov2RegRanker> {
    let path = resolve_dino_weights_path(options.dino_weights.as_deref())?;
    Dinov2RegRanker::load(&path, device, DINO_IMAGE_SIZE)
}

/// DINOv2-reg CLS-token ranker.
pub struct Dinov2RegRanker {
    encoder: Arc<Dinov2RegEncoder>,
    image_size: usize,
}

impl std::fmt::Debug for Dinov2RegRanker {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Dinov2RegRanker")
            .field("image_size", &self.image_size)
            .finish()
    }
}

impl Dinov2RegRanker {
    /// Load from a safetensors file with the raw DINOv2 pretrain tensor
    /// names.
    pub fn load(path: &Path, device: &Device, image_size: usize) -> Result<Self> {
        // Safety: the mmap must outlive every tensor cloned from the builder;
        // candle tensors clone the backing storage (Arc), so the encoder owns
        // the mapping for its whole lifetime.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!(
                        "VGGT auto query: failed to load DINOv2-reg weights {}: {source}",
                        path.display()
                    ),
                }
            })?
        };
        let encoder = Dinov2RegEncoder::load(vb, Dinov2RegConfig::default()).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT auto query: failed to build DINOv2-reg encoder: {source}"),
            }
        })?;
        Ok(Self {
            encoder: Arc::new(encoder),
            image_size,
        })
    }

    /// Test-only constructor from a tensor map (raw DINOv2 names).
    #[cfg(test)]
    fn from_tensors(
        tensors: HashMap<String, Tensor>,
        device: &Device,
        image_size: usize,
    ) -> Result<Self> {
        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        let encoder = Dinov2RegEncoder::load(vb, Dinov2RegConfig::default()).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "VGGT auto query: failed to build a test DINOv2-reg encoder: {source}"
                ),
            }
        })?;
        Ok(Self {
            encoder: Arc::new(encoder),
            image_size,
        })
    }

    /// CLS embeddings `[S, 768]` for `[S, 3, H, W]` frames in `[0, 1]`.
    pub fn frame_embeddings(&self, rgb_frames: &Tensor) -> Result<Tensor> {
        let (frames, channels, _height, _width) =
            rgb_frames
                .dims4()
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("VGGT auto query: DINO frames must be [S, 3, H, W]: {source}"),
                })?;
        if channels != 3 {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT auto query: DINO expects 3 channels, got {channels}"),
            });
        }
        let device = rgb_frames.device();
        // `F.interpolate(images, (336, 336), mode="bilinear",
        // align_corners=False)` over the whole frame stack.
        let resized = rgb_frames
            .to_dtype(DType::F32)
            .and_then(|frames| frames.upsample_bilinear2d(self.image_size, self.image_size, false))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT auto query: DINO resize failed: {source}"),
            })?;
        let mean = Tensor::from_slice(&IMAGENET_MEAN, (1, 3, 1, 1), device)
            .and_then(|tensor| tensor.to_dtype(DType::F32))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT auto query: DINO mean tensor failed: {source}"),
            })?;
        let std = Tensor::from_slice(&IMAGENET_STD, (1, 3, 1, 1), device)
            .and_then(|tensor| tensor.to_dtype(DType::F32))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT auto query: DINO std tensor failed: {source}"),
            })?;
        let normalized = resized
            .broadcast_sub(&mean)
            .and_then(|tensor| tensor.broadcast_div(&std))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT auto query: DINO normalization failed: {source}"),
            })?;
        self.encoder
            .forward(&normalized)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT auto query: DINO forward failed: {source}"),
            })
            .and_then(|tokens| {
                let returned = tokens.dim(0).unwrap_or(0);
                if returned != frames {
                    return Err(Lux3dError::CanonicalWeightsValidation {
                        message: format!(
                            "VGGT auto query: DINO returned {returned} frames, expected {frames}"
                        ),
                    });
                }
                Ok(tokens)
            })
    }

    /// Rank frames with `generate_rank_by_dino` semantics.
    pub fn rank_frames(&self, rgb_frames: &Tensor, query_frame_num: usize) -> Result<Vec<usize>> {
        let tokens = self.frame_embeddings(rgb_frames)?;
        let embeddings = tokens
            .to_device(&Device::Cpu)
            .and_then(|tokens| tokens.to_dtype(DType::F32))
            .and_then(|tokens| tokens.to_vec2::<f32>())
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT auto query: failed to read DINO embeddings: {source}"),
            })?;
        rank_frames_by_embeddings(&embeddings, query_frame_num)
    }
}

// ---------------------------------------------------------------------------
// Integration: ranked query frames + confidence-gated FPS points -> track head
// ---------------------------------------------------------------------------

fn conf_plane(conf: &Tensor, frame: usize) -> Result<Vec<Vec<f32>>> {
    // `conf` is `[S, H, W]` (the inference output has the batch dim removed
    // by the caller).
    let plane = conf
        .i(frame)
        .and_then(|plane| plane.to_device(&Device::Cpu))
        .and_then(|plane| plane.to_dtype(DType::F32))
        .and_then(|plane| plane.to_vec2::<f32>())
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("VGGT auto query: failed to read confidence frame {frame}: {source}"),
        })?;
    Ok(plane)
}

/// Run stage-A query selection and the existing explicit-query track head.
///
/// `frames` is the `[1, S, 3, H, W]` normalized model input, `rgb_frames`
/// the `[S, 3, H, W]` `[0, 1]` frames used by DINO, and `conf` the
/// `[S, H, W]` depth-confidence map used by the query gate. `ranker` is the
/// preflighted DINOv2-reg ranker (loaded before the VGGT forward by
/// [`preflight_dino_weights`]) and is reused here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_auto_tracking(
    track_head: &VggtTrackHead,
    ranker: &Dinov2RegRanker,
    cached_layers: &[Tensor],
    frames: &Tensor,
    rgb_frames: &Tensor,
    conf: &Tensor,
    patch_start_idx: usize,
    options: &VggtAutoQueryOptions,
    chunk_size: usize,
) -> Result<VggtAutoTrackResult> {
    if options.query_frame_num == 0 {
        return Err(Lux3dError::InvalidInput(
            "VGGT auto query: --query-frame-num must be at least 1",
        ));
    }
    if options.max_query_pts == 0 {
        return Err(Lux3dError::InvalidInput(
            "VGGT auto query: --max-query-pts must be at least 1",
        ));
    }
    if options.track_iters == 0 {
        return Err(Lux3dError::InvalidInput(
            "VGGT auto query: --track-iters must be at least 1",
        ));
    }
    let frame_count =
        rgb_frames
            .dim(0)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT auto query: failed to read the frame count: {source}"),
            })?;
    if frame_count == 0 {
        return Err(Lux3dError::InvalidInput(
            "VGGT auto query: the prepared sequence has no frames",
        ));
    }
    let (_, frames_dim, _, height, width) =
        frames
            .dims5()
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT auto query: model frames must be [1, S, 3, H, W]: {source}"),
            })?;
    if frames_dim != frame_count {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "VGGT auto query: rgb frames ({frame_count}) and model frames ({frames_dim}) disagree"
            ),
        });
    }
    if cached_layers.len() != 4 {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "VGGT auto query: expected 4 cached aggregator layers, got {}",
                cached_layers.len()
            ),
        });
    }

    let ranked = ranker.rank_frames(rgb_frames, options.query_frame_num)?;
    let ranked_query_frames = order_query_frames(&ranked);

    // The FPS spread over the model grid is deterministic and independent of
    // the query frame; only the confidence gate below is per frame. Compute
    // it once and reuse the pruned subsets.
    let query_points_base = fps_query_points(height, width, options.max_query_pts);

    let mut coords = Vec::with_capacity(ranked_query_frames.len());
    let mut visibility = Vec::with_capacity(ranked_query_frames.len());
    let mut confidence = Vec::with_capacity(ranked_query_frames.len());
    let mut query_frames = Vec::new();
    let mut query_points = Vec::new();
    let mut queries_per_frame = Vec::with_capacity(ranked_query_frames.len());
    let mut pruned_per_frame = Vec::with_capacity(ranked_query_frames.len());
    let mut reorder_indices = Vec::with_capacity(ranked_query_frames.len());

    for query_frame in &ranked_query_frames {
        let conf = conf_plane(conf, *query_frame)?;
        let (points, pruned) = prune_query_points_by_confidence(
            &query_points_base,
            &conf,
            width,
            QUERY_CONF_THRESHOLD,
            QUERY_CONF_MIN_POINTS,
        );
        if points.is_empty() {
            return Err(Lux3dError::InvalidInput(
                "VGGT auto query: confidence gating removed every query point",
            ));
        }
        let order = calculate_index_mappings(*query_frame, frame_count);
        let mut rotated_layers = Vec::with_capacity(cached_layers.len());
        for layer in cached_layers {
            rotated_layers.push(switch_tensor_order(layer, &order, 1)?);
        }
        let rotated_frames = switch_tensor_order(frames, &order, 1)?;
        let queries = points
            .iter()
            .map(|xy| VggtTrackQuery { frame: 0, xy: *xy })
            .collect::<Vec<_>>();
        let layer_refs = rotated_layers.iter().collect::<Vec<_>>();
        let tracked = track_head.track(
            &layer_refs,
            &rotated_frames,
            patch_start_idx,
            &queries,
            options.track_iters,
            chunk_size,
        )?;
        // `switch_tensor_order([pred_track, pred_vis], reorder_index, dim=1)`
        // (`track_predict.py:224`): undo the frame swap. The frame axis is
        // dim 0 of the batched track result.
        coords.push(switch_tensor_order(&tracked.coords, &order, 0)?);
        visibility.push(switch_tensor_order(&tracked.visibility, &order, 0)?);
        if let Some(value) = &tracked.confidence {
            confidence.push(switch_tensor_order(value, &order, 0)?);
        }
        query_frames.extend(std::iter::repeat(*query_frame).take(points.len()));
        query_points.extend_from_slice(&points);
        queries_per_frame.push(points.len());
        pruned_per_frame.push(pruned);
        reorder_indices.push(order);
    }

    let concat = |parts: &[Tensor]| -> Result<Tensor> {
        Tensor::cat(parts, 1).map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("VGGT auto query: failed to merge tracks: {source}"),
        })
    };
    let track = VggtTrackResult {
        coords: concat(&coords)?,
        visibility: concat(&visibility)?,
        confidence: if confidence.is_empty() {
            None
        } else {
            Some(concat(&confidence)?)
        },
    };
    Ok(VggtAutoTrackResult {
        track,
        query_frames,
        query_points,
        ranked_query_frames,
        queries_per_frame,
        pruned_per_frame,
        reorder_indices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedding_pattern(values: &[f32]) -> Vec<f32> {
        values.to_vec()
    }

    #[test]
    fn fps_is_deterministic_and_caps_at_frame_count() {
        let distance = vec![
            vec![0.0f32, 10.0, 5.0, 9.0],
            vec![10.0, 0.0, 6.0, 5.0],
            vec![5.0, 6.0, 0.0, 4.0],
            vec![9.0, 5.0, 4.0, 0.0],
        ];
        let first = farthest_point_sampling(&distance, 3, 0);
        let second = farthest_point_sampling(&distance, 3, 0);
        assert_eq!(first, second);
        assert_eq!(first.len(), 3);
        assert_eq!(first[0], 0);
        // Row 0 maxima: index 1 (10.0).
        assert_eq!(first[1], 1);
        // After selecting 1, row 1 has selected columns 0 and 1 zeroed:
        // values [0, 0, 6, 5] -> index 2.
        assert_eq!(first[2], 2);
        // Requesting more frames than exist stops at the frame count.
        assert_eq!(farthest_point_sampling(&distance, 99, 0).len(), 4);
    }

    #[test]
    fn ranking_prefers_the_similarity_cluster() {
        // Frames 0 and 1 are near-identical, frames 2 and 3 are distinct
        // directions. Frame 1 has the largest similarity sum (it is close to
        // frame 0 and slightly closer to the others), so FPS starts there and
        // then jumps to the least similar frame.
        let embeddings = vec![
            embedding_pattern(&[1.0, 0.0, 0.0]),
            embedding_pattern(&[0.99, 0.01, 0.0]),
            embedding_pattern(&[0.0, 1.0, 0.0]),
            embedding_pattern(&[0.0, 0.0, 1.0]),
        ];
        let ranked = rank_frames_by_embeddings(&embeddings, 2).expect("ranking");
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0], 1);
        // Distance from frame 1 is largest for frame 3 (exactly orthogonal);
        // frame 2 retains the 0.01 similarity component.
        assert_eq!(ranked[1], 3);
        assert_eq!(
            ranked,
            rank_frames_by_embeddings(&embeddings, 2).expect("repeat ranking")
        );
        // `order_query_frames` prepends frame 0 and drops the duplicate.
        assert_eq!(order_query_frames(&[2, 3, 0]), vec![0, 2, 3]);
        assert_eq!(order_query_frames(&[0, 1]), vec![0, 1]);
        assert_eq!(order_query_frames(&[1, 2]), vec![0, 1, 2]);
    }

    #[test]
    fn ranking_uses_the_first_maximum_for_ties() {
        let embeddings = vec![vec![1.0f32, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]];
        // Frames 0 and 2 are identical; the similarity sums rank frame 0
        // first (first maximum), then FPS picks frame 1.
        let ranked = rank_frames_by_embeddings(&embeddings, 2).expect("ranking");
        assert_eq!(ranked, vec![0, 1]);
    }

    #[test]
    fn reorder_mapping_is_an_involution() {
        let device = Device::Cpu;
        let tensor = Tensor::from_vec(
            (0..3 * 4 * 2).map(|value| value as f32).collect::<Vec<_>>(),
            (3, 4, 2),
            &device,
        )
        .expect("tensor");
        for query in 0..3 {
            let order = calculate_index_mappings(query, 3);
            assert_eq!(order[0], query);
            assert_eq!(order[query], 0);
            let permuted = switch_tensor_order(&tensor, &order, 0).expect("permute");
            let restored = switch_tensor_order(&permuted, &order, 0).expect("restore");
            let original = tensor
                .flatten_all()
                .expect("flat")
                .to_vec1::<f32>()
                .expect("vec");
            let restored = restored
                .flatten_all()
                .expect("flat")
                .to_vec1::<f32>()
                .expect("vec");
            assert_eq!(original, restored);
            // Once means "query frame first".
            let frame0 = permuted.i(0).expect("frame 0");
            let source = tensor.i(query).expect("source frame");
            let frame0 = frame0
                .flatten_all()
                .expect("flat")
                .to_vec1::<f32>()
                .expect("vec");
            let source = source
                .flatten_all()
                .expect("flat")
                .to_vec1::<f32>()
                .expect("vec");
            assert_eq!(frame0, source);
        }
    }

    #[test]
    fn fps_query_points_are_deterministic_and_capped() {
        let first = fps_query_points(32, 48, 17);
        let second = fps_query_points(32, 48, 17);
        assert_eq!(first, second);
        assert_eq!(first.len(), 17);
        for [x, y] in &first {
            assert!((0.0..=47.0).contains(x));
            assert!((0.0..=31.0).contains(y));
        }
        // More points than pixels returns every candidate.
        let all = fps_query_points(4, 4, 1000);
        assert_eq!(all.len(), 16);
        // The spread reaches the border at some point.
        assert!(first.iter().any(|[x, _]| *x == 0.0 || *x == 47.0));
        assert!(first.iter().any(|[_, y]| *y == 0.0 || *y == 31.0));
    }

    #[test]
    fn confidence_gate_applies_only_above_the_official_minimum() {
        // 1024 points, all high confidence except the first 100.
        let points = fps_query_points(32, 32, 1024);
        let conf = vec![vec![5.0f32; 32]; 32];
        let (kept, pruned) =
            prune_query_points_by_confidence(&points, &conf, 32, QUERY_CONF_THRESHOLD, 512);
        assert_eq!(kept.len(), points.len());
        assert_eq!(pruned, 0);

        // 600 points, 300 below the threshold: only 300 survive, which is
        // not more than 512, so the official gate keeps all of them.
        let points = fps_query_points(32, 32, 600);
        let mut conf = vec![vec![5.0f32; 32]; 32];
        for row in conf.iter_mut().take(15) {
            for value in row.iter_mut() {
                *value = 0.5;
            }
        }
        let (kept, pruned) =
            prune_query_points_by_confidence(&points, &conf, 32, QUERY_CONF_THRESHOLD, 512);
        assert_eq!(kept.len(), points.len());
        assert_eq!(pruned, 0);

        // 2048 points inside a 32x32 conf plane: low-confidence pixels are
        // pruned only when the survivors exceed 512.
        let points = fps_query_points(32, 32, 2048);
        let mut conf = vec![vec![5.0f32; 32]; 32];
        for row in conf.iter_mut().take(8) {
            for value in row.iter_mut() {
                *value = 0.5;
            }
        }
        let (kept, pruned) =
            prune_query_points_by_confidence(&points, &conf, 32, QUERY_CONF_THRESHOLD, 512);
        assert_eq!(pruned, points.len() - kept.len());
        assert!(kept.len() > 512);
        assert!(kept.len() < points.len());
    }

    fn synthetic_vitb14_tensors(device: &Device) -> HashMap<String, Tensor> {
        // Only the small test tensors need deterministic values; random
        // weights are fine because the assertions are structural.
        let mut tensors = HashMap::new();
        let dim = DINO_EMBED_DIM;
        let grid = DINO_PRETRAIN_IMAGE_SIZE / DINO_PATCH_SIZE;
        let random = |shape: (usize, usize, usize)| -> Tensor {
            Tensor::randn(0.0f32, 0.02, shape, device).expect("random tensor")
        };
        tensors.insert("cls_token".to_string(), random((1, 1, dim)));
        tensors.insert(
            "register_tokens".to_string(),
            random((1, DINO_NUM_REGISTER_TOKENS, dim)),
        );
        tensors.insert("pos_embed".to_string(), random((1, grid * grid + 1, dim)));
        tensors.insert(
            "patch_embed.proj.weight".to_string(),
            Tensor::randn(
                0.0f32,
                0.02,
                (dim, 3, DINO_PATCH_SIZE, DINO_PATCH_SIZE),
                device,
            )
            .expect("patch weight"),
        );
        tensors.insert(
            "patch_embed.proj.bias".to_string(),
            Tensor::zeros(dim, DType::F32, device).expect("patch bias"),
        );
        for layer in 0..DINO_DEPTH {
            let prefix = format!("blocks.{layer}");
            tensors.insert(
                format!("{prefix}.norm1.weight"),
                Tensor::ones(dim, DType::F32, device).expect("norm"),
            );
            tensors.insert(
                format!("{prefix}.norm1.bias"),
                Tensor::zeros(dim, DType::F32, device).expect("norm"),
            );
            tensors.insert(
                format!("{prefix}.attn.qkv.weight"),
                Tensor::randn(0.0f32, 0.02, (dim * 3, dim), device).expect("qkv"),
            );
            tensors.insert(
                format!("{prefix}.attn.qkv.bias"),
                Tensor::zeros(dim * 3, DType::F32, device).expect("qkv bias"),
            );
            tensors.insert(
                format!("{prefix}.attn.proj.weight"),
                Tensor::randn(0.0f32, 0.02, (dim, dim), device).expect("proj"),
            );
            tensors.insert(
                format!("{prefix}.attn.proj.bias"),
                Tensor::zeros(dim, DType::F32, device).expect("proj bias"),
            );
            tensors.insert(
                format!("{prefix}.ls1.gamma"),
                Tensor::ones(dim, DType::F32, device).expect("ls1"),
            );
            tensors.insert(
                format!("{prefix}.norm2.weight"),
                Tensor::ones(dim, DType::F32, device).expect("norm"),
            );
            tensors.insert(
                format!("{prefix}.norm2.bias"),
                Tensor::zeros(dim, DType::F32, device).expect("norm"),
            );
            tensors.insert(
                format!("{prefix}.mlp.fc1.weight"),
                Tensor::randn(0.0f32, 0.02, (dim * 4, dim), device).expect("fc1"),
            );
            tensors.insert(
                format!("{prefix}.mlp.fc1.bias"),
                Tensor::zeros(dim * 4, DType::F32, device).expect("fc1 bias"),
            );
            tensors.insert(
                format!("{prefix}.mlp.fc2.weight"),
                Tensor::randn(0.0f32, 0.02, (dim, dim * 4), device).expect("fc2"),
            );
            tensors.insert(
                format!("{prefix}.mlp.fc2.bias"),
                Tensor::zeros(dim, DType::F32, device).expect("fc2 bias"),
            );
            tensors.insert(
                format!("{prefix}.ls2.gamma"),
                Tensor::ones(dim, DType::F32, device).expect("ls2"),
            );
        }
        tensors.insert(
            "norm.weight".to_string(),
            Tensor::ones(dim, DType::F32, device).expect("norm"),
        );
        tensors.insert(
            "norm.bias".to_string(),
            Tensor::zeros(dim, DType::F32, device).expect("norm"),
        );
        tensors
    }

    #[test]
    fn dino_encoder_forward_is_structural() {
        let device = Device::Cpu;
        let tensors = synthetic_vitb14_tensors(&device);
        let ranker = Dinov2RegRanker::from_tensors(tensors, &device, 336).expect("ranker");
        let frames = Tensor::randn(0.0f32, 1.0, (2, 3, 336, 336), &device).expect("frames");
        let embeddings = ranker.frame_embeddings(&frames).expect("embeddings");
        assert_eq!(embeddings.dims(), &[2, DINO_EMBED_DIM]);
        let values = embeddings
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("vec");
        assert!(values.iter().all(|value| value.is_finite()));
        let ranked = ranker.rank_frames(&frames, 2).expect("rank");
        assert_eq!(ranked.len(), 2);
        assert!(ranked.iter().all(|index| *index < 2));
    }

    /// Serializes tests that mutate/read the DINO weight environment
    /// variables (`std::env::set_var` is process-global).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// A set-but-invalid environment path must be a hard error, never a silent
    /// fallback to the cwd/workspace defaults (or to the other env alias).
    #[test]
    fn invalid_dino_weights_env_path_is_an_error() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        let saved = [DINOV2_REG_WEIGHTS_ENV, DINOV2_REG_WEIGHTS_ENV_ALIAS]
            .map(|name| (name, std::env::var_os(name)));
        for (name, _) in &saved {
            // Safety: all tests in this module that touch these variables hold
            // ENV_LOCK, so no concurrent reader observes the temporary state.
            unsafe { std::env::remove_var(*name) };
        }
        let missing = std::env::temp_dir().join(format!(
            "lux3d-missing-dino-weights-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        unsafe { std::env::set_var(DINOV2_REG_WEIGHTS_ENV_ALIAS, &missing) };
        let result = resolve_dino_weights_path(None);
        for (name, value) in saved {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
        let error = result
            .expect_err("an invalid env DINO weights path must not fall back")
            .to_string();
        assert!(
            error.contains(DINOV2_REG_WEIGHTS_ENV_ALIAS),
            "error must name the offending env variable: {error}"
        );
        assert!(
            error.contains("does not silently fall back"),
            "error must explain the strict behavior: {error}"
        );
    }

    /// Preflight rejects a missing explicit path and a corrupt safetensors
    /// file before the VGGT forward is reached.
    #[test]
    fn preflight_rejects_missing_or_corrupt_dino_weights() {
        let missing = std::env::temp_dir().join(format!(
            "lux3d-missing-preflight-dino-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let options = VggtAutoQueryOptions {
            dino_weights: Some(missing.clone()),
            ..VggtAutoQueryOptions::default()
        };
        let error = preflight_dino_weights(&options, &Device::Cpu)
            .expect_err("missing explicit weights must fail preflight")
            .to_string();
        assert!(
            error.contains("DINOv2-reg weights not found"),
            "unexpected missing-weights error: {error}"
        );

        let corrupt = std::env::temp_dir().join(format!(
            "lux3d-corrupt-preflight-dino-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::write(&corrupt, b"not a safetensors archive").expect("write corrupt fixture");
        let options = VggtAutoQueryOptions {
            dino_weights: Some(corrupt.clone()),
            ..VggtAutoQueryOptions::default()
        };
        let error = preflight_dino_weights(&options, &Device::Cpu)
            .expect_err("corrupt weights must fail preflight")
            .to_string();
        let _ = std::fs::remove_file(&corrupt);
        assert!(
            error.contains("failed to load DINOv2-reg weights"),
            "unexpected corrupt-weights error: {error}"
        );
    }

    /// Numeric parity against the recorded PyTorch reference. Ignored by
    /// default because it needs the local DINOv2-reg safetensors; when run
    /// explicitly (`--ignored`) absent weights are a hard failure, never a
    /// silent pass.
    #[test]
    #[ignore = "requires local DINOv2-reg safetensors (models/dinov2_reg4/model.safetensors or LUX3D_DINOV2_REG_MODEL_PATH); run explicitly with --ignored"]
    fn dinov2_reg_matches_torch_reference_when_weights_are_local() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        let path = resolve_dino_weights_path(None).expect(
            "this ignored parity test requires local DINOv2-reg weights; set \
             LUX3D_DINOV2_REG_MODEL_PATH or place models/dinov2_reg4/model.safetensors",
        );
        let device = Device::Cpu;
        let ranker = Dinov2RegRanker::load(&path, &device, 336).expect("ranker");
        // Deterministic image used by the Python reference dump:
        // pixel[c, y, x] = ((x * 17 + y * 31 + c * 7) % 256) / 255.
        let mut host = Vec::with_capacity(3 * 336 * 336);
        for channel in 0..3usize {
            for y in 0..336usize {
                for x in 0..336usize {
                    host.push((((x * 17 + y * 31 + channel * 7) % 256) as f32) / 255.0);
                }
            }
        }
        let image = Tensor::from_vec(host, (3, 336, 336), &device).expect("image");
        let embeddings = ranker
            .frame_embeddings(&image.unsqueeze(0).expect("batch"))
            .expect("embeddings");
        let values = embeddings.i(0).expect("row").to_vec1::<f32>().expect("vec");
        // Golden values recorded from the PyTorch reference implementation
        // of `dinov2_vitb14_reg` (see the VG4 results file).
        let golden = [
            -0.12525344f32,
            -0.88138705,
            1.11139929,
            0.17161749,
            0.49460369,
            0.31286281,
            -0.63926065,
            -0.13607097,
            1.43241835,
            -0.63082206,
            0.77769226,
            0.06391314,
            -1.7753979,
            -0.15772742,
            1.30046642,
            0.92315674,
        ];
        for (index, expected) in golden.iter().enumerate() {
            let delta = (values[index] - expected).abs();
            assert!(
                delta < 5e-3,
                "DINOv2-reg CLS[{index}] = {} vs torch {expected} (delta {delta})",
                values[index]
            );
        }

        // Bicubic antialias position-embedding parity for the 37x37 -> 24x24
        // grid (recorded from the same PyTorch reference).
        let pos = ranker
            .encoder
            .pos_embed_for(24 * 24, 336, 336)
            .expect("interpolated pos embed")
            .to_vec3::<f32>()
            .expect("pos vec");
        let golden_pos_class = [-0.00015994f32, -0.00220644, -0.01724585, 0.00186159];
        for (index, expected) in golden_pos_class.iter().enumerate() {
            let delta = (pos[0][0][index] - expected).abs();
            assert!(
                delta < 2e-6,
                "DINOv2-reg class pos[{index}] = {} vs torch {expected}",
                pos[0][0][index]
            );
        }
        let golden_pos_first_patch = [0.00988477f32, -0.00011751, 0.01901283, 0.00297998];
        for (index, expected) in golden_pos_first_patch.iter().enumerate() {
            let delta = (pos[0][1][index] - expected).abs();
            assert!(
                delta < 2e-6,
                "DINOv2-reg patch pos[{index}] = {} vs torch {expected}",
                pos[0][1][index]
            );
        }
    }
}
