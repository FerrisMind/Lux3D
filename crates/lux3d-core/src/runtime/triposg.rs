//! TripoSG end-to-end pipeline assembly (Task 4): canonical weights ->
//! preprocess -> DINOv2 conditioning -> seeded rectified-flow latents ->
//! DiT denoise -> VAE SDF decode -> hierarchical marching cubes -> OBJ export.
//!
//! The pipeline mirrors [`TripoSrPipeline`](super::triposr::TripoSrPipeline):
//! a per-device model bundle is built once from the canonical F32 weight set
//! and cached in a [`DeviceLocalCache`], and every Candle failure is mapped
//! into `Lux3dError::CanonicalWeightsValidation` with a `TripoSG ... failed`
//! message.
//!
//! ## Seeded latent initialization
//!
//! The initial latents are sampled from a deterministic CPU RNG (SplitMix64 +
//! Box-Muller, f64 math rounded to f32) and only then moved to the target
//! device. Runs with the same `--seed` are therefore reproducible across
//! Lux3D invocations and backends, but they are **NOT bit-identical** to the
//! torch device RNG the reference pipeline uses (plan scope decision 3):
//! cross-framework parity is exercised by injecting the baseline
//! `latents_init` through `infer_with_latents` instead.
//!
//! ## Mesh extraction / export
//!
//! `extract_mesh` decodes the hierarchical SDF grid at the run's bounds (the
//! canonical `+/-1.005` cube unless `TripoSgRunOptions::bounds` overrides
//! them — the override threads through both the flash DiffDMC and the
//! dense/hierarchical MC path and moves the query window WITHOUT changing the
//! absolute world frame of the returned vertices) at level 0 (positive =
//! inside) and assembles the CPU mesh through the TripoSG geometry stage;
//! device mesh buffers are materialized on the latents' device. When
//! `TripoSgRunOptions::faces` is set, the reference `--faces` postprocess
//! (pymeshlab merge-close-vertices + quadric decimation to the target face
//! count) runs on the extracted mesh. `export_mesh` moves the buffers to CPU
//! and writes GLB/OBJ/PLY/STL according to the output extension; `export_obj`
//! keeps the legacy OBJ-only entry point.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use candle_core::{DType, Device, Tensor};

use super::{
    triposg_dino::TripoSgDinoEncoder,
    triposg_dit::TripoSgDiT,
    triposg_field::hierarchical_sdf_grid,
    triposg_flash::{compact_dmc_mesh, flash_dual_marching_cubes, flash_sdf_grid},
    triposg_text::{ClipBpeTokenizer, TripoSgClipTextEncoder},
    triposg_vae::{
        TripoSgLatentDistribution, TripoSgVaeConfig, TripoSgVaeDecoder, TripoSgVaeEncoder,
    },
    DeviceLocalCache, RmbgConfig, RmbgModel,
};
use crate::{
    error::Lux3dError,
    export::{MeshFormat, TripoSgExportStage},
    geometry::{merge_close_vertices, simplify_mesh, TripoSgGeometryStage, TripoSgMeshCpu},
    load_auxiliary_canonical_weights, load_canonical_weights,
    neural::TripoSgNeuralStage,
    preprocess::TripoSgPreprocessStage,
    weights::VarBuilder,
    AuxiliaryWeightSetPaths, CanonicalWeightSetPaths, ModelAssetOptions, ModelFamily, Result,
};

use super::triposg_dit::{
    denoise_dual_with_schedule, denoise_with_schedule, RectifiedFlowSchedule,
    TripoSgTransformerConfig, SCHEDULER_SHIFT,
};

/// TripoSG latent channel count (`transformer.in_channels` ==
/// `vae.latent_channels` == 64 in the shipped config).
pub(crate) const TRIPOSG_LATENT_CHANNELS: usize = 64;

/// Canonical SDF query bounds, `[-1.005, -1.005, -1.005, 1.005, 1.005, 1.005]`
/// (the reference `bbox_size = 2.01` cube; see `pipeline_defaults.bounds` in
/// the canonical `resolved_config.json`).
const CANONICAL_BOUNDS: [f32; 6] = [-1.005, -1.005, -1.005, 1.005, 1.005, 1.005];

/// `--faces` merge-close threshold as a fraction of the mesh bbox diagonal:
/// pymeshlab's `meshing_merge_close_vertices` default (`1/10000 of bounding
/// box diagonal`, per the official filter documentation).
const PYMESHLAB_MERGE_CLOSE_BBOX_DIAGONAL_FRACTION: f64 = 1.0 / 10_000.0;

/// Iso level the surface is extracted at (the decoder returns negated logits,
/// so positive = inside and the surface sits at 0).
const CANONICAL_SDF_LEVEL: f32 = 0.0;

/// Hard octree-depth cap for run options (grid side `2^depth` <= 1024, the
/// same cap `hierarchical_sdf_grid` enforces).
const MAX_OCTREE_DEPTH: u32 = 10;

// ---------------------------------------------------------------------------
// TripoSG-scribble defaults (`scripts/inference_triposg_scribble.py:32-40`
// and the checkpoint's `transformer/config.json`)
// ---------------------------------------------------------------------------

/// Scribble latent token count (`TripoSGScribblePipeline.__call__` default
/// `num_tokens=512`, `pipeline_triposg_scribble.py:192`).
pub const TRIPOSG_SCRIBBLE_NUM_TOKENS: usize = 512;
/// Reference scribble inference steps (`inference_triposg_scribble.py:25`).
pub const TRIPOSG_SCRIBBLE_STEPS: usize = 16;
/// Official scribble cross-attention scale for the image branch
/// (`--scribble-conf` default 0.4).
pub const TRIPOSG_SCRIBBLE_CONFIDENCE: f32 = 0.4;
/// Official scribble cross-attention scale for the text branch
/// (`--prompt-conf` default 1.0).
pub const TRIPOSG_SCRIBBLE_PROMPT_CONFIDENCE: f32 = 1.0;
/// Reference scribble dense octree depth (`inference_triposg_scribble.py:40`).
pub const TRIPOSG_SCRIBBLE_DENSE_OCTREE_DEPTH: u32 = 8;
/// Reference scribble hierarchical octree depth
/// (`inference_triposg_scribble.py:40`).
pub const TRIPOSG_SCRIBBLE_HIERARCHICAL_OCTREE_DEPTH: u32 = 8;
/// Default local checkpoint directory (Hugging Face `VAST-AI/TripoSG-scribble`).
pub const TRIPOSG_SCRIBBLE_DEFAULT_DIR: &str = "models/TripoSG-scribble";
/// Environment override for the local scribble checkpoint directory.
pub const TRIPOSG_SCRIBBLE_MODEL_DIR_ENV: &str = "LUX3D_TRIPOSG_SCRIBBLE_MODEL_DIR";
/// Files the raw diffusers-style scribble checkpoint must provide.
pub const TRIPOSG_SCRIBBLE_REQUIRED_FILES: [&str; 12] = [
    "model_index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "image_encoder_dinov2/config.json",
    "image_encoder_dinov2/model.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer/tokenizer_config.json",
];

/// Per-run options for the TripoSG-scribble pipeline. Defaults replicate the
/// shipped `run_triposg_scribble` call
/// (`scripts/inference_triposg_scribble.py:32-40`): 16 rectified-flow steps,
/// 512 latent tokens, dense depth 8 / hierarchical depth 8, no flash decoder,
/// `guidance_scale=0` (CFG-distilled model, so no classifier-free guidance),
/// and the two per-branch cross-attention scales.
#[derive(Debug, Clone, PartialEq)]
pub struct TripoSgScribbleRunOptions {
    /// Rectified-flow Euler steps. Ignored when [`Self::timesteps`] is `Some`.
    pub steps: usize,
    /// Latent-initialization seed (Lux3D deterministic RNG).
    pub seed: u64,
    /// Number of latent tokens (`[1, num_tokens, 64]`).
    pub num_tokens: usize,
    /// Dense octree depth of the hierarchical SDF grid.
    pub dense_octree_depth: u32,
    /// Final hierarchical octree depth.
    pub hierarchical_octree_depth: u32,
    /// Maximum SDF query points per decoder pass.
    pub query_chunk_size: usize,
    /// Explicit timestep schedule override (reference pipeline
    /// `timesteps=` argument); `steps` is ignored when set.
    pub timesteps: Option<Vec<f32>>,
    /// SDF query bounds override `[min_x, min_y, min_z, max_x, max_y, max_z]`.
    pub bounds: Option<[f32; 6]>,
    /// Target face count after mesh extraction (reference `--faces`).
    pub faces: Option<usize>,
    /// `cross_attention_2_scale`: weight of the scribble-image branch
    /// (`--scribble-conf`, default 0.4).
    pub scribble_confidence: f32,
    /// `cross_attention_scale`: weight of the CLIP text branch
    /// (`--prompt-conf`, default 1.0).
    pub prompt_confidence: f32,
}

impl Default for TripoSgScribbleRunOptions {
    fn default() -> Self {
        Self {
            steps: TRIPOSG_SCRIBBLE_STEPS,
            seed: 42,
            num_tokens: TRIPOSG_SCRIBBLE_NUM_TOKENS,
            dense_octree_depth: TRIPOSG_SCRIBBLE_DENSE_OCTREE_DEPTH,
            hierarchical_octree_depth: TRIPOSG_SCRIBBLE_HIERARCHICAL_OCTREE_DEPTH,
            query_chunk_size: 16384,
            timesteps: None,
            bounds: None,
            faces: None,
            scribble_confidence: TRIPOSG_SCRIBBLE_CONFIDENCE,
            prompt_confidence: TRIPOSG_SCRIBBLE_PROMPT_CONFIDENCE,
        }
    }
}

impl TripoSgScribbleRunOptions {
    pub fn validate(&self) -> Result<()> {
        if self.timesteps.is_none() && self.steps == 0 {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG scribble steps must be at least 1, got {}",
                    self.steps
                ),
            });
        }
        if self.num_tokens == 0 {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG scribble num_tokens must be at least 1, got {}",
                    self.num_tokens
                ),
            });
        }
        if self.dense_octree_depth > MAX_OCTREE_DEPTH {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG scribble dense_octree_depth {} exceeds the maximum \
                     {MAX_OCTREE_DEPTH}",
                    self.dense_octree_depth
                ),
            });
        }
        if self.hierarchical_octree_depth > MAX_OCTREE_DEPTH {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG scribble hierarchical_octree_depth {} exceeds the maximum \
                     {MAX_OCTREE_DEPTH}",
                    self.hierarchical_octree_depth
                ),
            });
        }
        if let Some(timesteps) = &self.timesteps {
            if timesteps.is_empty() {
                return Err(Lux3dError::CanonicalWeightsValidation {
                    message: "TripoSG scribble timesteps must not be empty".to_string(),
                });
            }
            if let Some(index) = timesteps.iter().position(|value| !value.is_finite()) {
                return Err(Lux3dError::CanonicalWeightsValidation {
                    message: format!(
                        "TripoSG scribble timesteps[{index}] must be finite, got {}",
                        timesteps[index]
                    ),
                });
            }
        }
        if let Some(bounds) = self.bounds {
            for (axis, window) in [
                ("x", (bounds[0], bounds[3])),
                ("y", (bounds[1], bounds[4])),
                ("z", (bounds[2], bounds[5])),
            ] {
                let (min, max) = window;
                if !min.is_finite() || !max.is_finite() {
                    return Err(Lux3dError::CanonicalWeightsValidation {
                        message: format!(
                            "TripoSG scribble bounds {axis} axis must be finite, got \
                             [{min}, {max}]"
                        ),
                    });
                }
                if min >= max {
                    return Err(Lux3dError::CanonicalWeightsValidation {
                        message: format!(
                            "TripoSG scribble bounds {axis} axis must have min < max, got \
                             [{min}, {max}]"
                        ),
                    });
                }
            }
        }
        if !self.scribble_confidence.is_finite() {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG scribble confidence must be finite, got {}",
                    self.scribble_confidence
                ),
            });
        }
        if !self.prompt_confidence.is_finite() {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG prompt confidence must be finite, got {}",
                    self.prompt_confidence
                ),
            });
        }
        Ok(())
    }

    /// The SDF query bounds this run decodes at (override or canonical).
    pub fn effective_bounds(&self) -> [f32; 6] {
        self.bounds.unwrap_or(CANONICAL_BOUNDS)
    }
}

/// Resolve the local `VAST-AI/TripoSG-scribble` checkpoint directory: an
/// explicit path wins, then `LUX3D_TRIPOSG_SCRIBBLE_MODEL_DIR`, then
/// [`TRIPOSG_SCRIBBLE_DEFAULT_DIR`] relative to the current directory.
pub fn resolve_triposg_scribble_model_dir(explicit: Option<&Path>) -> Result<PathBuf> {
    let candidate = match explicit {
        Some(path) => path.to_path_buf(),
        None => std::env::var_os(TRIPOSG_SCRIBBLE_MODEL_DIR_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(TRIPOSG_SCRIBBLE_DEFAULT_DIR)),
    };
    if !candidate.is_dir() {
        return Err(Lux3dError::InvalidInput(
            "TripoSG scribble weights were not found; download the Hugging Face \
             `VAST-AI/TripoSG-scribble` checkpoint and pass --model-path (or set \
             LUX3D_TRIPOSG_SCRIBBLE_MODEL_DIR). Expected the diffusers-style directory \
             with transformer/, vae/, image_encoder_dinov2/, text_encoder/ and tokenizer/.",
        ));
    }
    let missing = TRIPOSG_SCRIBBLE_REQUIRED_FILES
        .iter()
        .map(|relative| candidate.join(relative))
        .filter(|path| !path.is_file())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        let list = missing
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "TripoSG scribble checkpoint directory `{}` is incomplete; missing: {list}",
                candidate.display()
            ),
        });
    }
    Ok(candidate)
}

/// Surface points the reference `load_surface` draws before encoding
/// (`scripts/inference_vae.py:12-23`, `num_pc = 204800`; the shipped demo has
/// 500000 points, so the draw is without replacement).
pub(crate) const TRIPOSG_SURFACE_POINT_COUNT: usize = 204_800;

/// Deterministic surface-point subset selection for `--surface-input`.
///
/// The reference `load_surface` draws `num_pc` indices with
/// `np.random.default_rng().choice(n, num_pc, replace=False)`. Lux3D replaces
/// that draw with the documented SplitMix64 RNG (same distribution, not
/// draw-identical). Inputs smaller than `count` (test fixtures) are upsampled
/// with replacement instead of raising like the reference, because the
/// shipped demo is the only 204800-point file.
fn surface_selection_indices(total: usize, count: usize, seed: u64) -> Vec<usize> {
    let mut rng = SeededNormalRng::new(seed);
    if total >= count {
        // Partial Fisher-Yates: `count` uniform draws without replacement.
        let mut pool: Vec<usize> = (0..total).collect();
        for index in 0..count {
            let swap = index + rng.next_index(total - index);
            pool.swap(index, swap);
        }
        pool.truncate(count);
        pool
    } else {
        (0..count).map(|_| rng.next_index(total)).collect()
    }
}

/// Per-run options for the TripoSG pipeline. Defaults replicate the reference
/// `TripoSGPipeline` call in `pipeline_triposg.py` (`num_inference_steps=50`,
/// `guidance_scale=7.0`, `num_tokens=2048`, dense depth 8 / hierarchical
/// depth 9, `num_chunks`-equivalent query chunk 16384).
///
/// Upstream knobs without a field here:
/// - `num_shapes_per_prompt` (>1) is not supported: every TripoSG path in
///   Lux3D is batch-1 by contract (the VAE decoder and the CFG denoise loop
///   both enforce batch 1), so multi-shape generation would need a batched
///   rewrite of the whole decode stack. `num_shapes_per_prompt = 1` (the
///   reference default) is the only behavior.
/// - `latents` / `generator` / `attention_kwargs` / callback hooks: latents
///   are injected through `infer_with_latents`, the Lux3D RNG replaces the
///   torch generator (module docs), and the remaining plumbing has no
///   numerical effect.
#[derive(Debug, Clone, PartialEq)]
pub struct TripoSgRunOptions {
    /// Rectified-flow Euler steps (reference `num_inference_steps`).
    ///
    /// Ignored when [`Self::timesteps`] is `Some` (upstream
    /// `retrieve_timesteps` derives `num_inference_steps` from the explicit
    /// schedule length).
    pub steps: usize,
    /// Classifier-free guidance scale (batch order `[uncond, cond]`).
    pub guidance_scale: f32,
    /// Seed for the Lux3D-seeded latent initialization (see module docs).
    pub seed: u64,
    /// Number of latent tokens (`[1, num_tokens, 64]` init latents).
    pub num_tokens: usize,
    /// Dense octree depth of the hierarchical SDF grid (default 8 -> 256^3).
    pub dense_octree_depth: u32,
    /// Final hierarchical octree depth (default 9 -> 512^3 mesh grid).
    pub hierarchical_octree_depth: u32,
    /// Use the reference flash decoder + DiffDMC mesh path (default true,
    /// matching `pipeline_triposg.py`). When false, dense/hierarchical MC is
    /// used instead.
    pub use_flash_decoder: bool,
    /// Flash (DiffDMC) octree depth (default 9, reference
    /// `flash_octree_depth`).
    pub flash_octree_depth: u32,
    /// Maximum number of SDF query points per decoder pass.
    pub query_chunk_size: usize,
    /// Explicit timestep schedule override (reference
    /// `TripoSGPipeline.__call__(timesteps=...)` via `retrieve_timesteps`).
    ///
    /// When `Some`, the denoise loop runs EXACTLY these timesteps and
    /// [`Self::steps`] is ignored (upstream derives `num_inference_steps`
    /// from the list length); sigmas are `timestep / 1000` plus the
    /// scheduler shift (see
    /// [`RectifiedFlowSchedule::from_timesteps`](super::triposg_dit::RectifiedFlowSchedule::from_timesteps)
    /// for the full semantics and the shipped-scheduler caveat). Validation
    /// requires a non-empty list of finite values; the schedule is otherwise
    /// used as given (no monotonicity check, matching upstream). When
    /// `None`, the default linear schedule for `steps` is used and the
    /// behavior is bit-identical to the pre-override pipeline.
    pub timesteps: Option<Vec<f32>>,
    /// SDF query bounds override `[min_x, min_y, min_z, max_x, max_y,
    /// max_z]` (reference `TripoSGPipeline.__call__(bounds=...)`, default
    /// `(-1.005, ..., 1.005)` = [`CANONICAL_BOUNDS`]).
    ///
    /// `None` uses the canonical bounds. When set, BOTH decode paths (flash
    /// DiffDMC and dense/hierarchical marching cubes) query and map vertices
    /// inside this window; the returned mesh stays in the same ABSOLUTE
    /// world frame (no re-normalization). Must be finite with
    /// `min < max` per axis.
    pub bounds: Option<[f32; 6]>,
    /// Target face count after mesh extraction (reference `--faces` in
    /// `scripts/inference_triposg.py`).
    ///
    /// `Some(n)` with `n > 0` and a mesh that has more than `n` faces runs
    /// the upstream pymeshlab sequence — `meshing_merge_close_vertices()`
    /// followed by `meshing_decimation_quadric_edge_collapse(targetfacenum=n)`
    /// — implemented as [`merge_close_vertices`] (threshold = 1/10000 of the
    /// mesh bbox diagonal, the pymeshlab filter default) plus the meshopt
    /// quadric-style [`simplify_mesh`]. `Some(0)` and `None` leave the mesh
    /// unchanged (upstream's `if faces > 0` guard), and a target at or above
    /// the current face count is a no-op (upstream returns the mesh
    /// unchanged). The TripoSG mesh carries positions/faces only, so there
    /// are no per-vertex attributes to remap.
    ///
    /// [`merge_close_vertices`]: crate::geometry::merge_close_vertices
    /// [`simplify_mesh`]: crate::geometry::simplify_mesh
    pub faces: Option<usize>,
}

impl Default for TripoSgRunOptions {
    fn default() -> Self {
        Self {
            steps: 50,
            guidance_scale: 7.0,
            seed: 42,
            num_tokens: 2048,
            dense_octree_depth: 8,
            hierarchical_octree_depth: 9,
            use_flash_decoder: true,
            flash_octree_depth: 9,
            query_chunk_size: 16384,
            timesteps: None,
            bounds: None,
            faces: None,
        }
    }
}

impl TripoSgRunOptions {
    /// Validate the options before any weights are touched. The checks match
    /// the pipeline invariants: at least one denoise step and one latent
    /// token, octree depths within the 1024-side grid cap, a finite guidance
    /// scale, a non-empty finite custom timestep schedule (when given), and
    /// finite bounds with `min < max` per axis (when given).
    pub fn validate(&self) -> Result<()> {
        if self.timesteps.is_none() && self.steps == 0 {
            // With an explicit schedule, `steps` is ignored (upstream derives
            // num_inference_steps from the schedule length).
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSG steps must be at least 1, got {}", self.steps),
            });
        }
        if self.num_tokens == 0 {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG num_tokens must be at least 1, got {}",
                    self.num_tokens
                ),
            });
        }
        if self.dense_octree_depth > MAX_OCTREE_DEPTH {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG dense_octree_depth {} exceeds the maximum {MAX_OCTREE_DEPTH}",
                    self.dense_octree_depth
                ),
            });
        }
        if self.hierarchical_octree_depth > MAX_OCTREE_DEPTH {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG hierarchical_octree_depth {} exceeds the maximum \
                     {MAX_OCTREE_DEPTH}",
                    self.hierarchical_octree_depth
                ),
            });
        }
        if self.flash_octree_depth == 0 || self.flash_octree_depth > MAX_OCTREE_DEPTH {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG flash_octree_depth {} must be in 1..={MAX_OCTREE_DEPTH}",
                    self.flash_octree_depth
                ),
            });
        }
        if !self.guidance_scale.is_finite() {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG guidance_scale must be finite, got {}",
                    self.guidance_scale
                ),
            });
        }
        if let Some(timesteps) = &self.timesteps {
            if timesteps.is_empty() {
                return Err(Lux3dError::CanonicalWeightsValidation {
                    message: "TripoSG timesteps must not be empty".to_string(),
                });
            }
            if let Some(index) = timesteps.iter().position(|value| !value.is_finite()) {
                return Err(Lux3dError::CanonicalWeightsValidation {
                    message: format!(
                        "TripoSG timesteps[{index}] must be finite, got {}",
                        timesteps[index]
                    ),
                });
            }
        }
        if let Some(bounds) = self.bounds {
            for (axis, window) in [
                ("x", (bounds[0], bounds[3])),
                ("y", (bounds[1], bounds[4])),
                ("z", (bounds[2], bounds[5])),
            ] {
                let (min, max) = window;
                if !min.is_finite() || !max.is_finite() {
                    return Err(Lux3dError::CanonicalWeightsValidation {
                        message: format!(
                            "TripoSG bounds {axis} axis must be finite, got [{min}, {max}]"
                        ),
                    });
                }
                if min >= max {
                    return Err(Lux3dError::CanonicalWeightsValidation {
                        message: format!(
                            "TripoSG bounds {axis} axis must have min < max, got \
                             [{min}, {max}]"
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// The SDF query bounds this run decodes at (override or canonical).
    pub fn effective_bounds(&self) -> [f32; 6] {
        self.bounds.unwrap_or(CANONICAL_BOUNDS)
    }
}

/// Re-export of the preprocess stage output so callers can name the whole
/// TripoSG stage chain from `runtime`.
pub use crate::preprocess::TripoSgPreparedInputs;

/// TripoSG inference output: DINOv2 conditioning embeds, the seeded initial
/// latents and the denoised latents (all F32).
#[derive(Debug, Clone)]
pub struct TripoSgInferenceOutput {
    /// `[1, 257, 1024]` last hidden state of the DINOv2-large encoder.
    pub image_embeds: Tensor,
    /// `[1, num_tokens, 64]` seeded standard-normal init latents.
    pub latents_init: Tensor,
    /// `[1, num_tokens, 64]` latents after the rectified-flow Euler schedule
    /// (`steps` steps, or exactly `options.timesteps` when the explicit
    /// schedule override is set).
    pub latents: Tensor,
}

/// TripoSG-scribble inference output: CLIP text conditioning, DINOv2 scribble
/// conditioning, the seeded initial latents and the denoised latents (all
/// F32, `[1, ...]`).
#[derive(Debug, Clone)]
pub struct TripoSgScribbleInferenceOutput {
    /// Prompt the text embeds were produced from (for run reporting).
    pub prompt: String,
    /// `[1, model_max_length, 768]` CLIP `last_hidden_state` (PRE-projection).
    pub text_embeds: Tensor,
    /// `[1, 257, 1024]` last hidden state of the DINOv2-large encoder for the
    /// scribble image.
    pub image_embeds: Tensor,
    /// `[1, num_tokens, 64]` seeded standard-normal init latents.
    pub latents_init: Tensor,
    /// `[1, num_tokens, 64]` latents after the rectified-flow Euler schedule.
    pub latents: Tensor,
}

/// Device mesh buffers produced by `extract_mesh` (positions in the canonical
/// `+/-1.005` world space, zero-based triangle indices).
#[derive(Debug, Clone)]
pub struct TripoSgMeshBuffers {
    /// `[N, 3]` F32 vertex positions on the latents' device.
    pub vertices: Tensor,
    /// `[F, 3]` I64 zero-based triangle indices on the latents' device.
    pub faces: Tensor,
}

/// Everything the pipeline needs per device: the DINOv2 image encoder, the
/// 21-block DiT and the VAE SDF decoder, all loaded from the canonical F32
/// weight set.
#[derive(Debug)]
struct TripoSgModelBundle {
    dino: TripoSgDinoEncoder,
    dit: TripoSgDiT,
    vae: TripoSgVaeDecoder,
    vae_encoder: TripoSgVaeEncoder,
}

impl TripoSgModelBundle {
    fn load(weights: &CanonicalWeightSetPaths, device: &Device, dtype: DType) -> Result<Self> {
        // SAFETY: the canonical safetensors file is read-only and the mmap
        // stays alive through the VarBuilder's 'static borrow for the bundle's
        // lifetime (same pattern as TripoSR).
        let vb = unsafe {
            weights.var_builder(dtype, device).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to open TripoSG canonical weights: {source}"),
                }
            })?
        };
        let resolved = weights.read_resolved_config_json()?;
        let transformer_cfg =
            TripoSgTransformerConfig::from_resolved_config(&resolved).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to parse TripoSG transformer config: {source}"),
                }
            })?;
        let vae_cfg = TripoSgVaeConfig::from_resolved_config(&resolved).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to parse TripoSG VAE config: {source}"),
            }
        })?;
        Ok(Self {
            dino: TripoSgDinoEncoder::load(vb.pp("image_encoder")).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct TripoSG image encoder: {source}"),
                }
            })?,
            dit: TripoSgDiT::load(vb.pp("transformer"), &transformer_cfg).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct TripoSG DiT: {source}"),
                }
            })?,
            vae: TripoSgVaeDecoder::load(vb.pp("vae"), &vae_cfg).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct TripoSG VAE decoder: {source}"),
                }
            })?,
            vae_encoder: TripoSgVaeEncoder::load(vb.pp("vae"), &vae_cfg).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct TripoSG VAE encoder: {source}"),
                }
            })?,
        })
    }
}

/// Everything the scribble pipeline needs per device, loaded from the raw
/// diffusers-style `VAST-AI/TripoSG-scribble` checkpoint: the DINOv2 scribble
/// encoder, the dual-cross-attention DiT, the VAE SDF decoder and the CLIP
/// text encoder plus its byte-level BPE tokenizer.
#[derive(Debug)]
struct TripoSgScribbleModelBundle {
    dino: TripoSgDinoEncoder,
    dit: TripoSgDiT,
    vae: TripoSgVaeDecoder,
    text_encoder: TripoSgClipTextEncoder,
    tokenizer: ClipBpeTokenizer,
}

impl TripoSgScribbleModelBundle {
    fn load(dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let transformer_config = read_triposg_scribble_json(dir, "transformer/config.json")?;
        let vae_config = read_triposg_scribble_json(dir, "vae/config.json")?;
        let dino_config = read_triposg_scribble_json(dir, "image_encoder_dinov2/config.json")?;
        let text_config = read_triposg_scribble_json(dir, "text_encoder/config.json")?;

        let transformer_cfg = TripoSgTransformerConfig::from_resolved_config(&transformer_config)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to parse TripoSG-scribble transformer config: {source}"),
        })?;
        if !transformer_cfg.use_cross_attention_2 {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: "TripoSG-scribble transformer config does not enable \
                          use_cross_attention_2; the raw checkpoint is not a scribble model"
                    .to_string(),
            });
        }
        let vae_cfg = TripoSgVaeConfig::from_resolved_config(&vae_config).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to parse TripoSG-scribble VAE config: {source}"),
            }
        })?;
        let dino_hidden = dino_config
            .get("hidden_size")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize)
            .unwrap_or(1024);
        let dino_layers = dino_config
            .get("num_hidden_layers")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize)
            .unwrap_or(24);
        if dino_hidden != 1024 || dino_layers != 24 {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG-scribble image encoder must be DINOv2-large (hidden 1024, 24 \
                     layers), got hidden {dino_hidden}, {dino_layers} layers"
                ),
            });
        }
        if transformer_cfg.cross_attention_2_dim != Some(dino_hidden) {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG-scribble transformer cross_attention_2_dim {:?} does not match \
                     the image encoder hidden size {dino_hidden}",
                    transformer_cfg.cross_attention_2_dim
                ),
            });
        }

        let text_encoder = TripoSgClipTextEncoder::load(
            raw_safetensors_var_builder(dir, "text_encoder/model.safetensors", dtype, device)?,
            &text_config,
            dtype,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to construct TripoSG-scribble CLIP text encoder: {source}"),
        })?;
        if text_encoder.config().embed_dim != transformer_cfg.cross_attention_dim {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG-scribble text encoder hidden size {} does not match the \
                     transformer cross_attention_dim {}",
                    text_encoder.config().embed_dim,
                    transformer_cfg.cross_attention_dim
                ),
            });
        }

        Ok(Self {
            dino: TripoSgDinoEncoder::load(raw_safetensors_var_builder(
                dir,
                "image_encoder_dinov2/model.safetensors",
                dtype,
                device,
            )?)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to construct TripoSG-scribble image encoder: {source}"),
            })?,
            dit: TripoSgDiT::load(
                raw_safetensors_var_builder(
                    dir,
                    "transformer/diffusion_pytorch_model.safetensors",
                    dtype,
                    device,
                )?,
                &transformer_cfg,
            )
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to construct TripoSG-scribble DiT: {source}"),
            })?,
            vae: TripoSgVaeDecoder::load(
                raw_safetensors_var_builder(
                    dir,
                    "vae/diffusion_pytorch_model.safetensors",
                    dtype,
                    device,
                )?,
                &vae_cfg,
            )
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to construct TripoSG-scribble VAE decoder: {source}"),
            })?,
            text_encoder,
            tokenizer: ClipBpeTokenizer::from_dir(&dir.join("tokenizer")).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!(
                        "failed to load the TripoSG-scribble CLIP tokenizer from `{}`: {source}",
                        dir.join("tokenizer").display()
                    ),
                }
            })?,
        })
    }
}

fn read_triposg_scribble_json(dir: &Path, relative: &str) -> Result<serde_json::Value> {
    let path = dir.join(relative);
    let body =
        fs::read_to_string(&path).map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "failed to read TripoSG-scribble checkpoint file `{}`: {source}",
                path.display()
            ),
        })?;
    serde_json::from_str(&body).map_err(|source| Lux3dError::CanonicalWeightsValidation {
        message: format!(
            "failed to parse TripoSG-scribble checkpoint file `{}`: {source}",
            path.display()
        ),
    })
}

/// Open one raw safetensors shard as a `VarBuilder`.
fn raw_safetensors_var_builder(
    dir: &Path,
    relative: &str,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let path = dir.join(relative);
    // SAFETY: the checkpoint file is read-only and the mmap stays alive for
    // the 'static borrow owned by the bundle's VarBuilder (same contract as
    // the canonical TripoSG loader and the TRELLIS.2 raw-directory loader).
    unsafe { VarBuilder::from_mmaped_safetensors(&[&path], dtype, device) }.map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "failed to open TripoSG-scribble weights `{}`: {source}",
                path.display()
            ),
        }
    })
}

/// TripoSG-scribble pipeline: preprocess -> CLIP text embeds + DINOv2
/// scribble embeds -> seeded rectified-flow denoise with dual cross-attention
/// scales -> VAE SDF decode (dense/hierarchical marching cubes) -> GLB export.
#[derive(Debug)]
pub struct TripoSgScribblePipeline {
    /// Raw diffusers-style checkpoint directory (`VAST-AI/TripoSG-scribble`).
    pub model_dir: PathBuf,
    /// Optional separate RMBG-1.4 canonical package for inputs without a
    /// valid alpha channel (same fallback as the image pipeline).
    pub rmbg_weights: Option<AuxiliaryWeightSetPaths>,
    pub preprocess: TripoSgPreprocessStage,
    pub geometry: TripoSgGeometryStage,
    pub export: TripoSgExportStage,
    dtype: DType,
    bundle_cache: Arc<DeviceLocalCache<TripoSgScribbleModelBundle>>,
    rmbg_cache: Arc<DeviceLocalCache<RmbgModel>>,
}

impl TripoSgScribblePipeline {
    pub fn load(model_dir: impl Into<PathBuf>) -> Result<Self> {
        Self::load_with_rmbg_and_dtype(model_dir, None, DType::F32)
    }

    pub fn load_with_dtype(model_dir: impl Into<PathBuf>, dtype: DType) -> Result<Self> {
        Self::load_with_rmbg_and_dtype(model_dir, None, dtype)
    }

    pub fn load_with_rmbg_and_dtype(
        model_dir: impl Into<PathBuf>,
        rmbg_model_dir: Option<PathBuf>,
        dtype: DType,
    ) -> Result<Self> {
        let model_dir = model_dir.into();
        let resolved = resolve_triposg_scribble_model_dir(Some(&model_dir))?;
        let rmbg_weights = match rmbg_model_dir {
            Some(dir) => Some(load_auxiliary_canonical_weights(dir, "rmbg")?),
            None => None,
        };
        Ok(Self {
            model_dir: resolved,
            rmbg_weights,
            preprocess: TripoSgPreprocessStage::default(),
            geometry: TripoSgGeometryStage,
            export: TripoSgExportStage,
            dtype,
            bundle_cache: Arc::default(),
            rmbg_cache: Arc::default(),
        })
    }

    fn bundle_for(&self, device: &Device) -> Result<Arc<TripoSgScribbleModelBundle>> {
        self.bundle_cache.get_or_try_init(device, || {
            TripoSgScribbleModelBundle::load(&self.model_dir, device, self.dtype)
        })
    }

    fn rmbg_for(&self, device: &Device) -> Result<Option<Arc<RmbgModel>>> {
        let Some(weights) = &self.rmbg_weights else {
            return Ok(None);
        };
        self.rmbg_cache
            .get_or_try_init(device, || {
                let config =
                    RmbgConfig::from_resolved_config(&weights.read_resolved_config_json()?)
                        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                            message: format!("failed to parse RMBG config: {source}"),
                        })?;
                // SAFETY: canonical safetensors mmap, same lifetime contract
                // as the scribble bundle.
                let vb = unsafe { weights.var_builder(DType::F32, device) }.map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("failed to open RMBG canonical weights: {source}"),
                    }
                })?;
                RmbgModel::load(vb, &config).map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("failed to construct RMBG model: {source}"),
                    }
                })
            })
            .map(Some)
    }

    /// Same preprocessing contract as the image pipeline (see
    /// `TripoSgPipeline::prepare_inputs_from_path`); the scribble reference
    /// script feeds the raw RGB scribble through the DINOv2 image processor,
    /// so this path shares Lux3D's TripoSG preprocess (documented deviation:
    /// the reference skips the alpha/RMBG foreground step for scribbles).
    pub fn prepare_inputs_from_path(
        &self,
        source: &Path,
        device: &Device,
    ) -> Result<TripoSgPreparedInputs> {
        match self
            .preprocess
            .prepare_inputs_from_path_with_rmbg(source, device, None)
        {
            Ok(prepared) => Ok(prepared),
            Err(Lux3dError::InvalidInput(message))
                if message.starts_with("TripoSG input has no valid alpha channel")
                    && self.rmbg_weights.is_some() =>
            {
                let rmbg = self.rmbg_for(device)?;
                self.preprocess
                    .prepare_inputs_from_path_with_rmbg(source, device, rmbg.as_deref())
            }
            Err(error) => Err(error),
        }
    }

    /// `CLIPTextModelWithProjection(prompt).last_hidden_state`
    /// `[1, model_max_length, 768]` (PRE-projection), F32.
    pub fn text_embeds(&self, prompt: &str, device: &Device) -> Result<Tensor> {
        if prompt.trim().is_empty() {
            return Err(Lux3dError::InvalidInput(
                "TripoSG-scribble prompt must not be empty",
            ));
        }
        let bundle = self.bundle_for(device)?;
        bundle
            .text_encoder
            .encode_prompt(&bundle.tokenizer, prompt, device)
            .and_then(|embeds| embeds.to_dtype(DType::F32))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSG-scribble text encode failed: {source}"),
            })
    }

    /// DINOv2-large scribble conditioning embeds `[1, 257, 1024]` from
    /// prepared `[1, 3, 224, 224]` pixel values, F32.
    pub fn image_embeds(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let bundle = self.bundle_for(pixel_values.device())?;
        let pixel_values = pixel_values.to_dtype(self.dtype).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "failed to convert TripoSG-scribble image input to {dtype:?}: {source}",
                    dtype = self.dtype
                ),
            }
        })?;
        let embeds = bundle.dino.forward(&pixel_values).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSG-scribble image encoder forward failed: {source}"),
            }
        })?;
        embeds
            .to_dtype(DType::F32)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "failed to convert TripoSG-scribble image embeds to f32: {source}"
                ),
            })
    }

    /// Full scribble inference from files: preprocess the scribble image ->
    /// DINOv2 embeds -> CLIP text embeds -> seeded 512-token latents ->
    /// dual-conditioned Euler denoise with `guidance_scale=0` (no CFG) and
    /// the per-branch confidences from `options`.
    pub fn infer_from_path(
        &self,
        scribble: &Path,
        prompt: &str,
        device: &Device,
        options: &TripoSgScribbleRunOptions,
    ) -> Result<TripoSgScribbleInferenceOutput> {
        options.validate()?;
        let prepared = self.prepare_inputs_from_path(scribble, device)?;
        let text_embeds = self.text_embeds(prompt, device)?;
        let image_embeds = self.image_embeds(&prepared.pixel_values)?;
        let latents_init = TripoSgPipeline::sample_latents(
            options.seed,
            options.num_tokens,
            TRIPOSG_LATENT_CHANNELS,
            device,
        )?;
        let latents =
            self.infer_with_latents(&text_embeds, &image_embeds, &latents_init, options)?;
        Ok(TripoSgScribbleInferenceOutput {
            prompt: prompt.to_string(),
            text_embeds,
            image_embeds,
            latents_init,
            latents,
        })
    }

    /// Denoise caller-supplied initial latents with both conditionings. The
    /// per-branch attention scales are
    /// `cross_attention_scale = options.prompt_confidence` (text branch) and
    /// `cross_attention_2_scale = options.scribble_confidence` (image
    /// branch); guidance stays at 0 (single pass, no CFG), matching
    /// `pipeline_triposg_scribble.py`.
    pub fn infer_with_latents(
        &self,
        text_embeds: &Tensor,
        image_embeds: &Tensor,
        latents_init: &Tensor,
        options: &TripoSgScribbleRunOptions,
    ) -> Result<Tensor> {
        options.validate()?;
        TripoSgPipeline::validate_latents(latents_init, options.num_tokens)?;
        let bundle = self.bundle_for(latents_init.device())?;
        let scheduler = match &options.timesteps {
            Some(timesteps) => RectifiedFlowSchedule::from_timesteps(timesteps, SCHEDULER_SHIFT),
            None => RectifiedFlowSchedule::new(options.steps, SCHEDULER_SHIFT),
        };
        denoise_dual_with_schedule(
            &bundle.dit,
            latents_init,
            text_embeds,
            image_embeds,
            &scheduler,
            0.0,
            options.prompt_confidence,
            options.scribble_confidence,
            self.dtype,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("TripoSG-scribble denoise failed: {source}"),
        })
    }

    /// Decode the SDF field and assemble the marching-cubes mesh, always via
    /// the reference scribble path: `use_flash_decoder=False` with dense
    /// depth 8 / hierarchical depth 8 and level-0 marching cubes.
    pub fn extract_mesh(
        &self,
        latents: &Tensor,
        options: &TripoSgScribbleRunOptions,
    ) -> Result<TripoSgMeshBuffers> {
        options.validate()?;
        TripoSgPipeline::validate_latents(latents, options.num_tokens)?;
        let bundle = self.bundle_for(latents.device())?;
        let latents = latents.to_dtype(self.dtype).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "failed to convert TripoSG-scribble latents to {dtype:?}: {source}",
                    dtype = self.dtype
                ),
            }
        })?;
        let bounds = options.effective_bounds();
        let grid = hierarchical_sdf_grid(
            &bundle.vae,
            &latents,
            bounds,
            options.dense_octree_depth,
            options.hierarchical_octree_depth,
            options.query_chunk_size,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("TripoSG-scribble SDF grid decode failed: {source}"),
        })?;
        let mut mesh_cpu = self.geometry.assemble_cpu(&grid)?;

        if let Some(target_faces) = options.faces {
            if target_faces > 0 && mesh_cpu.faces.len() / 3 > target_faces {
                mesh_cpu = simplify_triposg_mesh(&mesh_cpu, target_faces)?;
            }
        }

        triposg_mesh_buffers(&mesh_cpu, latents.device())
    }

    /// Export mesh buffers in the format selected by `path`'s extension.
    pub fn export_mesh(&self, mesh: &TripoSgMeshBuffers, path: &Path) -> Result<MeshFormat> {
        let mesh_cpu = triposg_mesh_buffers_to_cpu(mesh)?;
        self.export.write_mesh(&mesh_cpu, path)
    }

    /// Export mesh buffers as ASCII OBJ.
    pub fn export_obj(&self, mesh: &TripoSgMeshBuffers, path: &Path) -> Result<()> {
        let mesh_cpu = triposg_mesh_buffers_to_cpu(mesh)?;
        self.export.write_obj(&mesh_cpu, path)
    }
}

/// TripoSG image-to-3D pipeline: preprocess -> DINOv2 conditioning ->
/// seeded rectified-flow denoise -> VAE SDF decode -> OBJ export.
#[derive(Debug)]
pub struct TripoSgPipeline {
    pub weights: CanonicalWeightSetPaths,
    /// Optional separate RMBG-1.4 canonical package. Required only when the
    /// input image has no valid alpha channel (the reference fallback).
    pub rmbg_weights: Option<AuxiliaryWeightSetPaths>,
    pub preprocess: TripoSgPreprocessStage,
    pub neural: TripoSgNeuralStage,
    pub geometry: TripoSgGeometryStage,
    pub export: TripoSgExportStage,
    /// Weight/activation dtype for DINOv2, DiT and the VAE. Public outputs
    /// are converted back to F32; `F16` is an inference-time mode.
    dtype: DType,
    bundle_cache: Arc<DeviceLocalCache<TripoSgModelBundle>>,
    rmbg_cache: Arc<DeviceLocalCache<RmbgModel>>,
}

impl TripoSgPipeline {
    pub fn load(model_assets: ModelAssetOptions) -> Result<Self> {
        Self::load_with_rmbg(model_assets, None)
    }

    /// Load TripoSG with an explicit neural-network dtype. `F32` is the
    /// default reference mode; `F16` halves weight memory for the DINOv2,
    /// DiT and VAE submodels on CUDA/Vulkan/WGPU.
    pub fn load_with_dtype(model_assets: ModelAssetOptions, dtype: DType) -> Result<Self> {
        Self::load_with_rmbg_and_dtype(model_assets, None, dtype)
    }

    /// Load TripoSG plus an optional separate RMBG-1.4 canonical package.
    ///
    /// RMBG is not part of the TripoSG weight file: its BRIA license differs,
    /// so it is distributed in its own canonical directory. Passing `None`
    /// keeps the alpha-only workflow; inputs without a valid alpha channel
    /// then fail with an explicit error, matching the reference's hard
    /// requirement for BriaRMBG.
    pub fn load_with_rmbg(
        model_assets: ModelAssetOptions,
        rmbg_model_dir: Option<std::path::PathBuf>,
    ) -> Result<Self> {
        Self::load_with_rmbg_and_dtype(model_assets, rmbg_model_dir, DType::F32)
    }

    pub fn load_with_rmbg_and_dtype(
        model_assets: ModelAssetOptions,
        rmbg_model_dir: Option<std::path::PathBuf>,
        dtype: DType,
    ) -> Result<Self> {
        let rmbg_weights = match rmbg_model_dir {
            Some(dir) => Some(load_auxiliary_canonical_weights(dir, "rmbg")?),
            None => None,
        };
        Ok(Self {
            weights: load_canonical_weights(ModelFamily::TripoSg, model_assets)?,
            rmbg_weights,
            preprocess: TripoSgPreprocessStage::default(),
            neural: TripoSgNeuralStage,
            geometry: TripoSgGeometryStage,
            export: TripoSgExportStage,
            dtype,
            bundle_cache: Arc::default(),
            rmbg_cache: Arc::default(),
        })
    }

    fn bundle_for(&self, device: &Device) -> Result<Arc<TripoSgModelBundle>> {
        self.bundle_cache.get_or_try_init(device, || {
            TripoSgModelBundle::load(&self.weights, device, self.dtype)
        })
    }

    fn rmbg_for(&self, device: &Device) -> Result<Option<Arc<RmbgModel>>> {
        let Some(weights) = &self.rmbg_weights else {
            return Ok(None);
        };
        self.rmbg_cache
            .get_or_try_init(device, || {
                let config =
                    RmbgConfig::from_resolved_config(&weights.read_resolved_config_json()?)
                        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                            message: format!("failed to parse RMBG config: {source}"),
                        })?;
                // SAFETY: canonical safetensors mmap, same lifetime contract
                // as the TripoSG bundle.
                let vb = unsafe { weights.var_builder(DType::F32, device) }.map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("failed to open RMBG canonical weights: {source}"),
                    }
                })?;
                RmbgModel::load(vb, &config).map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("failed to construct RMBG model: {source}"),
                    }
                })
            })
            .map(Some)
    }

    /// Deterministic Lux3D-seeded latent initialization: `[1, num_tokens,
    /// channels]` F32 standard-normal values generated on CPU (SplitMix64 +
    /// Box-Muller, f64 math rounded to f32) and moved to `device`. Same seed,
    /// same tensor; different seeds differ; NOT torch-RNG-identical (see the
    /// module docs).
    pub fn sample_latents(
        seed: u64,
        num_tokens: usize,
        channels: usize,
        device: &Device,
    ) -> Result<Tensor> {
        if num_tokens == 0 || channels == 0 {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG latent shape ({num_tokens}, {channels}) must be positive on \
                     both axes"
                ),
            });
        }
        let mut rng = SeededNormalRng::new(seed);
        let mut values = Vec::with_capacity(num_tokens * channels);
        for _ in 0..(num_tokens * channels) {
            values.push(rng.next_standard_normal() as f32);
        }
        Tensor::from_vec(values, (1, num_tokens, channels), &Device::Cpu)
            .and_then(|tensor| tensor.to_device(device))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to materialize TripoSG seeded latents: {source}"),
            })
    }

    pub fn prepare_inputs_from_path(
        &self,
        source: &Path,
        device: &Device,
    ) -> Result<TripoSgPreparedInputs> {
        // Match reference `load_image`: BriaRMBG is only constructed when the
        // source has no valid alpha channel. Eager loading costs ~176 MiB of
        // VRAM and pushes native Vulkan/WGPU backends over the 12 GiB limit.
        match self
            .preprocess
            .prepare_inputs_from_path_with_rmbg(source, device, None)
        {
            Ok(prepared) => Ok(prepared),
            Err(Lux3dError::InvalidInput(message))
                if message.starts_with("TripoSG input has no valid alpha channel")
                    && self.rmbg_weights.is_some() =>
            {
                let rmbg = self.rmbg_for(device)?;
                self.preprocess
                    .prepare_inputs_from_path_with_rmbg(source, device, rmbg.as_deref())
            }
            Err(error) => Err(error),
        }
    }

    /// DINOv2-large conditioning embeds `[B, 257, 1024]` from prepared
    /// `[B, 3, 224, 224]` pixel values.
    pub fn image_embeds(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let bundle = self.bundle_for(pixel_values.device())?;
        let pixel_values = pixel_values.to_dtype(self.dtype).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "failed to convert TripoSG image input to {dtype:?}: {source}",
                    dtype = self.dtype
                ),
            }
        })?;
        let embeds = bundle.dino.forward(&pixel_values).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSG image encoder forward failed: {source}"),
            }
        })?;
        embeds
            .to_dtype(DType::F32)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to convert TripoSG image embeds to f32: {source}"),
            })
    }

    /// Full inference from an image file: preprocess -> DINOv2 embeds ->
    /// seeded latents -> CFG denoise. The VAE/mesh stage is driven separately
    /// through [`Self::extract_mesh`] so callers can cache latents.
    pub fn infer_from_path(
        &self,
        source: &Path,
        device: &Device,
        options: &TripoSgRunOptions,
    ) -> Result<TripoSgInferenceOutput> {
        options.validate()?;
        let prepared = self.prepare_inputs_from_path(source, device)?;
        self.neural
            .infer_latents(self, &prepared.pixel_values, options)
    }

    /// Denoise caller-supplied initial latents with classifier-free guidance.
    /// The latents are NOT regenerated here: `latents_init` must already be
    /// `[1, num_tokens, 64]` (batch 1), e.g. from [`Self::sample_latents`] or
    /// a parity baseline. `image_embeds` is the conditional DINOv2 context;
    /// the uncond branch is `zeros_like` inside the denoise loop.
    ///
    /// The schedule is the linear rectified-flow Euler schedule for
    /// `options.steps`, or — when `options.timesteps` is `Some` — the exact
    /// explicit schedule (`RectifiedFlowSchedule::from_timesteps`), with the
    /// step count derived from the schedule length like upstream
    /// `retrieve_timesteps`.
    pub fn infer_with_latents(
        &self,
        image_embeds: &Tensor,
        latents_init: &Tensor,
        options: &TripoSgRunOptions,
    ) -> Result<Tensor> {
        options.validate()?;
        Self::validate_latents(latents_init, options.num_tokens)?;
        let bundle = self.bundle_for(latents_init.device())?;
        let scheduler = match &options.timesteps {
            Some(timesteps) => RectifiedFlowSchedule::from_timesteps(timesteps, SCHEDULER_SHIFT),
            None => RectifiedFlowSchedule::new(options.steps, SCHEDULER_SHIFT),
        };
        denoise_with_schedule(
            &bundle.dit,
            latents_init,
            image_embeds,
            &scheduler,
            options.guidance_scale,
            self.dtype,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("TripoSG denoise failed: {source}"),
        })
    }

    /// Encode a point cloud into the TripoSG latent distribution.
    ///
    /// `points [B, N, C]` with the first three channels interpreted as
    /// positions and the remaining channels as per-point features (TripoSG's
    /// reference script feeds `[positions, normals]`, C = 6). The reference
    /// encoder samples `num_tokens * 4` points, runs deterministic FPS at
    /// ratio 1/4 (`random_start=False` in eval mode), and returns the
    /// `DiagonalGaussianDistribution` from `vae.quant`.
    ///
    /// `seed` controls the Lux3D deterministic index sampler; pass `None` for
    /// a process-time seed. The transformer/media pipeline does not call this
    /// path, but it is the full reference encode API.
    pub fn encode_point_cloud(
        &self,
        points: &Tensor,
        num_tokens: usize,
        seed: Option<u64>,
    ) -> Result<TripoSgLatentDistribution> {
        let dims = points.dims();
        if dims.len() != 3 || dims[0] == 0 || dims[1] == 0 || dims[2] < 3 {
            return Err(Lux3dError::InvalidInput(
                "TripoSG VAE encode expects point features [B, N, C] with B/N > 0 and C >= 3",
            ));
        }
        let bundle = self.bundle_for(points.device())?;
        bundle
            .vae_encoder
            .encode(points, num_tokens, seed)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSG VAE encode failed: {source}"),
            })
    }

    /// Encode with an explicit sampled query subset (reference parity path).
    pub fn encode_point_cloud_sampled(
        &self,
        points: &Tensor,
        sampled_points: &Tensor,
    ) -> Result<Tensor> {
        let bundle = self.bundle_for(points.device())?;
        bundle
            .vae_encoder
            .encode_sampled(points, sampled_points)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSG VAE encode failed: {source}"),
            })
    }

    /// Encode with an explicit initial NumPy-choice index vector; the FPS
    /// half still runs. Used by the reference parity tests.
    pub fn encode_point_cloud_with_indices(
        &self,
        points: &Tensor,
        indices: &[usize],
        num_tokens: usize,
    ) -> Result<Tensor> {
        let bundle = self.bundle_for(points.device())?;
        let sampled = bundle
            .vae_encoder
            .sample_with_indices(points, indices, num_tokens)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSG VAE FPS sampling failed: {source}"),
            })?;
        bundle
            .vae_encoder
            .encode_sampled(points, &sampled)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSG VAE encode failed: {source}"),
            })
    }

    /// Reference `scripts/inference_vae.py` surface entry point: load the
    /// keyed `.npy` dict (`surface_points` / `surface_normals`), take the
    /// reference `num_pc = 204800` subset, run the VAE encoder and sample the
    /// latent distribution (`vae.encode(surface).latent_dist.sample()`).
    ///
    /// The image-to-3D preprocess, DINOv2 conditioning and denoise stages are
    /// skipped entirely; the returned latent feeds [`Self::extract_mesh`]
    /// directly. The sampled latent uses the Lux3D seeded RNG (documented RNG
    /// deviation: same distribution, not torch-draw-identical).
    pub fn encode_surface_from_path(
        &self,
        path: &Path,
        device: &Device,
        options: &TripoSgRunOptions,
    ) -> Result<Tensor> {
        options.validate()?;
        let surface = super::npy::read_surface_arrays(path).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "failed to load TripoSG surface input `{}`: {source}",
                    path.display()
                ),
            }
        })?;
        if surface.is_empty() {
            return Err(Lux3dError::InvalidInput(
                "TripoSG surface input contains no points",
            ));
        }
        let indices =
            surface_selection_indices(surface.len(), TRIPOSG_SURFACE_POINT_COUNT, options.seed);
        let mut features = Vec::with_capacity(TRIPOSG_SURFACE_POINT_COUNT * 6);
        for &index in &indices {
            let base = index * 3;
            features.extend_from_slice(&surface.points[base..base + 3]);
            features.extend_from_slice(&surface.normals[base..base + 3]);
        }
        let points = Tensor::from_vec(features, (1, TRIPOSG_SURFACE_POINT_COUNT, 6), &Device::Cpu)
            .and_then(|points| points.to_device(device))
            .and_then(|points| points.to_dtype(self.dtype))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to materialize TripoSG surface points: {source}"),
            })?;
        let distribution =
            self.encode_point_cloud(&points, options.num_tokens, Some(options.seed))?;
        distribution
            .sample(options.seed)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSG VAE latent sampling failed: {source}"),
            })
    }

    /// Decode the SDF field for `latents [1, num_tokens, 64]` at the run's
    /// bounds (canonical `+/-1.005` unless `options.bounds` overrides them)
    /// and assemble the level-0 marching-cubes mesh. The buffers are
    /// materialized on `latents.device()` in the SAME absolute world frame —
    /// a bounds override only moves the query window, it never re-normalizes
    /// the returned coordinates.
    ///
    /// When `options.faces` is `Some(n > 0)` and the extracted mesh has more
    /// than `n` faces, the upstream `--faces` postprocess runs
    /// (`meshing_merge_close_vertices` + quadric decimation to `n`; see the
    /// option docs).
    pub fn extract_mesh(
        &self,
        latents: &Tensor,
        options: &TripoSgRunOptions,
    ) -> Result<TripoSgMeshBuffers> {
        options.validate()?;
        Self::validate_latents(latents, options.num_tokens)?;
        let bundle = self.bundle_for(latents.device())?;
        let latents = latents.to_dtype(self.dtype).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "failed to convert TripoSG latents to {dtype:?}: {source}",
                    dtype = self.dtype
                ),
            }
        })?;
        let bounds = options.effective_bounds();
        let mut mesh_cpu = if options.use_flash_decoder {
            // Reference default (`use_flash_decoder=True`): hierarchical
            // flash SDF query schedule + clean-room DiffDMC.
            let grid = flash_sdf_grid(
                &bundle.vae,
                &latents,
                bounds,
                options.flash_octree_depth,
                options.query_chunk_size,
            )
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSG flash SDF grid decode failed: {source}"),
            })?;
            let mesh = flash_dual_marching_cubes(&grid).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("TripoSG flash DMC failed: {source}"),
                }
            })?;
            // Reference pipelines wrap the DMC output in
            // `trimesh.Trimesh(..., process=True)`, which drops vertices no
            // face references (NaN-padded boundary shells at depth >= 7).
            let mesh = compact_dmc_mesh(&mesh);
            TripoSgMeshCpu {
                resolution: grid.dims.saturating_sub(1) as u32,
                sdf_level: CANONICAL_SDF_LEVEL,
                bounds,
                vertices: mesh.vertices,
                faces: mesh.faces,
            }
        } else {
            let grid = hierarchical_sdf_grid(
                &bundle.vae,
                &latents,
                bounds,
                options.dense_octree_depth,
                options.hierarchical_octree_depth,
                options.query_chunk_size,
            )
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSG SDF grid decode failed: {source}"),
            })?;
            self.geometry.assemble_cpu(&grid)?
        };

        // Reference `--faces` (scripts/inference_triposg.py::simplify_mesh):
        // merge close vertices then quadric-decimate, only when the mesh has
        // MORE faces than the target (otherwise upstream returns it
        // unchanged). TripoSG meshes are positions+faces only, so no vertex
        // attributes need remapping.
        if let Some(target_faces) = options.faces {
            if target_faces > 0 && mesh_cpu.faces.len() / 3 > target_faces {
                let simplified = simplify_triposg_mesh(&mesh_cpu, target_faces)?;
                mesh_cpu = simplified;
            }
        }

        let device = latents.device().clone();
        let vertex_count = mesh_cpu.vertices.len() / 3;
        let face_count = mesh_cpu.faces.len() / 3;
        let vertices =
            Tensor::from_vec(mesh_cpu.vertices, (vertex_count, 3), &device).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to materialize TripoSG mesh vertices: {source}"),
                }
            })?;
        let faces =
            Tensor::from_vec(mesh_cpu.faces, (face_count, 3), &device).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to materialize TripoSG mesh faces: {source}"),
                }
            })?;
        Ok(TripoSgMeshBuffers { vertices, faces })
    }

    /// Export mesh buffers in the format selected by `path`'s extension:
    /// `.glb` (primary), `.obj`, `.ply` or `.stl`.
    pub fn export_mesh(&self, mesh: &TripoSgMeshBuffers, path: &Path) -> Result<MeshFormat> {
        let mesh_cpu = triposg_mesh_buffers_to_cpu(mesh)?;
        self.export.write_mesh(&mesh_cpu, path)
    }

    /// Export mesh buffers as ASCII OBJ (legacy entry point).
    pub fn export_obj(&self, mesh: &TripoSgMeshBuffers, path: &Path) -> Result<()> {
        let mesh_cpu = triposg_mesh_buffers_to_cpu(mesh)?;
        self.export.write_obj(&mesh_cpu, path)
    }

    /// `latents` must be `[1, num_tokens, 64]` (batch 1 — the VAE decoder and
    /// the CFG denoise loop are batch-1 paths).
    fn validate_latents(latents: &Tensor, num_tokens: usize) -> Result<()> {
        let dims = latents.dims();
        if dims.len() != 3
            || dims[0] != 1
            || dims[1] != num_tokens
            || dims[2] != TRIPOSG_LATENT_CHANNELS
        {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG latents must be [1, {num_tokens}, {TRIPOSG_LATENT_CHANNELS}] \
                     (batch 1), got {dims:?}"
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Shared mesh-buffer helpers (image and scribble pipelines)
// ---------------------------------------------------------------------------

/// Materialize a CPU mesh as device buffers on `device`.
fn triposg_mesh_buffers(mesh: &TripoSgMeshCpu, device: &Device) -> Result<TripoSgMeshBuffers> {
    let vertex_count = mesh.vertices.len() / 3;
    let face_count = mesh.faces.len() / 3;
    let vertices =
        Tensor::from_vec(mesh.vertices.clone(), (vertex_count, 3), device).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to materialize TripoSG mesh vertices: {source}"),
            }
        })?;
    let faces =
        Tensor::from_vec(mesh.faces.clone(), (face_count, 3), device).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to materialize TripoSG mesh faces: {source}"),
            }
        })?;
    Ok(TripoSgMeshBuffers { vertices, faces })
}

/// Gather device mesh buffers back to CPU. The device buffers don't carry
/// grid metadata (resolution and query bounds live in the SDF grid, which is
/// consumed inside `extract_mesh`); exporters only read vertices/faces, so
/// the bounds here are descriptive metadata only (canonical default).
fn triposg_mesh_buffers_to_cpu(mesh: &TripoSgMeshBuffers) -> Result<TripoSgMeshCpu> {
    let vertices = mesh
        .vertices
        .to_device(&Device::Cpu)
        .and_then(|tensor| tensor.flatten_all())
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to gather TripoSG mesh vertices on cpu: {source}"),
        })?;
    let faces = mesh
        .faces
        .to_device(&Device::Cpu)
        .and_then(|tensor| tensor.flatten_all())
        .and_then(|tensor| tensor.to_vec1::<i64>())
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to gather TripoSG mesh faces on cpu: {source}"),
        })?;
    Ok(TripoSgMeshCpu {
        resolution: 0,
        sdf_level: CANONICAL_SDF_LEVEL,
        bounds: CANONICAL_BOUNDS,
        vertices,
        faces,
    })
}

// ---------------------------------------------------------------------------
// `--faces` postprocess (pymeshlab merge-close + quadric decimation)
// ---------------------------------------------------------------------------

/// `simplify_mesh` from `scripts/inference_triposg.py`: pymeshlab
/// `meshing_merge_close_vertices()` (default threshold = 1/10000 of the mesh
/// bbox diagonal) followed by
/// `meshing_decimation_quadric_edge_collapse(targetfacenum=n)`. The decimation
/// uses meshopt's quadric-style simplifier (`simplify_mesh`, target-error 1.0
/// so the face count dominates) — a same-family algorithm, NOT pymeshlab's
/// exact solver, so the surviving topology can differ from a pymeshlab run at
/// equal face counts (the TRELLIS.2 export path made the same substitution).
///
/// The caller guarantees `target_faces > 0` and a face count above the
/// target; the mesh keeps its bounds/sdf-level metadata and the result stays
/// in the same absolute world frame.
fn simplify_triposg_mesh(mesh: &TripoSgMeshCpu, target_faces: usize) -> Result<TripoSgMeshCpu> {
    let vertex_count = mesh.vertices.len() / 3;
    let mut bbox_min = [f64::INFINITY; 3];
    let mut bbox_max = [f64::NEG_INFINITY; 3];
    for vertex in mesh.vertices.chunks_exact(3) {
        for axis in 0..3 {
            bbox_min[axis] = bbox_min[axis].min(f64::from(vertex[axis]));
            bbox_max[axis] = bbox_max[axis].max(f64::from(vertex[axis]));
        }
    }
    let diagonal = if vertex_count == 0 {
        0.0
    } else {
        (bbox_max[0] - bbox_min[0])
            .powi(2)
            .max((bbox_max[1] - bbox_min[1]).powi(2))
            .max((bbox_max[2] - bbox_min[2]).powi(2))
            .sqrt()
    };
    let epsilon = (diagonal * PYMESHLAB_MERGE_CLOSE_BBOX_DIAGONAL_FRACTION) as f32;

    // TripoSG mesh faces are zero-based and in range by construction
    // (both extraction paths emit validated indices); a defensive remap
    // keeps a malformed buffer from panicking.
    let faces_u32 = mesh
        .faces
        .iter()
        .map(|&index| {
            u32::try_from(index).map_err(|_| {
                Lux3dError::InvalidInput("TripoSG mesh face index is negative (mesh is corrupt)")
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let merged = merge_close_vertices(&mesh.vertices, &faces_u32, epsilon);
    let simplified = simplify_mesh(&merged.vertices, &merged.faces, target_faces);

    Ok(TripoSgMeshCpu {
        resolution: mesh.resolution,
        sdf_level: mesh.sdf_level,
        bounds: mesh.bounds,
        vertices: simplified.vertices,
        faces: simplified
            .faces
            .iter()
            .map(|&index| i64::from(index))
            .collect(),
    })
}

// ---------------------------------------------------------------------------
// Deterministic CPU RNG (SplitMix64 + Box-Muller)
// ---------------------------------------------------------------------------

/// SplitMix64 state + Box-Muller transform for the latent initialization.
///
/// Chosen for reproducibility, not statistical sophistication: every run with
/// the same seed walks the exact same f64 sequence, and rounding to f32 at the
/// end keeps the values stable across platforms (all math happens in f64).
pub(crate) struct SeededNormalRng {
    state: u64,
}

impl SeededNormalRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// SplitMix64 next output.
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` from the top 53 bits.
    fn next_uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Unbiased index in `0..n` via Lemire rejection (surface subset draw).
    fn next_index(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        let bound = n as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            let product = (value as u128) * (bound as u128);
            let low = product as u64;
            if low >= threshold {
                return (product >> 64) as usize;
            }
        }
    }

    /// One standard normal per two uniforms (Box-Muller, cosine branch).
    /// The first uniform is clamped away from zero so `ln` stays finite.
    fn next_standard_normal(&mut self) -> f64 {
        let u1 = self.next_uniform().max(f64::EPSILON);
        let u2 = self.next_uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod surface_selection_tests {
    use super::*;

    #[test]
    fn reference_point_count_is_204800() {
        assert_eq!(TRIPOSG_SURFACE_POINT_COUNT, 204_800);
    }

    #[test]
    fn selection_is_deterministic_and_seed_sensitive() {
        let first = surface_selection_indices(1_000, 64, 42);
        let second = surface_selection_indices(1_000, 64, 42);
        assert_eq!(first, second, "same seed must reproduce the subset");
        let other = surface_selection_indices(1_000, 64, 43);
        assert_ne!(first, other, "different seeds must change the subset");
    }

    #[test]
    fn selection_is_without_replacement_when_enough_points() {
        let selected = surface_selection_indices(10_000, 4_096, 7);
        assert_eq!(selected.len(), 4_096);
        let mut sorted = selected.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), selected.len(), "indices must be unique");
        assert!(selected.iter().all(|index| *index < 10_000));
    }

    #[test]
    fn selection_upsamples_small_fixtures() {
        let selected = surface_selection_indices(3, 16, 7);
        assert_eq!(selected.len(), 16);
        assert!(selected.iter().all(|index| *index < 3));
        // A small pool must repeat points (with-replacement path).
        let mut sorted = selected;
        sorted.sort_unstable();
        sorted.dedup();
        assert!(sorted.len() <= 3);
    }
}
