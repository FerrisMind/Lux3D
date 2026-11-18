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

use candle_core::{DType, Device, Tensor, quantized::gguf_file};

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
    weights::{VarBuilder, GGUF_VARIANT_KEY, TRIPOSG_SCRIBBLE_VARIANT, is_gguf_path},
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
/// Files the raw diffusers-style scribble checkpoint must provide. When a
/// quantized DiT GGUF is resolved (explicit `--dit-gguf` or the sibling
/// `transformer/model.gguf`), the float DiT shard
/// ([`TRIPOSG_SCRIBBLE_DIT_SHARD`]) is not required.
pub const TRIPOSG_SCRIBBLE_REQUIRED_FILES: [&str; 12] = [
    "model_index.json",
    "transformer/config.json",
    TRIPOSG_SCRIBBLE_DIT_SHARD,
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

/// Float DiT shard (3.13 GiB in the shipped checkpoint); replaced by a GGUF
/// override when one is resolved.
pub const TRIPOSG_SCRIBBLE_DIT_SHARD: &str = "transformer/diffusion_pytorch_model.safetensors";

/// Sibling quantized DiT package preferred over the float shard by the loader
/// (`weights quantize --scribble` default output).
pub const TRIPOSG_SCRIBBLE_DIT_GGUF: &str = "transformer/model.gguf";

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
///
/// A canonical `triposg-scribble` package is accepted directly. A sibling
/// `transformer/model.gguf` satisfies the DiT requirement, so a checkpoint
/// whose float DiT shard was replaced by the quantized package (the documented
/// `weights quantize --scribble` output) still resolves.
pub fn resolve_triposg_scribble_model_dir(explicit: Option<&Path>) -> Result<PathBuf> {
    let candidate = scribble_model_dir_candidate(explicit);
    if !candidate.is_dir() {
        return Err(scribble_model_dir_missing_error());
    }
    if scribble_canonical_package(&candidate).is_some() {
        return Ok(candidate);
    }
    // The loader prefers an explicit or sibling GGUF over the safetensors
    // shard, so do not require the 3.13 GiB float DiT file when one exists.
    let require_dit_safetensors = scribble_sibling_dit_gguf(&candidate).is_none();
    resolve_scribble_model_dir_with(&candidate, require_dit_safetensors)
}

fn scribble_model_dir_missing_error() -> Lux3dError {
    Lux3dError::InvalidInput(
        "TripoSG scribble weights were not found; download the Hugging Face \
         `VAST-AI/TripoSG-scribble` checkpoint and pass --model-path (or set \
         LUX3D_TRIPOSG_SCRIBBLE_MODEL_DIR). Expected the diffusers-style directory \
         with transformer/, vae/, image_encoder_dinov2/, text_encoder/ and tokenizer/.",
    )
}

fn scribble_model_dir_candidate(explicit: Option<&Path>) -> PathBuf {
    match explicit {
        Some(path) => path.to_path_buf(),
        None => std::env::var_os(TRIPOSG_SCRIBBLE_MODEL_DIR_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(TRIPOSG_SCRIBBLE_DEFAULT_DIR)),
    }
}

fn scribble_sibling_dit_gguf(dir: &Path) -> Option<PathBuf> {
    let sibling = dir.join(TRIPOSG_SCRIBBLE_DIT_GGUF);
    sibling.is_file().then_some(sibling)
}

/// Root name of a standalone scribble package (one GGUF with every
/// component), as opposed to the raw checkpoint's `transformer/model.gguf`.
const TRIPOSG_SCRIBBLE_STANDALONE_GGUF: &str = "model.gguf";
const TRIPOSG_SCRIBBLE_COMPONENT_PREFIXES: [&str; 4] =
    ["image_encoder.", "transformer.", "vae.", "text_encoder."];

/// Return the standalone GGUF for a `.gguf` file or a package directory.
///
/// A directory that still has `transformer/config.json` is treated as the raw
/// diffusers checkpoint even if it also contains a root `model.gguf`, so the
/// existing DiT-override layout keeps its meaning.
fn scribble_standalone_gguf(path: &Path) -> Option<PathBuf> {
    if path.is_file() {
        return (is_gguf_path(path) && scribble_gguf_is_standalone(path))
            .then(|| path.to_path_buf());
    }
    if path.is_dir() {
        let candidate = path.join(TRIPOSG_SCRIBBLE_STANDALONE_GGUF);
        if candidate.is_file()
            && !path.join("transformer/config.json").is_file()
            && scribble_gguf_is_standalone(&candidate)
        {
            return Some(candidate);
        }
    }
    None
}

/// Return the canonical `triposg-scribble` package directory when `path`
/// contains the four files written by `weights normalize ... triposg`.
fn scribble_canonical_package(path: &Path) -> Option<PathBuf> {
    let required = [
        "model.safetensors",
        "resolved_config.json",
        "manifest.json",
        "checksums.json",
    ];
    (path.is_dir()
        && required
            .iter()
            .all(|relative| path.join(relative).is_file()))
    .then(|| path.to_path_buf())
}

fn scribble_gguf_is_standalone(path: &Path) -> bool {
    let Ok(mut file) = fs::File::open(path) else {
        return false;
    };
    let Ok(content) = gguf_file::Content::read(&mut file) else {
        return false;
    };
    if let Some(gguf_file::Value::String(variant)) = content.metadata.get(GGUF_VARIANT_KEY) {
        if variant.as_str() == TRIPOSG_SCRIBBLE_VARIANT {
            return true;
        }
    }
    TRIPOSG_SCRIBBLE_COMPONENT_PREFIXES.iter().all(|prefix| {
        content
            .tensor_infos
            .keys()
            .any(|name| name.starts_with(prefix))
    })
}

/// Validate a resolved scribble checkpoint directory. `require_dit_safetensors`
/// is false when an explicit or sibling GGUF supplies the DiT weights.
fn resolve_scribble_model_dir_with(
    candidate: &Path,
    require_dit_safetensors: bool,
) -> Result<PathBuf> {
    let missing = TRIPOSG_SCRIBBLE_REQUIRED_FILES
        .iter()
        .filter(|relative| require_dit_safetensors || **relative != TRIPOSG_SCRIBBLE_DIT_SHARD)
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
    Ok(candidate.to_path_buf())
}

/// Surface points the reference `load_surface` draws before encoding
/// (`scripts/inference_vae.py:12-23`, `num_pc = 204800`; the shipped demo has
/// 500000 points, so the draw is without replacement).
pub(crate) const TRIPOSG_SURFACE_POINT_COUNT: usize = 204_800;

/// Deterministic surface-point subset selection for `--surface-input`.
///
/// The reference `load_surface` draws `num_pc` indices with
/// `np.random.default_rng().choice(n, num_pc, replace=False)`, which raises
/// when the input has fewer than `num_pc` points. Lux3D replaces that draw
/// with the documented SplitMix64 RNG (same distribution, not
/// draw-identical) and mirrors the reference precondition: inputs with fewer
/// than `count` points are rejected instead of being silently upsampled with
/// replacement.
fn surface_selection_indices(total: usize, count: usize, seed: u64) -> Result<Vec<usize>> {
    if total < count {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "TripoSG surface input contains {total} points, but at least {count} are \
                 required (the reference num_pc = {TRIPOSG_SURFACE_POINT_COUNT} samples \
                 without replacement and cannot upsample a smaller input)"
            ),
        });
    }
    let mut rng = SeededNormalRng::new(seed);
    // Partial Fisher-Yates: `count` uniform draws without replacement.
    let mut pool: Vec<usize> = (0..total).collect();
    for index in 0..count {
        let swap = index + rng.next_index(total - index);
        pool.swap(index, swap);
    }
    pool.truncate(count);
    Ok(pool)
}

/// Per-run options for the TripoSG pipeline. Defaults replicate the reference
/// `TripoSGPipeline` call in `pipeline_triposg.py` (`num_inference_steps=50`,
/// `guidance_scale=7.0`, `num_tokens=2048`, dense depth 8 / hierarchical
/// depth 9, `num_chunks`-equivalent query chunk 16384).
///
/// Upstream knobs without a field here:
/// - `num_shapes_per_prompt`: exposed as the explicit `num_shapes` argument
///   of [`TripoSgPipeline::infer_batch_from_paths`] (CLI `--num-shapes`).
///   The batch contract is `num_images * num_shapes`; the conditioning pass
///   expands via `repeat_interleave` (`pipeline_triposg.py:146`) and the DiT
///   runs batched, while the VAE decode loops per sample (the reference
///   decode helpers are effectively batch-1). The `num_shapes = 1` +
///   single-image path is byte-identical to the pre-batch pipeline.
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
/// latents and the denoised latents (all F32). The batch axis is
/// `num_images * num_shapes`, in image-major `repeat_interleave` order
/// (`pipeline_triposg.py:146, 233-241`); a single image with `num_shapes = 1`
/// keeps the historical `[1, ...]` shapes.
#[derive(Debug, Clone)]
pub struct TripoSgInferenceOutput {
    /// `[B, 257, 1024]` last hidden state of the DINOv2-large encoder after
    /// the `repeat_interleave(num_shapes)` conditioning expansion.
    pub image_embeds: Tensor,
    /// `[B, num_tokens, 64]` seeded standard-normal init latents (ONE draw).
    pub latents_init: Tensor,
    /// `[B, num_tokens, 64]` latents after the rectified-flow Euler schedule
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

/// `repeat_interleave(repeats, dim=0)` on the batch axis of `[N, S, D]`
/// conditioning embeds (`pipeline_triposg.py:146`): the output is
/// `[N * repeats, S, D]` with image-major order, i.e. rows
/// `[0, 0, ..., 1, 1, ..., N - 1, N - 1, ...]`.
///
/// `repeats == 1` returns the input unchanged (the historical single-shape
/// path stays byte-identical), and `repeats == 0` is rejected.
pub fn repeat_interleave_batch(embeds: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 0 {
        return Err(Lux3dError::InvalidInput(
            "TripoSG repeat_interleave repeats must be at least 1, got 0",
        ));
    }
    if repeats == 1 {
        return Ok(embeds.clone());
    }
    let batch = embeds
        .dim(0)
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("TripoSG conditioning embeds must be rank >= 1: {source}"),
        })?;
    if batch == 0 {
        return Err(Lux3dError::InvalidInput(
            "TripoSG conditioning batch must be at least 1, got 0",
        ));
    }
    let mut rows = Vec::with_capacity(batch * repeats);
    for index in 0..batch {
        let row = embeds.narrow(0, index, 1).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to slice TripoSG conditioning row {index}: {source}"),
            }
        })?;
        for _ in 0..repeats {
            rows.push(row.clone());
        }
    }
    let refs: Vec<&Tensor> = rows.iter().collect();
    Tensor::cat(&refs, 0).map_err(|source| Lux3dError::CanonicalWeightsValidation {
        message: format!("failed to expand TripoSG conditioning batch: {source}"),
    })
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
    fn load(dir: &Path, dit_gguf: Option<&Path>, device: &Device, dtype: DType) -> Result<Self> {
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
                scribble_dit_var_builder(dir, dit_gguf, dtype, device)?,
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

    /// Load every component from one standalone GGUF written by
    /// `weights quantize triposg --scribble --standalone`.
    fn load_standalone(gguf: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let standalone = crate::weights::read_standalone_gguf(gguf, "triposg")?;
        let config = &standalone.resolved_config;
        let vb = VarBuilder::from_gguf(gguf, dtype, device).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "failed to open standalone TripoSG-scribble GGUF `{}`: {source}",
                    gguf.display()
                ),
            }
        })?;
        let tokenizer = scribble_resolved_tokenizer(config, gguf)?;
        Self::load_from_vb(
            config,
            vb,
            tokenizer,
            dtype,
            &format!("standalone GGUF `{}`", gguf.display()),
        )
    }

    /// Load every component from a canonical `triposg-scribble` package.
    /// The package may also contain a quantized `model.gguf`; the canonical
    /// `VarBuilder` prefers it over `model.safetensors`.
    fn load_canonical(dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let paths = load_canonical_weights(
            ModelFamily::TripoSg,
            ModelAssetOptions {
                canonical_dir: Some(dir.to_path_buf()),
                ..Default::default()
            },
        )?;
        if paths.manifest.variant.as_deref() != Some(TRIPOSG_SCRIBBLE_VARIANT) {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "canonical package `{}` is not TripoSG-scribble; rebuild it with \
                     `weights normalize ... triposg` from the raw scribble checkpoint",
                    dir.display()
                ),
            });
        }
        let config = paths.read_resolved_config_json()?;
        let vb = unsafe { paths.var_builder(dtype, device) }.map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "failed to open canonical TripoSG-scribble weights in `{}`: {source}",
                    dir.display()
                ),
            }
        })?;
        let tokenizer = scribble_resolved_tokenizer(&config, dir)?;
        Self::load_from_vb(
            &config,
            vb,
            tokenizer,
            dtype,
            &format!("canonical package `{}`", dir.display()),
        )
    }

    fn load_from_vb(
        config: &serde_json::Value,
        vb: VarBuilder<'static>,
        tokenizer: ClipBpeTokenizer,
        dtype: DType,
        source: &str,
    ) -> Result<Self> {
        if config.get("variant").and_then(serde_json::Value::as_str)
            != Some(TRIPOSG_SCRIBBLE_VARIANT)
        {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!("{source} is not a TripoSG-scribble package"),
            });
        }
        let transformer_config =
            config
                .get("transformer")
                .ok_or_else(|| Lux3dError::CanonicalWeightsValidation {
                    message: format!("{source} has no transformer config"),
                })?;
        let vae_config = config
            .get("vae")
            .ok_or_else(|| Lux3dError::CanonicalWeightsValidation {
                message: format!("{source} has no VAE config"),
            })?;
        let dino_config =
            config
                .get("image_encoder")
                .ok_or_else(|| Lux3dError::CanonicalWeightsValidation {
                    message: format!("{source} has no image-encoder config"),
                })?;
        let text_config =
            config
                .get("text_encoder")
                .ok_or_else(|| Lux3dError::CanonicalWeightsValidation {
                    message: format!("{source} has no text-encoder config"),
                })?;

        let transformer_cfg = TripoSgTransformerConfig::from_resolved_config(transformer_config)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to parse TripoSG-scribble transformer config: {source}"),
            })?;
        if !transformer_cfg.use_cross_attention_2 {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "{source} transformer config does not enable use_cross_attention_2"
                ),
            });
        }
        let vae_cfg = TripoSgVaeConfig::from_resolved_config(vae_config).map_err(|source| {
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
                    "{source} image encoder must be DINOv2-large (hidden 1024, 24 layers), \
                     got hidden {dino_hidden}, {dino_layers} layers"
                ),
            });
        }
        if transformer_cfg.cross_attention_2_dim != Some(dino_hidden) {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "{source} transformer cross_attention_2_dim {:?} does not match the \
                     image encoder hidden size {dino_hidden}",
                    transformer_cfg.cross_attention_2_dim
                ),
            });
        }

        let text_encoder = TripoSgClipTextEncoder::load(vb.pp("text_encoder"), text_config, dtype)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to construct TripoSG-scribble CLIP text encoder: {source}"),
            })?;
        if text_encoder.config().embed_dim != transformer_cfg.cross_attention_dim {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "{source} text encoder hidden size {} does not match the transformer \
                     cross_attention_dim {}",
                    text_encoder.config().embed_dim,
                    transformer_cfg.cross_attention_dim
                ),
            });
        }

        Ok(Self {
            dino: TripoSgDinoEncoder::load(vb.pp("image_encoder")).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct TripoSG-scribble image encoder: {source}"),
                }
            })?,
            dit: TripoSgDiT::load(vb.pp("transformer"), &transformer_cfg).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct TripoSG-scribble DiT: {source}"),
                }
            })?,
            vae: TripoSgVaeDecoder::load(vb.pp("vae"), &vae_cfg).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct TripoSG-scribble VAE decoder: {source}"),
                }
            })?,
            text_encoder,
            tokenizer,
        })
    }
}

fn scribble_resolved_tokenizer(
    config: &serde_json::Value,
    source_path: &Path,
) -> Result<ClipBpeTokenizer> {
    let tokenizer =
        config
            .get("tokenizer")
            .ok_or_else(|| Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG-scribble package `{}` has no embedded tokenizer",
                    source_path.display()
                ),
            })?;
    let vocab = tokenizer
        .get("vocab_json")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "TripoSG-scribble package `{}` has no embedded vocab.json",
                source_path.display()
            ),
        })?;
    let merges = tokenizer
        .get("merges_txt")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "TripoSG-scribble package `{}` has no embedded merges.txt",
                source_path.display()
            ),
        })?;
    let tokenizer_config = tokenizer
        .get("config")
        .cloned()
        .unwrap_or_else(|| serde_json::Value::Object(Default::default()));
    let special_tokens_map = tokenizer.get("special_tokens_map");
    ClipBpeTokenizer::from_json(vocab, merges, &tokenizer_config, special_tokens_map).map_err(
        |source| Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "failed to load the TripoSG-scribble CLIP tokenizer from `{}`: {source}",
                source_path.display()
            ),
        },
    )
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

/// Resolve the scribble DiT weights: explicit GGUF override, then a sibling
/// `transformer/model.gguf` (the `weights quantize --scribble` default
/// output), then the raw diffusers safetensors shard. This mirrors
/// [`CanonicalWeightSetPaths::var_builder`], which prefers a sibling
/// `model.gguf` over `model.safetensors` for the image pipeline.
fn scribble_dit_var_builder(
    dir: &Path,
    dit_gguf: Option<&Path>,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let gguf_path = match dit_gguf {
        Some(path) => Some(path.to_path_buf()),
        None => scribble_sibling_dit_gguf(dir),
    };
    if let Some(path) = gguf_path {
        return VarBuilder::from_gguf(&path, dtype, device).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "failed to open TripoSG-scribble DiT GGUF `{}`: {source}",
                    path.display()
                ),
            }
        });
    }
    raw_safetensors_var_builder(dir, TRIPOSG_SCRIBBLE_DIT_SHARD, dtype, device)
}

/// Resolve a scribble run source: a canonical `triposg-scribble` package, a
/// raw checkpoint directory, a standalone GGUF (file or package directory), or
/// a legacy DiT `.gguf` (the remaining weights then come from
/// [`TRIPOSG_SCRIBBLE_MODEL_DIR_ENV`]/[`TRIPOSG_SCRIBBLE_DEFAULT_DIR`]). The
/// float DiT shard is not required when an explicit GGUF supplies the DiT.
pub fn resolve_triposg_scribble_model_source(
    explicit: Option<&Path>,
) -> Result<(PathBuf, Option<PathBuf>)> {
    if let Some(path) = explicit {
        if let Some(standalone) = scribble_standalone_gguf(path) {
            return Ok((standalone, None));
        }
        if path.is_file() {
            if !crate::weights::is_gguf_path(path) {
                return Err(Lux3dError::InvalidInput(
                    "TripoSG scribble `--model-path` file must be a standalone `.gguf` \
                     package or a quantized DiT `.gguf`; pass the raw checkpoint directory \
                     for the float checkpoint",
                ));
            }
            let fallback = scribble_model_dir_candidate(None);
            if !fallback.is_dir() {
                return Err(scribble_model_dir_missing_error());
            }
            return Ok((
                resolve_scribble_model_dir_with(&fallback, false)?,
                Some(path.to_path_buf()),
            ));
        }
    }
    let fallback = scribble_model_dir_candidate(explicit);
    if let Some(standalone) = scribble_standalone_gguf(&fallback) {
        return Ok((standalone, None));
    }
    Ok((resolve_triposg_scribble_model_dir(explicit)?, None))
}

/// TripoSG-scribble pipeline: preprocess -> CLIP text embeds + DINOv2
/// scribble embeds -> seeded rectified-flow denoise with dual cross-attention
/// scales -> VAE SDF decode (dense/hierarchical marching cubes) -> GLB export.
#[derive(Debug)]
pub struct TripoSgScribblePipeline {
    /// Raw diffusers-style checkpoint directory (`VAST-AI/TripoSG-scribble`)
    /// or a canonical `triposg-scribble` package directory.
    pub model_dir: PathBuf,
    /// Optional explicit DiT GGUF. When absent a sibling
    /// `transformer/model.gguf` is preferred over the safetensors shard
    /// (same precedence as the canonical image pipeline).
    pub dit_gguf: Option<PathBuf>,
    /// Standalone GGUF containing all four components. When set, the raw
    /// checkpoint directory is not read at all.
    pub standalone_gguf: Option<PathBuf>,
    /// Canonical `triposg-scribble` package directory. When set, all
    /// components come from `model.safetensors` (or a sibling `model.gguf`).
    pub canonical_dir: Option<PathBuf>,
    pub preprocess: TripoSgPreprocessStage,
    pub geometry: TripoSgGeometryStage,
    pub export: TripoSgExportStage,
    dtype: DType,
    bundle_cache: Arc<DeviceLocalCache<TripoSgScribbleModelBundle>>,
}

impl TripoSgScribblePipeline {
    pub fn load(model_dir: impl Into<PathBuf>) -> Result<Self> {
        Self::load_with_dit_gguf_and_dtype(model_dir, None, DType::F32)
    }

    pub fn load_with_dtype(model_dir: impl Into<PathBuf>, dtype: DType) -> Result<Self> {
        Self::load_with_dit_gguf_and_dtype(model_dir, None, dtype)
    }

    /// Load with an optional quantized DiT package. `dit_gguf` is an explicit
    /// override; `None` keeps the reference order: sibling
    /// `transformer/model.gguf`, then the raw safetensors shard.
    pub fn load_with_dit_gguf_and_dtype(
        model_dir: impl Into<PathBuf>,
        dit_gguf: Option<PathBuf>,
        dtype: DType,
    ) -> Result<Self> {
        let model_dir = model_dir.into();
        if let Some(path) = dit_gguf.as_ref() {
            if !path.is_file() {
                return Err(Lux3dError::MissingCanonicalArtifact { path: path.clone() });
            }
            if scribble_standalone_gguf(path).is_some() {
                return Ok(Self::from_standalone(path.clone(), dtype));
            }
        }
        if let Some(standalone) = scribble_standalone_gguf(&model_dir) {
            return Ok(Self::from_standalone(standalone, dtype));
        }
        if let Some(canonical) = scribble_canonical_package(&model_dir) {
            if dit_gguf.is_some() {
                return Err(Lux3dError::InvalidInput(
                    "an explicit TripoSG-scribble DiT GGUF cannot be combined with a \
                     canonical scribble package; put `model.gguf` inside the package or \
                     pass the raw checkpoint directory",
                ));
            }
            return Ok(Self::from_canonical(canonical, dtype));
        }
        if !model_dir.is_dir() {
            return Err(scribble_model_dir_missing_error());
        }
        // An explicit GGUF or the sibling `transformer/model.gguf` replaces the
        // float DiT shard; every other (small) file is still required.
        let require_dit_safetensors =
            dit_gguf.is_none() && scribble_sibling_dit_gguf(&model_dir).is_none();
        let resolved = resolve_scribble_model_dir_with(&model_dir, require_dit_safetensors)?;
        Ok(Self {
            model_dir: resolved,
            dit_gguf,
            standalone_gguf: None,
            canonical_dir: None,
            preprocess: TripoSgPreprocessStage::default(),
            geometry: TripoSgGeometryStage,
            export: TripoSgExportStage,
            dtype,
            bundle_cache: Arc::default(),
        })
    }

    fn from_standalone(gguf: PathBuf, dtype: DType) -> Self {
        let model_dir = gguf.parent().map(Path::to_path_buf).unwrap_or_default();
        Self {
            model_dir,
            dit_gguf: None,
            standalone_gguf: Some(gguf),
            canonical_dir: None,
            preprocess: TripoSgPreprocessStage::default(),
            geometry: TripoSgGeometryStage,
            export: TripoSgExportStage,
            dtype,
            bundle_cache: Arc::default(),
        }
    }

    fn from_canonical(dir: PathBuf, dtype: DType) -> Self {
        Self {
            model_dir: dir.clone(),
            dit_gguf: None,
            standalone_gguf: None,
            canonical_dir: Some(dir),
            preprocess: TripoSgPreprocessStage::default(),
            geometry: TripoSgGeometryStage,
            export: TripoSgExportStage,
            dtype,
            bundle_cache: Arc::default(),
        }
    }

    fn bundle_for(&self, device: &Device) -> Result<Arc<TripoSgScribbleModelBundle>> {
        self.bundle_cache.get_or_try_init(device, || {
            if let Some(dir) = self.canonical_dir.as_deref() {
                TripoSgScribbleModelBundle::load_canonical(dir, device, self.dtype)
            } else if let Some(gguf) = self.standalone_gguf.as_deref() {
                TripoSgScribbleModelBundle::load_standalone(gguf, device, self.dtype)
            } else {
                TripoSgScribbleModelBundle::load(
                    &self.model_dir,
                    self.dit_gguf.as_deref(),
                    device,
                    self.dtype,
                )
            }
        })
    }

    /// Official scribble preprocessing: `convert("RGB")` + the DINOv2
    /// `BitImageProcessor` only (`scripts/inference_triposg_scribble.py:22`,
    /// `pipeline_triposg_scribble.py:186-188`). Alpha/RMBG foreground
    /// handling belongs to the image-to-3D path and is not used here.
    pub fn prepare_inputs_from_path(
        &self,
        source: &Path,
        device: &Device,
    ) -> Result<TripoSgPreparedInputs> {
        self.preprocess
            .prepare_scribble_inputs_from_path(source, device)
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
        Self::sample_latents_batched(seed, 1, num_tokens, channels, device)
    }

    /// Batched variant of [`Self::sample_latents`]: ONE `[batch, num_tokens,
    /// channels]` draw from the Lux3D RNG, matching the reference
    /// `prepare_latents` shape (`pipeline_triposg.py:164-174`: a single
    /// `randn_tensor((batch_size * num_shapes_per_prompt, num_tokens,
    /// channels), generator=...)`). `batch = 1` is **bit-identical** to
    /// [`Self::sample_latents`] (same RNG sequence, same tensor shape), and
    /// for `batch > 1` sample `i` occupies rows `[i, ...]` in draw order —
    /// the documented RNG deviation still applies (values are
    /// distribution-equivalent, not torch-draw-identical).
    pub fn sample_latents_batched(
        seed: u64,
        batch: usize,
        num_tokens: usize,
        channels: usize,
        device: &Device,
    ) -> Result<Tensor> {
        if batch == 0 {
            return Err(Lux3dError::InvalidInput(
                "TripoSG latent batch must be at least 1, got 0",
            ));
        }
        if num_tokens == 0 || channels == 0 {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG latent shape ({num_tokens}, {channels}) must be positive on \
                     both axes"
                ),
            });
        }
        let mut rng = SeededNormalRng::new(seed);
        let mut values = Vec::with_capacity(batch * num_tokens * channels);
        for _ in 0..(batch * num_tokens * channels) {
            values.push(rng.next_standard_normal() as f32);
        }
        Tensor::from_vec(values, (batch, num_tokens, channels), &Device::Cpu)
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

    /// Stack the DINOv2 `pixel_values` of every source into `[N, 3, 224,
    /// 224]`: each source goes through the full per-image preprocess (alpha
    /// or RMBG foreground path), matching the reference script calling
    /// `prepare_image` per input before the pipeline consumes a batch
    /// (`scripts/inference_triposg.py:33` + `pipeline_triposg.py:203-210`).
    /// A single source returns its tensor unchanged.
    pub fn prepare_batch_from_paths(&self, sources: &[PathBuf], device: &Device) -> Result<Tensor> {
        if sources.is_empty() {
            return Err(Lux3dError::InvalidInput(
                "TripoSG batch inference requires at least one image source",
            ));
        }
        if sources.len() == 1 {
            return Ok(self
                .prepare_inputs_from_path(&sources[0], device)?
                .pixel_values);
        }
        let mut pixel_values = Vec::with_capacity(sources.len());
        for source in sources {
            pixel_values.push(self.prepare_inputs_from_path(source, device)?.pixel_values);
        }
        let refs: Vec<&Tensor> = pixel_values.iter().collect();
        Tensor::cat(&refs, 0).map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to stack TripoSG batch pixel values: {source}"),
        })
    }

    /// Batched inference over `N` image sources with `num_shapes` samples per
    /// image: total batch `B = N * num_shapes`.
    ///
    /// Mirrors the reference pipeline call shape
    /// (`pipeline_triposg.py:203-241`): ONE conditioning pass over the
    /// stacked `[N, 3, 224, 224]` pixel values, `repeat_interleave` on the
    /// batch axis by `num_shapes` (`:146`), one `[B, num_tokens, 64]` latent
    /// draw (`:164-174, :233-241`) and the batched DiT denoise loop
    /// (`:243-275`). The single-source + `num_shapes = 1` case delegates to
    /// [`Self::infer_from_path`], so that path is byte-identical to the
    /// pre-batch pipeline.
    pub fn infer_batch_from_paths(
        &self,
        sources: &[PathBuf],
        device: &Device,
        num_shapes: usize,
        options: &TripoSgRunOptions,
    ) -> Result<TripoSgInferenceOutput> {
        options.validate()?;
        if num_shapes == 0 {
            return Err(Lux3dError::InvalidInput(
                "TripoSG num_shapes must be at least 1, got 0",
            ));
        }
        if sources.is_empty() {
            return Err(Lux3dError::InvalidInput(
                "TripoSG batch inference requires at least one image source",
            ));
        }
        if sources.len() == 1 && num_shapes == 1 {
            return self.infer_from_path(&sources[0], device, options);
        }
        let pixel_values = self.prepare_batch_from_paths(sources, device)?;
        self.neural
            .infer_latents_batched(self, &pixel_values, num_shapes, options)
    }

    /// Denoise caller-supplied initial latents with classifier-free guidance.
    /// The latents are NOT regenerated here: `latents_init` must already be
    /// `[B, num_tokens, 64]` (`B = num_images * num_shapes`; `B = 1` is the
    /// historical single-sample contract), e.g. from [`Self::sample_latents`]
    /// / [`Self::sample_latents_batched`] or a parity baseline.
    /// `image_embeds` is the conditional DINOv2 context and must have the
    /// same batch dimension; the uncond branch is `zeros_like` inside the
    /// denoise loop.
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
        Self::validate_batched_latents(latents_init, options.num_tokens)?;
        let latent_batch =
            latents_init
                .dim(0)
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("TripoSG latents batch dim unavailable: {source}"),
                })?;
        let (cond_batch, _cond_seq, _cond_dim) =
            image_embeds
                .dims3()
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("TripoSG image embeds must be [B, S, D]: {source}"),
                })?;
        if cond_batch != latent_batch {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG conditioning batch {cond_batch} does not match latent batch \
                     {latent_batch} (expected num_images * num_shapes)"
                ),
            });
        }
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
    /// Inputs with fewer than [`TRIPOSG_SURFACE_POINT_COUNT`] points are
    /// rejected with an explicit error: the reference draws without
    /// replacement and raises for undersized inputs, so silently duplicating
    /// points would bias the encoded latent.
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
            surface_selection_indices(surface.len(), TRIPOSG_SURFACE_POINT_COUNT, options.seed)?;
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
        let mesh_cpu = self.decode_mesh(latents, options)?;
        triposg_mesh_buffers(&mesh_cpu, latents.device())
    }

    /// Batched decode for `latents [B, num_tokens, 64]` (`B = num_images *
    /// num_shapes`): one mesh per sample, returned in batch order (which is
    /// image-major after the `repeat_interleave` conditioning expansion).
    ///
    /// The reference decode helpers are effectively batch-1 (`flash_extract_
    /// geometry` squeezes the latent batch, `inference_utils.py:384-385`;
    /// the hierarchical path feeds one dense query grid through a batch-1 kv
    /// cache), so Lux3D loops the existing single-sample decode per row —
    /// exactly the same flash/DiffDMC and dense/hierarchical marching-cubes
    /// code, `--faces` postprocess included. Conditioning and the DiT stay
    /// batched.
    pub fn extract_meshes(
        &self,
        latents: &Tensor,
        options: &TripoSgRunOptions,
    ) -> Result<Vec<TripoSgMeshBuffers>> {
        options.validate()?;
        Self::validate_batched_latents(latents, options.num_tokens)?;
        let batch = latents
            .dim(0)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSG latents batch dim unavailable: {source}"),
            })?;
        let device = latents.device().clone();
        let mut meshes = Vec::with_capacity(batch);
        for index in 0..batch {
            let sample = latents.narrow(0, index, 1).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to slice TripoSG latent sample {index}: {source}"),
                }
            })?;
            let mesh_cpu = self.decode_mesh(&sample, options)?;
            meshes.push(triposg_mesh_buffers(&mesh_cpu, &device)?);
        }
        Ok(meshes)
    }

    /// Shared per-sample decode body (latents `[1, num_tokens, 64]` already
    /// validated/converted by the callers below): flash or hierarchical SDF
    /// grid -> mesh -> `--faces` postprocess.
    fn decode_mesh(&self, latents: &Tensor, options: &TripoSgRunOptions) -> Result<TripoSgMeshCpu> {
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
        Ok(mesh_cpu)
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

    /// `latents` must be `[1, num_tokens, 64]` (batch 1 — the single-sample
    /// VAE decode path).
    fn validate_latents(latents: &Tensor, num_tokens: usize) -> Result<()> {
        Self::validate_latents_shape(latents, num_tokens, Some(1))
    }

    /// `latents` must be `[B, num_tokens, 64]` with `B >= 1` (batched decode
    /// and batched denoise).
    fn validate_batched_latents(latents: &Tensor, num_tokens: usize) -> Result<()> {
        Self::validate_latents_shape(latents, num_tokens, None)
    }

    fn validate_latents_shape(
        latents: &Tensor,
        num_tokens: usize,
        expected_batch: Option<usize>,
    ) -> Result<()> {
        let dims = latents.dims();
        let batch_ok = match expected_batch {
            Some(expected) => dims.first() == Some(&expected),
            None => dims.first().is_some_and(|batch| *batch >= 1),
        };
        if dims.len() != 3
            || !batch_ok
            || dims[1] != num_tokens
            || dims[2] != TRIPOSG_LATENT_CHANNELS
        {
            let batch_description = match expected_batch {
                Some(expected) => format!("{expected}"),
                None => "B >= 1".to_string(),
            };
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG latents must be [{batch_description}, {num_tokens}, \
                     {TRIPOSG_LATENT_CHANNELS}], got {dims:?}"
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
        let first = surface_selection_indices(1_000, 64, 42).expect("enough points");
        let second = surface_selection_indices(1_000, 64, 42).expect("enough points");
        assert_eq!(first, second, "same seed must reproduce the subset");
        let other = surface_selection_indices(1_000, 64, 43).expect("enough points");
        assert_ne!(first, other, "different seeds must change the subset");
    }

    #[test]
    fn selection_is_without_replacement_when_enough_points() {
        let selected = surface_selection_indices(10_000, 4_096, 7).expect("enough points");
        assert_eq!(selected.len(), 4_096);
        let mut sorted = selected.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), selected.len(), "indices must be unique");
        assert!(selected.iter().all(|index| *index < 10_000));
    }

    #[test]
    fn selection_accepts_exactly_reference_count() {
        let selected = surface_selection_indices(4_096, 4_096, 7).expect("exact count");
        assert_eq!(selected.len(), 4_096);
        let mut sorted = selected;
        sorted.sort_unstable();
        assert_eq!(sorted, (0..4_096).collect::<Vec<_>>());
    }

    #[test]
    fn selection_rejects_small_inputs_instead_of_upsampling() {
        let error =
            surface_selection_indices(3, TRIPOSG_SURFACE_POINT_COUNT, 7).expect_err("small input");
        let message = error.to_string();
        assert!(
            message.contains("204800"),
            "error must name the reference point count: {message}"
        );
        assert!(
            message.contains("contains 3 points"),
            "error must name the actual point count: {message}"
        );
        assert!(
            message.contains("without replacement"),
            "error must explain the reference sampling contract: {message}"
        );
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;

    fn embeds(rows: &[[f32; 2]]) -> Tensor {
        let values = rows
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (rows.len(), 1, 2), &Device::Cpu).unwrap()
    }

    fn embed_values(tensor: &Tensor) -> Vec<f32> {
        tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    #[test]
    fn repeat_interleave_matches_reference_order() {
        // Reference: `image_embeds.repeat_interleave(num_shapes, dim=0)`
        // (pipeline_triposg.py:146) -> image-major, each image repeated.
        let input = embeds(&[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]);
        let expanded = repeat_interleave_batch(&input, 2).unwrap();
        assert_eq!(expanded.dims(), &[6, 1, 2]);
        assert_eq!(
            embed_values(&expanded),
            vec![0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0, 4.0, 5.0]
        );

        // Factor 1 is the identity (keeps the historical path byte-identical).
        let identity = repeat_interleave_batch(&input, 1).unwrap();
        assert_eq!(identity.dims(), input.dims());
        assert_eq!(embed_values(&identity), embed_values(&input));

        // Total batch is always N * num_shapes.
        for repeats in 1..=4 {
            let expanded = repeat_interleave_batch(&input, repeats).unwrap();
            assert_eq!(expanded.dims()[0], 3 * repeats);
        }
    }

    #[test]
    fn repeat_interleave_rejects_zero() {
        let input = embeds(&[[0.0, 1.0]]);
        let error = repeat_interleave_batch(&input, 0).unwrap_err();
        assert!(error.to_string().contains("repeats must be at least 1"));
    }

    #[test]
    fn batched_latents_have_one_draw_per_sample() {
        let device = Device::Cpu;
        let single = TripoSgPipeline::sample_latents(42, 8, 4, &device).unwrap();
        assert_eq!(single.dims(), &[1, 8, 4]);

        let batch = TripoSgPipeline::sample_latents_batched(42, 3, 8, 4, &device).unwrap();
        assert_eq!(batch.dims(), &[3, 8, 4]);

        // Batch = 1 is exactly the historical single-sample draw.
        let batch_one = TripoSgPipeline::sample_latents_batched(42, 1, 8, 4, &device).unwrap();
        assert_eq!(
            batch_one.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            single.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );

        // Sample 0 of a batch starts from the same RNG sequence as the
        // single-sample draw: the batch draw is one continuation, not a
        // per-row reseed.
        let first_row = batch
            .narrow(0, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(
            first_row,
            single.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );

        // Batch rows are independent draws (no accidental broadcast).
        let rows = batch
            .narrow(0, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let second = batch
            .narrow(0, 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_ne!(rows, second);
    }

    #[test]
    fn batched_latents_reject_empty_batch() {
        let error = TripoSgPipeline::sample_latents_batched(42, 0, 8, 4, &Device::Cpu).unwrap_err();
        assert!(error
            .to_string()
            .contains("latent batch must be at least 1"));
    }

    #[test]
    fn batched_latent_shape_validation() {
        let device = Device::Cpu;
        // [2, 8, 64] is the batched shape (`num_images * num_shapes`).
        let batched = Tensor::zeros((2, 8, TRIPOSG_LATENT_CHANNELS), DType::F32, &device).unwrap();
        TripoSgPipeline::validate_batched_latents(&batched, 8).unwrap();
        // The single-sample decode path still requires batch 1.
        TripoSgPipeline::validate_latents(&batched, 8).unwrap_err();
        TripoSgPipeline::validate_latents(&batched.narrow(0, 0, 1).unwrap(), 8).unwrap();

        // Wrong token count / channel count / rank are rejected on both.
        let wrong_tokens =
            Tensor::zeros((2, 7, TRIPOSG_LATENT_CHANNELS), DType::F32, &device).unwrap();
        TripoSgPipeline::validate_batched_latents(&wrong_tokens, 8).unwrap_err();
        let wrong_channels = Tensor::zeros((2, 8, 63), DType::F32, &device).unwrap();
        TripoSgPipeline::validate_batched_latents(&wrong_channels, 8).unwrap_err();
        let wrong_rank = Tensor::zeros((2, 8), DType::F32, &device).unwrap();
        TripoSgPipeline::validate_batched_latents(&wrong_rank, 8).unwrap_err();
        TripoSgPipeline::validate_latents(&wrong_rank, 8).unwrap_err();
    }
}

#[cfg(test)]
mod scribble_dit_loader_tests {
    use super::*;
    use candle_core::quantized::GgmlDType;
    use std::collections::HashMap;

    fn scratch_dir(tag: &str) -> PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "lux3d-scribble-dit-{tag}-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn tiny_transformer_checkpoint(dir: &Path) -> (PathBuf, PathBuf) {
        let transformer = dir.join("transformer");
        fs::create_dir_all(&transformer).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert(
            "proj_in.weight".to_string(),
            candle_core::Tensor::rand(-0.05f32, 0.05f32, (8, 32), &Device::Cpu).unwrap(),
        );
        let source = transformer.join("diffusion_pytorch_model.safetensors");
        candle_core::safetensors::save(&tensors, &source).unwrap();
        let gguf = transformer.join("model.gguf");
        crate::weights::quantize_checkpoint(
            &source,
            &gguf,
            ModelFamily::TripoSg,
            crate::weights::QuantizeOptions {
                dtype: GgmlDType::Q8_0,
                ..Default::default()
            },
        )
        .unwrap();
        (source, gguf)
    }

    #[test]
    fn dit_var_builder_prefers_sibling_gguf_then_safetensors() {
        let dir = scratch_dir("sibling");
        let (source, gguf) = tiny_transformer_checkpoint(&dir);

        let vb = scribble_dit_var_builder(&dir, None, DType::F32, &Device::Cpu).unwrap();
        let qt = vb
            .get_quantized("proj_in.weight")
            .unwrap()
            .expect("sibling model.gguf must supply the packed tensor");
        assert_eq!(qt.dtype(), GgmlDType::Q8_0);

        // Remove the GGUF: the raw safetensors shard is the fallback.
        fs::remove_file(&gguf).unwrap();
        let vb = scribble_dit_var_builder(&dir, None, DType::F32, &Device::Cpu).unwrap();
        assert!(
            vb.get_quantized("proj_in.weight").unwrap().is_none(),
            "safetensors fallback must be dense"
        );
        let weight = vb.get((8, 32), "proj_in.weight").unwrap();
        assert_eq!(weight.dims(), &[8, 32]);
        let _ = source;
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn dit_var_builder_explicit_gguf_overrides_sibling() {
        let dir = scratch_dir("explicit");
        let (_source, gguf) = tiny_transformer_checkpoint(&dir);
        let explicit = dir.join("explicit-q8_0.gguf");
        fs::copy(&gguf, &explicit).unwrap();

        let vb = scribble_dit_var_builder(&dir, Some(&explicit), DType::F32, &Device::Cpu).unwrap();
        assert!(
            vb.get_quantized("proj_in.weight").unwrap().is_some(),
            "explicit DiT GGUF must be selected"
        );

        let missing = dir.join("missing.gguf");
        let err = match scribble_dit_var_builder(&dir, Some(&missing), DType::F32, &Device::Cpu) {
            Ok(_) => panic!("missing explicit GGUF must fail"),
            Err(error) => error,
        };
        assert!(err.to_string().contains("missing.gguf"));
        let _ = fs::remove_dir_all(&dir);
    }

    fn tiny_standalone_scribble_gguf(dir: &Path) -> PathBuf {
        let path = dir.join("model.gguf");
        let mut file = fs::File::create(&path).unwrap();
        let metadata = [
            (
                "general.architecture",
                gguf_file::Value::String("triposg".to_string()),
            ),
            (
                "lux3d.variant",
                gguf_file::Value::String(TRIPOSG_SCRIBBLE_VARIANT.to_string()),
            ),
        ];
        let metadata_refs: Vec<(&str, &gguf_file::Value)> = metadata
            .iter()
            .map(|(name, value)| (*name, value))
            .collect();
        let tensors: Vec<(&str, &candle_core::quantized::QTensor)> = Vec::new();
        gguf_file::write(&mut file, &metadata_refs, &tensors).unwrap();
        path
    }

    #[test]
    fn standalone_detection_accepts_variant_metadata_file_and_package_dir() {
        let dir = scratch_dir("standalone-detect");
        let gguf = tiny_standalone_scribble_gguf(&dir);
        assert_eq!(scribble_standalone_gguf(&gguf), Some(gguf.clone()));

        let package = dir.join("package");
        fs::create_dir_all(&package).unwrap();
        fs::copy(&gguf, package.join("model.gguf")).unwrap();
        assert_eq!(
            scribble_standalone_gguf(&package),
            Some(package.join("model.gguf"))
        );

        let (resolved, dit) = resolve_triposg_scribble_model_source(Some(&gguf)).unwrap();
        assert_eq!(resolved, gguf);
        assert!(dit.is_none());

        // A raw checkpoint keeps its DiT-override meaning even when a root
        // `model.gguf` exists.
        fs::create_dir_all(package.join("transformer")).unwrap();
        fs::write(package.join("transformer").join("config.json"), b"{}").unwrap();
        assert_eq!(scribble_standalone_gguf(&package), None);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn standalone_package_loads_all_components_when_available() {
        let path = std::env::var_os("LUX3D_TRIPOSG_SCRIBBLE_STANDALONE")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                crate::test_support::runtime_root()
                    .join("candle-3d")
                    .join("out")
                    .join("scribble-standalone")
                    .join("triposg-scribble-q8_0.gguf")
            });
        if !path.is_file() {
            eprintln!(
                "skipping standalone scribble load: no package at {}",
                path.display()
            );
            return;
        }

        let pipeline = TripoSgScribblePipeline::load_with_dtype(path.clone(), DType::F32)
            .expect("standalone scribble pipeline load");
        assert_eq!(pipeline.standalone_gguf.as_deref(), Some(path.as_path()));
        assert!(pipeline.dit_gguf.is_none());

        // Force the lazy bundle load: this constructs DINOv2, the DiT, the VAE
        // and the CLIP text encoder from the single GGUF and parses the
        // embedded tokenizer. It would fail on any missing component prefix.
        let bundle = pipeline
            .bundle_for(&Device::Cpu)
            .expect("load all standalone components");
        let ids = bundle.tokenizer.encode("a photo of a cat");
        assert_eq!(ids.len(), 77);
        assert_eq!(ids[0], 49406);
        assert_eq!(ids[6], 49407);
    }

    #[test]
    fn canonical_scribble_package_loads_all_components_when_available() {
        let dir = std::env::var_os("LUX3D_TRIPOSG_SCRIBBLE_CANONICAL")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                crate::test_support::runtime_root()
                    .join("3d")
                    .join("canonical-weights")
                    .join("triposg-scribble")
            });
        if !dir.is_dir() {
            eprintln!(
                "skipping canonical scribble load: no package at {}",
                dir.display()
            );
            return;
        }

        let (resolved, dit) = resolve_triposg_scribble_model_source(Some(&dir))
            .expect("resolve canonical scribble package");
        assert_eq!(resolved, dir);
        assert!(dit.is_none());

        let pipeline = TripoSgScribblePipeline::load_with_dtype(dir.clone(), DType::F16)
            .expect("canonical scribble pipeline load");
        assert_eq!(pipeline.canonical_dir.as_deref(), Some(dir.as_path()));
        assert!(pipeline.standalone_gguf.is_none());
        assert!(pipeline.dit_gguf.is_none());

        let bundle = pipeline
            .bundle_for(&Device::Cpu)
            .expect("load all canonical components");
        let ids = bundle.tokenizer.encode("a photo of a cat");
        assert_eq!(ids.len(), 77);
        assert_eq!(ids[0], 49406);
        assert_eq!(ids[6], 49407);
    }

    /// Create every required scribble file except the float DiT shard.
    fn placeholder_scribble_checkpoint_without_dit(dir: &Path) {
        for relative in TRIPOSG_SCRIBBLE_REQUIRED_FILES {
            if relative == TRIPOSG_SCRIBBLE_DIT_SHARD {
                continue;
            }
            let path = dir.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, b"{}").unwrap();
        }
    }

    #[test]
    fn scribble_pipeline_loads_with_sibling_gguf_without_dit_safetensors() {
        let dir = scratch_dir("gguf-sibling");
        placeholder_scribble_checkpoint_without_dit(&dir);
        let (source, _gguf) = tiny_transformer_checkpoint(&dir);
        // The quantized sibling replaces the float shard: remove it and show
        // that the checkpoint still resolves and the pipeline still loads.
        fs::remove_file(&source).unwrap();

        let resolved = resolve_triposg_scribble_model_dir(Some(&dir)).unwrap();
        assert_eq!(resolved, dir);
        let pipeline = TripoSgScribblePipeline::load(&dir).unwrap();
        assert_eq!(pipeline.model_dir, dir);
        assert!(
            pipeline.dit_gguf.is_none(),
            "sibling GGUF is resolved lazily by the bundle, not as an explicit override"
        );

        // Without the sibling GGUF the float DiT shard is required again.
        fs::remove_file(dir.join(TRIPOSG_SCRIBBLE_DIT_GGUF)).unwrap();
        let error = TripoSgScribblePipeline::load(&dir).unwrap_err();
        assert!(
            error.to_string().contains(TRIPOSG_SCRIBBLE_DIT_SHARD),
            "missing DiT shard must be named when no GGUF is present: {error}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn scribble_pipeline_loads_with_explicit_gguf_without_dit_safetensors() {
        let dir = scratch_dir("gguf-explicit-standalone");
        placeholder_scribble_checkpoint_without_dit(&dir);
        let (source, gguf) = tiny_transformer_checkpoint(&dir);
        let explicit = dir.join("explicit-q8_0.gguf");
        fs::copy(&gguf, &explicit).unwrap();
        fs::remove_file(&source).unwrap();
        fs::remove_file(&gguf).unwrap();

        let pipeline = TripoSgScribblePipeline::load_with_dit_gguf_and_dtype(
            &dir,
            Some(explicit.clone()),
            DType::F32,
        )
        .unwrap();
        assert_eq!(pipeline.dit_gguf.as_deref(), Some(explicit.as_path()));

        // Every other small weight/config file is still enforced.
        fs::remove_file(dir.join("tokenizer/vocab.json")).unwrap();
        let error =
            TripoSgScribblePipeline::load_with_dit_gguf_and_dtype(&dir, Some(explicit), DType::F32)
                .unwrap_err();
        assert!(
            error.to_string().contains("vocab.json"),
            "non-DiT required files must still be validated: {error}"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
