use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand};
use lux3d_core::export::{
    load_mesh, write_feedforward_colmap, VggtColmapOptions, DEFAULT_CONF_THRESHOLD,
    DEFAULT_MAX_POINTS,
};
use lux3d_core::geometry::VggtPointCloudCpu;
use lux3d_core::preprocess::VggtPreprocessMode;
use lux3d_core::runtime::{
    Pi3Pipeline, Pi3xInjectConditions, Pi3xPipeline, Pi3xVoPipeline,
    Trellis2MeshPostprocessOptions, Trellis2Pipeline, Trellis2RunOptions, Trellis2SamplerOverrides,
    Trellis2TextureExportOptions, Trellis2TexturingOptions, TripoSgPipeline, TripoSgRunOptions,
    TripoSrPipeline, VggtInferenceOutput, VggtPipeline, VggtRunOptions, VggtTrackQuery,
    VggtTrackResult,
};
use lux3d_core::{
    dtype_from_name, ensure_canonical_weights, quantize_auxiliary_checkpoint, quantize_checkpoint,
    ModelAssetOptions, ModelFamily, QuantizeOptions, Trellis2PipelineType, WeightLocator,
};

#[derive(Debug, Parser, PartialEq)]
#[command(
    name = "lux3d",
    about = "CUDA-first runtime and weight tools for Pi3, Pi3X, TripoSR, VGGT, TripoSG, and TRELLIS.2."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand, PartialEq)]
pub enum Command {
    Inspect(InspectArgs),
    Weights(WeightsArgs),
    Run(RunArgs),
}

#[derive(Debug, Clone, Args, PartialEq, Eq)]
pub struct InspectArgs {
    #[arg(long)]
    pub repo_root: PathBuf,

    #[arg(value_enum)]
    pub family: Family,
}

#[derive(Debug, Clone, Args, PartialEq, Eq)]
pub struct WeightsArgs {
    #[command(subcommand)]
    pub command: WeightsCommand,
}

#[derive(Debug, Clone, Subcommand, PartialEq, Eq)]
pub enum WeightsCommand {
    Normalize(NormalizeArgs),
    Quantize(QuantizeArgs),
}

#[derive(Debug, Clone, Args, PartialEq, Eq)]
pub struct QuantizeArgs {
    #[arg(
        value_enum,
        required_unless_present = "aux_family",
        help = "Output model family (triposg, vggt, ...); omit when using `--aux-family`."
    )]
    pub family: Option<Family>,

    #[arg(
        long,
        conflicts_with = "family",
        help = "Auxiliary package family label, e.g. `rmbg` or `rmbg2`; makes FAMILY optional."
    )]
    pub aux_family: Option<String>,

    #[arg(long, help = "Input canonical safetensors (`model.safetensors`).")]
    pub source: PathBuf,

    #[arg(
        long,
        help = "Output GGUF path; use the package's `model.gguf` name for auto-detection."
    )]
    pub output: PathBuf,

    #[arg(
        long,
        default_value = "q8_0",
        help = "Quantization dtype for 2D linear weights (f16, q8_0, q4_0, q4_k, q5_k, q6_k, ...)."
    )]
    pub dtype: String,

    #[arg(
        long,
        default_value = "f16",
        help = "Dtype for norms, biases, embeddings, convolutions, and non-divisible weights."
    )]
    pub fallback_dtype: String,

    #[arg(
        long,
        help = "Quantize every rank-2 tensor instead of only names ending in `.weight`."
    )]
    pub all_2d: bool,

    #[arg(
        long,
        help = "Only quantize tensors whose name starts with this prefix (repeatable)."
    )]
    pub prefix: Vec<String>,
}

#[derive(Debug, Clone, Args, PartialEq, Eq)]
pub struct NormalizeArgs {
    #[arg(long)]
    pub repo_root: PathBuf,

    #[arg(
        long,
        help = "Path to the upstream raw model directory used only for maintainer-side canonicalization."
    )]
    pub raw_model_dir: PathBuf,

    #[arg(long)]
    pub output_dir: PathBuf,

    #[arg(value_enum)]
    pub family: Family,
}

#[derive(Debug, Clone, Args, PartialEq)]
pub struct RunArgs {
    #[arg(value_enum)]
    pub family: Family,

    #[arg(long, value_enum, default_value = "cuda")]
    pub device: DeviceBackend,

    #[arg(
        long,
        required_unless_present = "surface_input",
        help = "Input source: a frame directory (.png/.jpg/.jpeg), a video file (.mp4/.mov/.avi/.mkv), or - for `vggt` - a single image file (.png/.jpg/.jpeg/.webp/.bmp/.tif/.tiff). Mutually exclusive with `--surface-input`."
    )]
    pub source: Option<PathBuf>,

    #[arg(long)]
    pub interval: Option<usize>,

    /// VGGT-only run flags; absent flags keep the historical behavior
    /// (F32 weights, crop preprocessing, one forward pass, PLY output only).
    #[command(flatten)]
    pub vggt: VggtRunArgs,

    #[arg(long)]
    pub conditions: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    pub vo: bool,

    #[arg(
        long,
        help = "Pi3X VO chunk size; for `triposg` this doubles as the SDF query chunk size (default 16384)."
    )]
    pub chunk_size: Option<usize>,

    #[arg(long)]
    pub overlap: Option<usize>,

    #[arg(long)]
    pub conf_threshold: Option<f32>,

    #[arg(long, value_delimiter = ',')]
    pub inject_condition: Vec<InjectCondition>,

    #[arg(long)]
    pub mc_resolution: Option<u32>,

    #[arg(long)]
    pub mc_threshold: Option<f32>,

    #[arg(
        long,
        help = "Denoise steps (TripoSG rectified-flow default 50; TRELLIS.2 sparse-structure/shape/texture Euler default 12 each)."
    )]
    pub steps: Option<usize>,

    #[arg(long, help = "TripoSG classifier-free guidance scale (default 7.0).")]
    pub guidance_scale: Option<f32>,

    #[arg(
        long,
        help = "TripoSG/TRELLIS.2 latent-initialization seed (default 42)."
    )]
    pub seed: Option<u64>,

    #[arg(long, help = "TripoSG latent token count (default 2048).")]
    pub num_tokens: Option<usize>,

    #[arg(
        long,
        help = "TripoSG dense octree depth of the hierarchical SDF grid (default 8)."
    )]
    pub dense_octree_depth: Option<u32>,

    #[arg(long, help = "TripoSG final hierarchical octree depth (default 9).")]
    pub hierarchical_octree_depth: Option<u32>,

    #[arg(
        long,
        num_args = 0..=1,
        default_missing_value = "true",
        help = "TripoSG flash decoder + DiffDMC path (reference default: true). Pass --flash-decoder false to use dense/hierarchical marching cubes."
    )]
    pub flash_decoder: Option<bool>,

    #[arg(
        long,
        help = "TripoSG flash octree depth used by the reference default flash path (default 9)."
    )]
    pub flash_octree_depth: Option<u32>,

    #[arg(
        long,
        help = "TripoSG target face count after mesh extraction (upstream `--faces`): the postprocess runs `meshing_merge_close_vertices` + quadric decimation ONLY when the extracted mesh has MORE faces than the target; a target at/above the current face count is a no-op. 0 disables simplification (the upstream `-1` equivalent)."
    )]
    pub faces: Option<usize>,

    #[arg(
        long,
        value_enum,
        help = "TripoSG weight/activation dtype for the DINOv2/DiT/VAE pipeline load (default f32 = the reference mode; f16 halves weight memory on CUDA/Vulkan/WGPU and is also supported on CPU; public outputs stay f32)."
    )]
    pub dtype: Option<DTypeKind>,

    #[arg(
        long,
        allow_hyphen_values = true,
        help = "TripoSG SDF query bounds override: either a single scalar `b` (symmetric cube [-b, b], upstream float shorthand) or `min_x,min_y,min_z,max_x,max_y,max_z` in the normalized model frame (default -1.005,-1.005,-1.005,1.005,1.005,1.005). The override moves the decode window without changing the absolute frame of the returned mesh."
    )]
    pub bounds: Option<String>,

    #[arg(
        long,
        allow_hyphen_values = true,
        help = "TripoSG explicit timestep schedule override as comma-separated floats (e.g. 999,777,500,250,0): the denoise loop runs EXACTLY these timesteps and `--steps` is ignored (upstream `retrieve_timesteps` derives the step count from the list length)."
    )]
    pub timesteps: Option<String>,

    #[arg(
        long,
        conflicts_with = "source",
        help = "TripoSG VAE surface encode input (reference `scripts/inference_vae.py`): a `.npy` holding the pickled dict written by `np.save(path, {\"surface_points\": [N, 3], \"surface_normals\": [N, 3]})`. Skips image-to-3D/DINO/denoise, encodes the sampled 204800-point subset, and decodes the latent with the reference hierarchical extractor (flash decoder off, dense depth 8, hierarchical depth 9 unless overridden). Mutually exclusive with `--source`."
    )]
    pub surface_input: Option<PathBuf>,

    #[arg(
        long,
        value_enum,
        help = "TRELLIS.2 pipeline type: `512`, `1024`, `1024_cascade` (reference default), or `1536_cascade`."
    )]
    pub pipeline_type: Option<Trellis2PipelineKind>,

    #[arg(
        long,
        help = "TRELLIS.2 cascade token budget; the reference loop stops at 1024 (default 49152). The 1536 cascade can reduce 1536 -> ... -> 1024."
    )]
    pub max_num_tokens: Option<usize>,

    #[arg(
        long,
        num_args = 0..=1,
        default_missing_value = "true",
        help = "TRELLIS.2 texture stages: texture SLAT + guided VAE + PBR vertex colors (reference default: true). Pass --texture false for geometry-only output."
    )]
    pub texture: Option<bool>,

    #[arg(
        long,
        help = "TRELLIS.2 textured-GLB bake edge length in texels (default 2048). Accepts any value in 64..=8192; the core additionally requires multiples of 8 (nvdiffrast parity) and the upstream reference allows only 1024..4096 in steps of 1024. Applies to .glb outputs when the texture stages ran."
    )]
    pub texture_size: Option<usize>,

    #[arg(
        long,
        default_value_t = false,
        help = "TRELLIS.2 disable the textured-GLB export stage (upstream to_glb UV unwrap + PBR bake; default: enabled when --texture is on). The run falls back to the vertex-color mesh + PBR sidecar pair."
    )]
    pub no_texture_export: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "TRELLIS.2 skip the xatlas UV unwrap (texture baking needs UVs, so the textured mesh is skipped entirely); the run falls back to the vertex-color mesh + PBR sidecar pair."
    )]
    pub no_uv_unwrap: bool,

    #[arg(
        long,
        help = "TRELLIS.2 texturing mode: take this existing mesh file (.glb/.obj/.ply/.stl) instead of running image-to-3D, and texture it from --source (upstream Trellis2TexturingPipeline). Requires a .glb output."
    )]
    pub texture_mesh: Option<PathBuf>,

    #[arg(
        long,
        help = "TRELLIS.2 texturing voxelization/encode resolution (512 or 1024; default 1024). Selects tex_dit_512 vs tex_dit_1024 and the matching DINOv3 conditioning resolution."
    )]
    pub texture_resolution: Option<u32>,

    #[arg(
        long,
        default_value_t = false,
        help = "TRELLIS.2 texturing: skip the reference mesh preprocessing (center/scale to [-0.5, 0.5]^3 and y/z axis swap); the input mesh must already be inside that frame."
    )]
    pub no_preprocess_mesh: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "TRELLIS.2 texturing: use the condition image as-is (no foreground crop/RMBG); it must be square and is resized to the conditioning resolution."
    )]
    pub no_preprocess_image: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "TRELLIS.2 disable the mesh post-process (upstream to_glb cleaning: hole filling, decimation, small-component removal, orientation unification; default: enabled)."
    )]
    pub no_postprocess: bool,

    #[arg(
        long,
        help = "TRELLIS.2 post-process decimation target in triangles for both simplify passes (default 1000000; 0 skips simplification and runs cleanup only)."
    )]
    pub decimation_target: Option<usize>,

    #[arg(
        long,
        help = "TRELLIS.2 post-process max hole perimeter (default 3e-2; 0 disables hole filling)."
    )]
    pub fill_hole_perimeter: Option<f32>,

    #[arg(
        long,
        help = "TRELLIS.2 post-process min connected-component triangle area (default 1e-5; 0 keeps every component)."
    )]
    pub min_component_area: Option<f32>,

    #[arg(
        long,
        help = "TRELLIS.2 sparse-structure Euler steps (overrides --steps for the SS stage only)."
    )]
    pub ss_steps: Option<usize>,

    #[arg(
        long,
        help = "TRELLIS.2 shape SLAT Euler steps (overrides --steps for both cascade shape stages)."
    )]
    pub shape_steps: Option<usize>,

    #[arg(
        long,
        help = "TRELLIS.2 texture SLAT Euler steps (overrides --steps for the texture stage only)."
    )]
    pub tex_steps: Option<usize>,

    #[arg(
        long,
        help = "TRELLIS.2 sparse-structure CFG guidance strength (pipeline.json default 7.5)."
    )]
    pub ss_guidance: Option<f32>,

    #[arg(
        long,
        help = "TRELLIS.2 shape SLAT CFG guidance strength (pipeline.json default 7.5)."
    )]
    pub shape_guidance: Option<f32>,

    #[arg(
        long,
        help = "TRELLIS.2 texture SLAT CFG guidance strength (pipeline.json default 1.0 = cond-only)."
    )]
    pub tex_guidance: Option<f32>,

    #[arg(
        long,
        help = "TRELLIS.2 sparse-structure CFG rescale (pipeline.json default 0.7; 0 disables)."
    )]
    pub ss_rescale: Option<f32>,

    #[arg(
        long,
        help = "TRELLIS.2 shape SLAT CFG rescale (pipeline.json default 0.5; 0 disables)."
    )]
    pub shape_rescale: Option<f32>,

    #[arg(
        long,
        help = "TRELLIS.2 texture SLAT CFG rescale (pipeline.json default 0.0; 0 disables)."
    )]
    pub tex_rescale: Option<f32>,

    #[arg(
        long,
        help = "TRELLIS.2 sparse-structure guidance interval `lo,hi` (t range where CFG applies; default 0.6,1.0)."
    )]
    pub ss_interval: Option<String>,

    #[arg(
        long,
        help = "TRELLIS.2 shape SLAT guidance interval `lo,hi` (default 0.6,1.0)."
    )]
    pub shape_interval: Option<String>,

    #[arg(
        long,
        help = "TRELLIS.2 texture SLAT guidance interval `lo,hi` (default 0.6,0.9)."
    )]
    pub tex_interval: Option<String>,

    #[arg(
        long,
        help = "TRELLIS.2 sample count (default 1). Values > 1 run the deterministic stages sequentially with per-sample seed offset (seed + i) over the same conditioning — NOT batched inference — and write one output per sample with a `_0`, `_1`, ... suffix."
    )]
    pub num_samples: Option<usize>,

    #[arg(
        long,
        help = "TRELLIS.2 `return_latent` surface: write the final shape/texture latents to this `.npz` path (shape_latent/tex_latent/coords/resolution/lr_shape_latent/meta). With --num-samples > 1 one file per sample gets the same `_i` suffix as the mesh output."
    )]
    pub return_latent: Option<PathBuf>,

    #[arg(
        long,
        help = "Path to a canonical package directory, a directory containing `model.gguf`, or a standalone `.gguf` file."
    )]
    pub model_path: Option<PathBuf>,

    #[arg(
        long,
        help = "Path to the separate RMBG-1.4 package directory or its `.gguf` file (required for inputs without a valid alpha channel)."
    )]
    pub rmbg_model_path: Option<PathBuf>,

    #[arg(
        long,
        help = "Override the Hugging Face cache root used when --model-path is not supplied."
    )]
    pub cache_dir: Option<PathBuf>,

    #[arg(
        long,
        help = "Output artifact path. Mesh families dispatch on extension: `.glb` (default), `.obj`, `.ply`, `.stl`. Point clouds use `.ply` (default) or `.xyz`."
    )]
    pub output: PathBuf,

    #[arg(
        long,
        help = "Dump raw inference tensors (.npz) for quantization-fidelity metrics (vggt only)."
    )]
    pub dump_npz: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum Family {
    Pi3,
    Pi3x,
    Triposr,
    Triposg,
    Vggt,
    /// TRELLIS.2-4B image-to-3D: preprocess -> DINOv3 -> SS flow -> shape SLAT
    /// (512 or 1024 cascade) -> texture SLAT -> dual-grid mesh + PBR sidecar.
    Trellis2,
}

/// TRELLIS.2 pipeline variant (`--pipeline-type`); names mirror the reference
/// config strings (`512`, `1024`, `1024_cascade`, `1536_cascade`).
#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum Trellis2PipelineKind {
    #[value(name = "512")]
    R512,
    #[value(name = "1024")]
    R1024,
    #[value(name = "1024_cascade")]
    Cascade1024,
    #[value(name = "1536_cascade")]
    Cascade1536,
}

impl From<Trellis2PipelineKind> for Trellis2PipelineType {
    fn from(kind: Trellis2PipelineKind) -> Self {
        match kind {
            Trellis2PipelineKind::R512 => Self::R512,
            Trellis2PipelineKind::R1024 => Self::R1024,
            Trellis2PipelineKind::Cascade1024 => Self::Cascade1024,
            Trellis2PipelineKind::Cascade1536 => Self::Cascade1536,
        }
    }
}

/// Neural-network dtype for the TripoSG pipeline load (`--dtype`); names
/// mirror the `backend_parity_dump` example (`f32`, `f16`, `bf16`).
#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum DTypeKind {
    F32,
    F16,
    Bf16,
}

impl From<DTypeKind> for candle_core::DType {
    fn from(kind: DTypeKind) -> Self {
        match kind {
            DTypeKind::F32 => Self::F32,
            DTypeKind::F16 => Self::F16,
            DTypeKind::Bf16 => Self::BF16,
        }
    }
}

/// VGGT input preprocessing mode (`--preprocess-mode`); names mirror the
/// upstream `load_and_preprocess_images` `mode` strings.
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum, PartialEq, Eq)]
pub enum VggtPreprocessKind {
    /// Upstream default: resize to width 518, center-crop rows taller than
    /// 518, then white-pad to the batch maximum.
    #[default]
    Crop,
    /// Lossless: resize so the largest side is 518 (aspect preserved) and
    /// center-pad to a white 518x518 square; no pixels are discarded.
    Pad,
}

impl From<VggtPreprocessKind> for VggtPreprocessMode {
    fn from(kind: VggtPreprocessKind) -> Self {
        match kind {
            VggtPreprocessKind::Crop => Self::Crop,
            VggtPreprocessKind::Pad => Self::Pad,
        }
    }
}

/// VGGT COLMAP 3D-point source (`--colmap-point-source`).
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum, PartialEq, Eq)]
pub enum VggtColmapPointSourceKind {
    /// Depth-unprojected points (`unproject_depth_map_to_point_map`), the
    /// upstream `demo_colmap.py` feedforward source.
    #[default]
    Depth,
    /// Point-head `world_points`.
    Pointmap,
}

impl From<VggtColmapPointSourceKind> for lux3d_core::export::colmap::VggtColmapPointSource {
    fn from(kind: VggtColmapPointSourceKind) -> Self {
        match kind {
            VggtColmapPointSourceKind::Depth => Self::Depth,
            VggtColmapPointSourceKind::Pointmap => Self::Pointmap,
        }
    }
}

/// VGGT-only run flags (flattened into [`RunArgs`]). Every flag is rejected
/// for the other families; the defaults keep the historical `vggt` behavior:
/// F32 weights, crop preprocessing, one forward pass over all frames, and a
/// single confidence-filtered point-cloud output.
#[derive(Debug, Clone, Args, PartialEq)]
pub struct VggtRunArgs {
    #[arg(
        long,
        value_enum,
        default_value_t = VggtPreprocessKind::Crop,
        help = "VGGT input preprocessing (vggt only): `crop` (upstream default: width 518 + center crop, then batch white pad) or `pad` (largest side 518, center-padded white 518x518 square, no pixels discarded)."
    )]
    pub preprocess_mode: VggtPreprocessKind,

    #[arg(
        long,
        value_enum,
        help = "VGGT weight/activation dtype (vggt only; default f32 = the reference mode; f16 halves weight memory; public outputs stay f32). Separate from the TripoSG `--dtype` flag."
    )]
    pub vggt_dtype: Option<DTypeKind>,

    #[arg(
        long,
        default_value_t = 1,
        help = "VGGT frames per forward pass (vggt only, default 1; 0 rejected): the frame list is split into SEQUENTIAL chunks of this size and each chunk runs as its own B=1 pass. This bounds peak memory; it is NOT one attention pass over all frames - cross-frame attention only sees frames inside their chunk, so chunked results differ from a single full-scene pass whenever the scene spans multiple chunks. A value >= the frame count is a plain single pass."
    )]
    pub batch_size: usize,

    #[arg(
        long,
        help = "VGGT COLMAP feedforward export directory (vggt only): writes cameras.bin/images.bin/points3D.bin plus the sibling points.ply visualization cloud (upstream demo_colmap.py feedforward mode: one PINHOLE camera per image, depth-confidence-gated depth-unprojected points, intrinsics rescaled to the original resolution)."
    )]
    pub colmap_dir: Option<PathBuf>,

    #[arg(
        long,
        default_value_t = DEFAULT_CONF_THRESHOLD,
        help = "VGGT COLMAP depth-confidence gate `depth_conf >= threshold` (upstream demo default 5.0). Modifies --colmap-dir."
    )]
    pub colmap_conf_threshold: f32,

    #[arg(
        long,
        default_value_t = DEFAULT_MAX_POINTS,
        help = "VGGT COLMAP 3D-point cap (upstream demo default 100000). Without --colmap-seed the cap keeps a deterministic evenly-spaced subsample; with a seed it samples randomly (upstream `randomly_limit_trues`). Modifies --colmap-dir."
    )]
    pub colmap_max_points: usize,

    #[arg(
        long,
        value_enum,
        default_value_t = VggtColmapPointSourceKind::Depth,
        help = "VGGT COLMAP 3D-point source (vggt only, upstream demo default `depth`): `depth` unprojects the depth maps (`unproject_depth_map_to_point_map`, official feedforward semantics) or `pointmap` uses the point-head world_points. Modifies --colmap-dir."
    )]
    pub colmap_point_source: VggtColmapPointSourceKind,

    #[arg(
        long,
        help = "VGGT COLMAP random seed for point limiting (vggt only, upstream demo `--seed` 42): when set, up to --colmap-max-points points are drawn randomly from the confidence candidates; without it the deterministic evenly-spaced subsample is kept. The draw stream is distribution-equivalent to numpy's, not bit-identical. Modifies --colmap-dir."
    )]
    pub colmap_seed: Option<u32>,

    #[arg(
        long,
        help = "VGGT COLMAP extra `world_points_conf >= threshold` gate on top of --colmap-conf-threshold (vggt only; unset = no extra gate). Modifies --colmap-dir."
    )]
    pub colmap_world_conf_threshold: Option<f32>,

    #[arg(
        long,
        help = "VGGT depth-unprojection point cloud path (vggt only): unprojects the depth maps to world coordinates (reference `unproject_depth_map_to_point_map`) and writes a .ply/.xyz point cloud gated by `depth_conf > --depth-conf-threshold`."
    )]
    pub depth_points: Option<PathBuf>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "VGGT depth point-cloud confidence gate `depth_conf > threshold` (strict, default 1.0). Modifies --depth-points."
    )]
    pub depth_conf_threshold: f32,

    #[arg(
        long,
        allow_hyphen_values = true,
        help = "VGGT explicit track queries as a JSON array (vggt only) of {\"frame\": N, \"x\": F, \"y\": F} objects or [N, F, F] arrays: pixel coordinates in the model input resolution, `x` in [0, 518) and `y` in [0, 518) (prepared frames are at most 518x518) and the query frame must be 0 (rotate the clip so the desired query frame comes first). Automatic query selection (ALIKED/SP keypoints, DINO query-frame ranking) is NOT available - explicit queries are required."
    )]
    pub track_queries: Option<String>,

    #[arg(
        long,
        help = "VGGT track output JSON path (vggt only; default `<source-stem>_tracks.json` when --track-queries is given): writes {\"coords\": [[[x, y], ...], ...], \"visibility\": [[...], ...], \"confidence\": [[...], ...] | null} per frame per query."
    )]
    pub track_output: Option<PathBuf>,

    #[arg(
        long,
        default_value_t = 4,
        help = "VGGT track refinement iterations (upstream TrackHead.iters default 4). Modifies --track-queries."
    )]
    pub track_iters: usize,

    #[arg(
        long,
        help = "VGGT query-point PLY path (vggt only, requires --track-queries): writes one colored vertex per query - the frame-0 world point sampled at the query pixel (nearest) with its RGB color."
    )]
    pub track_points_ply: Option<PathBuf>,
}

impl Default for VggtRunArgs {
    fn default() -> Self {
        Self {
            preprocess_mode: VggtPreprocessKind::Crop,
            vggt_dtype: None,
            batch_size: 1,
            colmap_dir: None,
            colmap_conf_threshold: DEFAULT_CONF_THRESHOLD,
            colmap_max_points: DEFAULT_MAX_POINTS,
            colmap_point_source: VggtColmapPointSourceKind::Depth,
            colmap_seed: None,
            colmap_world_conf_threshold: None,
            depth_points: None,
            depth_conf_threshold: 1.0,
            track_queries: None,
            track_output: None,
            track_iters: 4,
            track_points_ply: None,
        }
    }
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum InjectCondition {
    Pose,
    Depth,
    Ray,
    Intrinsic,
}

/// Backend device used to run inference.
#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum DeviceBackend {
    Cpu,
    Cuda,
    Wgpu,
    Vulkan,
}

impl DeviceBackend {
    /// Resolve the backend into a candle-core device.
    pub fn to_device(self) -> anyhow::Result<candle_core::Device> {
        match self {
            DeviceBackend::Cpu => Ok(candle_core::Device::Cpu),
            DeviceBackend::Cuda => Ok(candle_core::Device::new_cuda(0)?),
            DeviceBackend::Wgpu => Ok(candle_core::Device::new_wgpu(0)?),
            DeviceBackend::Vulkan => Ok(candle_core::Device::new_vulkan(0)?),
        }
    }
}

/// Human-readable device identity for demos/logs (GPU name, not just backend).
fn describe_device(backend: DeviceBackend, device: &candle_core::Device) -> String {
    let loc = device.location();
    match backend {
        DeviceBackend::Cpu => "[device] backend=cpu".to_string(),
        DeviceBackend::Cuda => {
            let gpu_id = match loc {
                candle_core::DeviceLocation::Cuda { gpu_id } => gpu_id,
                _ => 0,
            };
            #[cfg(feature = "cuda")]
            {
                match device.as_cuda_device() {
                    Ok(cuda) => {
                        let stream = cuda.cuda_stream();
                        let ctx = stream.context();
                        let name = ctx.name().unwrap_or_else(|_| format!("cuda:{gpu_id}"));
                        let cc = ctx
                            .compute_capability()
                            .map(|(maj, min)| format!("{maj}.{min}"))
                            .unwrap_or_else(|_| "?".to_string());
                        let mem = ctx
                            .total_mem()
                            .map(|bytes| {
                                format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
                            })
                            .unwrap_or_else(|_| "?".to_string());
                        return format!(
                            "[device] backend=cuda gpu=\"{name}\" compute={cc} mem={mem} id={gpu_id}"
                        );
                    }
                    Err(_) => {}
                }
            }
            format!("[device] backend=cuda id={gpu_id}")
        }
        DeviceBackend::Wgpu => {
            let gpu_id = match loc {
                candle_core::DeviceLocation::Wgpu { gpu_id } => gpu_id,
                _ => 0,
            };
            #[cfg(feature = "wgpu")]
            {
                if let Ok(wgpu) = device.as_wgpu_device() {
                    let driver = wgpu.adapter_driver();
                    let driver_info = wgpu.adapter_driver_info();
                    let driver_bit = if driver.is_empty() && driver_info.is_empty() {
                        String::new()
                    } else if driver_info.is_empty() {
                        format!(" driver=\"{driver}\"")
                    } else if driver.is_empty() {
                        format!(" driver_info=\"{driver_info}\"")
                    } else {
                        format!(" driver=\"{driver}\" driver_info=\"{driver_info}\"")
                    };
                    return format!(
                        "[device] backend=wgpu gpu=\"{}\" api={}{driver_bit} id={gpu_id}",
                        wgpu.adapter_name(),
                        wgpu.adapter_backend(),
                    );
                }
            }
            format!("[device] backend=wgpu id={gpu_id}")
        }
        DeviceBackend::Vulkan => {
            let gpu_id = match loc {
                candle_core::DeviceLocation::Vulkan { gpu_id } => gpu_id,
                _ => 0,
            };
            #[cfg(feature = "vulkan")]
            {
                if let Ok(vk) = device.as_vulkan_device() {
                    return format!(
                        "[device] backend=vulkan gpu=\"{}\" type={:?} id={gpu_id}",
                        vk.physical_device_name(),
                        vk.physical_device_type(),
                    );
                }
            }
            format!("[device] backend=vulkan id={gpu_id}")
        }
    }
}

impl Family {
    pub const fn model_family(self) -> ModelFamily {
        match self {
            Self::Pi3 => ModelFamily::Pi3,
            Self::Pi3x => ModelFamily::Pi3x,
            Self::Triposr => ModelFamily::TripoSr,
            Self::Triposg => ModelFamily::TripoSg,
            Self::Vggt => ModelFamily::Vggt,
            Self::Trellis2 => ModelFamily::Trellis2,
        }
    }
}

impl Command {
    pub fn family_name(&self) -> &str {
        match self {
            Self::Inspect(args) => match args.family {
                Family::Pi3 => "pi3",
                Family::Pi3x => "pi3x",
                Family::Triposr => "triposr",
                Family::Triposg => "triposg",
                Family::Vggt => "vggt",
                Family::Trellis2 => "trellis2",
            },
            Self::Weights(args) => match args.command {
                WeightsCommand::Normalize(ref family) => match family.family {
                    Family::Pi3 => "pi3",
                    Family::Pi3x => "pi3x",
                    Family::Triposr => "triposr",
                    Family::Triposg => "triposg",
                    Family::Vggt => "vggt",
                    Family::Trellis2 => "trellis2",
                },
                WeightsCommand::Quantize(ref args) => {
                    if let Some(aux) = args.aux_family.as_deref() {
                        return aux;
                    }
                    match args.family {
                        Some(Family::Pi3) => "pi3",
                        Some(Family::Pi3x) => "pi3x",
                        Some(Family::Triposr) => "triposr",
                        Some(Family::Triposg) => "triposg",
                        Some(Family::Vggt) => "vggt",
                        Some(Family::Trellis2) => "trellis2",
                        None => "aux",
                    }
                }
            },
            Self::Run(args) => match args.family {
                Family::Pi3 => "pi3",
                Family::Pi3x => "pi3x",
                Family::Triposr => "triposr",
                Family::Triposg => "triposg",
                Family::Vggt => "vggt",
                Family::Trellis2 => "trellis2",
            },
        }
    }

    pub fn family(&self) -> Option<ModelFamily> {
        match self {
            Self::Inspect(args) => Some(args.family.model_family()),
            Self::Weights(args) => match args.command {
                WeightsCommand::Normalize(ref family) => Some(family.family.model_family()),
                WeightsCommand::Quantize(ref args) => args.family.map(Family::model_family),
            },
            Self::Run(args) => Some(args.family.model_family()),
        }
    }

    pub fn repo_root(&self) -> Option<&PathBuf> {
        match self {
            Self::Inspect(args) => Some(&args.repo_root),
            Self::Weights(args) => match &args.command {
                WeightsCommand::Normalize(family) => Some(&family.repo_root),
                WeightsCommand::Quantize(_) => None,
            },
            Self::Run(_) => None,
        }
    }
}

pub fn inspect_model(repo_root: PathBuf, family: ModelFamily) -> anyhow::Result<String> {
    let spec = lux3d_core::ModelSpec::inspect(repo_root, family)?;
    Ok(serde_json::to_string_pretty(&spec)?)
}

pub fn quantize_weights(args: QuantizeArgs) -> anyhow::Result<String> {
    let dtype = dtype_from_name(&args.dtype)
        .ok_or_else(|| anyhow::anyhow!("unknown quantization dtype `{}`", args.dtype))?;
    let fallback = dtype_from_name(&args.fallback_dtype).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown fallback quantization dtype `{}`",
            args.fallback_dtype
        )
    })?;
    let report = if let Some(aux_family) = args.aux_family.as_deref() {
        quantize_auxiliary_checkpoint(
            &args.source,
            &args.output,
            aux_family,
            QuantizeOptions {
                dtype,
                fallback,
                all_2d: args.all_2d,
                prefixes: args.prefix.clone(),
            },
        )?
    } else {
        let family = args
            .family
            .ok_or_else(|| anyhow::anyhow!("FAMILY is required unless `--aux-family` is used"))?;
        quantize_checkpoint(
            &args.source,
            &args.output,
            family.model_family(),
            QuantizeOptions {
                dtype,
                fallback,
                all_2d: args.all_2d,
                prefixes: args.prefix.clone(),
            },
        )?
    };
    Ok(report.render())
}

pub fn normalize_weights(
    repo_root: PathBuf,
    family: ModelFamily,
    raw_model_dir: PathBuf,
    output_dir: PathBuf,
) -> anyhow::Result<String> {
    let python = python_executable(&repo_root, family);
    let script = normalizer_script(&repo_root)?;

    let family_name = family.as_str();
    let output = std::process::Command::new(&python)
        .arg(&script)
        .arg("--family")
        .arg(family_name)
        .arg("--raw-model-dir")
        .arg(&raw_model_dir)
        .arg("--output-dir")
        .arg(&output_dir)
        .output()
        .map_err(|source| {
            anyhow::anyhow!(
                "failed to start Python interpreter `{}` for `{family_name}`: {source}; \
                 install Python or set LUX3D_PYTHON",
                python.display()
            )
        })?;

    if !output.status.success() {
        anyhow::bail!(
            "python normalizer failed for `{family_name}`:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let locator = WeightLocator::new(raw_model_dir, output_dir);
    let plan = locator.locate(family)?;
    let canonical = ensure_canonical_weights(&plan)?;
    Ok(serde_json::to_string_pretty(&canonical.manifest)?)
}

pub fn run_model(args: RunArgs) -> anyhow::Result<PathBuf> {
    // Must happen before device creation and any worker threads: candle reads
    // these switches once per process. Only validated families opt in — see
    // `reduced_precision_gemm_validated_for`.
    lux3d_core::enable_reduced_precision_gemm_opt_ins_for(args.family.model_family());
    let model_assets = ModelAssetOptions {
        canonical_dir: args.model_path.clone(),
        cache_dir: args.cache_dir.clone(),
    };
    let device = args.device.to_device()?;
    eprintln!("{}", describe_device(args.device, &device));
    if let Some(parent) = args
        .output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }

    if args.family.model_family() != ModelFamily::Vggt && args.dump_npz.is_some() {
        anyhow::bail!("`--dump-npz` is only supported for `vggt`");
    }

    match args.family.model_family() {
        ModelFamily::Pi3 => {
            reject_triposg_only_options(&args, args.family)?;
            reject_vggt_only_options(&args, args.family)?;
            reject_trellis2_only_options(&args, args.family)?;
            run_pi3(
                model_assets,
                required_source(&args, "pi3")?,
                args.interval,
                &args.output,
                &device,
            )?
        }
        ModelFamily::Pi3x => {
            reject_triposg_only_options(&args, args.family)?;
            reject_vggt_only_options(&args, args.family)?;
            reject_trellis2_only_options(&args, args.family)?;
            run_pi3x(
                model_assets,
                required_source(&args, "pi3x")?,
                args.conditions.as_deref(),
                args.interval,
                args.vo,
                args.chunk_size,
                args.overlap,
                args.conf_threshold,
                &args.inject_condition,
                &args.output,
                &device,
            )?
        }
        ModelFamily::TripoSr => {
            reject_triposg_only_options(&args, args.family)?;
            reject_vggt_only_options(&args, args.family)?;
            reject_trellis2_only_options(&args, args.family)?;
            run_triposr(
                model_assets,
                required_source(&args, "triposr")?,
                args.mc_resolution.unwrap_or(256),
                args.mc_threshold.unwrap_or(25.0),
                &args.output,
                &device,
            )?
        }
        ModelFamily::TripoSg => {
            reject_unsupported_triposg_options(&args)?;
            reject_vggt_only_options(&args, args.family)?;
            reject_trellis2_only_options(&args, args.family)?;
            let options = triposg_run_options_from_args(&args)?;
            let rmbg_model_path = resolve_rmbg_model_path(&args);
            let dtype = args
                .dtype
                .map(candle_core::DType::from)
                .unwrap_or(candle_core::DType::F32);
            let input = match (args.source.as_deref(), args.surface_input.as_deref()) {
                (_, Some(surface)) => TripoSgRunInput::Surface(surface),
                (Some(source), None) => TripoSgRunInput::Image(source),
                (None, None) => {
                    anyhow::bail!("`--source` or `--surface-input` is required for `triposg`")
                }
            };
            run_triposg(
                model_assets,
                rmbg_model_path,
                input,
                options,
                &args.output,
                &device,
                dtype,
            )?
        }
        ModelFamily::Vggt => {
            reject_unsupported_vggt_options(&args)?;
            reject_triposg_only_options(&args, args.family)?;
            reject_trellis2_only_options(&args, args.family)?;
            run_vggt(model_assets, &args, &device)?
        }
        ModelFamily::Trellis2 => {
            reject_unsupported_trellis2_options(&args)?;
            reject_vggt_only_options(&args, args.family)?;
            validate_trellis2_texturing_mode(&args)?;
            let rmbg_model_path = resolve_trellis2_rmbg_model_path(&args);
            let source = required_source(&args, "trellis2")?;
            if let Some(mesh_path) = args.texture_mesh.as_deref() {
                let options = trellis2_texturing_options_from_args(&args)?;
                run_trellis2_texturing(
                    model_assets,
                    rmbg_model_path,
                    source,
                    mesh_path,
                    &options,
                    &args.output,
                    &device,
                )?
            } else {
                let options = trellis2_run_options_from_args(&args)?;
                run_trellis2(
                    model_assets,
                    rmbg_model_path,
                    source,
                    options,
                    &args.output,
                    args.return_latent.as_deref(),
                    &device,
                )?
            }
        }
    }

    // Backend CPU-fallback observability: nonzero values = hidden host/CPU compute.
    match args.device {
        DeviceBackend::Wgpu => {
            eprintln!(
                "[candle-obs] wgpu cpu_fallback={} host_compute={}",
                candle_core::wgpu_cpu_fallback_count(),
                candle_core::wgpu_host_compute_count(),
            );
            #[cfg(feature = "wgpu")]
            if let Some(rows) = candle_core::wgpu_gpu_profile_report() {
                let _ = device.synchronize();
                eprintln!("[gpu-profile] wgpu kernels={}", rows.len());
                for (name, count, total_ms) in rows.iter().take(40) {
                    eprintln!(
                        "[gpu-profile] {name:44} {count:>7}x {total_ms:>10.1}ms {:>8.1}us/ea",
                        total_ms * 1000.0 / (*count as f64)
                    );
                }
            }
        }
        #[cfg(feature = "vulkan")]
        DeviceBackend::Vulkan => {
            eprintln!(
                "[candle-obs] vulkan cpu_fallback={} host_compute={}",
                candle_core::vulkan_cpu_fallback_count(),
                candle_core::vulkan_host_compute_count(),
            );
            if std::env::var_os("LUX3D_GPU_PROFILE").is_some() {
                device.synchronize()?;
                if let Some((wall, rows)) = candle_core::vulkan_gpu_profile_report() {
                    eprintln!(
                        "[gpu-profile] wall={wall:.2}s (aggregation start -> report) kernels={}",
                        rows.len()
                    );
                    for (name, count, total_ms) in rows.iter().take(50) {
                        eprintln!(
                            "[gpu-profile] {name:44} {count:>7}x {total_ms:>10.1}ms {:>8.1}us/ea",
                            total_ms * 1000.0 / (*count as f64)
                        );
                    }
                }
            }
            if let Some(rows) = candle_core::vulkan_flush_reason_report() {
                eprintln!("[cpu-profile] vulkan flush reasons:");
                for (name, count) in rows {
                    eprintln!("[cpu-profile] flush:{name:28} {count:>8}x");
                }
            }
            if let Some(rows) = candle_core::vulkan_cpu_profile_report() {
                eprintln!("[cpu-profile] vulkan dispatch hot path:");
                for (name, count, total_ms) in rows {
                    eprintln!(
                        "[cpu-profile] {name:24} {count:>8}x {total_ms:>10.1}ms {:>8.1}us/ea",
                        total_ms * 1000.0 / (count as f64)
                    );
                }
            }
        }
        _ => {}
    }

    Ok(args.output)
}

/// Resolve the required `--source` for families that do not accept
/// `--surface-input`. Clap only makes `--source` optional when
/// `--surface-input` is present, and the family guards reject that
/// combination first; this is the defensive fallback.
fn required_source<'a>(args: &'a RunArgs, family: &str) -> anyhow::Result<&'a Path> {
    args.source
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("`--source` is required for `{family}`"))
}

fn run_pi3(
    model_assets: ModelAssetOptions,
    source: &Path,
    interval: Option<usize>,
    output: &Path,
    device: &candle_core::Device,
) -> anyhow::Result<()> {
    let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
    let t0 = std::time::Instant::now();
    let pipeline = Pi3Pipeline::load(model_assets)?;
    if stage_time {
        eprintln!("[stage] load: {:.2}s", t0.elapsed().as_secs_f64());
    }
    let t1 = std::time::Instant::now();
    let inference = pipeline.infer_from_path_with_interval(source, interval, device)?;
    if stage_time {
        eprintln!("[stage] infer: {:.2}s", t1.elapsed().as_secs_f64());
    }
    let t2 = std::time::Instant::now();
    pipeline.export_point_cloud(&inference, output)?;
    if stage_time {
        eprintln!("[stage] export: {:.2}s", t2.elapsed().as_secs_f64());
    }
    Ok(())
}

/// VGGT exposes point/depth/camera inference only. Reject Pi3X VO and
/// TripoSR meshing options instead of silently ignoring them.
fn reject_unsupported_vggt_options(args: &RunArgs) -> anyhow::Result<()> {
    if args.conditions.is_some() {
        anyhow::bail!("`--conditions` is not supported for `vggt`");
    }
    if args.vo {
        anyhow::bail!("`--vo` is not supported for `vggt`");
    }
    if args.chunk_size.is_some() {
        anyhow::bail!("`--chunk-size` is not supported for `vggt`");
    }
    if args.overlap.is_some() {
        anyhow::bail!("`--overlap` is not supported for `vggt`");
    }
    if !args.inject_condition.is_empty() {
        anyhow::bail!("`--inject-condition` is not supported for `vggt`");
    }
    if args.mc_resolution.is_some() {
        anyhow::bail!("`--mc-resolution` is not supported for `vggt`");
    }
    if args.mc_threshold.is_some() {
        anyhow::bail!("`--mc-threshold` is not supported for `vggt`");
    }
    Ok(())
}

/// The `--vggt-*` / preprocess-mode / batch-size / colmap / depth-points /
/// track flags are VGGT-only; every other family must reject them (when they
/// differ from their default) instead of silently ignoring them. Default-valued
/// flags cannot be distinguished from explicitly-passed defaults, so a bare
/// `--preprocess-mode crop` on another family is accepted as a no-op.
fn reject_vggt_only_options(args: &RunArgs, family: Family) -> anyhow::Result<()> {
    let vggt = &args.vggt;
    let set: [(&str, bool); 15] = [
        (
            "--preprocess-mode",
            vggt.preprocess_mode != VggtPreprocessKind::Crop,
        ),
        ("--vggt-dtype", vggt.vggt_dtype.is_some()),
        ("--batch-size", vggt.batch_size != 1),
        ("--colmap-dir", vggt.colmap_dir.is_some()),
        (
            "--colmap-conf-threshold",
            vggt.colmap_conf_threshold != DEFAULT_CONF_THRESHOLD,
        ),
        (
            "--colmap-max-points",
            vggt.colmap_max_points != DEFAULT_MAX_POINTS,
        ),
        (
            "--colmap-point-source",
            vggt.colmap_point_source != VggtColmapPointSourceKind::Depth,
        ),
        ("--colmap-seed", vggt.colmap_seed.is_some()),
        (
            "--colmap-world-conf-threshold",
            vggt.colmap_world_conf_threshold.is_some(),
        ),
        ("--depth-points", vggt.depth_points.is_some()),
        ("--depth-conf-threshold", vggt.depth_conf_threshold != 1.0),
        ("--track-queries", vggt.track_queries.is_some()),
        ("--track-output", vggt.track_output.is_some()),
        ("--track-iters", vggt.track_iters != 4),
        ("--track-points-ply", vggt.track_points_ply.is_some()),
    ];
    for (flag, is_set) in set {
        if is_set {
            anyhow::bail!("`{flag}` is not supported for `{}`", family.model_family());
        }
    }
    Ok(())
}

/// TripoSG exposes image-to-mesh diffusion only. Reject the Pi3/Pi3X/VGGT
/// conditioning and point-cloud options plus the TripoSR marching-cubes knobs
/// instead of silently ignoring them. `--chunk-size` stays allowed: it feeds
/// the TripoSG SDF query chunk.
fn reject_unsupported_triposg_options(args: &RunArgs) -> anyhow::Result<()> {
    if args.interval.is_some() {
        anyhow::bail!("`--interval` is not supported for `triposg`");
    }
    if args.conditions.is_some() {
        anyhow::bail!("`--conditions` is not supported for `triposg`");
    }
    if args.vo {
        anyhow::bail!("`--vo` is not supported for `triposg`");
    }
    if args.overlap.is_some() {
        anyhow::bail!("`--overlap` is not supported for `triposg`");
    }
    if args.conf_threshold.is_some() {
        anyhow::bail!("`--conf-threshold` is not supported for `triposg`");
    }
    if !args.inject_condition.is_empty() {
        anyhow::bail!("`--inject-condition` is not supported for `triposg`");
    }
    if args.mc_resolution.is_some() {
        anyhow::bail!("`--mc-resolution` is not supported for `triposg`");
    }
    if args.mc_threshold.is_some() {
        anyhow::bail!("`--mc-threshold` is not supported for `triposg`");
    }
    Ok(())
}

/// The `--surface-input`/`--steps`/`--guidance-scale`/`--seed`/`--num-tokens`/
/// `--dense-octree-depth`/`--hierarchical-octree-depth` flags are TripoSG-only;
/// every other family must reject them instead of silently ignoring them.
fn reject_triposg_only_options(args: &RunArgs, family: Family) -> anyhow::Result<()> {
    if args.surface_input.is_some() {
        anyhow::bail!(
            "`--surface-input` is not supported for `{}`",
            family.model_family()
        );
    }
    if args.steps.is_some() {
        anyhow::bail!("`--steps` is not supported for `{}`", family.model_family());
    }
    if args.guidance_scale.is_some() {
        anyhow::bail!(
            "`--guidance-scale` is not supported for `{}`",
            family.model_family()
        );
    }
    if args.seed.is_some() {
        anyhow::bail!("`--seed` is not supported for `{}`", family.model_family());
    }
    if args.num_tokens.is_some() {
        anyhow::bail!(
            "`--num-tokens` is not supported for `{}`",
            family.model_family()
        );
    }
    if args.dense_octree_depth.is_some() {
        anyhow::bail!(
            "`--dense-octree-depth` is not supported for `{}`",
            family.model_family()
        );
    }
    if args.hierarchical_octree_depth.is_some() {
        anyhow::bail!(
            "`--hierarchical-octree-depth` is not supported for `{}`",
            family.model_family()
        );
    }
    if args.flash_decoder.is_some() {
        anyhow::bail!(
            "`--flash-decoder` is not supported for `{}`",
            family.model_family()
        );
    }
    if args.flash_octree_depth.is_some() {
        anyhow::bail!(
            "`--flash-octree-depth` is not supported for `{}`",
            family.model_family()
        );
    }
    if args.faces.is_some() {
        anyhow::bail!("`--faces` is not supported for `{}`", family.model_family());
    }
    if args.dtype.is_some() {
        anyhow::bail!("`--dtype` is not supported for `{}`", family.model_family());
    }
    if args.bounds.is_some() {
        anyhow::bail!(
            "`--bounds` is not supported for `{}`",
            family.model_family()
        );
    }
    if args.timesteps.is_some() {
        anyhow::bail!(
            "`--timesteps` is not supported for `{}`",
            family.model_family()
        );
    }
    Ok(())
}

/// Resolve the RMBG package: explicit `--rmbg-model-path` wins, otherwise a
/// sibling `rmbg-1.4` canonical directory next to an explicit TripoSG
/// `--model-path` is used when present. (The canonical directories are
/// distributed separately because the BRIA license differs from TripoSG's
/// MIT license.)
fn resolve_rmbg_model_path(args: &RunArgs) -> Option<PathBuf> {
    if args.rmbg_model_path.is_some() {
        return args.rmbg_model_path.clone();
    }
    args.model_path
        .as_ref()
        .and_then(|path| path.parent())
        .map(|parent| parent.join("rmbg-1.4"))
        .filter(|candidate| candidate.is_dir())
}

/// Resolve the RMBG-2.0/BiRefNet package used by TRELLIS.2 for RGB inputs:
/// explicit `--rmbg-model-path` wins, otherwise a sibling `rmbg-2.0`
/// canonical directory (or raw `RMBG-2.0` checkpoint directory) next to an
/// explicit `--model-path` is used when present. The package is distributed
/// separately because its CC-BY-NC license differs from TRELLIS.2's MIT
/// license.
fn resolve_trellis2_rmbg_model_path(args: &RunArgs) -> Option<PathBuf> {
    if args.rmbg_model_path.is_some() {
        return args.rmbg_model_path.clone();
    }
    let parent = args.model_path.as_ref().and_then(|path| path.parent())?;
    ["rmbg-2.0", "RMBG-2.0"]
        .iter()
        .map(|name| parent.join(name))
        .find(|candidate| candidate.is_dir())
}

/// TRELLIS.2 exposes image-to-mesh diffusion + texture only. Reject the
/// Pi3/Pi3X/VGGT conditioning and point-cloud options, the TripoSR
/// marching-cubes knobs, and the TripoSG octree/flash/latent knobs it does
/// not share instead of silently ignoring them. `--rmbg-model-path` is
/// accepted: the reference runs RMBG-2.0/BiRefNet for inputs without a
/// usable alpha channel. `--steps`/`--seed`/`--pipeline-type`/
/// `--max-num-tokens`/`--texture`/`--num-samples`/`--return-latent` and the
/// per-stage `--ss-*`/`--shape-*`/`--tex-*` sampler flags stay allowed;
/// `--guidance-scale` has no runtime override (per-stage guidance comes from
/// the dedicated flags; `pipeline.json` defaults otherwise).
fn reject_unsupported_trellis2_options(args: &RunArgs) -> anyhow::Result<()> {
    if args.interval.is_some() {
        anyhow::bail!("`--interval` is not supported for `trellis2`");
    }
    if args.conditions.is_some() {
        anyhow::bail!("`--conditions` is not supported for `trellis2`");
    }
    if args.vo {
        anyhow::bail!("`--vo` is not supported for `trellis2`");
    }
    if args.chunk_size.is_some() {
        anyhow::bail!("`--chunk-size` is not supported for `trellis2`");
    }
    if args.overlap.is_some() {
        anyhow::bail!("`--overlap` is not supported for `trellis2`");
    }
    if args.surface_input.is_some() {
        anyhow::bail!("`--surface-input` is not supported for `trellis2`");
    }
    if args.conf_threshold.is_some() {
        anyhow::bail!("`--conf-threshold` is not supported for `trellis2`");
    }
    if !args.inject_condition.is_empty() {
        anyhow::bail!("`--inject-condition` is not supported for `trellis2`");
    }
    if args.mc_resolution.is_some() {
        anyhow::bail!("`--mc-resolution` is not supported for `trellis2`");
    }
    if args.mc_threshold.is_some() {
        anyhow::bail!("`--mc-threshold` is not supported for `trellis2`");
    }
    if args.guidance_scale.is_some() {
        anyhow::bail!("`--guidance-scale` is not supported for `trellis2`");
    }
    if args.num_tokens.is_some() {
        anyhow::bail!("`--num-tokens` is not supported for `trellis2`; use `--max-num-tokens`");
    }
    if args.dense_octree_depth.is_some() {
        anyhow::bail!("`--dense-octree-depth` is not supported for `trellis2`");
    }
    if args.hierarchical_octree_depth.is_some() {
        anyhow::bail!("`--hierarchical-octree-depth` is not supported for `trellis2`");
    }
    if args.flash_decoder.is_some() {
        anyhow::bail!("`--flash-decoder` is not supported for `trellis2`");
    }
    if args.flash_octree_depth.is_some() {
        anyhow::bail!("`--flash-octree-depth` is not supported for `trellis2`");
    }
    if args.faces.is_some() {
        anyhow::bail!("`--faces` is not supported for `trellis2`");
    }
    if args.dtype.is_some() {
        anyhow::bail!("`--dtype` is not supported for `trellis2`");
    }
    if args.bounds.is_some() {
        anyhow::bail!("`--bounds` is not supported for `trellis2`");
    }
    if args.timesteps.is_some() {
        anyhow::bail!("`--timesteps` is not supported for `trellis2`");
    }
    Ok(())
}

/// The `--pipeline-type`/`--max-num-tokens`/`--texture`/textured-GLB
/// (`--texture-size`/`--no-texture-export`/`--no-uv-unwrap`)/mesh
/// post-process/per-stage sampler/`--num-samples`/`--return-latent` flags
/// are TRELLIS.2-only; every other family must reject them instead of
/// silently ignoring them (same fail-fast pattern as the TripoSG-only and
/// VGGT-only flags).
fn reject_trellis2_only_options(args: &RunArgs, family: Family) -> anyhow::Result<()> {
    let set: [(&str, bool); 28] = [
        ("--pipeline-type", args.pipeline_type.is_some()),
        ("--max-num-tokens", args.max_num_tokens.is_some()),
        ("--texture", args.texture.is_some()),
        ("--texture-size", args.texture_size.is_some()),
        ("--no-texture-export", args.no_texture_export),
        ("--no-uv-unwrap", args.no_uv_unwrap),
        ("--texture-mesh", args.texture_mesh.is_some()),
        ("--texture-resolution", args.texture_resolution.is_some()),
        ("--no-preprocess-mesh", args.no_preprocess_mesh),
        ("--no-preprocess-image", args.no_preprocess_image),
        ("--no-postprocess", args.no_postprocess),
        ("--decimation-target", args.decimation_target.is_some()),
        ("--fill-hole-perimeter", args.fill_hole_perimeter.is_some()),
        ("--min-component-area", args.min_component_area.is_some()),
        ("--ss-steps", args.ss_steps.is_some()),
        ("--shape-steps", args.shape_steps.is_some()),
        ("--tex-steps", args.tex_steps.is_some()),
        ("--ss-guidance", args.ss_guidance.is_some()),
        ("--shape-guidance", args.shape_guidance.is_some()),
        ("--tex-guidance", args.tex_guidance.is_some()),
        ("--ss-rescale", args.ss_rescale.is_some()),
        ("--shape-rescale", args.shape_rescale.is_some()),
        ("--tex-rescale", args.tex_rescale.is_some()),
        ("--ss-interval", args.ss_interval.is_some()),
        ("--shape-interval", args.shape_interval.is_some()),
        ("--tex-interval", args.tex_interval.is_some()),
        ("--num-samples", args.num_samples.is_some()),
        ("--return-latent", args.return_latent.is_some()),
    ];
    for (flag, is_set) in set {
        if is_set {
            anyhow::bail!("`{flag}` is not supported for `{}`", family.model_family());
        }
    }
    Ok(())
}

/// Parse a `lo,hi` guidance-interval argument (upstream
/// `guidance_interval`).
fn parse_interval_pair(value: &str) -> anyhow::Result<(f32, f32)> {
    let (low, high) = value
        .split_once(',')
        .ok_or_else(|| anyhow::anyhow!("expected `lo,hi` (e.g. `0.6,1.0`), got `{value}`"))?;
    let low: f32 = low
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid interval low `{low}`"))?;
    let high: f32 = high
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid interval high `{high}`"))?;
    if !(0.0..=1.0).contains(&low) || !(0.0..=1.0).contains(&high) || low > high {
        anyhow::bail!("guidance interval must be 0 <= lo <= hi <= 1, got `{value}`");
    }
    Ok((low, high))
}

/// Assemble one stage's [`Trellis2SamplerOverrides`] from its CLI flags;
/// `None` when no flag was set (keep the pipeline defaults).
fn trellis2_stage_override(
    steps: Option<usize>,
    guidance: Option<f32>,
    rescale: Option<f32>,
    interval: Option<&String>,
) -> anyhow::Result<Option<Trellis2SamplerOverrides>> {
    let interval = match interval {
        Some(value) => Some(parse_interval_pair(value)?),
        None => None,
    };
    let overrides = Trellis2SamplerOverrides {
        steps,
        guidance_strength: guidance,
        guidance_rescale: rescale,
        guidance_interval: interval,
        rescale_t: None,
    };
    Ok((overrides != Trellis2SamplerOverrides::default()).then_some(overrides))
}

/// Build the TRELLIS.2 run options from the reference defaults plus explicit
/// CLI overrides. `--steps` is the base step count for all three Euler
/// samplers; `--ss-steps`/`--shape-steps`/`--tex-steps` and the
/// `--ss-*`/`--shape-*`/`--tex-*` guidance flags map onto the per-stage
/// sampler overrides, which win over `--steps`/the pipeline defaults at
/// resolution time (`resolve_stage_params`). `--guidance-scale` has no
/// runtime override. `--no-postprocess` disables the geometry post-process
/// and `--decimation-target`/`--fill-hole-perimeter`/`--min-component-area`
/// override its parameters (defaults mirror the upstream `to_glb`
/// signature; validated before the run starts). The textured-GLB export
/// stage stays enabled by default (`--no-texture-export` disables it,
/// `--no-uv-unwrap` skips just the xatlas unwrap) and `--texture-size`
/// overrides the bake edge length: the CLI accepts any value in 64..=8192
/// (the upstream reference allows 1024..4096 in steps of 1024), while the
/// core `validate` adds the nvdiffrast multiple-of-8 granularity.
fn trellis2_run_options_from_args(args: &RunArgs) -> anyhow::Result<Trellis2RunOptions> {
    let defaults = Trellis2RunOptions::default();
    let num_samples = args.num_samples.unwrap_or(1);
    if num_samples == 0 {
        anyhow::bail!("`--num-samples` must be >= 1");
    }
    let mut postprocess = Trellis2MeshPostprocessOptions::default();
    if args.no_postprocess {
        postprocess.enabled = false;
    }
    if let Some(value) = args.decimation_target {
        postprocess.decimation_target = value;
    }
    if let Some(value) = args.fill_hole_perimeter {
        postprocess.fill_hole_perimeter = value;
    }
    if let Some(value) = args.min_component_area {
        postprocess.min_component_area = value;
    }
    postprocess.validate().map_err(anyhow::Error::from)?;
    let mut texture_export = Trellis2TextureExportOptions::default();
    if args.no_texture_export {
        texture_export.enabled = false;
    }
    if args.no_uv_unwrap {
        texture_export.uv_unwrap = false;
    }
    if let Some(size) = args.texture_size {
        if !(64..=8192).contains(&size) {
            anyhow::bail!(
                "`--texture-size` must be in 64..=8192 texels (got {size}); the upstream \
                 reference allows 1024..=4096 in steps of 1024"
            );
        }
        texture_export.texture_size = size;
    }
    texture_export.validate().map_err(anyhow::Error::from)?;
    let options = Trellis2RunOptions {
        seed: args.seed.unwrap_or(defaults.seed),
        pipeline_type: args
            .pipeline_type
            .map(Trellis2PipelineType::from)
            .unwrap_or(defaults.pipeline_type),
        ss_steps: args.steps.unwrap_or(defaults.ss_steps),
        shape_steps: args.steps.unwrap_or(defaults.shape_steps),
        tex_steps: args.steps.unwrap_or(defaults.tex_steps),
        ss_sampler: trellis2_stage_override(
            args.ss_steps,
            args.ss_guidance,
            args.ss_rescale,
            args.ss_interval.as_ref(),
        )?,
        shape_sampler: trellis2_stage_override(
            args.shape_steps,
            args.shape_guidance,
            args.shape_rescale,
            args.shape_interval.as_ref(),
        )?,
        tex_sampler: trellis2_stage_override(
            args.tex_steps,
            args.tex_guidance,
            args.tex_rescale,
            args.tex_interval.as_ref(),
        )?,
        with_texture: args.texture.unwrap_or(defaults.with_texture),
        max_num_tokens: args.max_num_tokens.unwrap_or(defaults.max_num_tokens),
        preprocess: defaults.preprocess,
        num_samples,
        postprocess,
        texture_export,
    };
    Ok(options)
}

/// Build the TRELLIS.2 texturing-pipeline options (upstream
/// `Trellis2TexturingPipeline.run`): the condition image comes from
/// `--source`, the mesh from `--texture-mesh`; `--texture-size`, `--seed`
/// and the `--tex-*` sampler flags are shared with the image-to-3D surface.
fn trellis2_texturing_options_from_args(
    args: &RunArgs,
) -> anyhow::Result<Trellis2TexturingOptions> {
    let resolution = args.texture_resolution.unwrap_or(1024);
    if resolution != 512 && resolution != 1024 {
        anyhow::bail!("`--texture-resolution` must be 512 or 1024 (got {resolution})");
    }
    let texture_size = args.texture_size.unwrap_or(2048);
    if !(64..=8192).contains(&texture_size) {
        anyhow::bail!(
            "`--texture-size` must be in 64..=8192 texels (got {texture_size}); the upstream \
             reference allows 1024..=4096 in steps of 1024"
        );
    }
    let options = Trellis2TexturingOptions {
        seed: args.seed.unwrap_or(42),
        resolution,
        tex_sampler: trellis2_stage_override(
            args.tex_steps,
            args.tex_guidance,
            args.tex_rescale,
            args.tex_interval.as_ref(),
        )?
        .unwrap_or_default(),
        texture_size,
        preprocess_image: !args.no_preprocess_image,
        preprocess_mesh: !args.no_preprocess_mesh,
    };
    options.validate().map_err(anyhow::Error::from)?;
    Ok(options)
}

/// Validate the texturing-mode CLI surface: the textured GLB writer is the
/// only output that can carry the baked textures, and the image-to-3D-only
/// flags have no meaning when an existing mesh is supplied.
fn validate_trellis2_texturing_mode(args: &RunArgs) -> anyhow::Result<()> {
    if args.texture_mesh.is_none() {
        return Ok(());
    }
    if !matches!(
        lux3d_core::export::MeshFormat::from_path(&args.output),
        Ok(lux3d_core::export::MeshFormat::Glb)
    ) {
        anyhow::bail!(
            "`--texture-mesh` requires a `.glb` output: the textured GLB embeds the baked \
             base-color + metallic-roughness PNGs (got `{}`)",
            args.output.display()
        );
    }
    let incompatible: [(&str, bool); 9] = [
        ("--pipeline-type", args.pipeline_type.is_some()),
        ("--max-num-tokens", args.max_num_tokens.is_some()),
        ("--return-latent", args.return_latent.is_some()),
        ("--num-samples", args.num_samples.is_some()),
        ("--no-texture-export", args.no_texture_export),
        ("--no-uv-unwrap", args.no_uv_unwrap),
        ("--no-postprocess", args.no_postprocess),
        ("--decimation-target", args.decimation_target.is_some()),
        ("--texture false", args.texture == Some(false)),
    ];
    for (flag, is_set) in incompatible {
        if is_set {
            anyhow::bail!(
                "`{flag}` is not supported with `--texture-mesh` (the texturing pipeline \
                 uses the input mesh topology and always bakes a textured GLB)"
            );
        }
    }
    Ok(())
}

/// the extension (`out.glb` -> `out_0.glb`, `out.latents.npz` ->
/// `out.latents_0.npz`; extension-less paths get the suffix at the end).
fn sample_output_path(output: &Path, index: usize, num_samples: usize) -> PathBuf {
    if num_samples <= 1 {
        return output.to_path_buf();
    }
    let file_name = output
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| format!("sample_{index}"));
    let suffixed = match file_name.rsplit_once('.') {
        Some((stem, extension)) if !stem.is_empty() && !extension.is_empty() => {
            format!("{stem}_{index}.{extension}")
        }
        _ => format!("{file_name}_{index}"),
    };
    output.with_file_name(suffixed)
}

/// Parse `--bounds` into the six-float `[min_x, min_y, min_z, max_x, max_y,
/// max_z]` SDF query window (upstream `TripoSGPipeline.__call__(bounds=...)`).
///
/// A single scalar `b` expands to the symmetric cube
/// `[-b, -b, -b, b, b, b]`, matching the upstream `Union[Tuple, List, float]`
/// contract (`inference_utils.py` expands a float exactly this way); six
/// comma-separated floats pass through unchanged. The min < max sanity check
/// lives in `TripoSgRunOptions::validate`.
fn parse_bounds_arg(value: &str) -> anyhow::Result<[f32; 6]> {
    let mut components = Vec::new();
    for part in value.split(',') {
        let component: f32 = part
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid `--bounds` component `{part}` in `{value}`"))?;
        components.push(component);
    }
    if components.len() == 1 {
        let half = components[0];
        return Ok([-half, -half, -half, half, half, half]);
    }
    if components.len() != 6 {
        anyhow::bail!(
            "`--bounds` expects a single scalar (symmetric cube `[-b, b]`) or \
             exactly six comma-separated floats \
             min_x,min_y,min_z,max_x,max_y,max_z, got {} value(s) in `{value}`",
            components.len()
        );
    }
    Ok([
        components[0],
        components[1],
        components[2],
        components[3],
        components[4],
        components[5],
    ])
}

/// Parse `--timesteps` into the explicit schedule (upstream
/// `TripoSGPipeline.__call__(timesteps=...)` via `retrieve_timesteps`). The
/// list order is preserved; the non-empty / finiteness checks live in
/// `TripoSgRunOptions::validate`.
fn parse_timesteps_arg(value: &str) -> anyhow::Result<Vec<f32>> {
    value
        .split(',')
        .map(|part| {
            part.trim().parse::<f32>().map_err(|_| {
                anyhow::anyhow!("invalid `--timesteps` component `{part}` in `{value}`")
            })
        })
        .collect()
}

/// Build the TripoSG run options from the reference defaults plus explicit
/// CLI overrides; `--chunk-size` doubles as the SDF query chunk. With
/// `--surface-input` the decode defaults follow `scripts/inference_vae.py`
/// (`hierarchical_extract_geometry`: flash off, dense 8 / hierarchical 9)
/// unless the corresponding flags override them.
fn triposg_run_options_from_args(args: &RunArgs) -> anyhow::Result<TripoSgRunOptions> {
    let defaults = TripoSgRunOptions::default();
    let surface_decode_default = args.surface_input.is_none();
    let options = TripoSgRunOptions {
        steps: args.steps.unwrap_or(defaults.steps),
        guidance_scale: args.guidance_scale.unwrap_or(defaults.guidance_scale),
        seed: args.seed.unwrap_or(defaults.seed),
        num_tokens: args.num_tokens.unwrap_or(defaults.num_tokens),
        dense_octree_depth: args
            .dense_octree_depth
            .unwrap_or(defaults.dense_octree_depth),
        hierarchical_octree_depth: args
            .hierarchical_octree_depth
            .unwrap_or(defaults.hierarchical_octree_depth),
        use_flash_decoder: args.flash_decoder.unwrap_or(surface_decode_default),
        flash_octree_depth: args
            .flash_octree_depth
            .unwrap_or(defaults.flash_octree_depth),
        query_chunk_size: args.chunk_size.unwrap_or(defaults.query_chunk_size),
        timesteps: args
            .timesteps
            .as_deref()
            .map(parse_timesteps_arg)
            .transpose()?,
        bounds: args.bounds.as_deref().map(parse_bounds_arg).transpose()?,
        // `--faces 0` is the documented "no simplification" value (the
        // upstream `-1` equivalent); it maps to `None` like the flag default.
        faces: args.faces.filter(|&faces| faces > 0),
    };
    options.validate().map_err(anyhow::Error::from)?;
    Ok(options)
}

/// Map the VGGT CLI flags onto [`VggtRunOptions`]. `--batch-size` defaults to
/// 1 (sequential single-frame chunks; see the flag help for the semantics)
/// and 0 is rejected before any weights are touched.
fn vggt_run_options_from_args(args: &RunArgs) -> anyhow::Result<VggtRunOptions> {
    if args.vggt.batch_size == 0 {
        anyhow::bail!("`--batch-size` must be at least 1");
    }
    Ok(VggtRunOptions {
        dtype: args.vggt.vggt_dtype.map(candle_core::DType::from),
        preprocess_mode: VggtPreprocessMode::from(args.vggt.preprocess_mode),
        interval: args.interval,
        batch_size: Some(args.vggt.batch_size),
    })
}

/// Map the COLMAP export flags onto [`VggtColmapOptions`].
fn vggt_colmap_options_from_args(vggt: &VggtRunArgs) -> VggtColmapOptions {
    VggtColmapOptions {
        conf_threshold: vggt.colmap_conf_threshold,
        max_points: vggt.colmap_max_points,
        confidence: vggt.colmap_world_conf_threshold,
        point_source: vggt.colmap_point_source.into(),
        seed: vggt.colmap_seed,
    }
}

/// Highest valid query coordinate: prepared VGGT frames are at most 518x518,
/// so pixel coordinates live in `[0, 518)`.
const VGGT_QUERY_COORD_LIMIT: f32 = 518.0;

/// Parse `--track-queries`: a JSON array of `{"frame": N, "x": F, "y": F}`
/// objects or `[N, F, F]` arrays. Pixel coordinates are in the model input
/// resolution (`x`/`y` in `[0, 518)`) and the query frame must be 0 (the
/// tracker's query frame; rotate the clip to track from another frame).
fn parse_vggt_track_queries(json: &str) -> anyhow::Result<Vec<VggtTrackQuery>> {
    let value: serde_json::Value = serde_json::from_str(json)
        .map_err(|error| anyhow::anyhow!("`--track-queries` is not valid JSON: {error}"))?;
    let items = value
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("`--track-queries` must be a JSON array of queries"))?;
    if items.is_empty() {
        anyhow::bail!("`--track-queries` must contain at least one query point");
    }
    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let (frame, x, y) = match item {
                serde_json::Value::Object(map) => {
                    let missing = |key: &str| {
                        anyhow::anyhow!(
                            "`--track-queries` query {index} is missing the `{key}` field"
                        )
                    };
                    let frame = map.get("frame").ok_or_else(|| missing("frame"))?;
                    let x = map.get("x").ok_or_else(|| missing("x"))?;
                    let y = map.get("y").ok_or_else(|| missing("y"))?;
                    (frame.as_u64(), x.as_f64(), y.as_f64())
                }
                serde_json::Value::Array(values) if values.len() == 3 => {
                    (
                        values[0].as_u64(),
                        values[1].as_f64(),
                        values[2].as_f64(),
                    )
                }
                _ => {
                    anyhow::bail!(
                        "`--track-queries` query {index} must be an object with `frame`/`x`/`y` or a `[frame, x, y]` array"
                    );
                }
            };
            let frame = frame.ok_or_else(|| {
                anyhow::anyhow!(
                    "`--track-queries` query {index}: `frame` must be a non-negative integer (the tracker's query frame is 0)"
                )
            })?;
            if frame != 0 {
                anyhow::bail!(
                    "`--track-queries` query {index}: `frame` must be 0 (the tracker's query frame; rotate the clip so the desired query frame comes first), got {frame}"
                );
            }
            let coordinate = |axis: &str, value: Option<f64>| -> anyhow::Result<f32> {
                let value = value.ok_or_else(|| {
                    anyhow::anyhow!(
                        "`--track-queries` query {index}: `{axis}` must be a JSON number"
                    )
                })?;
                let value = value as f32;
                if !value.is_finite() {
                    anyhow::bail!(
                        "`--track-queries` query {index}: `{axis}` must be finite, got {value}"
                    );
                }
                if !(0.0..VGGT_QUERY_COORD_LIMIT).contains(&value) {
                    anyhow::bail!(
                        "`--track-queries` query {index}: `{axis}` must be a pixel coordinate in [0, {}), got {value}",
                        VGGT_QUERY_COORD_LIMIT
                    );
                }
                Ok(value)
            };
            Ok(VggtTrackQuery {
                frame: frame as usize,
                xy: [coordinate("x", x)?, coordinate("y", y)?],
            })
        })
        .collect()
}

/// Default track output path: `<source-stem>_tracks.json` in the working
/// directory (sources without a stem fall back to `tracks.json`).
fn vggt_default_track_output(source: &Path) -> PathBuf {
    match source.file_stem() {
        Some(stem) => PathBuf::from(format!("{}_tracks.json", stem.to_string_lossy())),
        None => PathBuf::from("tracks.json"),
    }
}

/// COLMAP image names for a directory source, mirroring the core preprocess
/// listing exactly (png/jpg/jpeg filter, Python-`sorted` order by lowercased
/// file name then path, `interval` stride). Video/single-image sources return
/// an empty list (`write_feedforward_colmap` then writes `image_N` names).
/// The caller must verify the count against the frame count.
fn vggt_colmap_image_names(source: &Path, interval: usize) -> Vec<String> {
    if !source.is_dir() {
        return Vec::new();
    }
    let mut entries: Vec<PathBuf> = match std::fs::read_dir(source) {
        Ok(entries) => entries
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg"))
                    .unwrap_or(false)
            })
            .collect(),
        Err(_) => return Vec::new(),
    };
    entries.sort_by(|lhs, rhs| {
        let lhs_name = lhs
            .file_name()
            .and_then(|value| value.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_else(|| lhs.to_string_lossy().to_ascii_lowercase());
        let rhs_name = rhs
            .file_name()
            .and_then(|value| value.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_else(|| rhs.to_string_lossy().to_ascii_lowercase());
        lhs_name.cmp(&rhs_name).then_with(|| lhs.cmp(rhs))
    });
    entries
        .into_iter()
        .step_by(interval.max(1))
        .filter_map(|path| {
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .collect()
}

/// Serialize a track result as `{"coords": [S][Q][2], "visibility": [S][Q],
/// "confidence": [S][Q] | null}` (F32 JSON arrays; `coords` are
/// model-resolution pixel coordinates with `coords[0] == queries`).
fn vggt_track_result_to_json(track: &VggtTrackResult) -> anyhow::Result<serde_json::Value> {
    let device = candle_core::Device::Cpu;
    let coords = track
        .coords
        .to_device(&device)
        .and_then(|tensor| tensor.to_dtype(candle_core::DType::F32))
        .and_then(|tensor| tensor.contiguous())
        .and_then(|tensor| tensor.to_vec3::<f32>())
        .map_err(|error| anyhow::anyhow!("failed to read VGGT track coords: {error}"))?;
    let visibility = track
        .visibility
        .to_device(&device)
        .and_then(|tensor| tensor.to_dtype(candle_core::DType::F32))
        .and_then(|tensor| tensor.contiguous())
        .and_then(|tensor| tensor.to_vec2::<f32>())
        .map_err(|error| anyhow::anyhow!("failed to read VGGT track visibility: {error}"))?;
    let confidence = match &track.confidence {
        Some(confidence) => serde_json::Value::from(
            confidence
                .to_device(&device)
                .and_then(|tensor| tensor.to_dtype(candle_core::DType::F32))
                .and_then(|tensor| tensor.contiguous())
                .and_then(|tensor| tensor.to_vec2::<f32>())
                .map_err(|error| {
                    anyhow::anyhow!("failed to read VGGT track confidence: {error}")
                })?,
        ),
        None => serde_json::Value::Null,
    };
    Ok(serde_json::json!({
        "coords": coords,
        "visibility": visibility,
        "confidence": confidence,
    }))
}

/// Flatten a tensor to a CPU f32 vector.
fn vggt_cpu_f32(tensor: &candle_core::Tensor, name: &str) -> anyhow::Result<Vec<f32>> {
    tensor
        .to_device(&candle_core::Device::Cpu)
        .and_then(|tensor| tensor.to_dtype(candle_core::DType::F32))
        .and_then(|tensor| tensor.contiguous())
        .and_then(|tensor| tensor.flatten_all())
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(|error| anyhow::anyhow!("failed to read VGGT {name}: {error}"))
}

/// Drop a leading singleton batch dim (`[1, S, ...]` -> `[S, ...]`).
fn vggt_squeeze_batch(
    tensor: &candle_core::Tensor,
    name: &str,
) -> anyhow::Result<candle_core::Tensor> {
    let dims = tensor.dims();
    if dims.len() >= 4 && dims.first() == Some(&1) {
        tensor
            .squeeze(0)
            .map_err(|error| anyhow::anyhow!("failed to squeeze VGGT {name} batch dim: {error}"))
    } else {
        Ok(tensor.clone())
    }
}

/// Depth-unprojection point cloud: `world_points_from_depth()` gated by the
/// strict `depth_conf > conf_threshold` mask, with colors from the prepared
/// RGB frames (matching the point-head PLY export's gate semantics).
fn vggt_depth_point_cloud(
    output: &VggtInferenceOutput,
    conf_threshold: f32,
) -> anyhow::Result<VggtPointCloudCpu> {
    let world = output
        .world_points_from_depth()
        .map_err(anyhow::Error::from)?;
    let dims = world.dims().to_vec();
    if dims.len() != 4 || dims[3] != 3 {
        anyhow::bail!("VGGT depth unprojection returned shape {dims:?}, expected [S, H, W, 3]");
    }
    let (frames, height, width) = (dims[0], dims[1], dims[2]);
    let points = vggt_cpu_f32(&world, "depth world points")?;
    let depth_conf = vggt_cpu_f32(
        &vggt_squeeze_batch(&output.depth_conf, "depth_conf")?,
        "depth confidence",
    )?;
    let rgb = vggt_cpu_f32(
        &output
            .rgb_frames
            .permute((0, 2, 3, 1))
            .map_err(|error| anyhow::anyhow!("failed to permute VGGT rgb frames: {error}"))?,
        "rgb frames",
    )?;
    if depth_conf.len() != frames * height * width || rgb.len() != points.len() {
        anyhow::bail!(
            "VGGT depth point cloud inputs disagree: {} conf, {} rgb, {} world entries",
            depth_conf.len(),
            rgb.len(),
            points.len()
        );
    }

    let vertex_count = depth_conf
        .iter()
        .filter(|&&conf| conf > conf_threshold)
        .count();
    let mut filtered_points = Vec::with_capacity(vertex_count * 3);
    let mut filtered_colors = Vec::with_capacity(vertex_count * 3);
    for (index, &conf) in depth_conf.iter().enumerate() {
        if conf <= conf_threshold {
            continue;
        }
        let offset = index * 3;
        filtered_points.extend_from_slice(&points[offset..offset + 3]);
        filtered_colors.extend(
            rgb[offset..offset + 3]
                .iter()
                .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8),
        );
    }
    Ok(VggtPointCloudCpu {
        points: filtered_points,
        colors: filtered_colors,
        vertex_count,
    })
}

/// One colored vertex per query: the frame-0 world point sampled at the query
/// pixel (nearest) with its prepared-frame RGB color.
fn vggt_query_point_cloud(
    output: &VggtInferenceOutput,
    queries: &[VggtTrackQuery],
) -> anyhow::Result<VggtPointCloudCpu> {
    let dims = output.world_points.dims().to_vec();
    let (frames, height, width) = match dims.len() {
        5 if dims[0] == 1 && dims[4] == 3 => (dims[1], dims[2], dims[3]),
        4 if dims[3] == 3 => (dims[0], dims[1], dims[2]),
        _ => anyhow::bail!(
            "VGGT world_points shape {dims:?} must be [1,S,H,W,3] or [S,H,W,3] for query points"
        ),
    };
    if frames == 0 || height == 0 || width == 0 {
        anyhow::bail!("VGGT world_points grid {frames}x{height}x{width} is empty");
    }
    let points = vggt_cpu_f32(&output.world_points, "world points")?;
    let rgb = vggt_cpu_f32(&output.rgb_frames, "rgb frames")?;
    let plane = height * width;
    if points.len() != frames * plane * 3 || rgb.len() != frames * 3 * plane {
        anyhow::bail!(
            "VGGT query point cloud inputs disagree: {} world, {} rgb, grid {frames}x{height}x{width}",
            points.len(),
            rgb.len()
        );
    }

    let mut cloud = VggtPointCloudCpu {
        points: Vec::with_capacity(queries.len() * 3),
        colors: Vec::with_capacity(queries.len() * 3),
        vertex_count: queries.len(),
    };
    for query in queries {
        let frame = query.frame.min(frames - 1);
        let px = (query.xy[0].round().clamp(0.0, (width - 1) as f32)) as usize;
        let py = (query.xy[1].round().clamp(0.0, (height - 1) as f32)) as usize;
        let world_offset = (frame * plane + py * width + px) * 3;
        cloud
            .points
            .extend_from_slice(&points[world_offset..world_offset + 3]);
        let rgb_offset = frame * 3 * plane + py * width + px;
        for channel in 0..3 {
            let value = rgb[rgb_offset + channel * plane];
            cloud
                .colors
                .push((value.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    Ok(cloud)
}

/// VGGT run surface: point/depth/camera inference with optional COLMAP
/// feedforward export (`--colmap-dir`), depth-unprojected point cloud
/// (`--depth-points`) and explicit-query tracking (`--track-queries`, which
/// runs a single pass over all frames because the track head requires the
/// query frame first; `--batch-size` chunking > 1 is rejected in that case).
fn run_vggt(
    model_assets: ModelAssetOptions,
    args: &RunArgs,
    device: &candle_core::Device,
) -> anyhow::Result<()> {
    let source = required_source(args, "vggt")?;
    let vggt = &args.vggt;
    let options = vggt_run_options_from_args(args)?;
    let track_queries = match vggt.track_queries.as_deref() {
        Some(json) => Some(parse_vggt_track_queries(json)?),
        None => None,
    };
    if track_queries.is_some() && vggt.track_iters == 0 {
        anyhow::bail!("`--track-iters` must be at least 1");
    }
    if track_queries.is_some() && vggt.batch_size > 1 {
        anyhow::bail!(
            "`--track-queries` runs a single pass over all frames (the track head requires the query frame first); `--batch-size` chunking greater than 1 cannot be combined with tracking"
        );
    }
    if vggt.track_output.is_some() && track_queries.is_none() {
        anyhow::bail!("`--track-output` requires `--track-queries`");
    }
    if vggt.track_points_ply.is_some() && track_queries.is_none() {
        anyhow::bail!("`--track-points-ply` requires `--track-queries`");
    }
    let track_output = match &vggt.track_output {
        Some(path) => Some(path.clone()),
        None if track_queries.is_some() => Some(vggt_default_track_output(source)),
        None => None,
    };

    let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
    let t0 = std::time::Instant::now();
    let pipeline = VggtPipeline::load_with_options(model_assets, options.clone())?;
    if stage_time {
        eprintln!("[stage] load: {:.2}s", t0.elapsed().as_secs_f64());
    }
    let t1 = std::time::Instant::now();
    let prepared = pipeline.prepare_inputs_from_path_with_mode(
        source,
        options.interval,
        options.preprocess_mode,
        device,
    )?;
    let (inference, track) = match &track_queries {
        Some(queries) => {
            let (inference, track) =
                pipeline.run_with_tracks(&prepared, queries, Some(vggt.track_iters))?;
            (inference, Some(track))
        }
        None => (
            pipeline.infer_prepared_with_options(&prepared, &options)?,
            None,
        ),
    };
    if stage_time {
        eprintln!("[stage] infer: {:.2}s", t1.elapsed().as_secs_f64());
    }

    let t2 = std::time::Instant::now();
    if let Some(npz_path) = args.dump_npz.as_deref() {
        pipeline
            .export_inference_npz(&inference, npz_path)
            .map_err(anyhow::Error::from)?;
    }
    if let Some(colmap_dir) = vggt.colmap_dir.as_deref() {
        let mut image_names = vggt_colmap_image_names(source, prepared.interval);
        if image_names.len() != prepared.original_coords.len() {
            // Frame names that disagree with the prepared frame order would
            // label the wrong cameras; fall back to `image_N` names.
            image_names.clear();
        }
        let stats = write_feedforward_colmap(
            colmap_dir,
            &inference,
            &prepared,
            &image_names,
            &vggt_colmap_options_from_args(vggt),
        )
        .map_err(anyhow::Error::from)?;
        eprintln!(
            "[colmap] cameras={} images={} points3d={} point2d={} conf_candidates={} -> {}",
            stats.camera_count,
            stats.image_count,
            stats.point3d_count,
            stats.point2d_count,
            stats.confidence_candidates,
            colmap_dir.display()
        );
    }
    if let Some(depth_path) = vggt.depth_points.as_deref() {
        let point_cloud = vggt_depth_point_cloud(&inference, vggt.depth_conf_threshold)?;
        pipeline
            .export
            .write_point_cloud(&point_cloud, depth_path)
            .map_err(anyhow::Error::from)?;
        eprintln!(
            "[depth-points] {} points -> {}",
            point_cloud.vertex_count,
            depth_path.display()
        );
    }
    if let Some(track_path) = track_output.as_deref() {
        let track = track.as_ref().expect("tracks exist when queries are set");
        let json = vggt_track_result_to_json(track)?;
        std::fs::write(track_path, serde_json::to_string_pretty(&json)?).map_err(|source| {
            anyhow::anyhow!(
                "failed to write VGGT tracks to {}: {source}",
                track_path.display()
            )
        })?;
        eprintln!(
            "[tracks] {} queries x {} frames, iters={} -> {}",
            track_queries.as_ref().expect("queries are set").len(),
            track.coords.dims().first().copied().unwrap_or(0),
            vggt.track_iters,
            track_path.display()
        );
    }
    if let Some(query_ply) = vggt.track_points_ply.as_deref() {
        let queries = track_queries.as_ref().expect("queries are set");
        let point_cloud = vggt_query_point_cloud(&inference, queries)?;
        pipeline
            .export
            .write_point_cloud(&point_cloud, query_ply)
            .map_err(anyhow::Error::from)?;
        eprintln!(
            "[track-points] {} query points -> {}",
            point_cloud.vertex_count,
            query_ply.display()
        );
    }
    pipeline.export_point_cloud_with_threshold(
        &inference,
        &args.output,
        args.conf_threshold.unwrap_or(1.0),
    )?;
    if stage_time {
        eprintln!("[stage] export: {:.2}s", t2.elapsed().as_secs_f64());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_pi3x(
    model_assets: ModelAssetOptions,
    source: &Path,
    conditions: Option<&Path>,
    interval: Option<usize>,
    vo: bool,
    chunk_size: Option<usize>,
    overlap: Option<usize>,
    conf_threshold: Option<f32>,
    inject_condition: &[InjectCondition],
    output: &Path,
    device: &candle_core::Device,
) -> anyhow::Result<()> {
    let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
    if vo {
        let inject_conditions = Pi3xInjectConditions {
            pose: inject_condition
                .iter()
                .any(|item| matches!(item, InjectCondition::Pose)),
            depth: inject_condition
                .iter()
                .any(|item| matches!(item, InjectCondition::Depth)),
            ray: inject_condition
                .iter()
                .any(|item| matches!(item, InjectCondition::Ray | InjectCondition::Intrinsic)),
        };
        let t0 = std::time::Instant::now();
        let pipeline = Pi3xVoPipeline::load(model_assets)?;
        if stage_time {
            eprintln!("[stage] load: {:.2}s", t0.elapsed().as_secs_f64());
        }
        let t1 = std::time::Instant::now();
        let inference = pipeline.infer_from_path(
            source,
            interval,
            chunk_size,
            overlap,
            conf_threshold,
            inject_conditions,
            device,
        )?;
        if stage_time {
            eprintln!("[stage] infer: {:.2}s", t1.elapsed().as_secs_f64());
        }
        let t2 = std::time::Instant::now();
        pipeline.export_point_cloud(&inference, output)?;
        if stage_time {
            eprintln!("[stage] export: {:.2}s", t2.elapsed().as_secs_f64());
        }
    } else {
        let t0 = std::time::Instant::now();
        let pipeline = Pi3xPipeline::load(model_assets)?;
        if stage_time {
            eprintln!("[stage] load: {:.2}s", t0.elapsed().as_secs_f64());
        }
        let t1 = std::time::Instant::now();
        let inference = pipeline.infer_from_path(source, conditions, interval, device)?;
        if stage_time {
            eprintln!("[stage] infer: {:.2}s", t1.elapsed().as_secs_f64());
        }
        let t2 = std::time::Instant::now();
        pipeline.export_point_cloud(&inference, output)?;
        if stage_time {
            eprintln!("[stage] export: {:.2}s", t2.elapsed().as_secs_f64());
        }
    }
    Ok(())
}

fn run_triposr(
    model_assets: ModelAssetOptions,
    source: &Path,
    mc_resolution: u32,
    mc_threshold: f32,
    output: &Path,
    device: &candle_core::Device,
) -> anyhow::Result<()> {
    anyhow::ensure!(mc_resolution >= 2, "--mc-resolution must be at least 2");
    let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
    let t0 = std::time::Instant::now();
    let pipeline = TripoSrPipeline::load(model_assets)?;
    if stage_time {
        eprintln!("[stage] load: {:.2}s", t0.elapsed().as_secs_f64());
    }
    let t1 = std::time::Instant::now();
    let inference = pipeline.infer_from_path(source, device)?;
    if stage_time {
        eprintln!("[stage] infer: {:.2}s", t1.elapsed().as_secs_f64());
    }
    let t2 = std::time::Instant::now();
    let mesh = pipeline.extract_mesh(&inference.scene_codes, mc_resolution, mc_threshold, 8192)?;
    if stage_time {
        eprintln!("[stage] mesh: {:.2}s", t2.elapsed().as_secs_f64());
    }
    let t3 = std::time::Instant::now();
    pipeline.export_mesh(&mesh, output)?;
    if stage_time {
        eprintln!("[stage] export: {:.2}s", t3.elapsed().as_secs_f64());
    }
    Ok(())
}

/// TripoSG input selection: the image-to-3D path (`--source`) or the
/// reference VAE surface encode path (`--surface-input`).
enum TripoSgRunInput<'a> {
    Image(&'a Path),
    Surface(&'a Path),
}

fn run_triposg(
    model_assets: ModelAssetOptions,
    rmbg_model_path: Option<PathBuf>,
    input: TripoSgRunInput<'_>,
    options: TripoSgRunOptions,
    output: &Path,
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> anyhow::Result<()> {
    options.validate().map_err(anyhow::Error::from)?;
    let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
    let t0 = std::time::Instant::now();
    // `--dtype` selects the pipeline load dtype (default F32 = the reference
    // mode); an unsupported dtype/device combination surfaces the core error
    // verbatim (no silent fallback).
    let pipeline = TripoSgPipeline::load_with_rmbg_and_dtype(model_assets, rmbg_model_path, dtype)?;
    if stage_time {
        eprintln!("[stage] load: {:.2}s", t0.elapsed().as_secs_f64());
    }
    let t1 = std::time::Instant::now();
    // `--surface-input` is the reference `scripts/inference_vae.py` path: the
    // VAE encoder produces the latents directly, so preprocess/DINO/denoise
    // are skipped. Clap guarantees exactly one of source/surface-input.
    let is_surface = matches!(input, TripoSgRunInput::Surface(_));
    let latents = match input {
        TripoSgRunInput::Surface(surface_path) => {
            pipeline.encode_surface_from_path(surface_path, device, &options)?
        }
        TripoSgRunInput::Image(source) => {
            pipeline.infer_from_path(source, device, &options)?.latents
        }
    };
    if stage_time {
        let label = if is_surface {
            "surface encode"
        } else {
            "infer"
        };
        eprintln!("[stage] {label}: {:.2}s", t1.elapsed().as_secs_f64());
    }
    let t2 = std::time::Instant::now();
    let mesh = pipeline.extract_mesh(&latents, &options)?;
    if stage_time {
        eprintln!("[stage] mesh: {:.2}s", t2.elapsed().as_secs_f64());
    }
    // `extract_mesh` already returned the post-simplified buffers when
    // `--faces` triggered the merge-close + quadric pass, so these counts
    // (and the mesh written by `export_mesh` below) reflect the
    // simplification.
    eprintln!(
        "[mesh] triposg: {} vertices, {} faces{}",
        mesh.vertices.dims()[0],
        mesh.faces.dims()[0],
        match options.faces {
            Some(faces) => format!(" (target {faces})"),
            None => String::new(),
        }
    );
    let t3 = std::time::Instant::now();
    pipeline.export_mesh(&mesh, output)?;
    if stage_time {
        eprintln!("[stage] export: {:.2}s", t3.elapsed().as_secs_f64());
    }
    Ok(())
}

/// Load the TRELLIS.2 pipeline from `--model-path`. A raw TRELLIS.2 checkpoint
/// directory (`pipeline.json` + `ckpts/`, i.e. the upstream
/// `models/TRELLIS.2-4B` layout) uses the per-stage raw loader; anything else
/// (canonical package dir, GGUF, or HF cache auto-download) goes through the
/// canonical weights path. `rmbg_model_path` is the separate RMBG-2.0
/// canonical package/GGUF used for RGB inputs.
fn load_trellis2_pipeline(
    model_assets: &ModelAssetOptions,
    rmbg_model_path: Option<&Path>,
    device: &candle_core::Device,
) -> anyhow::Result<Trellis2Pipeline> {
    if let Some(dir) = &model_assets.canonical_dir {
        let looks_raw =
            dir.is_dir() && (dir.join("pipeline.json").is_file() || dir.join("ckpts").is_dir());
        if looks_raw {
            return Trellis2Pipeline::load_from_raw_dir_with_rmbg(dir, device, rmbg_model_path)
                .map_err(anyhow::Error::from);
        }
    }
    Trellis2Pipeline::load_with_rmbg(model_assets.clone(), rmbg_model_path, device)
        .map_err(anyhow::Error::from)
}

/// Whether this sample writes the upstream-parity textured GLB: the run's
/// texture-export stage produced a textured atlas mesh (`Trellis2Stages::
/// textured`) AND the sample output extension selects the GLB writer
/// (case-insensitive, same resolution as `MeshFormat::from_path`). A
/// textured atlas on a `.obj`/`.ply`/`.stl` output keeps the plain mesh
/// behavior — the embedded-PNG textured GLB has no OBJ/PLY counterpart.
fn should_write_textured_glb(path: &Path, has_textured_mesh: bool) -> bool {
    has_textured_mesh
        && matches!(
            lux3d_core::export::MeshFormat::from_path(path),
            Ok(lux3d_core::export::MeshFormat::Glb)
        )
}

/// The `<parent>/<stem>` prefix the PBR sidecar writer extends into
/// `<stem>.pbrvox.bin` / `<stem>.pbr.json` (the same stem rule
/// `Trellis2ExportStage::write_mesh_with_pbr` applies internally).
fn pbr_sidecar_stem(path: &Path) -> PathBuf {
    path.parent().unwrap_or_else(|| Path::new(".")).join(
        path.file_stem()
            .unwrap_or_else(|| std::ffi::OsStr::new("mesh")),
    )
}

/// TRELLIS.2 texturing mode (`--texture-mesh`): texture an existing mesh
/// from the condition image (upstream `Trellis2TexturingPipeline.run`).
fn run_trellis2_texturing(
    model_assets: ModelAssetOptions,
    rmbg_model_path: Option<PathBuf>,
    source: &Path,
    mesh_path: &Path,
    options: &Trellis2TexturingOptions,
    output: &Path,
    device: &candle_core::Device,
) -> anyhow::Result<()> {
    let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
    let t0 = std::time::Instant::now();
    let imported = load_mesh(mesh_path)?;
    if imported.faces.is_empty() {
        anyhow::bail!(
            "`--texture-mesh` input `{}` contains no triangles",
            mesh_path.display()
        );
    }
    if stage_time {
        eprintln!(
            "[stage] import: {:.2}s ({} vertices, {} faces)",
            t0.elapsed().as_secs_f64(),
            imported.vertices.len() / 3,
            imported.faces.len() / 3
        );
    }
    let pipeline = load_trellis2_pipeline(&model_assets, rmbg_model_path.as_deref(), device)?;
    let t1 = std::time::Instant::now();
    let textured = pipeline
        .texture_mesh_from_path(&imported.vertices, &imported.faces, source, options, device)
        .map_err(anyhow::Error::from)?;
    if stage_time {
        eprintln!("[stage] texture: {:.2}s", t1.elapsed().as_secs_f64());
    }
    let t2 = std::time::Instant::now();
    pipeline
        .export
        .write_textured_glb(&textured.mesh, output)
        .map_err(anyhow::Error::from)?;
    if stage_time {
        eprintln!("[stage] export: {:.2}s", t2.elapsed().as_secs_f64());
    }
    let bake = &textured.mesh.texture;
    eprintln!(
        "[texture] {} vertices, {} faces -> {} ({} charts, {} atlas(es), {} texels, {} valid)",
        imported.vertices.len() / 3,
        imported.faces.len() / 3,
        output.display(),
        textured.mesh.chart_count,
        textured.mesh.atlas_count,
        bake.size,
        bake.stats.valid_texels
    );
    Ok(())
}

fn run_trellis2(
    model_assets: ModelAssetOptions,
    rmbg_model_path: Option<PathBuf>,
    source: &Path,
    options: Trellis2RunOptions,
    output: &Path,
    return_latent: Option<&Path>,
    device: &candle_core::Device,
) -> anyhow::Result<()> {
    let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
    let t0 = std::time::Instant::now();
    let pipeline = load_trellis2_pipeline(&model_assets, rmbg_model_path.as_deref(), device)?;
    if stage_time {
        eprintln!("[stage] load: {:.2}s", t0.elapsed().as_secs_f64());
    }
    let t1 = std::time::Instant::now();
    // num_samples = 1 runs the single-sample surface (bit-identical to the
    // pre-num_samples behavior); > 1 runs the sequential seed-offset
    // emulation and writes one output per sample with a `_i` suffix.
    let stages_list = if options.num_samples == 1 {
        vec![pipeline
            .run_from_path_with_stages(source, device, &options)
            .map_err(anyhow::Error::from)?]
    } else {
        pipeline
            .run_many_from_path(source, device, &options)
            .map_err(anyhow::Error::from)?
    };
    if stage_time {
        eprintln!("[stage] infer: {:.2}s", t1.elapsed().as_secs_f64());
    }
    let t2 = std::time::Instant::now();
    let export = lux3d_core::export::Trellis2ExportStage;
    for (index, stages) in stages_list.iter().enumerate() {
        let sample_output = sample_output_path(output, index, stages_list.len());
        let mesh = stages
            .mesh
            .as_ref()
            .expect("trellis2 run always produces a mesh");
        let textured = stages.textured.as_ref();
        if should_write_textured_glb(&sample_output, textured.is_some()) {
            // Upstream `to_glb` textured export: embedded base-color /
            // metallic-roughness PNGs, TEXCOORD_0 UVs, converted normals and
            // the glTF axis conversion. The plain vertex-color mesh writer
            // does NOT run over it; the PBR voxel sidecar pair still records
            // the per-voxel attributes next to the GLB.
            let textured = textured.expect("textured mesh checked above");
            export
                .write_textured_glb(textured, &sample_output)
                .map_err(anyhow::Error::from)?;
            eprintln!("[export] textured glb: {}", sample_output.display());
            let stats = &textured.texture.stats;
            eprintln!(
                "[bake] charts {} atlases {} atlas_size {} texture_size {} valid {}/{} texels inpainted {}",
                textured.chart_count,
                textured.atlas_count,
                textured.atlas_size,
                textured.texture.size,
                stats.valid_texels,
                stats.texel_count,
                stats.inpainted_texels,
            );
            if options.with_texture
                && mesh.voxel_attr_channels == lux3d_core::export::pbrvox::CHANNELS
            {
                let sidecar = export
                    .write_pbr_vox(mesh, &pbr_sidecar_stem(&sample_output))
                    .map_err(anyhow::Error::from)?;
                eprintln!("[export] pbr sidecar: {}", sidecar.display());
            }
        } else {
            if textured.is_some() {
                eprintln!(
                    "[export] textured glb requires a .glb output; {} keeps the vertex-color mesh",
                    sample_output.display()
                );
            }
            if options.with_texture
                && mesh.voxel_attr_channels == lux3d_core::export::pbrvox::CHANNELS
            {
                let sidecar = pipeline
                    .export_mesh_with_pbr(&mesh, &sample_output)
                    .map_err(anyhow::Error::from)?;
                eprintln!("[export] pbr sidecar: {}", sidecar.display());
            } else {
                pipeline
                    .export_mesh(&mesh, &sample_output)
                    .map_err(anyhow::Error::from)?;
            }
        }
        if let Some(latent_path) = return_latent {
            let sample_latent = sample_output_path(latent_path, index, stages_list.len());
            stages
                .write_latent_npz(&sample_latent, options.pipeline_type)
                .map_err(anyhow::Error::from)?;
            eprintln!("[export] latents: {}", sample_latent.display());
        }
        if stages_list.len() > 1 {
            println!("[export] sample {index}: {}", sample_output.display());
        }
    }
    if stage_time {
        eprintln!("[stage] export: {:.2}s", t2.elapsed().as_secs_f64());
    }
    Ok(())
}

fn normalizer_script(repo_root: &std::path::Path) -> anyhow::Result<PathBuf> {
    let candidates = [
        repo_root
            .join("tools")
            .join("python_baseline")
            .join("normalize_weights.py"),
        repo_root
            .join("candle-3d")
            .join("tools")
            .join("python_baseline")
            .join("normalize_weights.py"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("tools")
            .join("python_baseline")
            .join("normalize_weights.py"),
        repo_root
            .join("3d-rs")
            .join("tools")
            .join("python_baseline")
            .join("normalize_weights.py"),
    ];
    for candidate in candidates {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    anyhow::bail!(
        "missing normalizer script `tools/python_baseline/normalize_weights.py` under `{}`",
        repo_root.display()
    )
}

fn python_executable(repo_root: &std::path::Path, family: ModelFamily) -> PathBuf {
    if let Some(override_path) = std::env::var_os("LUX3D_PYTHON") {
        return PathBuf::from(override_path);
    }
    let env_root = repo_root
        .join("3d")
        .join("_generated")
        .join("python-envs")
        .join(family.as_str());
    for candidate in [
        env_root.join("Scripts").join("python.exe"),
        env_root.join("bin").join("python"),
    ] {
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from(if cfg!(windows) { "python" } else { "python3" })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::{fs, path::Path};

    use clap::Parser;

    use super::{inspect_model, normalize_weights, run_model, Cli, Command, RunArgs};
    use lux3d_core::{
        test_support::{
            canonical_package_dir, resolve_raw_model_dir_for_tests, runtime_root, GpuTestLock,
        },
        ModelFamily,
    };

    fn repo_root() -> PathBuf {
        runtime_root()
    }

    fn temp_output_dir(family: ModelFamily) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("current time")
            .as_nanos();
        std::env::temp_dir().join(format!("lux3d-cli-{}-{unique}", family.as_str()))
    }

    fn cleanup_dir(path: &Path) {
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn routes_inspect_pi3_family() {
        let cli = Cli::parse_from([
            "lux3d",
            "inspect",
            "--repo-root",
            repo_root().to_string_lossy().as_ref(),
            "pi3",
        ]);
        assert_eq!("pi3", cli.command.family_name());
    }

    #[test]
    fn routes_inspect_vggt_family() {
        let cli = Cli::parse_from([
            "lux3d",
            "inspect",
            "--repo-root",
            repo_root().to_string_lossy().as_ref(),
            "vggt",
        ]);
        assert_eq!("vggt", cli.command.family_name());
    }

    #[test]
    fn routes_run_vggt_family() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "vggt",
            "--source",
            repo_root()
                .join("test-assets")
                .join("pi3-frames")
                .to_string_lossy()
                .as_ref(),
            "--model-path",
            canonical_package_dir(ModelFamily::Vggt)
                .to_string_lossy()
                .as_ref(),
            "--conf-threshold",
            "1.0",
            "--output",
            repo_root()
                .join("3d")
                .join("_generated")
                .join("vggt.ply")
                .to_string_lossy()
                .as_ref(),
        ]);
        assert_eq!("vggt", cli.command.family_name());
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(Some(1.0), args.conf_threshold);
        super::reject_unsupported_vggt_options(&args).expect("vggt supported options");
    }

    #[test]
    fn run_vggt_rejects_pi3x_only_options() {
        let cli = Cli::parse_from([
            "lux3d", "run", "vggt", "--source", "frames", "--vo", "--output", "out.ply",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let error = super::reject_unsupported_vggt_options(&args)
            .expect_err("--vo must be rejected for vggt");
        assert!(error.to_string().contains("--vo"));
    }

    /// The VGGT-only flags parse, map onto `VggtRunOptions` / the COLMAP
    /// options, and parse explicit track queries in both JSON shapes.
    #[test]
    fn vggt_flags_parse_and_map_to_run_options() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "vggt",
            "--source",
            "frames",
            "--interval",
            "3",
            "--preprocess-mode",
            "pad",
            "--vggt-dtype",
            "f16",
            "--batch-size",
            "4",
            "--colmap-dir",
            "sparse",
            "--colmap-conf-threshold",
            "2.5",
            "--colmap-max-points",
            "500",
            "--colmap-point-source",
            "pointmap",
            "--colmap-seed",
            "42",
            "--colmap-world-conf-threshold",
            "0.5",
            "--depth-points",
            "depth.ply",
            "--depth-conf-threshold",
            "0.75",
            "--track-queries",
            r#"[{"frame":0,"x":10.5,"y":20.25},[0,1.0,2.0]]"#,
            "--track-output",
            "tracks.json",
            "--track-iters",
            "6",
            "--track-points-ply",
            "queries.ply",
            "--output",
            "out.ply",
        ]);
        assert_eq!("vggt", cli.command.family_name());
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        super::reject_unsupported_vggt_options(&args).expect("vggt supported options");

        assert_eq!(super::VggtPreprocessKind::Pad, args.vggt.preprocess_mode);
        assert_eq!(Some(super::DTypeKind::F16), args.vggt.vggt_dtype);
        assert_eq!(4, args.vggt.batch_size);
        assert_eq!(
            Some(std::path::PathBuf::from("sparse")),
            args.vggt.colmap_dir
        );
        assert_eq!(2.5, args.vggt.colmap_conf_threshold);
        assert_eq!(500, args.vggt.colmap_max_points);
        assert_eq!(
            super::VggtColmapPointSourceKind::Pointmap,
            args.vggt.colmap_point_source
        );
        assert_eq!(Some(42), args.vggt.colmap_seed);
        assert_eq!(Some(0.5), args.vggt.colmap_world_conf_threshold);
        assert_eq!(
            Some(std::path::PathBuf::from("depth.ply")),
            args.vggt.depth_points
        );
        assert_eq!(0.75, args.vggt.depth_conf_threshold);
        assert!(args.vggt.track_queries.is_some());
        assert_eq!(
            Some(std::path::PathBuf::from("tracks.json")),
            args.vggt.track_output
        );
        assert_eq!(6, args.vggt.track_iters);
        assert_eq!(
            Some(std::path::PathBuf::from("queries.ply")),
            args.vggt.track_points_ply
        );

        let options = super::vggt_run_options_from_args(&args).expect("run options");
        assert_eq!(Some(candle_core::DType::F16), options.dtype);
        assert_eq!(
            lux3d_core::preprocess::VggtPreprocessMode::Pad,
            options.preprocess_mode
        );
        assert_eq!(Some(3), options.interval);
        assert_eq!(Some(4), options.batch_size);

        let colmap = super::vggt_colmap_options_from_args(&args.vggt);
        assert_eq!(2.5, colmap.conf_threshold);
        assert_eq!(500, colmap.max_points);
        assert_eq!(Some(0.5), colmap.confidence);
        assert_eq!(
            lux3d_core::export::colmap::VggtColmapPointSource::Pointmap,
            colmap.point_source
        );
        assert_eq!(Some(42), colmap.seed);

        let queries =
            super::parse_vggt_track_queries(args.vggt.track_queries.as_deref().expect("queries"))
                .expect("queries parse");
        assert_eq!(
            vec![
                super::VggtTrackQuery {
                    frame: 0,
                    xy: [10.5, 20.25]
                },
                super::VggtTrackQuery {
                    frame: 0,
                    xy: [1.0, 2.0]
                },
            ],
            queries
        );
    }

    /// Defaults keep the historical behavior: f32, crop mode, a single forward
    /// pass (batch-size 1) and no optional exports; the COLMAP options match
    /// the upstream demo defaults.
    #[test]
    fn vggt_flags_default_to_historical_behavior() {
        let cli = Cli::parse_from([
            "lux3d", "run", "vggt", "--source", "frames", "--output", "out.ply",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(super::VggtPreprocessKind::Crop, args.vggt.preprocess_mode);
        assert_eq!(None, args.vggt.vggt_dtype);
        assert_eq!(1, args.vggt.batch_size);
        assert_eq!(None, args.vggt.colmap_dir);
        assert_eq!(5.0, args.vggt.colmap_conf_threshold);
        assert_eq!(100_000, args.vggt.colmap_max_points);
        assert_eq!(
            super::VggtColmapPointSourceKind::Depth,
            args.vggt.colmap_point_source
        );
        assert_eq!(None, args.vggt.colmap_seed);
        assert_eq!(None, args.vggt.colmap_world_conf_threshold);
        assert_eq!(None, args.vggt.depth_points);
        assert_eq!(1.0, args.vggt.depth_conf_threshold);
        assert_eq!(None, args.vggt.track_queries);
        assert_eq!(None, args.vggt.track_output);
        assert_eq!(4, args.vggt.track_iters);
        assert_eq!(None, args.vggt.track_points_ply);

        let options = super::vggt_run_options_from_args(&args).expect("run options");
        assert_eq!(None, options.dtype);
        assert_eq!(
            lux3d_core::preprocess::VggtPreprocessMode::Crop,
            options.preprocess_mode
        );
        assert_eq!(None, options.interval);
        assert_eq!(Some(1), options.batch_size);

        let colmap = super::vggt_colmap_options_from_args(&args.vggt);
        let defaults = lux3d_core::export::VggtColmapOptions::default();
        assert_eq!(defaults.conf_threshold, colmap.conf_threshold);
        assert_eq!(defaults.max_points, colmap.max_points);
        assert_eq!(None, colmap.confidence);
        assert_eq!(defaults.point_source, colmap.point_source);
        assert_eq!(None, colmap.seed);
    }

    /// `--batch-size 0` is rejected before any weights are touched.
    #[test]
    fn vggt_batch_size_zero_is_rejected_at_dispatch() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "vggt",
            "--device",
            "cpu",
            "--source",
            "does-not-exist",
            "--batch-size",
            "0",
            "--output",
            "out.ply",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let error = run_model(args).expect_err("batch-size 0 must be rejected");
        let message = error.to_string();
        assert!(
            message.contains("`--batch-size` must be at least 1"),
            "unexpected message: {message}"
        );
    }

    /// `--track-output` / `--track-points-ply` without `--track-queries` are
    /// rejected (explicit queries are required), as is `--batch-size` chunking
    /// greater than 1 combined with tracking and `--track-iters 0`.
    #[test]
    fn vggt_track_flags_require_queries_and_single_pass() {
        let base = |extra: &[&str]| {
            let mut argv = vec![
                "lux3d",
                "run",
                "vggt",
                "--device",
                "cpu",
                "--source",
                "does-not-exist",
                "--output",
                "out.ply",
            ];
            argv.extend_from_slice(extra);
            let cli = Cli::parse_from(argv);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            args
        };

        // Output flags without queries must fail fast.
        for (flag, value) in [
            ("--track-output", "tracks.json"),
            ("--track-points-ply", "queries.ply"),
        ] {
            let args = base(&[flag, value]);
            let error = run_model(args).expect_err(&format!("{flag} requires queries"));
            assert!(
                error.to_string().contains("requires `--track-queries`"),
                "unexpected message for {flag}: {error}"
            );
        }

        // With queries present, batch chunking > 1 and zero iters are rejected
        // before the pipeline load.
        let queries: &str = r#"[{"frame":0,"x":1.0,"y":2.0}]"#;
        for (extra, expected) in [
            (
                vec!["--track-queries", queries, "--batch-size", "4"],
                "cannot be combined with tracking",
            ),
            (
                vec!["--track-queries", queries, "--track-iters", "0"],
                "`--track-iters` must be at least 1",
            ),
        ] {
            let args = base(&extra);
            let error = run_model(args).expect_err("invalid track options must be rejected");
            assert!(
                error.to_string().contains(expected),
                "unexpected message for {extra:?}: {error}"
            );
        }
    }

    /// Track queries parse from both the object and `[frame, x, y]` shapes.
    #[test]
    fn vggt_track_queries_parse_object_and_array_formats() {
        let object = super::parse_vggt_track_queries(r#"[{"frame":0,"x":5.0,"y":6.5}]"#)
            .expect("object query");
        assert_eq!(1, object.len());
        assert_eq!([5.0, 6.5], object[0].xy);
        assert_eq!(0, object[0].frame);

        let array = super::parse_vggt_track_queries("[[0, 3.25, 4.0]]").expect("array query");
        assert_eq!(1, array.len());
        assert_eq!([3.25, 4.0], array[0].xy);

        let mixed = super::parse_vggt_track_queries(r#"[{"frame":0,"x":1,"y":2},[0, 7.0, 8.0]]"#)
            .expect("mixed queries");
        assert_eq!(2, mixed.len());
        assert_eq!([1.0, 2.0], mixed[0].xy);
        assert_eq!([7.0, 8.0], mixed[1].xy);

        // Fractional pixel coordinates stay within [0, 518).
        let edge = super::parse_vggt_track_queries("[[0, 0.0, 517.9]]").expect("edge query");
        assert_eq!([0.0, 517.9], edge[0].xy);
    }

    /// Invalid query JSON is rejected with a message naming the problem.
    #[test]
    fn vggt_track_queries_reject_invalid_input() {
        let cases: [(&str, &str); 9] = [
            ("not json", "not valid JSON"),
            ("42", "must be a JSON array"),
            ("[]", "at least one query"),
            (r#"[{"frame":0,"x":1.0}]"#, "missing the `y` field"),
            (r#"[{"frame":1,"x":1.0,"y":2.0}]"#, "`frame` must be 0"),
            ("[[3, 1.0, 2.0]]", "`frame` must be 0"),
            ("[[0, -0.5, 2.0]]", "`x` must be a pixel coordinate"),
            ("[[0, 518.0, 2.0]]", "`x` must be a pixel coordinate"),
            (
                r#"[{"frame":0,"x":"a","y":2.0}]"#,
                "`x` must be a JSON number",
            ),
        ];
        for (input, expected) in cases {
            let error = super::parse_vggt_track_queries(input)
                .expect_err(&format!("`{input}` must be rejected"));
            let message = error.to_string();
            assert!(
                message.contains(expected) && message.contains("--track-queries"),
                "unexpected message for `{input}`: {message}"
            );
        }
        // Wrong-shape arrays and non-integer frames are rejected too.
        for input in ["[[0, 1.0]]", "[[0, 1.0, 2.0, 3.0]]", "[[0.5, 1.0, 2.0]]"] {
            assert!(
                super::parse_vggt_track_queries(input).is_err(),
                "`{input}` must be rejected"
            );
        }
        // The 518.0 x check fires against the x message text; make sure a
        // large y is caught as well.
        let error =
            super::parse_vggt_track_queries("[[0, 1.0, 600.0]]").expect_err("y out of range");
        assert!(
            error.to_string().contains("`y` must be a pixel coordinate"),
            "unexpected message: {error}"
        );
    }

    /// Track results serialize to `{coords, visibility, confidence}` with
    /// `confidence: null` when the head did not predict it.
    #[test]
    fn vggt_track_result_serializes_coords_visibility_confidence() {
        let device = candle_core::Device::Cpu;
        let coords = candle_core::Tensor::from_vec(
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            (2, 3, 2),
            &device,
        )
        .expect("coords");
        let visibility =
            candle_core::Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], (2, 3), &device)
                .expect("visibility");
        let confidence =
            candle_core::Tensor::from_vec(vec![0.9f32, 0.8, 0.7, 0.6, 0.5, 0.4], (2, 3), &device)
                .expect("confidence");

        let with_conf = super::vggt_track_result_to_json(&super::VggtTrackResult {
            coords: coords.clone(),
            visibility: visibility.clone(),
            confidence: Some(confidence),
        })
        .expect("json");
        // serde_json converts the f32 tensor values to f64 exactly as the
        // literal conversion below, so the expected value uses f32 literals.
        assert_eq!(
            serde_json::json!({
                "coords": [[
                    [1.0f32, 2.0f32],
                    [3.0f32, 4.0f32],
                    [5.0f32, 6.0f32]
                ], [
                    [7.0f32, 8.0f32],
                    [9.0f32, 10.0f32],
                    [11.0f32, 12.0f32]
                ]],
                "visibility": [[0.1f32, 0.2f32, 0.3f32], [0.4f32, 0.5f32, 0.6f32]],
                "confidence": [[0.9f32, 0.8f32, 0.7f32], [0.6f32, 0.5f32, 0.4f32]],
            }),
            with_conf
        );

        let without_conf = super::vggt_track_result_to_json(&super::VggtTrackResult {
            coords,
            visibility,
            confidence: None,
        })
        .expect("json");
        assert_eq!(serde_json::Value::Null, without_conf["confidence"]);
        assert_eq!(2, without_conf["coords"].as_array().expect("frames").len());
    }

    /// Every VGGT-only flag (when it differs from its default) is rejected for
    /// the other families instead of being silently ignored.
    #[test]
    fn vggt_only_flags_rejected_for_other_families() {
        let flag_options: [(&str, &[&str]); 15] = [
            ("--preprocess-mode", &["--preprocess-mode", "pad"]),
            ("--vggt-dtype", &["--vggt-dtype", "f16"]),
            ("--batch-size", &["--batch-size", "4"]),
            ("--colmap-dir", &["--colmap-dir", "sparse"]),
            (
                "--colmap-conf-threshold",
                &["--colmap-conf-threshold", "2.5"],
            ),
            ("--colmap-max-points", &["--colmap-max-points", "500"]),
            (
                "--colmap-point-source",
                &["--colmap-point-source", "pointmap"],
            ),
            ("--colmap-seed", &["--colmap-seed", "42"]),
            (
                "--colmap-world-conf-threshold",
                &["--colmap-world-conf-threshold", "0.5"],
            ),
            ("--depth-points", &["--depth-points", "depth.ply"]),
            (
                "--depth-conf-threshold",
                &["--depth-conf-threshold", "0.75"],
            ),
            (
                "--track-queries",
                &["--track-queries", r#"[{"frame":0,"x":1.0,"y":2.0}]"#],
            ),
            ("--track-output", &["--track-output", "tracks.json"]),
            ("--track-iters", &["--track-iters", "6"]),
            ("--track-points-ply", &["--track-points-ply", "queries.ply"]),
        ];
        for family in ["pi3", "pi3x", "triposr", "triposg", "trellis2"] {
            for (flag, extra) in flag_options {
                let mut argv = vec![
                    "lux3d", "run", family, "--source", "input", "--output", "out",
                ];
                argv.extend_from_slice(extra);
                let cli = Cli::parse_from(argv);
                let Command::Run(args) = cli.command else {
                    panic!("expected run command");
                };
                let error = super::reject_vggt_only_options(&args, args.family)
                    .expect_err(&format!("{flag} must be rejected for {family}"));
                let message = error.to_string();
                assert!(
                    message.contains(flag) && message.contains("is not supported"),
                    "unexpected message for {flag} on {family}: `{message}`"
                );
            }
        }
        // VGGT itself accepts every flag (defaults included).
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "vggt",
            "--source",
            "frames",
            "--preprocess-mode",
            "crop",
            "--batch-size",
            "1",
            "--track-iters",
            "4",
            "--output",
            "out.ply",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        super::reject_vggt_only_options(&args, args.family).expect("vggt accepts its own flags");
    }

    /// The default track output path uses the source stem; stem-less sources
    /// fall back to `tracks.json`.
    #[test]
    fn vggt_default_track_output_uses_source_stem() {
        assert_eq!(
            std::path::PathBuf::from("frames_tracks.json"),
            super::vggt_default_track_output(Path::new("input/frames"))
        );
        assert_eq!(
            std::path::PathBuf::from("video_tracks.json"),
            super::vggt_default_track_output(Path::new("input/video.mp4"))
        );
        assert_eq!(
            std::path::PathBuf::from("tracks.json"),
            super::vggt_default_track_output(Path::new("/"))
        );
    }

    /// COLMAP image names mirror the core preprocess listing: png/jpg/jpeg
    /// only, Python-`sorted` order by lowercased file name, interval stride;
    /// non-directory sources produce no names (fallback `image_N`).
    #[test]
    fn vggt_colmap_image_names_mirror_preprocess_listing() {
        let root = std::env::temp_dir().join(format!(
            "lux3d-cli-vggt-names-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).expect("temp dir");
        for name in ["b.jpg", "A.png", "c.txt", "D.jpeg", "e.PNG"] {
            std::fs::write(root.join(name), b"x").expect("frame file");
        }

        assert_eq!(
            vec!["A.png", "b.jpg", "D.jpeg", "e.PNG"],
            super::vggt_colmap_image_names(&root, 1)
        );
        assert_eq!(
            vec!["A.png", "D.jpeg"],
            super::vggt_colmap_image_names(&root, 2)
        );
        assert!(
            super::vggt_colmap_image_names(Path::new("does-not-exist.png"), 1).is_empty(),
            "non-directory sources have no image names"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn routes_run_triposg_family_with_reference_defaults() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "triposg",
            "--source",
            "image.png",
            "--output",
            "out.obj",
        ]);
        assert_eq!("triposg", cli.command.family_name());
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(ModelFamily::TripoSg, args.family.model_family());
        super::reject_unsupported_triposg_options(&args).expect("default triposg options");
        let options = super::triposg_run_options_from_args(&args).expect("default run options");
        assert_eq!(super::TripoSgRunOptions::default(), options);
        // New core options default to "unset" so absent flags keep the
        // pre-flag behavior bit-identical.
        assert_eq!(None, options.timesteps);
        assert_eq!(None, options.bounds);
        assert_eq!(None, options.faces);
        assert_eq!(None, args.dtype);
        assert_eq!(None, args.faces);
        assert_eq!(None, args.bounds);
        assert_eq!(None, args.timesteps);
        let defaults = super::TripoSgRunOptions::default();
        assert_eq!(
            (50usize, 7.0f32, 42u64, 2048usize, 8u32, 9u32, true, 9u32, 16384usize),
            (
                defaults.steps,
                defaults.guidance_scale,
                defaults.seed,
                defaults.num_tokens,
                defaults.dense_octree_depth,
                defaults.hierarchical_octree_depth,
                defaults.use_flash_decoder,
                defaults.flash_octree_depth,
                defaults.query_chunk_size
            )
        );
    }

    #[test]
    fn triposg_run_options_map_cli_overrides() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "triposg",
            "--source",
            "image.png",
            "--output",
            "out.obj",
            "--steps",
            "12",
            "--guidance-scale",
            "3.5",
            "--seed",
            "7",
            "--num-tokens",
            "1024",
            "--dense-octree-depth",
            "5",
            "--hierarchical-octree-depth",
            "6",
            "--flash-decoder",
            "false",
            "--flash-octree-depth",
            "7",
            "--chunk-size",
            "4096",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let options = super::triposg_run_options_from_args(&args).expect("run options");
        assert_eq!(12, options.steps);
        assert_eq!(3.5, options.guidance_scale);
        assert_eq!(7, options.seed);
        assert_eq!(1024, options.num_tokens);
        assert_eq!(5, options.dense_octree_depth);
        assert_eq!(6, options.hierarchical_octree_depth);
        assert!(!options.use_flash_decoder);
        assert_eq!(7, options.flash_octree_depth);
        assert_eq!(4096, options.query_chunk_size);
    }

    /// The new TripoSG core options flow through the CLI funnel exactly the
    /// way `run_model` passes it into `run_triposg` -> the pipeline:
    /// `--faces` -> `TripoSgRunOptions::faces`, `--dtype` -> the pipeline
    /// load dtype, `--bounds`/`--timesteps` -> the run-option overrides.
    #[test]
    fn triposg_run_options_map_new_core_flags() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "triposg",
            "--source",
            "image.png",
            "--output",
            "out.glb",
            "--faces",
            "50000",
            "--dtype",
            "f16",
            "--bounds",
            "-1.2,-1.3,-1.4,1.2,1.3,1.4",
            "--timesteps",
            "999,777,500,250,0",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        super::reject_unsupported_triposg_options(&args).expect("new triposg flags accepted");
        assert_eq!(Some(50000), args.faces);
        assert_eq!(Some(super::DTypeKind::F16), args.dtype);
        assert_eq!(Some("-1.2,-1.3,-1.4,1.2,1.3,1.4".to_string()), args.bounds);
        assert_eq!(Some("999,777,500,250,0".to_string()), args.timesteps);
        let options = super::triposg_run_options_from_args(&args).expect("run options");
        assert_eq!(Some(50000), options.faces);
        assert_eq!(Some([-1.2, -1.3, -1.4, 1.2, 1.3, 1.4]), options.bounds);
        // List order is preserved exactly as given (upstream runs the
        // schedule as-is).
        assert_eq!(
            Some(vec![999.0, 777.0, 500.0, 250.0, 0.0]),
            options.timesteps
        );
        // `--steps` stays at its default (it is ignored once the explicit
        // schedule is set) and every untouched flag keeps its default.
        assert_eq!(50, options.steps);
        assert_eq!(7.0, options.guidance_scale);
        assert_eq!(Some(super::DTypeKind::F16), args.dtype);
        assert_eq!(
            candle_core::DType::F16,
            candle_core::DType::from(args.dtype.expect("f16"))
        );
    }

    /// `--faces 0` is the documented "no simplification" value (the upstream
    /// `-1` equivalent): it maps to `options.faces = None` like the flag
    /// default, and a positive target passes through unchanged.
    #[test]
    fn triposg_faces_zero_maps_to_no_simplification() {
        for (value, expected) in [("0", None), ("1", Some(1usize)), ("50000", Some(50000))] {
            let cli = Cli::parse_from([
                "lux3d",
                "run",
                "triposg",
                "--source",
                "image.png",
                "--output",
                "out.glb",
                "--faces",
                value,
            ]);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let options = super::triposg_run_options_from_args(&args).expect("run options");
            assert_eq!(expected, options.faces, "`--faces {value}`");
            options
                .validate()
                .map_err(anyhow::Error::from)
                .expect("validated run options");
        }
    }

    /// `--bounds` accepts one scalar or exactly six comma-separated floats;
    /// any other count is rejected. Non-finite values parse (Rust `f32`
    /// accepts `nan`/`inf`) and are then rejected by the core run-option
    /// validation, as are min >= max windows.
    #[test]
    fn triposg_bounds_flag_rejects_wrong_count_and_non_finite() {
        for value in ["1,2,3", "-1,-1,-1,1,1", "1,2,3,4,5,6,7", ""] {
            let cli = Cli::parse_from([
                "lux3d",
                "run",
                "triposg",
                "--source",
                "image.png",
                "--output",
                "out.glb",
                "--bounds",
                value,
            ]);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let error = super::triposg_run_options_from_args(&args)
                .expect_err(&format!("`--bounds {value}` must be rejected"));
            let message = error.to_string();
            assert!(
                message.contains("--bounds"),
                "unexpected message for `{value}`: `{message}`"
            );
        }
        for value in ["nan,0,0,1,1,1", "-1,0,0,inf,1,1"] {
            let cli = Cli::parse_from([
                "lux3d",
                "run",
                "triposg",
                "--source",
                "image.png",
                "--output",
                "out.glb",
                "--bounds",
                value,
            ]);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let error = super::triposg_run_options_from_args(&args)
                .expect_err(&format!("non-finite `--bounds {value}` must be rejected"));
            assert!(
                error.to_string().contains("must be finite"),
                "unexpected message for `{value}`: `{error}`"
            );
        }
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "triposg",
            "--source",
            "image.png",
            "--output",
            "out.glb",
            "--bounds",
            "-1,-1,-1,1,1,-1",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let error = super::triposg_run_options_from_args(&args)
            .expect_err("min >= max bounds must be rejected");
        assert!(
            error.to_string().contains("min < max"),
            "unexpected message: `{error}`"
        );
    }

    /// A single `--bounds` scalar is the upstream symmetric-cube shorthand
    /// (`bounds=1.25` -> `[-1.25, -1.25, -1.25, 1.25, 1.25, 1.25]`).
    #[test]
    fn triposg_bounds_flag_accepts_scalar_shorthand() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "triposg",
            "--source",
            "image.png",
            "--output",
            "out.glb",
            "--bounds",
            "1.25",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let options = super::triposg_run_options_from_args(&args).expect("run options");
        assert_eq!(
            Some([-1.25, -1.25, -1.25, 1.25, 1.25, 1.25]),
            options.bounds
        );
        options
            .validate()
            .map_err(anyhow::Error::from)
            .expect("scalar bounds validate");
    }

    /// `--timesteps` preserves the given order, tolerates spaces around the
    /// commas, and fails on empty components or non-finite values (via the
    /// core validation, matching upstream's finite check).
    #[test]
    fn triposg_timesteps_flag_parses_order_and_rejects_bad_values() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "triposg",
            "--source",
            "image.png",
            "--output",
            "out.glb",
            "--timesteps",
            " 999 , 500 , 0 ",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let options = super::triposg_run_options_from_args(&args).expect("run options");
        assert_eq!(Some(vec![999.0, 500.0, 0.0]), options.timesteps);

        for value in ["999,,500", "", "abc"] {
            let cli = Cli::parse_from([
                "lux3d",
                "run",
                "triposg",
                "--source",
                "image.png",
                "--output",
                "out.glb",
                "--timesteps",
                value,
            ]);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let error = super::triposg_run_options_from_args(&args)
                .expect_err(&format!("`--timesteps {value}` must be rejected"));
            assert!(
                error.to_string().contains("--timesteps"),
                "unexpected message for `{value}`: `{error}`"
            );
        }
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "triposg",
            "--source",
            "image.png",
            "--output",
            "out.glb",
            "--timesteps",
            "999,nan,0",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let error = super::triposg_run_options_from_args(&args)
            .expect_err("non-finite timesteps must be rejected");
        assert!(
            error.to_string().contains("must be finite"),
            "unexpected message: `{error}`"
        );
    }

    /// Unknown `--dtype` names are rejected by clap with the possible values
    /// listed; the known names map onto the candle load dtypes.
    #[test]
    fn triposg_dtype_flag_rejects_unknown_names() {
        let parsed = Cli::try_parse_from([
            "lux3d",
            "run",
            "triposg",
            "--source",
            "image.png",
            "--output",
            "out.glb",
            "--dtype",
            "q8_0",
        ]);
        let error = parsed.expect_err("unknown dtype must be rejected");
        let message = error.to_string();
        assert!(
            message.contains("invalid value") && message.contains("possible values"),
            "unexpected message: `{message}`"
        );
    }

    #[test]
    fn triposg_dtype_kind_maps_to_candle_dtypes() {
        assert_eq!(
            candle_core::DType::F32,
            candle_core::DType::from(super::DTypeKind::F32)
        );
        assert_eq!(
            candle_core::DType::F16,
            candle_core::DType::from(super::DTypeKind::F16)
        );
        assert_eq!(
            candle_core::DType::BF16,
            candle_core::DType::from(super::DTypeKind::Bf16)
        );
    }

    #[test]
    fn routes_run_trellis2_family_with_reference_defaults() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--output",
            "out.glb",
        ]);
        assert_eq!("trellis2", cli.command.family_name());
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(ModelFamily::Trellis2, args.family.model_family());
        super::reject_unsupported_trellis2_options(&args).expect("default trellis2 options");
        let options =
            super::trellis2_run_options_from_args(&args).expect("default trellis2 run options");
        assert_eq!(super::Trellis2RunOptions::default(), options);
        let defaults = super::Trellis2RunOptions::default();
        assert_eq!(
            (
                42u64,
                "1024_cascade",
                12usize,
                12usize,
                12usize,
                true,
                49152usize,
                1usize
            ),
            (
                defaults.seed,
                defaults.pipeline_type.as_str(),
                defaults.ss_steps,
                defaults.shape_steps,
                defaults.tex_steps,
                defaults.with_texture,
                defaults.max_num_tokens,
                defaults.num_samples
            )
        );
        assert_eq!(None, defaults.ss_sampler);
        assert_eq!(None, defaults.shape_sampler);
        assert_eq!(None, defaults.tex_sampler);
    }

    #[test]
    fn trellis2_run_options_map_cli_overrides() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--output",
            "out.glb",
            "--pipeline-type",
            "1536_cascade",
            "--steps",
            "4",
            "--seed",
            "7",
            "--max-num-tokens",
            "8192",
            "--texture",
            "false",
            "--model-path",
            "models/TRELLIS.2-4B",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(Some(super::Family::Trellis2), Some(args.family));
        assert_eq!(Some(4), args.steps);
        assert_eq!(Some(7), args.seed);
        assert_eq!(Some(8192), args.max_num_tokens);
        assert_eq!(Some(false), args.texture);
        let options = super::trellis2_run_options_from_args(&args).expect("trellis2 run options");
        assert_eq!(7, options.seed);
        assert_eq!(
            super::Trellis2PipelineType::Cascade1536,
            options.pipeline_type
        );
        assert_eq!(4, options.ss_steps);
        assert_eq!(4, options.shape_steps);
        assert_eq!(4, options.tex_steps);
        assert!(!options.with_texture);
        assert_eq!(8192, options.max_num_tokens);
        assert_eq!(1, options.num_samples);
    }

    #[test]
    fn trellis2_pipeline_type_accepts_reference_names() {
        for (name, expected) in [
            ("512", super::Trellis2PipelineType::R512),
            ("1024", super::Trellis2PipelineType::R1024),
            ("1024_cascade", super::Trellis2PipelineType::Cascade1024),
            ("1536_cascade", super::Trellis2PipelineType::Cascade1536),
        ] {
            let cli = Cli::parse_from([
                "lux3d",
                "run",
                "trellis2",
                "--source",
                "image.png",
                "--pipeline-type",
                name,
                "--output",
                "out.glb",
            ]);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let options = super::trellis2_run_options_from_args(&args).expect("run options");
            assert_eq!(expected, options.pipeline_type, "pipeline type `{name}`");
        }
    }

    #[test]
    fn trellis2_texture_flag_defaults_to_true_without_value() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--texture",
            "--output",
            "out.glb",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(Some(true), args.texture);
        let options = super::trellis2_run_options_from_args(&args).expect("run options");
        assert!(options.with_texture);
    }

    /// Mesh post-process flags default to the upstream `to_glb` parameters
    /// with the pass enabled.
    #[test]
    fn trellis2_postprocess_flags_default_to_upstream_to_glb() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--output",
            "out.glb",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert!(!args.no_postprocess);
        assert_eq!(None, args.decimation_target);
        assert_eq!(None, args.fill_hole_perimeter);
        assert_eq!(None, args.min_component_area);
        let options = super::trellis2_run_options_from_args(&args).expect("run options");
        assert_eq!(
            super::Trellis2MeshPostprocessOptions::default(),
            options.postprocess
        );
        assert!(options.postprocess.enabled);
        assert_eq!(3e-2, options.postprocess.fill_hole_perimeter);
        assert_eq!(1_000_000, options.postprocess.decimation_target);
        assert_eq!(1e-5, options.postprocess.min_component_area);
    }

    /// `--no-postprocess` / `--decimation-target` / `--fill-hole-perimeter` /
    /// `--min-component-area` map onto the run options; a target of 0 means
    /// "skip simplification" (cleanup-only post-processing) and is valid.
    #[test]
    fn trellis2_postprocess_flags_map_cli_overrides() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--output",
            "out.glb",
            "--no-postprocess",
            "--decimation-target",
            "4096",
            "--fill-hole-perimeter",
            "0.05",
            "--min-component-area",
            "1e-4",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert!(args.no_postprocess);
        assert_eq!(Some(4096), args.decimation_target);
        assert_eq!(Some(0.05), args.fill_hole_perimeter);
        assert_eq!(Some(1e-4), args.min_component_area);
        let options = super::trellis2_run_options_from_args(&args).expect("run options");
        assert!(!options.postprocess.enabled);
        assert_eq!(4096, options.postprocess.decimation_target);
        assert_eq!(0.05, options.postprocess.fill_hole_perimeter);
        assert_eq!(1e-4, options.postprocess.min_component_area);

        // Without --no-postprocess the overrides keep the pass enabled; a
        // zero target is the documented "skip simplification" value.
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--output",
            "out.glb",
            "--decimation-target",
            "0",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let options = super::trellis2_run_options_from_args(&args).expect("run options");
        assert!(options.postprocess.enabled);
        assert_eq!(0, options.postprocess.decimation_target);
    }

    /// Invalid post-process thresholds fail before any weights load.
    /// `-`-prefixed values never reach option assembly (clap rejects them as
    /// flag-looking arguments); `nan`/`inf` parse but fail the run-option
    /// validation. Negative thresholds are covered directly by the core
    /// `Trellis2MeshPostprocessOptions::validate` unit tests.
    #[test]
    fn trellis2_postprocess_flags_reject_invalid_thresholds() {
        for (flag, value) in [
            ("--fill-hole-perimeter", "-0.1"),
            ("--min-component-area", "-1e-6"),
        ] {
            let parsed = Cli::try_parse_from([
                "lux3d",
                "run",
                "trellis2",
                "--source",
                "image.png",
                "--output",
                "out.glb",
                flag,
                value,
            ]);
            assert!(parsed.is_err(), "{flag} {value} must be rejected by clap");
        }
        for (flag, value) in [
            ("--fill-hole-perimeter", "nan"),
            ("--fill-hole-perimeter", "inf"),
            ("--min-component-area", "nan"),
            ("--min-component-area", "inf"),
        ] {
            let cli = Cli::parse_from([
                "lux3d",
                "run",
                "trellis2",
                "--source",
                "image.png",
                "--output",
                "out.glb",
                flag,
                value,
            ]);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let error = super::trellis2_run_options_from_args(&args)
                .expect_err(&format!("{flag} {value} must be rejected"));
            assert!(
                error.to_string().contains("finite non-negative"),
                "unexpected message for {flag} {value}: `{error}`"
            );
        }
    }

    /// Per-stage sampler overrides: the stage flags map onto
    /// `ss_sampler`/`shape_sampler`/`tex_sampler` and leave the legacy
    /// `*_steps` base (fed by `--steps`) untouched; `--num-samples` and
    /// `--return-latent` route through the run options / run dispatch.
    #[test]
    fn trellis2_per_stage_sampler_flags_map_to_overrides() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--output",
            "out.glb",
            "--steps",
            "9",
            "--ss-steps",
            "8",
            "--shape-guidance",
            "5.0",
            "--tex-rescale",
            "0.2",
            "--shape-interval",
            "0.4,0.9",
            "--num-samples",
            "3",
            "--return-latent",
            "latents.npz",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        super::reject_unsupported_trellis2_options(&args).expect("per-stage flags accepted");
        assert_eq!(Some(3), args.num_samples);
        let options = super::trellis2_run_options_from_args(&args).expect("run options");
        assert_eq!(3, options.num_samples);
        // Base steps stay 9 on every stage (override resolution picks the
        // per-stage steps from the overrides).
        assert_eq!(9, options.ss_steps);
        assert_eq!(9, options.shape_steps);
        assert_eq!(9, options.tex_steps);
        let ss = options.ss_sampler.expect("ss overrides");
        assert_eq!(Some(8), ss.steps);
        assert_eq!(None, ss.guidance_strength);
        let shape = options.shape_sampler.expect("shape overrides");
        assert_eq!(Some(5.0), shape.guidance_strength);
        assert_eq!(Some((0.4, 0.9)), shape.guidance_interval);
        assert_eq!(None, shape.steps);
        let tex = options.tex_sampler.expect("tex overrides");
        assert_eq!(Some(0.2), tex.guidance_rescale);
        assert_eq!(None, tex.guidance_interval);
    }

    /// Stages without flags keep `None` overrides (pipeline defaults).
    #[test]
    fn trellis2_stage_overrides_default_to_none() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--output",
            "out.glb",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let options = super::trellis2_run_options_from_args(&args).expect("run options");
        assert_eq!(None, options.ss_sampler);
        assert_eq!(None, options.shape_sampler);
        assert_eq!(None, options.tex_sampler);
    }

    #[test]
    fn trellis2_interval_flag_parses_lo_hi_and_rejects_bad_input() {
        assert_eq!(
            (0.6, 1.0),
            super::parse_interval_pair("0.6,1.0").expect("interval")
        );
        assert_eq!(
            (0.4, 0.9),
            super::parse_interval_pair("0.4, 0.9").expect("spaced interval")
        );
        assert!(super::parse_interval_pair("0.6").is_err(), "missing hi");
        assert!(super::parse_interval_pair("0.9,0.4").is_err(), "lo > hi");
        assert!(
            super::parse_interval_pair("1.4,1.9").is_err(),
            "out of range"
        );
    }

    /// Textured-GLB export flags default to the upstream `to_glb` behavior
    /// (enabled, bake 2048, unwrap on) when nothing is passed.
    #[test]
    fn trellis2_texture_export_flags_default_to_upstream_to_glb() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--output",
            "out.glb",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(None, args.texture_size);
        assert!(!args.no_texture_export);
        assert!(!args.no_uv_unwrap);
        super::reject_unsupported_trellis2_options(&args).expect("defaults accepted");
        let options = super::trellis2_run_options_from_args(&args).expect("run options");
        assert_eq!(
            super::Trellis2TextureExportOptions::default(),
            options.texture_export
        );
        assert!(options.texture_export.enabled);
        assert_eq!(2048, options.texture_export.texture_size);
        assert!(options.texture_export.uv_unwrap);
    }

    /// `--texture-size` / `--no-texture-export` / `--no-uv-unwrap` map onto
    /// `Trellis2RunOptions.texture_export`; an explicit size is kept while
    /// `--no-texture-export` disables the stage (the values are unused then).
    #[test]
    fn trellis2_texture_export_flags_map_cli_overrides() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--output",
            "out.glb",
            "--texture-size",
            "1024",
            "--no-uv-unwrap",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(Some(1024), args.texture_size);
        assert!(args.no_uv_unwrap);
        assert!(!args.no_texture_export);
        let options = super::trellis2_run_options_from_args(&args).expect("run options");
        assert!(options.texture_export.enabled);
        assert_eq!(1024, options.texture_export.texture_size);
        assert!(!options.texture_export.uv_unwrap);

        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--output",
            "out.glb",
            "--no-texture-export",
            "--texture-size",
            "8192",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert!(args.no_texture_export);
        assert_eq!(Some(8192), args.texture_size);
        let options = super::trellis2_run_options_from_args(&args).expect("run options");
        assert!(!options.texture_export.enabled);
        assert_eq!(8192, options.texture_export.texture_size);
        assert!(options.texture_export.uv_unwrap);
    }

    /// Out-of-range `--texture-size` values fail at option assembly (before
    /// any weights load); the boundary values 64 and 8192 are accepted.
    #[test]
    fn trellis2_texture_size_rejects_out_of_range_values() {
        for size in ["0", "32", "63", "8193", "16384"] {
            let cli = Cli::parse_from([
                "lux3d",
                "run",
                "trellis2",
                "--source",
                "image.png",
                "--output",
                "out.glb",
                "--texture-size",
                size,
            ]);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let error = super::trellis2_run_options_from_args(&args)
                .expect_err(&format!("--texture-size {size} must be rejected"));
            let message = error.to_string();
            assert!(
                message.contains("--texture-size") && message.contains("64..=8192"),
                "unexpected message for {size}: `{message}`"
            );
        }
        for size in ["64", "8192"] {
            let cli = Cli::parse_from([
                "lux3d",
                "run",
                "trellis2",
                "--source",
                "image.png",
                "--output",
                "out.glb",
                "--texture-size",
                size,
            ]);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let options = super::trellis2_run_options_from_args(&args).expect("run options");
            assert_eq!(
                size.parse::<usize>().expect("usize"),
                options.texture_export.texture_size
            );
        }
    }

    /// The core nvdiffrast granularity (multiple of 8) applies on top of the
    /// CLI range: an in-range non-multiple still fails validation.
    #[test]
    fn trellis2_texture_size_keeps_core_multiple_of_eight_rule() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--output",
            "out.glb",
            "--texture-size",
            "100",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let error = super::trellis2_run_options_from_args(&args)
            .expect_err("in-range non-multiple of 8 must be rejected by the core validate");
        assert!(
            error.to_string().contains("multiple of 8"),
            "unexpected message: `{error}`"
        );
    }

    /// Export-format selection: the textured GLB writer runs only for `.glb`
    /// outputs (case-insensitive) when a textured atlas mesh exists.
    #[test]
    fn should_write_textured_glb_selects_glb_outputs_with_textured_mesh() {
        assert!(super::should_write_textured_glb(Path::new("out.glb"), true));
        assert!(!super::should_write_textured_glb(
            Path::new("out.glb"),
            false
        ));
        assert!(super::should_write_textured_glb(Path::new("out.GLB"), true));
        assert!(super::should_write_textured_glb(
            Path::new("nested/out.glb"),
            true
        ));
        // Non-GLB outputs keep the plain mesh behavior even when textured.
        for path in ["out.obj", "out.ply", "out.stl", "out", "out.npz"] {
            assert!(
                !super::should_write_textured_glb(Path::new(path), true),
                "{path} must not select the textured GLB writer"
            );
        }
    }

    /// The sidecar stem keeps the parent directory and drops the extension,
    /// so `write_pbr_vox` lands `<stem>.pbrvox.bin` next to the sample output
    /// (the same stem rule `write_mesh_with_pbr` applies internally).
    #[test]
    fn pbr_sidecar_stem_matches_the_mesh_with_pbr_stem_rule() {
        assert_eq!(
            PathBuf::from("out"),
            super::pbr_sidecar_stem(Path::new("out.glb"))
        );
        assert_eq!(
            PathBuf::from("tmp/out"),
            super::pbr_sidecar_stem(Path::new("tmp/out.glb"))
        );
        assert_eq!(
            PathBuf::from("mesh"),
            super::pbr_sidecar_stem(Path::new("mesh"))
        );
        assert_eq!(
            PathBuf::from("out.latents"),
            super::pbr_sidecar_stem(Path::new("out.latents.npz"))
        );
    }

    /// The TRELLIS.2-only flags (pipeline/texture/textured-GLB/post-process/
    /// per-stage sampler/num-samples/return-latent) must be rejected for
    /// every other family instead of being silently ignored.
    #[test]
    fn trellis2_only_flags_rejected_for_other_families() {
        let flag_options: [(&str, &[&str]); 24] = [
            ("--pipeline-type", &["--pipeline-type", "512"]),
            ("--max-num-tokens", &["--max-num-tokens", "8192"]),
            ("--texture", &["--texture", "false"]),
            ("--texture-size", &["--texture-size", "1024"]),
            ("--no-texture-export", &["--no-texture-export"]),
            ("--no-uv-unwrap", &["--no-uv-unwrap"]),
            ("--no-postprocess", &["--no-postprocess"]),
            ("--decimation-target", &["--decimation-target", "4096"]),
            ("--fill-hole-perimeter", &["--fill-hole-perimeter", "0.05"]),
            ("--min-component-area", &["--min-component-area", "1e-4"]),
            ("--ss-steps", &["--ss-steps", "6"]),
            ("--shape-steps", &["--shape-steps", "6"]),
            ("--tex-steps", &["--tex-steps", "6"]),
            ("--ss-guidance", &["--ss-guidance", "4.5"]),
            ("--shape-guidance", &["--shape-guidance", "4.5"]),
            ("--tex-guidance", &["--tex-guidance", "4.5"]),
            ("--ss-rescale", &["--ss-rescale", "0.3"]),
            ("--shape-rescale", &["--shape-rescale", "0.3"]),
            ("--tex-rescale", &["--tex-rescale", "0.3"]),
            ("--ss-interval", &["--ss-interval", "0.5,0.9"]),
            ("--shape-interval", &["--shape-interval", "0.5,0.9"]),
            ("--tex-interval", &["--tex-interval", "0.5,0.9"]),
            ("--num-samples", &["--num-samples", "2"]),
            ("--return-latent", &["--return-latent", "latents.npz"]),
        ];
        for family in ["pi3", "pi3x", "triposr", "triposg", "vggt"] {
            for (flag, extra) in flag_options {
                let mut argv = vec![
                    "lux3d", "run", family, "--source", "input", "--output", "out",
                ];
                argv.extend_from_slice(extra);
                let cli = Cli::parse_from(argv);
                let Command::Run(args) = cli.command else {
                    panic!("expected run command");
                };
                let error = super::reject_trellis2_only_options(&args, args.family)
                    .expect_err(&format!("{flag} must be rejected for {family}"));
                let message = error.to_string();
                assert!(
                    message.contains(flag)
                        && message.contains(&format!("is not supported for `{family}`")),
                    "unexpected message for {flag} on {family}: `{message}`"
                );
            }
        }
    }

    /// Multi-sample output naming: single sample keeps the exact path,
    /// multiple samples insert `_i` before the extension.
    #[test]
    fn sample_output_path_inserts_index_suffix_for_multi_sample() {
        let one = Path::new("out.glb");
        assert_eq!(
            PathBuf::from("out.glb"),
            super::sample_output_path(one, 0, 1)
        );
        assert_eq!(
            PathBuf::from("out_0.glb"),
            super::sample_output_path(one, 0, 3)
        );
        assert_eq!(
            PathBuf::from("out_1.glb"),
            super::sample_output_path(one, 1, 3)
        );
        assert_eq!(
            PathBuf::from("latents_2.npz"),
            super::sample_output_path(Path::new("latents.npz"), 2, 4)
        );
        // Multi-dot names suffix before the LAST extension; dotted stems keep
        // the leading dots.
        assert_eq!(
            PathBuf::from("out.latents_0.npz"),
            super::sample_output_path(Path::new("out.latents.npz"), 0, 2)
        );
        assert_eq!(
            PathBuf::from("mesh_1"),
            super::sample_output_path(Path::new("mesh"), 1, 2)
        );
        assert_eq!(
            PathBuf::from(".hidden_0.glb"),
            super::sample_output_path(Path::new(".hidden.glb"), 0, 2)
        );
    }

    #[test]
    fn run_trellis2_rejects_foreign_family_options() {
        let flag_options: [(&str, &[&str]); 19] = [
            ("--interval", &["--interval", "2"]),
            ("--conditions", &["--conditions", "conditions.npz"]),
            ("--vo", &["--vo"]),
            ("--chunk-size", &["--chunk-size", "8"]),
            ("--overlap", &["--overlap", "4"]),
            ("--conf-threshold", &["--conf-threshold", "0.5"]),
            ("--inject-condition", &["--inject-condition", "pose"]),
            ("--mc-resolution", &["--mc-resolution", "256"]),
            ("--mc-threshold", &["--mc-threshold", "25.0"]),
            ("--guidance-scale", &["--guidance-scale", "3.5"]),
            ("--num-tokens", &["--num-tokens", "1024"]),
            ("--dense-octree-depth", &["--dense-octree-depth", "8"]),
            (
                "--hierarchical-octree-depth",
                &["--hierarchical-octree-depth", "9"],
            ),
            ("--flash-decoder", &["--flash-decoder", "false"]),
            ("--flash-octree-depth", &["--flash-octree-depth", "9"]),
            ("--faces", &["--faces", "50000"]),
            ("--dtype", &["--dtype", "f16"]),
            ("--bounds", &["--bounds", "-1,-1,-1,1,1,1"]),
            ("--timesteps", &["--timesteps", "999,500,0"]),
        ];
        for (flag, extra) in flag_options {
            let mut argv = vec![
                "lux3d",
                "run",
                "trellis2",
                "--source",
                "image.png",
                "--output",
                "out.glb",
            ];
            argv.extend_from_slice(extra);
            let cli = Cli::parse_from(argv);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let error = super::reject_unsupported_trellis2_options(&args)
                .expect_err(&format!("{flag} must be rejected for trellis2"));
            let message = error.to_string();
            assert!(
                message.contains("is not supported for `trellis2`"),
                "unexpected message for {flag}: `{message}`"
            );
            assert!(message.contains(flag), "`{flag}` missing in `{message}`");
        }
    }

    #[test]
    fn run_trellis2_accepts_rmbg_model_path_and_finds_rmbg2_sibling() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--rmbg-model-path",
            "models/RMBG-2.0",
            "--output",
            "out.glb",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        super::reject_unsupported_trellis2_options(&args).expect("rmbg accepted");
        assert_eq!(
            Some(std::path::PathBuf::from("models/RMBG-2.0")),
            super::resolve_trellis2_rmbg_model_path(&args)
        );

        let root = std::env::temp_dir().join(format!(
            "lux3d-trellis2-rmbg-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let model_dir = root.join("TRELLIS.2-4B");
        let rmbg_dir = root.join("rmbg-2.0");
        std::fs::create_dir_all(&model_dir).expect("model dir");
        std::fs::create_dir_all(&rmbg_dir).expect("rmbg dir");
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--model-path",
            model_dir.to_str().expect("utf8 model path"),
            "--output",
            "out.glb",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(
            Some(rmbg_dir.clone()),
            super::resolve_trellis2_rmbg_model_path(&args)
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn run_triposg_rejects_foreign_family_options() {
        let flag_options: [(&str, &[&str]); 8] = [
            ("--interval", &["--interval", "2"]),
            ("--conditions", &["--conditions", "conditions.npz"]),
            ("--vo", &["--vo"]),
            ("--overlap", &["--overlap", "4"]),
            ("--conf-threshold", &["--conf-threshold", "0.5"]),
            ("--inject-condition", &["--inject-condition", "pose"]),
            ("--mc-resolution", &["--mc-resolution", "256"]),
            ("--mc-threshold", &["--mc-threshold", "25.0"]),
        ];
        for (flag, extra) in flag_options {
            let mut argv = vec![
                "lux3d",
                "run",
                "triposg",
                "--source",
                "image.png",
                "--output",
                "out.obj",
            ];
            argv.extend_from_slice(extra);
            let cli = Cli::parse_from(argv);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let error = super::reject_unsupported_triposg_options(&args)
                .expect_err(&format!("{flag} must be rejected for triposg"));
            let message = error.to_string();
            assert!(
                message.contains("is not supported for `triposg`"),
                "unexpected message for {flag}: `{message}`"
            );
            assert!(message.contains(flag), "`{flag}` missing in `{message}`");
        }
    }

    #[test]
    fn triposg_only_steps_rejected_for_other_families() {
        for family in ["pi3", "pi3x", "triposr", "vggt"] {
            let cli = Cli::parse_from([
                "lux3d", "run", family, "--source", "input", "--steps", "10", "--output", "out",
            ]);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let error = super::reject_triposg_only_options(&args, args.family)
                .expect_err(&format!("--steps must be rejected for {family}"));
            let message = error.to_string();
            assert!(
                message.contains("--steps"),
                "`--steps` missing in `{message}`"
            );
            assert!(
                message.contains(&format!("is not supported for `{family}`")),
                "unexpected message: `{message}`"
            );
        }
    }

    #[test]
    fn run_triposg_surface_input_parses_without_source() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "triposg",
            "--surface-input",
            "surface.npy",
            "--output",
            "out.glb",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(args.surface_input, Some(PathBuf::from("surface.npy")));
        assert_eq!(args.source, None);
        let options = super::triposg_run_options_from_args(&args).expect("surface options");
        // `scripts/inference_vae.py:50-58` decodes with
        // `hierarchical_extract_geometry` (no flash decoder, depths 8/9).
        assert!(!options.use_flash_decoder);
        assert_eq!(options.dense_octree_depth, 8);
        assert_eq!(options.hierarchical_octree_depth, 9);
        assert_eq!(options.num_tokens, 2048);
        assert_eq!(options.seed, 42);
    }

    #[test]
    fn run_triposg_surface_input_keeps_explicit_decoder_overrides() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "triposg",
            "--surface-input",
            "surface.npy",
            "--flash-decoder",
            "true",
            "--dense-octree-depth",
            "6",
            "--hierarchical-octree-depth",
            "7",
            "--output",
            "out.glb",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let options = super::triposg_run_options_from_args(&args).expect("options");
        assert!(options.use_flash_decoder);
        assert_eq!(options.dense_octree_depth, 6);
        assert_eq!(options.hierarchical_octree_depth, 7);
    }

    #[test]
    fn run_triposg_surface_input_conflicts_with_source() {
        let error = Cli::try_parse_from([
            "lux3d",
            "run",
            "triposg",
            "--source",
            "image.png",
            "--surface-input",
            "surface.npy",
            "--output",
            "out.glb",
        ])
        .expect_err("`--source` and `--surface-input` must conflict");
        let message = error.to_string();
        assert!(
            message.contains("cannot be used with"),
            "unexpected conflict message: `{message}`"
        );
    }

    #[test]
    fn run_triposg_requires_source_or_surface_input() {
        let error = Cli::try_parse_from(["lux3d", "run", "triposg", "--output", "out.glb"])
            .expect_err("one input source is required");
        let message = error.to_string();
        assert!(
            message.contains("--source"),
            "unexpected missing-input message: `{message}`"
        );
    }

    #[test]
    fn surface_input_rejected_for_non_triposg_families() {
        for family in ["pi3", "pi3x", "triposr", "vggt"] {
            let cli = Cli::parse_from([
                "lux3d",
                "run",
                family,
                "--surface-input",
                "surface.npy",
                "--output",
                "out",
            ]);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let error = super::reject_triposg_only_options(&args, args.family)
                .expect_err(&format!("--surface-input must be rejected for {family}"));
            let message = error.to_string();
            assert!(
                message.contains("--surface-input"),
                "`--surface-input` missing in `{message}`"
            );
            assert!(
                message.contains(&format!("is not supported for `{family}`")),
                "unexpected message: `{message}`"
            );
        }

        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--surface-input",
            "surface.npy",
            "--output",
            "out.glb",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let error = super::reject_unsupported_trellis2_options(&args)
            .expect_err("--surface-input must be rejected for trellis2");
        assert!(
            error.to_string().contains("--surface-input"),
            "unexpected message: `{error}`"
        );
    }

    #[test]
    fn all_triposg_only_flags_rejected_for_triposr() {
        let flag_options: [&[&str]; 6] = [
            &["--steps", "10"],
            &["--guidance-scale", "7.0"],
            &["--seed", "1"],
            &["--num-tokens", "1024"],
            &["--dense-octree-depth", "5"],
            &["--hierarchical-octree-depth", "6"],
        ];
        for extra in flag_options {
            let mut argv = vec![
                "lux3d",
                "run",
                "triposr",
                "--source",
                "horse.png",
                "--output",
                "out.obj",
            ];
            argv.extend_from_slice(extra);
            let cli = Cli::parse_from(argv);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let error = super::reject_triposg_only_options(&args, args.family)
                .expect_err(&format!("{} must be rejected for triposr", extra[0]));
            let message = error.to_string();
            assert!(
                message.contains("is not supported for `triposr`"),
                "unexpected message: `{message}`"
            );
            assert!(
                message.contains(extra[0]),
                "`{}` missing in `{message}`",
                extra[0]
            );
        }
    }

    /// The new TripoSG-only core flags (`--faces` / `--dtype` / `--bounds` /
    /// `--timesteps`) must be rejected for every non-TripoSG family instead
    /// of being silently ignored.
    #[test]
    fn new_triposg_only_flags_rejected_for_other_families() {
        let flag_options: [&[&str]; 4] = [
            &["--faces", "50000"],
            &["--dtype", "f16"],
            &["--bounds", "-1,-1,-1,1,1,1"],
            &["--timesteps", "999,500,0"],
        ];
        for extra in flag_options {
            for family in ["pi3", "pi3x", "triposr", "vggt"] {
                let cli = Cli::parse_from({
                    let mut argv = vec![
                        "lux3d", "run", family, "--source", "input", "--output", "out",
                    ];
                    argv.extend_from_slice(extra);
                    argv
                });
                let Command::Run(args) = cli.command else {
                    panic!("expected run command");
                };
                let error = super::reject_triposg_only_options(&args, args.family)
                    .expect_err(&format!("{} must be rejected for {family}", extra[0]));
                let message = error.to_string();
                assert!(
                    message.contains(extra[0]),
                    "`{}` missing in `{message}`",
                    extra[0]
                );
                assert!(
                    message.contains(&format!("is not supported for `{family}`")),
                    "unexpected message: `{message}`"
                );
            }
        }
    }

    #[test]
    fn routes_weights_normalize_triposr_family() {
        let cli = Cli::parse_from([
            "lux3d",
            "weights",
            "normalize",
            "--repo-root",
            repo_root().to_string_lossy().as_ref(),
            "--raw-model-dir",
            r"C:\raw-models\triposr",
            "--output-dir",
            canonical_package_dir(ModelFamily::TripoSr)
                .to_string_lossy()
                .as_ref(),
            "triposr",
        ]);
        assert_eq!("triposr", cli.command.family_name());
    }

    #[test]
    fn routes_run_pi3_family() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "pi3",
            "--source",
            repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("house")
                .to_string_lossy()
                .as_ref(),
            "--model-path",
            canonical_package_dir(ModelFamily::Pi3)
                .to_string_lossy()
                .as_ref(),
            "--interval",
            "10",
            "--output",
            repo_root()
                .join("3d")
                .join("_generated")
                .join("pi3.ply")
                .to_string_lossy()
                .as_ref(),
        ]);
        assert_eq!("pi3", cli.command.family_name());
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(Some(10), args.interval);
        assert_eq!(
            Some(canonical_package_dir(ModelFamily::Pi3)),
            args.model_path
        );
    }

    #[test]
    fn routes_run_pi3x_family() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "pi3x",
            "--source",
            repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("room")
                .join("rgb")
                .to_string_lossy()
                .as_ref(),
            "--conditions",
            repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("room")
                .join("condition.npz")
                .to_string_lossy()
                .as_ref(),
            "--model-path",
            canonical_package_dir(ModelFamily::Pi3x)
                .to_string_lossy()
                .as_ref(),
            "--output",
            repo_root()
                .join("3d")
                .join("_generated")
                .join("pi3x.ply")
                .to_string_lossy()
                .as_ref(),
        ]);
        assert_eq!("pi3x", cli.command.family_name());
    }

    #[test]
    fn routes_run_pi3x_vo_inject_family() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "pi3x",
            "--source",
            repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("skating.mp4")
                .to_string_lossy()
                .as_ref(),
            "--model-path",
            canonical_package_dir(ModelFamily::Pi3x)
                .to_string_lossy()
                .as_ref(),
            "--vo",
            "--chunk-size",
            "8",
            "--overlap",
            "4",
            "--conf-threshold",
            "0.05",
            "--inject-condition",
            "pose,depth,ray",
            "--output",
            repo_root()
                .join("3d")
                .join("_generated")
                .join("pi3x-vo.ply")
                .to_string_lossy()
                .as_ref(),
        ]);
        assert_eq!("pi3x", cli.command.family_name());
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert!(args.vo);
        assert_eq!(Some(8), args.chunk_size);
        assert_eq!(Some(4), args.overlap);
        assert_eq!(Some(0.05), args.conf_threshold);
        assert_eq!(
            vec![
                super::InjectCondition::Pose,
                super::InjectCondition::Depth,
                super::InjectCondition::Ray
            ],
            args.inject_condition
        );
    }

    #[test]
    fn routes_run_pi3x_vo_inject_rejects_unknown_token() {
        let result = Cli::try_parse_from([
            "lux3d",
            "run",
            "pi3x",
            "--source",
            repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("skating.mp4")
                .to_string_lossy()
                .as_ref(),
            "--model-path",
            canonical_package_dir(ModelFamily::Pi3x)
                .to_string_lossy()
                .as_ref(),
            "--vo",
            "--inject-condition",
            "pose,unknown",
            "--output",
            repo_root()
                .join("3d")
                .join("_generated")
                .join("pi3x-vo.ply")
                .to_string_lossy()
                .as_ref(),
        ]);
        assert!(
            result.is_err(),
            "unknown inject-condition token must be rejected"
        );
    }

    #[test]
    fn renders_pi3_model_spec_as_json() {
        let json = inspect_model(repo_root(), ModelFamily::Pi3).expect("pi3 inspection");
        assert!(json.contains("\"family\": \"pi3\""));
        assert!(json.contains("\"canonical_filename\": \"model.safetensors\""));
        for forbidden in [
            "3d/models",
            "3d/canonical-weights",
            "yyfz233-Pi3",
            "yyfz233-Pi3X",
            "stabilityai-TripoSR",
        ] {
            assert!(
                !json.contains(forbidden),
                "inspect output must stay path-free, found `{forbidden}`"
            );
        }
    }

    #[test]
    fn resolves_normalizer_script_in_this_checkout() {
        let script = super::normalizer_script(&repo_root()).expect("normalizer script");
        assert!(script.is_file(), "resolved script must exist");
        assert!(
            script.ends_with("tools/python_baseline/normalize_weights.py"),
            "unexpected script path: {}",
            script.display()
        );
    }

    #[test]
    fn normalize_weights_runs_python_canonicalizer_for_pi3() {
        let raw_model_dir =
            resolve_raw_model_dir_for_tests(ModelFamily::Pi3).expect("resolve pi3 raw model dir");
        let output_dir = temp_output_dir(ModelFamily::Pi3);
        let result = normalize_weights(
            repo_root(),
            ModelFamily::Pi3,
            raw_model_dir,
            output_dir.clone(),
        )
        .expect("pi3 canonical weights");
        cleanup_dir(&output_dir);
        assert!(result.contains("\"family\": \"pi3\""));
        assert!(result.contains("\"tensor_count\": 1210"));
        assert!(result.contains("\"raw_files\": [\n    \"model.safetensors\"\n  ]"));
        assert!(result.contains("\"canonical_file\": \"model.safetensors\""));
        assert!(!result.contains("3d/models"));
    }

    #[test]
    fn run_pi3_smoke_emits_ply_output() {
        let _guard = GpuTestLock::acquire().expect("gpu test lock");
        let output = std::env::temp_dir().join(format!("lux3d-cli-pi3-{}.ply", std::process::id()));
        let args = RunArgs {
            family: super::Family::Pi3,
            device: super::DeviceBackend::Cuda,
            source: Some(
                repo_root()
                    .join("tp")
                    .join("3d")
                    .join("Pi3")
                    .join("examples")
                    .join("house"),
            ),
            surface_input: None,
            interval: None,
            conditions: None,
            vo: false,
            chunk_size: None,
            overlap: None,
            conf_threshold: None,
            inject_condition: Vec::new(),
            mc_resolution: None,
            mc_threshold: None,
            steps: None,
            guidance_scale: None,
            seed: None,
            num_tokens: None,
            dense_octree_depth: None,
            hierarchical_octree_depth: None,
            flash_decoder: None,
            flash_octree_depth: None,
            faces: None,
            dtype: None,
            bounds: None,
            timesteps: None,
            model_path: Some(canonical_package_dir(ModelFamily::Pi3)),
            rmbg_model_path: None,
            cache_dir: None,
            output: output.clone(),
            dump_npz: None,
            vggt: Default::default(),
            pipeline_type: None,
            max_num_tokens: None,
            texture: None,
            ss_steps: None,
            shape_steps: None,
            tex_steps: None,
            ss_guidance: None,
            shape_guidance: None,
            tex_guidance: None,
            ss_rescale: None,
            shape_rescale: None,
            tex_rescale: None,
            ss_interval: None,
            shape_interval: None,
            tex_interval: None,
            num_samples: None,
            return_latent: None,
            texture_size: None,
            no_texture_export: false,
            no_uv_unwrap: false,
            texture_mesh: None,
            texture_resolution: None,
            no_preprocess_mesh: false,
            no_preprocess_image: false,
            no_postprocess: false,
            decimation_target: None,
            fill_hole_perimeter: None,
            min_component_area: None,
        };

        let emitted = run_model(args).expect("pi3 cli run");
        let body = fs::read_to_string(&emitted).expect("read pi3 output");
        let _ = fs::remove_file(&emitted);

        assert!(body.starts_with("ply\nformat ascii 1.0\n"));
        assert!(body.contains("property float nx"));
    }

    #[test]
    fn run_triposr_smoke_emits_obj_output() {
        let _guard = GpuTestLock::acquire().expect("gpu test lock");
        let output =
            std::env::temp_dir().join(format!("lux3d-cli-triposr-{}.obj", std::process::id()));
        let args = RunArgs {
            family: super::Family::Triposr,
            device: super::DeviceBackend::Cuda,
            source: Some(
                repo_root()
                    .join("tp")
                    .join("3d")
                    .join("TripoSR")
                    .join("examples")
                    .join("horse.png"),
            ),
            surface_input: None,
            interval: None,
            conditions: None,
            vo: false,
            chunk_size: None,
            overlap: None,
            conf_threshold: None,
            inject_condition: Vec::new(),
            mc_resolution: Some(256),
            mc_threshold: Some(25.0),
            steps: None,
            guidance_scale: None,
            seed: None,
            num_tokens: None,
            dense_octree_depth: None,
            hierarchical_octree_depth: None,
            flash_decoder: None,
            flash_octree_depth: None,
            faces: None,
            dtype: None,
            bounds: None,
            timesteps: None,
            model_path: Some(canonical_package_dir(ModelFamily::TripoSr)),
            rmbg_model_path: None,
            cache_dir: None,
            output: output.clone(),
            dump_npz: None,
            vggt: Default::default(),
            pipeline_type: None,
            max_num_tokens: None,
            texture: None,
            ss_steps: None,
            shape_steps: None,
            tex_steps: None,
            ss_guidance: None,
            shape_guidance: None,
            tex_guidance: None,
            ss_rescale: None,
            shape_rescale: None,
            tex_rescale: None,
            ss_interval: None,
            shape_interval: None,
            tex_interval: None,
            num_samples: None,
            return_latent: None,
            texture_size: None,
            no_texture_export: false,
            no_uv_unwrap: false,
            texture_mesh: None,
            texture_resolution: None,
            no_preprocess_mesh: false,
            no_preprocess_image: false,
            no_postprocess: false,
            decimation_target: None,
            fill_hole_perimeter: None,
            min_component_area: None,
        };

        let emitted = run_model(args).expect("triposr cli run");
        let body = fs::read_to_string(&emitted).expect("read triposr output");
        let _ = fs::remove_file(&emitted);

        assert!(body.starts_with("# https://github.com/mikedh/trimesh\n"));
        assert!(body.lines().any(|line| line.starts_with("v ")));
        assert!(body.lines().any(|line| line.starts_with("f ")));
    }

    #[test]
    fn run_triposr_rejects_invalid_marching_cubes_resolution() {
        let output = std::env::temp_dir().join(format!(
            "lux3d-cli-triposr-invalid-{}.obj",
            std::process::id()
        ));
        let args = RunArgs {
            family: super::Family::Triposr,
            device: super::DeviceBackend::Cuda,
            source: Some(
                repo_root()
                    .join("tp")
                    .join("3d")
                    .join("TripoSR")
                    .join("examples")
                    .join("horse.png"),
            ),
            surface_input: None,
            interval: None,
            conditions: None,
            vo: false,
            chunk_size: None,
            overlap: None,
            conf_threshold: None,
            inject_condition: Vec::new(),
            mc_resolution: Some(1),
            mc_threshold: Some(25.0),
            steps: None,
            guidance_scale: None,
            seed: None,
            num_tokens: None,
            dense_octree_depth: None,
            hierarchical_octree_depth: None,
            flash_decoder: None,
            flash_octree_depth: None,
            faces: None,
            dtype: None,
            bounds: None,
            timesteps: None,
            model_path: Some(canonical_package_dir(ModelFamily::TripoSr)),
            rmbg_model_path: None,
            cache_dir: None,
            output,
            dump_npz: None,
            vggt: Default::default(),
            pipeline_type: None,
            max_num_tokens: None,
            texture: None,
            ss_steps: None,
            shape_steps: None,
            tex_steps: None,
            ss_guidance: None,
            shape_guidance: None,
            tex_guidance: None,
            ss_rescale: None,
            shape_rescale: None,
            tex_rescale: None,
            ss_interval: None,
            shape_interval: None,
            tex_interval: None,
            num_samples: None,
            return_latent: None,
            texture_size: None,
            no_texture_export: false,
            no_uv_unwrap: false,
            texture_mesh: None,
            texture_resolution: None,
            no_preprocess_mesh: false,
            no_preprocess_image: false,
            no_postprocess: false,
            decimation_target: None,
            fill_hole_perimeter: None,
            min_component_area: None,
        };

        let err = run_model(args).expect_err("invalid marching-cubes resolution must fail");
        assert!(err
            .to_string()
            .contains("--mc-resolution must be at least 2"));
    }

    #[test]
    fn run_pi3x_smoke_emits_ply_output() {
        let _guard = GpuTestLock::acquire().expect("gpu test lock");
        let output =
            std::env::temp_dir().join(format!("lux3d-cli-pi3x-{}.ply", std::process::id()));
        let args = RunArgs {
            family: super::Family::Pi3x,
            device: super::DeviceBackend::Cuda,
            source: Some(
                repo_root()
                    .join("tp")
                    .join("3d")
                    .join("Pi3")
                    .join("examples")
                    .join("room")
                    .join("rgb"),
            ),
            surface_input: None,
            interval: Some(2),
            conditions: Some(
                repo_root()
                    .join("tp")
                    .join("3d")
                    .join("Pi3")
                    .join("examples")
                    .join("room")
                    .join("condition.npz"),
            ),
            vo: false,
            chunk_size: None,
            overlap: None,
            conf_threshold: None,
            inject_condition: Vec::new(),
            mc_resolution: None,
            mc_threshold: None,
            steps: None,
            guidance_scale: None,
            seed: None,
            num_tokens: None,
            dense_octree_depth: None,
            hierarchical_octree_depth: None,
            flash_decoder: None,
            flash_octree_depth: None,
            faces: None,
            dtype: None,
            bounds: None,
            timesteps: None,
            model_path: Some(canonical_package_dir(ModelFamily::Pi3x)),
            rmbg_model_path: None,
            cache_dir: None,
            output: output.clone(),
            dump_npz: None,
            vggt: Default::default(),
            pipeline_type: None,
            max_num_tokens: None,
            texture: None,
            ss_steps: None,
            shape_steps: None,
            tex_steps: None,
            ss_guidance: None,
            shape_guidance: None,
            tex_guidance: None,
            ss_rescale: None,
            shape_rescale: None,
            tex_rescale: None,
            ss_interval: None,
            shape_interval: None,
            tex_interval: None,
            num_samples: None,
            return_latent: None,
            texture_size: None,
            no_texture_export: false,
            no_uv_unwrap: false,
            texture_mesh: None,
            texture_resolution: None,
            no_preprocess_mesh: false,
            no_preprocess_image: false,
            no_postprocess: false,
            decimation_target: None,
            fill_hole_perimeter: None,
            min_component_area: None,
        };

        let emitted = run_model(args).expect("pi3x cli run");
        let body = fs::read_to_string(&emitted).expect("read pi3x output");
        let _ = fs::remove_file(&emitted);

        assert!(body.starts_with("ply\nformat ascii 1.0\n"));
        assert!(body.contains("property float nx"));
    }

    #[test]
    fn run_pi3x_vo_inject_smoke_emits_ply_output() {
        let _guard = GpuTestLock::acquire().expect("gpu test lock");
        let output =
            std::env::temp_dir().join(format!("lux3d-cli-pi3x-vo-{}.ply", std::process::id()));
        let args = RunArgs {
            family: super::Family::Pi3x,
            device: super::DeviceBackend::Cuda,
            source: Some(
                repo_root()
                    .join("tp")
                    .join("3d")
                    .join("Pi3")
                    .join("examples")
                    .join("skating.mp4"),
            ),
            surface_input: None,
            interval: None,
            conditions: None,
            vo: true,
            chunk_size: Some(8),
            overlap: Some(4),
            conf_threshold: Some(0.05),
            inject_condition: vec![
                super::InjectCondition::Pose,
                super::InjectCondition::Depth,
                super::InjectCondition::Ray,
            ],
            mc_resolution: None,
            mc_threshold: None,
            steps: None,
            guidance_scale: None,
            seed: None,
            num_tokens: None,
            dense_octree_depth: None,
            hierarchical_octree_depth: None,
            flash_decoder: None,
            flash_octree_depth: None,
            faces: None,
            dtype: None,
            bounds: None,
            timesteps: None,
            model_path: Some(canonical_package_dir(ModelFamily::Pi3x)),
            rmbg_model_path: None,
            cache_dir: None,
            output: output.clone(),
            dump_npz: None,
            vggt: Default::default(),
            pipeline_type: None,
            max_num_tokens: None,
            texture: None,
            ss_steps: None,
            shape_steps: None,
            tex_steps: None,
            ss_guidance: None,
            shape_guidance: None,
            tex_guidance: None,
            ss_rescale: None,
            shape_rescale: None,
            tex_rescale: None,
            ss_interval: None,
            shape_interval: None,
            tex_interval: None,
            num_samples: None,
            return_latent: None,
            texture_size: None,
            no_texture_export: false,
            no_uv_unwrap: false,
            texture_mesh: None,
            texture_resolution: None,
            no_preprocess_mesh: false,
            no_preprocess_image: false,
            no_postprocess: false,
            decimation_target: None,
            fill_hole_perimeter: None,
            min_component_area: None,
        };

        let emitted = run_model(args).expect("pi3x vo cli run");
        let body = fs::read_to_string(&emitted).expect("read pi3x vo output");
        let _ = fs::remove_file(&emitted);

        assert!(body.starts_with("ply\nformat ascii 1.0\n"));
        assert!(body.contains("property float nx"));
    }

    /// `--texture-mesh` switches to the upstream texturing pipeline; the
    /// defaults match `Trellis2TexturingOptions::default()`.
    #[test]
    fn trellis2_texturing_mode_defaults_to_upstream_pipeline() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--output",
            "out.glb",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert!(args.texture_mesh.is_none());
        assert_eq!(None, args.texture_resolution);
        super::validate_trellis2_texturing_mode(&args).expect("default mode");
        assert_eq!(
            lux3d_core::runtime::Trellis2TexturingOptions::default(),
            super::trellis2_texturing_options_from_args(&args).expect("options")
        );
    }

    /// Texturing flags map onto the core options and the output/flag
    /// contract is enforced before any model load.
    #[test]
    fn trellis2_texturing_mode_maps_flags_and_validates_output() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--texture-mesh",
            "input.glb",
            "--texture-resolution",
            "512",
            "--texture-size",
            "1024",
            "--seed",
            "7",
            "--tex-steps",
            "3",
            "--tex-guidance",
            "1.5",
            "--tex-rescale",
            "0.25",
            "--tex-interval",
            "0.5,0.8",
            "--no-preprocess-mesh",
            "--no-preprocess-image",
            "--output",
            "out.glb",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(Some(PathBuf::from("input.glb")), args.texture_mesh);
        super::validate_trellis2_texturing_mode(&args).expect("texturing mode");
        let options = super::trellis2_texturing_options_from_args(&args).expect("options");
        assert_eq!(7, options.seed);
        assert_eq!(512, options.resolution);
        assert_eq!(1024, options.texture_size);
        assert!(!options.preprocess_mesh);
        assert!(!options.preprocess_image);
        assert_eq!(Some(3), options.tex_sampler.steps);
        assert_eq!(Some(1.5), options.tex_sampler.guidance_strength);
        assert_eq!(Some(0.25), options.tex_sampler.guidance_rescale);
        assert_eq!(Some((0.5, 0.8)), options.tex_sampler.guidance_interval);

        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--texture-mesh",
            "input.glb",
            "--output",
            "out.obj",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let error =
            super::validate_trellis2_texturing_mode(&args).expect_err("non-GLB output must fail");
        assert!(error.to_string().contains("requires a `.glb` output"));

        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--texture-mesh",
            "input.glb",
            "--texture-resolution",
            "256",
            "--output",
            "out.glb",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert!(
            super::trellis2_texturing_options_from_args(&args).is_err(),
            "unsupported resolution must fail"
        );

        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--texture-mesh",
            "input.glb",
            "--pipeline-type",
            "512",
            "--output",
            "out.glb",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let error = super::validate_trellis2_texturing_mode(&args)
            .expect_err("image-to-3D flags must be rejected");
        assert!(error.to_string().contains("--pipeline-type"));
    }
}
