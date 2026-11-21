use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

use clap::{Args, Parser, Subcommand};
use lux3d_core::export::{
    DEFAULT_CONF_THRESHOLD, DEFAULT_GLB_CONF_PERCENTILE, DEFAULT_MAX_POINTS,
    DEFAULT_MAX_REPROJ_ERROR, DEFAULT_MIN_INLIER_PER_FRAME, DEFAULT_VIS_THRESHOLD, VggtColmapMode,
    VggtColmapOptions, VggtGlbOptions, VggtGlbPredictions, VggtTrackColmapOptions,
    build_vggt_glb_scene, load_mesh, write_feedforward_colmap, write_track_colmap_from_auto,
    write_vggt_glb,
};
use lux3d_core::geometry::VggtPointCloudCpu;
use lux3d_core::preprocess::VggtPreprocessMode;
use lux3d_core::runtime::{
    DEFAULT_MAX_QUERY_PTS, DEFAULT_QUERY_FRAME_NUM, Pi3Pipeline, Pi3xInjectConditions,
    Pi3xPipeline, Pi3xVoPipeline, Trellis2MeshPostprocessOptions, Trellis2Pipeline,
    Trellis2RunOptions, Trellis2SamplerOverrides, Trellis2TextureExportOptions,
    Trellis2TexturingOptions, TripoSgPipeline, TripoSgRunOptions, TripoSgScribblePipeline,
    TripoSgScribbleRunOptions, TripoSrPipeline, VggtAutoQueryOptions, VggtInferenceOutput,
    VggtKeypointSource, VggtPipeline, VggtRunOptions, VggtTrackQuery, VggtTrackResult,
    resolve_triposg_scribble_model_source,
};
use lux3d_core::{
    ModelAssetOptions, ModelFamily, QuantizeMergeSource, QuantizeOptions, Trellis2PipelineType,
    WeightLocator, dtype_from_name, ensure_canonical_weights, quantize_auxiliary_checkpoint,
    quantize_checkpoint, quantize_checkpoint_with_merges, quantize_scribble_standalone,
    resolve_scribble_dit_source,
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
    Ovox(OvoxArgs),
}

#[derive(Debug, Clone, Args, PartialEq)]
pub struct OvoxArgs {
    #[command(subcommand)]
    pub command: OvoxCommand,
}

#[derive(Debug, Clone, Subcommand, PartialEq)]
pub enum OvoxCommand {
    #[command(
        name = "mesh2ovox",
        about = "Voxelize a textured mesh into a sparse PBR .vxz volume."
    )]
    Mesh2Ovox(Mesh2OvoxArgs),
    #[command(
        name = "ovox2mesh",
        about = "Reconstruct a .ply mesh (with base-color vertex colors) from an o_voxel volume."
    )]
    Ovox2Mesh(Ovox2MeshArgs),
    #[command(
        name = "ovox2glb",
        about = "Reconstruct, clean, UV-unwrap and bake a textured .glb from an o_voxel volume."
    )]
    Ovox2Glb(Ovox2GlbArgs),
}

#[derive(Debug, Clone, Args, PartialEq)]
pub struct Mesh2OvoxArgs {
    #[arg(
        value_name = "INPUT",
        help = "Input textured mesh: .glb (full PBR) or .obj/.ply/.stl (single default material)."
    )]
    pub input: PathBuf,

    #[arg(
        value_name = "OUTPUT",
        help = "Output sparse volume. .vxz is the official format; .npz/.ply are also accepted."
    )]
    pub output: PathBuf,

    #[arg(
        long = "grid-size",
        value_delimiter = ',',
        value_name = "N[,N,N]",
        help = "Voxel resolution per axis; one value repeats, three values set X,Y,Z. Default 512 (official RES) when neither resolution flag is given. If --grid-size is omitted and --voxel-size is supplied, the grid range is derived from the mesh bounds (the core auto-derives round(extent / voxel_size))."
    )]
    pub grid_size: Option<Vec<u32>>,

    #[arg(
        long = "voxel-size",
        value_delimiter = ',',
        value_name = "S[,S,S]",
        help = "Explicit voxel size per axis; one value repeats, three values set X,Y,Z. When set, the kernel uses it verbatim and --grid-size (or the auto grid) gives the index range."
    )]
    pub voxel_size: Vec<f32>,

    #[arg(
        long,
        value_delimiter = ',',
        value_name = "MINX,MINY,MINZ,MAXX,MAXY,MAXZ",
        help = "Voxel box in the (normalized) mesh frame. Default after normalization is [-0.5,0.5]^3; with --no-normalize the wrapper auto-pads the mesh bounds."
    )]
    pub aabb: Option<Vec<f32>>,

    #[arg(
        long = "no-normalize",
        help = "Skip the official mesh2ovox unit-cube normalization (center + 0.99999/max_extent)."
    )]
    pub no_normalize: bool,

    #[arg(
        long = "mip-level-offset",
        default_value_t = 0.0,
        help = "Texture mip-level offset (official textured_mesh_to_volumetric_attr default 0.0)."
    )]
    pub mip_level_offset: f32,

    #[arg(
        long = "face-weight",
        default_value_t = 1.0,
        help = "QEF face-term weight (official example 1.0)."
    )]
    pub face_weight: f32,

    #[arg(
        long = "boundary-weight",
        default_value_t = 0.2,
        help = "QEF boundary-term weight (official example 0.2)."
    )]
    pub boundary_weight: f32,

    #[arg(
        long = "regularization-weight",
        default_value_t = 1e-2,
        help = "QEF regularization weight (official example 1e-2)."
    )]
    pub regularization_weight: f32,

    #[arg(
        long,
        value_enum,
        default_value_t = OvoxCompressionArg::Deflate,
        help = "Compression for .vxz output: none or deflate (LZMA writing is not supported; LZMA files can be read)."
    )]
    pub compression: OvoxCompressionArg,
}

#[derive(Debug, Clone, Args, PartialEq)]
pub struct Ovox2MeshArgs {
    #[arg(value_name = "INPUT", help = "Input o_voxel volume (.vxz/.npz/.ply).")]
    pub input: PathBuf,

    #[arg(
        value_name = "OUTPUT",
        help = "Output .ply (ASCII, base-color vertex colors when present)."
    )]
    pub output: PathBuf,

    #[arg(
        long = "grid-size",
        value_delimiter = ',',
        value_name = "N[,N,N]",
        default_value = "512",
        help = "Grid resolution the asset was voxelized with (not stored in .vxz; official example uses 512)."
    )]
    pub grid_size: Vec<u32>,

    #[arg(
        long,
        value_delimiter = ',',
        value_name = "MINX,MINY,MINZ,MAXX,MAXY,MAXZ",
        default_value = "-0.5,-0.5,-0.5,0.5,0.5,0.5",
        help = "Voxel box the asset was voxelized with (not stored in .vxz)."
    )]
    pub aabb: Vec<f32>,
}

#[derive(Debug, Clone, Args, PartialEq)]
pub struct Ovox2GlbArgs {
    #[arg(value_name = "INPUT", help = "Input o_voxel volume (.vxz/.npz/.ply).")]
    pub input: PathBuf,

    #[arg(value_name = "OUTPUT", help = "Output textured .glb.")]
    pub output: PathBuf,

    #[arg(
        long = "grid-size",
        value_delimiter = ',',
        value_name = "N[,N,N]",
        default_value = "512",
        help = "Grid resolution the asset was voxelized with."
    )]
    pub grid_size: Vec<u32>,

    #[arg(
        long,
        value_delimiter = ',',
        value_name = "MINX,MINY,MINZ,MAXX,MAXY,MAXZ",
        default_value = "-0.5,-0.5,-0.5,0.5,0.5,0.5",
        help = "Voxel box the asset was voxelized with."
    )]
    pub aabb: Vec<f32>,

    #[arg(
        long = "decimation-target",
        default_value_t = 100_000,
        help = "Post-process simplification target (official ovox2glb example: 100000)."
    )]
    pub decimation_target: usize,

    #[arg(
        long = "texture-size",
        default_value_t = 2048,
        help = "Bake texture edge length (official ovox2glb example: 2048)."
    )]
    pub texture_size: usize,

    #[arg(
        long = "fill-hole-perimeter",
        default_value_t = 3e-2,
        help = "Initial/final hole-fill perimeter threshold (upstream to_glb: 3e-2)."
    )]
    pub fill_hole_perimeter: f32,

    #[arg(
        long = "min-component-area",
        default_value_t = 1e-5,
        help = "Small-connected-component removal area threshold (upstream to_glb: 1e-5)."
    )]
    pub min_component_area: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OvoxCompressionArg {
    None,
    Deflate,
}

impl From<OvoxCompressionArg> for lux3d_core::export::ovox::vxz::VxzCompression {
    fn from(value: OvoxCompressionArg) -> Self {
        match value {
            OvoxCompressionArg::None => Self::None,
            OvoxCompressionArg::Deflate => Self::Deflate,
        }
    }
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

    #[arg(
        long,
        value_name = "PATH",
        requires = "merge_prefix",
        help = "Additional safetensors file merged into the standalone GGUF, e.g. the TRELLIS.2 `shape_enc_next_dc_f16c32_fp16.safetensors` stage that is outside the canonical package."
    )]
    pub merge_source: Option<PathBuf>,

    #[arg(
        long,
        value_name = "PREFIX",
        requires = "merge_source",
        help = "Tensor-name prefix applied to `--merge-source` tensors (e.g. `shape_enc`)."
    )]
    pub merge_prefix: Option<String>,

    #[arg(
        long,
        help = "Quantize the TripoSG-scribble DiT. `--source` may be the raw `VAST-AI/TripoSG-scribble` checkpoint directory (the transformer/diffusion_pytorch_model.safetensors shard is resolved) or the shard itself; FAMILY must be `triposg`. Without `--standalone` this writes a DiT-only override (place it at `<checkpoint>/transformer/model.gguf` or pass it as `--model-path`). With `--standalone` it writes one self-contained package."
    )]
    pub scribble: bool,

    #[arg(
        long,
        requires = "scribble",
        help = "Build one standalone TripoSG-scribble GGUF containing the DINOv2 image encoder, DiT, VAE decoder, CLIP text encoder and tokenizer. `--source` must be the raw `VAST-AI/TripoSG-scribble` checkpoint directory. By default rank-2 linear weights in the DiT and VAE are quantized (the VAE Q8_0 path is measured near-lossless and much faster for SDF mesh decode); DINOv2/CLIP stay in the fallback dtype because their Q8_0 conditioning drift reaches the mesh. Pass `--prefix` to override the set."
    )]
    pub standalone: bool,
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
        help = "Input source: a frame directory (.png/.jpg/.jpeg), a video file (.mp4/.mov/.avi/.mkv), or - for `vggt` - a single image file (.png/.jpg/.jpeg/.webp/.bmp/.tif/.tiff). Repeatable for `triposg` image mode only: each `--source` adds one image to the batch (total batch = N images * `--num-shapes`). Mutually exclusive with `--surface-input`."
    )]
    pub source: Vec<PathBuf>,

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
        help = "Pi3X VO chunk size; for `triposg` this doubles as the SDF query chunk size (default 32768; the flash decoder caps it at 16384)."
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
        help = "TripoSG number of shapes per prompt (upstream `num_shapes_per_prompt`, default 1). With N > 1 the conditioning is expanded with repeat_interleave, one batched DiT denoise pass draws all samples, and one output mesh is written per sample (total batch = number of `--source` images * N) with a `_0`, `_1`, ... suffix."
    )]
    pub num_shapes: Option<usize>,

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
        default_value = "image",
        help = "TripoSG mode: `image` (DINOv2 image-to-3D diffusion, default) or `scribble` (TripoSG-scribble: CLIP prompt + scribble-image dual conditioning; requires `--prompt` and the `VAST-AI/TripoSG-scribble` raw checkpoint)."
    )]
    pub mode: TriposgMode,

    #[arg(
        long,
        help = "TripoSG-scribble text prompt (required with `--mode scribble`)."
    )]
    pub prompt: Option<String>,

    #[arg(
        long,
        help = "TripoSG-scribble `cross_attention_2_scale` for the scribble-image branch (reference `--scribble-conf` default 0.4)."
    )]
    pub scribble_conf: Option<f32>,

    #[arg(
        long,
        help = "TripoSG-scribble `cross_attention_scale` for the CLIP text branch (reference `--prompt-conf` default 1.0)."
    )]
    pub prompt_conf: Option<f32>,

    #[arg(
        long,
        value_enum,
        help = "TRELLIS.2 pipeline type: `512`, `1024`, `1024_cascade` (reference default), or `1536_cascade`."
    )]
    pub pipeline_type: Option<Trellis2PipelineKind>,

    #[arg(
        long,
        default_value_t = false,
        help = "TRELLIS.2 keep every loaded stage resident on the device for the whole run instead of the default per-stage load/drop (upstream `low_vram=False`, trellis2_image_to_3d.py:55, 107, 119-125). Speed-only: the outputs are identical to the default for the same seed. Fails with an actionable error naming this flag and the device memory limit when the resident stage set does not fit; never falls back silently."
    )]
    pub no_low_vram: bool,

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
        help = "TRELLIS.2 texturing voxelization/encode resolution: 512, 1024, or 1536 (default 1024). 512 selects tex_dit_512 and 512 DINOv3 conditioning; 1024 and 1536 use tex_dit_1024 with 1024 conditioning (1536 voxelizes/encodes the mesh at 1536)."
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
        help = "TRELLIS.2 disable the mesh post-process (upstream to_glb branches: hole filling, remesh or cleanup, decimation; default: enabled)."
    )]
    pub no_postprocess: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "TRELLIS.2 skip the upstream to_glb narrow-band dual-contouring remesh and run the previous cleanup/simplify chain instead (byte-identical to the pre-remesh pipeline). The default matches the shipped entry points (example.py:43, app.py:503): remesh on, band 1, project 0."
    )]
    pub no_remesh: bool,

    #[arg(
        long,
        help = "TRELLIS.2 remesh narrow-band half-width in grid cells (upstream remesh_band; default 1). Only used while remeshing."
    )]
    pub remesh_band: Option<f32>,

    #[arg(
        long,
        help = "TRELLIS.2 remesh project-back lerp factor toward the original surface (upstream remesh_project; default 0 = disabled, the shipped entry-point value; 1 snaps onto the surface). Only used while remeshing."
    )]
    pub remesh_project: Option<f32>,

    #[arg(
        long,
        help = "TRELLIS.2 post-process decimation target in triangles (default 1000000; 0 skips simplification and keeps the remeshed/cleaned topology)."
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
        help = "TRELLIS.2 sample count (default 1). Values > 1 share one conditioning pass and one joint batched noise draw (a `[S,8,R,R,R]` sparse-structure tensor plus concatenated SLAT rows; sample 0 matches a single-sample run) and write one output per sample with a `_0`, `_1`, ... suffix."
    )]
    pub num_samples: Option<usize>,

    #[arg(
        long,
        help = "TRELLIS.2 `return_latent` surface: write the final shape/texture latents to this `.npz` path (shape_latent/tex_latent/coords/resolution/lr_shape_latent/meta). With --num-samples > 1 one file per sample gets the same `_i` suffix as the mesh output."
    )]
    pub return_latent: Option<PathBuf>,

    #[arg(
        long,
        help = "Path to a canonical package directory, a directory containing `model.gguf`, or a standalone `.gguf` file. For `triposg --mode scribble` this is a canonical `triposg-scribble` package, a standalone scribble package (`.gguf` file or a directory containing only `model.gguf`), the raw `VAST-AI/TripoSG-scribble` checkpoint directory, or a legacy DiT-only `.gguf` (remaining weights resolve from LUX3D_TRIPOSG_SCRIBBLE_MODEL_DIR / models/TripoSG-scribble)."
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

/// TripoSG pipeline mode (`--mode`): the image-to-3D DiT (default) or the
/// TripoSG-scribble dual-conditioning DiT.
#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum TriposgMode {
    Image,
    Scribble,
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

/// VGGT input preprocessing mode (`--preprocess-mode`); `crop`/`pad` mirror
/// the upstream `load_and_preprocess_images` `mode` strings, `square` is the
/// official COLMAP demo's `load_and_preprocess_images_square` path.
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum, PartialEq, Eq)]
pub enum VggtPreprocessKind {
    /// Upstream default: resize to width 518, center-crop rows taller than
    /// 518, then white-pad to the batch maximum.
    #[default]
    Crop,
    /// Lossless: resize so the largest side is 518 (aspect preserved) and
    /// center-pad to a white 518x518 square; no pixels are discarded.
    Pad,
    /// Official `demo_colmap.py` path: black-pad to a 1024x1024 square
    /// (integer `// 2` offsets, no pixels discarded), load at 1024, then
    /// bilinear-downsample to the 518x518 model input; the official
    /// `[x1, y1, x2, y2, width, height]` coordinates drive the
    /// original-resolution export.
    Square,
}

impl From<VggtPreprocessKind> for VggtPreprocessMode {
    fn from(kind: VggtPreprocessKind) -> Self {
        match kind {
            VggtPreprocessKind::Crop => Self::Crop,
            VggtPreprocessKind::Pad => Self::Pad,
            VggtPreprocessKind::Square => Self::Square,
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

/// VGGT COLMAP export mode (`--colmap-mode`).
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum, PartialEq, Eq)]
pub enum VggtColmapModeKind {
    /// Feedforward single-observation model
    /// (`batch_np_matrix_to_pycolmap_wo_track`).
    #[default]
    Feedforward,
    /// Track-based reconstruction without BA (`batch_np_matrix_to_pycolmap`).
    Track,
}

impl From<VggtColmapModeKind> for VggtColmapMode {
    fn from(kind: VggtColmapModeKind) -> Self {
        match kind {
            VggtColmapModeKind::Feedforward => Self::Feedforward,
            VggtColmapModeKind::Track => Self::Track,
        }
    }
}

/// VGGT stage-A query-point source (`--track-keypoints`).
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum, PartialEq, Eq)]
pub enum VggtKeypointSourceKind {
    /// Official `keypoint_extractor="aliked+sp"` (ALIKED-n16 then SuperPoint,
    /// concatenated and confidence-gated).
    #[default]
    #[value(name = "aliked-sp")]
    AlikedSp,
    /// Deterministic FPS grid over the model-resolution pixels; requires no
    /// detector checkpoints (previous behavior and explicit opt-out).
    Fps,
}

impl From<VggtKeypointSourceKind> for VggtKeypointSource {
    fn from(kind: VggtKeypointSourceKind) -> Self {
        match kind {
            VggtKeypointSourceKind::AlikedSp => Self::AlikedSuperPoint,
            VggtKeypointSourceKind::Fps => Self::Fps,
        }
    }
}

/// VGGT GLB point source (`--glb-point-source`): the official demo's
/// "Pointmap Branch" vs "Depthmap and Camera Branch" selector
/// (`visual_util.py:68-80`, `demo_gradio.py:466-472`).
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum, PartialEq, Eq)]
pub enum VggtGlbPointSourceKind {
    /// Depth-unprojected `world_points_from_depth` + `depth_conf`, the Gradio
    /// UI's default branch (`demo_gradio.py:469`).
    #[default]
    Depth,
    /// Point-head `world_points` + `world_points_conf`.
    Pointmap,
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
        help = "VGGT input preprocessing (vggt only): `crop` (upstream default: width 518 + center crop, then batch white pad), `pad` (largest side 518, center-padded white 518x518 square, no pixels discarded) or `square` (official COLMAP demo: black-pad to a 1024 square, model input is its 518x518 bilinear downsample, and the official x1/y1/x2/y2/width/height coordinates drive the original-resolution export)."
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
        value_enum,
        default_value_t = VggtColmapModeKind::Feedforward,
        help = "VGGT COLMAP export mode (vggt only): `feedforward` (default) writes the depth-confidence-gated single-observation model (`batch_np_matrix_to_pycolmap_wo_track`, one PINHOLE camera per image); `track` writes the track-based reconstruction (`batch_np_matrix_to_pycolmap` semantics, SIMPLE_PINHOLE cameras, multi-frame tracks) WITHOUT bundle adjustment by default - a COLMAP-ready reconstruction for external BA. `track` requires --colmap-dir and --track-auto; add --colmap-ba to run the external `colmap bundle_adjuster` in the same run. Modifies --colmap-dir."
    )]
    pub colmap_mode: VggtColmapModeKind,

    #[arg(
        long,
        help = "VGGT COLMAP track-mode bundle adjustment (vggt only; upstream demo `--use_ba` default is false). After writing the track reconstruction, runs the external `colmap bundle_adjuster --input_path <dir> --output_path <fresh temp dir>` and atomically replaces cameras.bin/images.bin/points3D.bin (plus rigs.bin/frames.bin when the installed COLMAP writes them) in --colmap-dir with the adjusted model; other files (points.ply) are preserved. The binary comes from LUX3D_COLMAP_BIN or PATH. Requires --colmap-mode track and --colmap-dir (the track export must run first); on failure the unadjusted model is left untouched. Modifies --colmap-dir."
    )]
    pub colmap_ba: bool,

    #[arg(
        long,
        default_value_t = DEFAULT_VIS_THRESHOLD,
        help = "VGGT COLMAP track visibility gate `vis > threshold` (track mode only; upstream --vis_thresh default 0.2). Requires --colmap-mode track."
    )]
    pub vis_thresh: f32,

    #[arg(
        long,
        default_value_t = DEFAULT_MAX_REPROJ_ERROR,
        help = "VGGT COLMAP track reprojection-error gate `||project(seed 3D) - track|| < threshold`, in reconstruction pixels (track mode only; upstream --max_reproj_error default 8.0; square preprocess evaluates it in the official 1024 loaded-image space). Requires --colmap-mode track."
    )]
    pub max_reproj_error: f32,

    #[arg(
        long,
        help = "VGGT COLMAP track export shares one SIMPLE_PINHOLE camera across all images (track mode only; upstream --shared_camera). Requires --colmap-mode track."
    )]
    pub shared_camera: bool,

    #[arg(
        long,
        default_value_t = DEFAULT_MIN_INLIER_PER_FRAME,
        help = "VGGT COLMAP track per-frame inlier floor (track mode only; upstream min_inlier_per_frame default 64): the export fails when any frame keeps fewer visibility+reprojection inliers, exactly like the official builder. Lower only for tiny debug sequences. Requires --colmap-mode track."
    )]
    pub colmap_min_inlier_per_frame: usize,

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
        help = "VGGT explicit track queries as a JSON array (vggt only) of {\"frame\": N, \"x\": F, \"y\": F} objects or [N, F, F] arrays: pixel coordinates in the model input resolution, `x` in [0, 518) and `y` in [0, 518) (prepared frames are at most 518x518) and the query frame must be 0 (rotate the clip so the desired query frame comes first). Mutually exclusive with --track-auto."
    )]
    pub track_queries: Option<String>,

    #[arg(
        long,
        help = "VGGT automatic query selection (vggt only, tracking stage A): ranks frames with DINOv2 ViT-B/14-with-registers CLS similarity + FPS (official --query-frame-num), picks query points per query frame with ALIKED-n16 + SuperPoint (official `keypoint_extractor=\"aliked+sp\"`, `--track-keypoints fps` selects the weight-free FPS grid instead), confidence-gates them like track_predict.py (conf > 1.2 when more than 512 survive), rotates each query frame to position 0 for the track head, then rotates the tracks back. Requires the DINOv2-reg safetensors at models/dinov2_reg4/model.safetensors or LUX3D_DINOV2_REG_MODEL_PATH, and the detector safetensors at models/aliked-superpoint/ (override with LUX3D_ALIKED_WEIGHTS / LUX3D_SUPERPOINT_WEIGHTS) unless `--track-keypoints fps`. Mutually exclusive with --track-queries. Rejected for other families."
    )]
    pub track_auto: bool,

    #[arg(
        long,
        value_enum,
        default_value_t = VggtKeypointSourceKind::AlikedSp,
        help = "VGGT stage-A query points (vggt only, requires --track-auto): `aliked-sp` (default, official `keypoint_extractor=\"aliked+sp\"`: ALIKED-n16 + SuperPoint keypoints concatenated in that order) or `fps` (deterministic farthest-point grid over the model-resolution pixels; no detector checkpoints, previous behavior)."
    )]
    pub track_keypoints: VggtKeypointSourceKind,

    #[arg(
        long,
        default_value_t = DEFAULT_QUERY_FRAME_NUM,
        help = "VGGT automatic query frame count (vggt only, requires --track-auto; official demo default 8): DINO similarity + FPS ranks this many frames, then frame 0 is prepended when it is missing (track_predict.py), so the track head may run one extra query frame."
    )]
    pub query_frame_num: usize,

    #[arg(
        long,
        default_value_t = DEFAULT_MAX_QUERY_PTS,
        help = "VGGT automatic per-frame query-point cap (vggt only, requires --track-auto; official demo default 4096): passed to each detector as `max_num_keypoints` for `--track-keypoints aliked-sp` (so ALIKED+SuperPoint may contribute up to 2x this before confidence gating), or the FPS target for `--track-keypoints fps`."
    )]
    pub max_query_pts: usize,

    #[arg(
        long,
        help = "VGGT track output JSON path (vggt only; default `<source-stem>_tracks.json` when --track-queries or --track-auto is given): writes {\"coords\": [[[x, y], ...], ...], \"visibility\": [[...], ...], \"confidence\": [[...], ...] | null} per frame per query, plus \"query_frames\" for --track-auto."
    )]
    pub track_output: Option<PathBuf>,

    #[arg(
        long,
        default_value_t = 4,
        help = "VGGT track refinement iterations (upstream TrackHead.iters default 4). Modifies --track-queries and --track-auto."
    )]
    pub track_iters: usize,

    #[arg(
        long,
        help = "VGGT query-point PLY path (vggt only, requires --track-queries or --track-auto): writes one colored vertex per query - the query frame's world point sampled at the query pixel (nearest) with its RGB color."
    )]
    pub track_points_ply: Option<PathBuf>,

    #[arg(
        long,
        help = "VGGT downloadable GLB scene path (vggt only): writes the demo-gradio artifact (`predictions_to_glb`, visual_util.py:18-215) - the confidence-filtered colored point cloud plus, with --glb-show-cam, one camera-cone mesh per frame, all aligned to the first camera with the official OpenGL +/-y/z convention."
    )]
    pub glb: Option<PathBuf>,

    #[arg(
        long,
        default_value_t = DEFAULT_GLB_CONF_PERCENTILE,
        help = "VGGT GLB confidence percentile in [0, 100] (vggt only, official demo slider default 50): drop this percentage of the lowest-confidence points (`np.percentile`); `0` disables the percentile gate and keeps every `conf > 1e-5` point. Modifies --glb."
    )]
    pub glb_conf_percentile: f64,

    #[arg(
        long,
        help = "Include the per-frame camera-cone meshes in the VGGT GLB scene (vggt only, requires --glb): the official demo shows them by default; this CLI opts in."
    )]
    pub glb_show_cam: bool,

    #[arg(
        long,
        value_enum,
        default_value_t = VggtGlbPointSourceKind::Depth,
        help = "VGGT GLB point source (vggt only, requires --glb): `depth` unprojects the depth maps (`world_points_from_depth` + `depth_conf`, the Gradio UI default branch) or `pointmap` uses the point-head world_points + world_points_conf. Modifies --glb."
    )]
    pub glb_point_source: VggtGlbPointSourceKind,
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
            colmap_mode: VggtColmapModeKind::Feedforward,
            colmap_ba: false,
            vis_thresh: DEFAULT_VIS_THRESHOLD,
            max_reproj_error: DEFAULT_MAX_REPROJ_ERROR,
            shared_camera: false,
            colmap_min_inlier_per_frame: DEFAULT_MIN_INLIER_PER_FRAME,
            depth_points: None,
            depth_conf_threshold: 1.0,
            track_queries: None,
            track_auto: false,
            track_keypoints: VggtKeypointSourceKind::AlikedSp,
            query_frame_num: DEFAULT_QUERY_FRAME_NUM,
            max_query_pts: DEFAULT_MAX_QUERY_PTS,
            track_output: None,
            track_iters: 4,
            track_points_ply: None,
            glb: None,
            glb_conf_percentile: DEFAULT_GLB_CONF_PERCENTILE,
            glb_show_cam: false,
            glb_point_source: VggtGlbPointSourceKind::Depth,
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
            Self::Ovox(_) => "ovox",
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
            Self::Ovox(_) => None,
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
            Self::Ovox(_) => None,
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
    let options = QuantizeOptions {
        dtype,
        fallback,
        all_2d: args.all_2d,
        prefixes: args.prefix.clone(),
    };
    let report = if args.scribble {
        if args.aux_family.is_some() {
            anyhow::bail!("`--scribble` cannot be combined with `--aux-family`");
        }
        let family = args
            .family
            .ok_or_else(|| anyhow::anyhow!("FAMILY is required with `--scribble`"))?;
        if family != Family::Triposg {
            anyhow::bail!(
                "`--scribble` quantizes the TripoSG-scribble DiT; pass `triposg` as FAMILY"
            );
        }
        if args.standalone {
            quantize_scribble_standalone(&args.source, &args.output, options)?
        } else {
            let source = resolve_scribble_dit_source(&args.source)?;
            quantize_checkpoint(&source, &args.output, ModelFamily::TripoSg, options)?
        }
    } else if let Some(aux_family) = args.aux_family.as_deref() {
        if args.merge_source.is_some() {
            anyhow::bail!("`--merge-source` is not supported with `--aux-family`");
        }
        quantize_auxiliary_checkpoint(&args.source, &args.output, aux_family, options)?
    } else {
        let family = args
            .family
            .ok_or_else(|| anyhow::anyhow!("FAMILY is required unless `--aux-family` is used"))?;
        match (&args.merge_source, &args.merge_prefix) {
            (Some(source), Some(prefix)) => quantize_checkpoint_with_merges(
                &args.source,
                &args.output,
                family.model_family(),
                options,
                &[QuantizeMergeSource::new(source, prefix)],
            )?,
            (None, None) => {
                quantize_checkpoint(&args.source, &args.output, family.model_family(), options)?
            }
            _ => unreachable!("clap enforces --merge-source/--merge-prefix pairing"),
        }
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

fn expand3<T: Copy>(values: &[T], default: T) -> anyhow::Result<[T; 3]> {
    match values {
        [] => Ok([default; 3]),
        [v] => Ok([*v; 3]),
        [x, y, z] => Ok([*x, *y, *z]),
        _ => anyhow::bail!("--grid-size/--voxel-size accept 1 or 3 comma-separated values"),
    }
}

fn parse_aabb(values: &[f32]) -> anyhow::Result<[[f32; 3]; 2]> {
    match values {
        [x0, y0, z0, x1, y1, z1] => Ok([[*x0, *y0, *z0], [*x1, *y1, *z1]]),
        _ => anyhow::bail!(
            "--aabb needs exactly 6 comma-separated values: MINX,MINY,MINZ,MAXX,MAXY,MAXZ"
        ),
    }
}

fn resolve_voxel_size(values: &[f32]) -> anyhow::Result<Option<[f32; 3]>> {
    if values.is_empty() {
        return Ok(None);
    }
    expand3(values, values[0]).map(Some)
}

/// Map `mesh2ovox --grid-size` / `--voxel-size` onto the core option pair.
///
/// An explicit `--grid-size` always wins. When it is absent and `--voxel-size`
/// is supplied, the grid is left `None` so the core derives the index range
/// from the mesh bounds and the requested voxel size. With neither flag the
/// official default (512) applies.
fn resolve_mesh2ovox_grid_size(
    grid_size: Option<&[u32]>,
    voxel_size: &[f32],
) -> anyhow::Result<Option<[u32; 3]>> {
    match grid_size {
        Some(values) => expand3(values, 512).map(Some),
        None if !voxel_size.is_empty() => Ok(None),
        None => Ok(Some([512; 3])),
    }
}

/// Run one `ovox` subcommand and return its JSON stats report.
pub fn run_ovox(args: OvoxArgs) -> anyhow::Result<String> {
    use lux3d_core::export::{
        MeshToOvoxOptions, OvoxDecodeOptions, OvoxToGlbOptions, mesh2ovox, ovox2glb, ovox2mesh,
    };

    match args.command {
        OvoxCommand::Mesh2Ovox(a) => {
            let options = MeshToOvoxOptions {
                grid_size: resolve_mesh2ovox_grid_size(a.grid_size.as_deref(), &a.voxel_size)?,
                voxel_size: resolve_voxel_size(&a.voxel_size)?,
                aabb: a.aabb.as_deref().map(parse_aabb).transpose()?,
                normalize_to_unit_cube: !a.no_normalize,
                mip_level_offset: a.mip_level_offset,
                face_weight: a.face_weight,
                boundary_weight: a.boundary_weight,
                regularization_weight: a.regularization_weight,
                compression: a.compression.into(),
            };
            let stats = mesh2ovox(&a.input, &a.output, &options)?;
            Ok(serde_json::to_string_pretty(&stats)?)
        }
        OvoxCommand::Ovox2Mesh(a) => {
            let options = OvoxDecodeOptions {
                grid_size: expand3(&a.grid_size, 512)?,
                aabb: parse_aabb(&a.aabb)?,
            };
            let stats = ovox2mesh(&a.input, &a.output, &options)?;
            Ok(serde_json::to_string_pretty(&stats)?)
        }
        OvoxCommand::Ovox2Glb(a) => {
            let options = OvoxToGlbOptions {
                decode: OvoxDecodeOptions {
                    grid_size: expand3(&a.grid_size, 512)?,
                    aabb: parse_aabb(&a.aabb)?,
                },
                decimation_target: a.decimation_target,
                texture_size: a.texture_size,
                fill_hole_perimeter: a.fill_hole_perimeter,
                min_component_area: a.min_component_area,
            };
            let stats = ovox2glb(&a.input, &a.output, &options)?;
            Ok(serde_json::to_string_pretty(&stats)?)
        }
    }
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
            reject_unsupported_pi3_options(&args)?;
            reject_triposg_only_options(&args, args.family)?;
            reject_vggt_only_options(&args, args.family)?;
            reject_trellis2_only_options(&args, args.family)?;
            run_pi3(
                model_assets,
                required_source(&args, "pi3")?,
                args.interval,
                args.conf_threshold,
                &args.output,
                &device,
            )?
        }
        ModelFamily::Pi3x => {
            reject_unsupported_pi3x_options(&args)?;
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
            reject_unsupported_triposr_options(&args)?;
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
            match args.mode {
                TriposgMode::Scribble => {
                    reject_scribble_mode_conflicts(&args)?;
                    let prompt = args
                        .prompt
                        .clone()
                        .filter(|prompt| !prompt.trim().is_empty())
                        .ok_or_else(|| {
                            anyhow::anyhow!("`--prompt` is required for `triposg --mode scribble`")
                        })?;
                    let options = triposg_scribble_run_options_from_args(&args)?;
                    let (model_dir, dit_gguf) =
                        resolve_triposg_scribble_model_source(args.model_path.as_deref())?;
                    let dtype = args
                        .dtype
                        .map(candle_core::DType::from)
                        .unwrap_or(candle_core::DType::F32);
                    let source = match args.source.as_slice() {
                        [source] => source.as_path(),
                        [] => anyhow::bail!("`--source` is required for `triposg --mode scribble`"),
                        sources => anyhow::bail!(
                            "`triposg --mode scribble` accepts exactly one `--source`, got {}: \
                             the official scribble script runs a single scribble image \
                             (`scripts/inference_triposg_scribble.py:30`)",
                            sources.len()
                        ),
                    };
                    run_triposg_scribble(
                        model_dir,
                        dit_gguf,
                        source,
                        &prompt,
                        options,
                        &args.output,
                        &device,
                        dtype,
                    )?
                }
                TriposgMode::Image => {
                    let options = triposg_run_options_from_args(&args)?;
                    let rmbg_model_path = resolve_rmbg_model_path(&args);
                    let dtype = args
                        .dtype
                        .map(candle_core::DType::from)
                        .unwrap_or(candle_core::DType::F32);
                    let input = match (args.source.as_slice(), args.surface_input.as_deref()) {
                        (_, Some(surface)) => TripoSgRunInput::Surface(surface),
                        (sources, None) if !sources.is_empty() => TripoSgRunInput::Image(sources),
                        (_, None) => {
                            anyhow::bail!(
                                "`--source` or `--surface-input` is required for `triposg`"
                            )
                        }
                    };
                    run_triposg(
                        model_assets,
                        rmbg_model_path,
                        input,
                        options,
                        args.num_shapes.unwrap_or(1),
                        &args.output,
                        &device,
                        dtype,
                    )?
                }
            }
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
///
/// `--source` is repeatable at the clap layer because `triposg` image mode
/// accepts a batch; every other family accepts exactly one image/frame
/// source and fails fast on extra values instead of silently ignoring them.
fn required_source<'a>(args: &'a RunArgs, family: &str) -> anyhow::Result<&'a Path> {
    match args.source.as_slice() {
        [] => Err(anyhow::anyhow!("`--source` is required for `{family}`")),
        [source] => Ok(source),
        sources => anyhow::bail!(
            "`{family}` accepts exactly one `--source`, got {} (multiple sources are \
             only supported for `triposg` image mode)",
            sources.len()
        ),
    }
}

fn run_pi3(
    model_assets: ModelAssetOptions,
    source: &Path,
    interval: Option<usize>,
    conf_threshold: Option<f32>,
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
    let mut inference = pipeline.infer_from_path_with_interval(source, interval, device)?;
    if let Some(confidence_threshold) = conf_threshold {
        inference.export_mask = pipeline.export.export_mask_with_threshold(
            &inference.local_points,
            &inference.confidence_logits,
            confidence_threshold,
        )?;
    }
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

/// Pi3 exposes single-pass point-cloud inference only. Reject the Pi3X
/// VO/conditioning options and the TripoSG/TRELLIS.2 RMBG override instead of
/// silently ignoring them. `--interval` is genuinely consumed by the Pi3 frame
/// sampler and `--conf-threshold` gates the export mask, so both stay accepted.
fn reject_unsupported_pi3_options(args: &RunArgs) -> anyhow::Result<()> {
    let set: [(&str, bool); 6] = [
        ("--vo", args.vo),
        ("--conditions", args.conditions.is_some()),
        ("--chunk-size", args.chunk_size.is_some()),
        ("--overlap", args.overlap.is_some()),
        ("--inject-condition", !args.inject_condition.is_empty()),
        ("--rmbg-model-path", args.rmbg_model_path.is_some()),
    ];
    for (flag, is_set) in set {
        if is_set {
            anyhow::bail!("`{flag}` is not supported for `pi3`");
        }
    }
    Ok(())
}

/// Pi3X accepts the VO-only knobs (`--chunk-size`, `--overlap`,
/// `--conf-threshold`, `--inject-condition`) only when `--vo` selects the VO
/// pipeline. Without `--vo` the non-VO branch calls
/// `infer_from_path(source, conditions, interval, device)` and never reads
/// them, so reject them instead of silently ignoring the user's intent.
fn reject_unsupported_pi3x_options(args: &RunArgs) -> anyhow::Result<()> {
    if args.vo {
        return Ok(());
    }
    let set: [(&str, bool); 4] = [
        ("--chunk-size", args.chunk_size.is_some()),
        ("--overlap", args.overlap.is_some()),
        ("--conf-threshold", args.conf_threshold.is_some()),
        ("--inject-condition", !args.inject_condition.is_empty()),
    ];
    for (flag, is_set) in set {
        if is_set {
            anyhow::bail!("`{flag}` requires `--vo` for `pi3x`");
        }
    }
    Ok(())
}

/// TripoSR exposes image-to-mesh only (`--mc-resolution` / `--mc-threshold` are
/// consumed). Reject the Pi3/Pi3X/conditioning options, the Pi3 video
/// `--interval`, the confidence gate and the RMBG override instead of silently
/// ignoring them.
fn reject_unsupported_triposr_options(args: &RunArgs) -> anyhow::Result<()> {
    let set: [(&str, bool); 8] = [
        ("--vo", args.vo),
        ("--conditions", args.conditions.is_some()),
        ("--chunk-size", args.chunk_size.is_some()),
        ("--overlap", args.overlap.is_some()),
        ("--inject-condition", !args.inject_condition.is_empty()),
        ("--interval", args.interval.is_some()),
        ("--rmbg-model-path", args.rmbg_model_path.is_some()),
        ("--conf-threshold", args.conf_threshold.is_some()),
    ];
    for (flag, is_set) in set {
        if is_set {
            anyhow::bail!("`{flag}` is not supported for `triposr`");
        }
    }
    Ok(())
}

/// VGGT exposes point/depth/camera inference only. Reject Pi3X VO and
/// TripoSR meshing options plus the Pi3/TripoSR confidence gate and the RMBG
/// override instead of silently ignoring them (VGGT's own confidence knob is
/// `--depth-conf-threshold` / `--colmap-conf-threshold`).
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
    if args.conf_threshold.is_some() {
        anyhow::bail!(
            "`--conf-threshold` is not supported for `vggt`; use `--depth-conf-threshold` or \
             `--colmap-conf-threshold`"
        );
    }
    if args.rmbg_model_path.is_some() {
        anyhow::bail!(
            "`--rmbg-model-path` is not supported for `vggt` (VGGT consumes raw images without \
             an RMBG foreground step)"
        );
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
    let set: [(&str, bool); 29] = [
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
        (
            "--colmap-mode",
            vggt.colmap_mode != VggtColmapModeKind::Feedforward,
        ),
        ("--colmap-ba", vggt.colmap_ba),
        ("--vis-thresh", vggt.vis_thresh != DEFAULT_VIS_THRESHOLD),
        (
            "--max-reproj-error",
            vggt.max_reproj_error != DEFAULT_MAX_REPROJ_ERROR,
        ),
        ("--shared-camera", vggt.shared_camera),
        (
            "--colmap-min-inlier-per-frame",
            vggt.colmap_min_inlier_per_frame != DEFAULT_MIN_INLIER_PER_FRAME,
        ),
        ("--depth-points", vggt.depth_points.is_some()),
        ("--depth-conf-threshold", vggt.depth_conf_threshold != 1.0),
        ("--track-queries", vggt.track_queries.is_some()),
        ("--track-auto", vggt.track_auto),
        (
            "--track-keypoints",
            vggt.track_keypoints != VggtKeypointSourceKind::AlikedSp,
        ),
        (
            "--query-frame-num",
            vggt.query_frame_num != DEFAULT_QUERY_FRAME_NUM,
        ),
        (
            "--max-query-pts",
            vggt.max_query_pts != DEFAULT_MAX_QUERY_PTS,
        ),
        ("--track-output", vggt.track_output.is_some()),
        ("--track-iters", vggt.track_iters != 4),
        ("--track-points-ply", vggt.track_points_ply.is_some()),
        ("--glb", vggt.glb.is_some()),
        (
            "--glb-conf-percentile",
            vggt.glb_conf_percentile != DEFAULT_GLB_CONF_PERCENTILE,
        ),
        ("--glb-show-cam", vggt.glb_show_cam),
        (
            "--glb-point-source",
            vggt.glb_point_source != VggtGlbPointSourceKind::Depth,
        ),
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
    // `--surface-input` encodes the VAE latents directly
    // (`scripts/inference_vae.py`), skipping image conditioning and the DiT
    // denoise loop, so the denoise schedule knobs have no effect.
    if args.surface_input.is_some() {
        let set: [(&str, bool); 3] = [
            ("--steps", args.steps.is_some()),
            ("--guidance-scale", args.guidance_scale.is_some()),
            ("--timesteps", args.timesteps.is_some()),
        ];
        for (flag, is_set) in set {
            if is_set {
                anyhow::bail!(
                    "`{flag}` has no effect with `--surface-input`: the surface encode path skips \
                     the denoise loop (the reference `scripts/inference_vae.py` extracts the \
                     latent directly)"
                );
            }
        }
    }
    if args.mode == TriposgMode::Image {
        if args.surface_input.is_some() && args.num_shapes.is_some() {
            anyhow::bail!(
                "`--num-shapes` is not supported with `--surface-input`: the reference VAE \
                 surface-encode path (`scripts/inference_vae.py`) produces a single latent"
            );
        }
        if args.prompt.is_some() {
            anyhow::bail!("`--prompt` requires `triposg --mode scribble`");
        }
        if args.scribble_conf.is_some() {
            anyhow::bail!("`--scribble-conf` requires `triposg --mode scribble`");
        }
        if args.prompt_conf.is_some() {
            anyhow::bail!("`--prompt-conf` requires `triposg --mode scribble`");
        }
    }
    Ok(())
}

/// The scribble pipeline is a single-pass distilled model: the reference
/// script hardcodes `guidance_scale=0` (no CFG), `use_flash_decoder=False`,
/// dense/hierarchical depth 8 and a raw diffusers-style checkpoint directory.
/// Reject the flags that would silently contradict those semantics. Shared
/// flags (`--steps`, `--seed`, `--num-tokens`, the octree depths, `--faces`,
/// `--bounds`, `--timesteps`, `--chunk-size`, `--dtype`) stay allowed.
/// `--rmbg-model-path` is rejected: the official scribble path consumes the
/// raw RGB image through the DINOv2 `BitImageProcessor` and never runs RMBG.
fn reject_scribble_mode_conflicts(args: &RunArgs) -> anyhow::Result<()> {
    if args.surface_input.is_some() {
        anyhow::bail!("`--surface-input` is not supported for `triposg --mode scribble`");
    }
    if args.num_shapes.is_some() {
        anyhow::bail!(
            "`--num-shapes` is not supported for `triposg --mode scribble`: the official \
             scribble script runs a single shape (`scripts/inference_triposg_scribble.py:32-41`)"
        );
    }
    if args.rmbg_model_path.is_some() {
        anyhow::bail!(
            "`--rmbg-model-path` is not supported for `triposg --mode scribble`: the \
             reference feeds the raw RGB scribble through the DINOv2 BitImageProcessor \
             without any alpha/RMBG foreground step"
        );
    }
    if args.guidance_scale.is_some() {
        anyhow::bail!(
            "`--guidance-scale` is not supported for `triposg --mode scribble` (the \
             distilled scribble model runs without classifier-free guidance)"
        );
    }
    if args.flash_decoder.is_some() {
        anyhow::bail!(
            "`--flash-decoder` is not supported for `triposg --mode scribble` (the \
             reference always uses the dense/hierarchical decoder)"
        );
    }
    if args.flash_octree_depth.is_some() {
        anyhow::bail!("`--flash-octree-depth` is not supported for `triposg --mode scribble`");
    }
    if args.cache_dir.is_some() {
        anyhow::bail!(
            "`--cache-dir` is not supported for `triposg --mode scribble`: the scribble \
             checkpoint has no HF auto-download; pass --model-path (canonical package, \
             standalone GGUF or raw checkpoint) or set LUX3D_TRIPOSG_SCRIBBLE_MODEL_DIR"
        );
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
    if args.num_shapes.is_some() {
        anyhow::bail!(
            "`--num-shapes` is not supported for `{}`",
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
    if args.mode != TriposgMode::Image {
        anyhow::bail!(
            "`--mode scribble` is not supported for `{}`",
            family.model_family()
        );
    }
    if args.prompt.is_some() {
        anyhow::bail!(
            "`--prompt` is not supported for `{}`",
            family.model_family()
        );
    }
    if args.scribble_conf.is_some() {
        anyhow::bail!(
            "`--scribble-conf` is not supported for `{}`",
            family.model_family()
        );
    }
    if args.prompt_conf.is_some() {
        anyhow::bail!(
            "`--prompt-conf` is not supported for `{}`",
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
    if args.num_shapes.is_some() {
        anyhow::bail!("`--num-shapes` is not supported for `trellis2`; use `--num-samples`");
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
    let set: [(&str, bool); 32] = [
        ("--pipeline-type", args.pipeline_type.is_some()),
        ("--no-low-vram", args.no_low_vram),
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
        ("--no-remesh", args.no_remesh),
        ("--remesh-band", args.remesh_band.is_some()),
        ("--remesh-project", args.remesh_project.is_some()),
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
    if args.no_remesh {
        postprocess.remesh = false;
    }
    if let Some(value) = args.remesh_band {
        if args.no_remesh {
            anyhow::bail!(
                "`--remesh-band` requires the remesh branch; it conflicts with `--no-remesh`"
            );
        }
        postprocess.remesh_band = value;
    }
    if let Some(value) = args.remesh_project {
        if args.no_remesh {
            anyhow::bail!(
                "`--remesh-project` requires the remesh branch; it conflicts with `--no-remesh`"
            );
        }
        postprocess.remesh_project = value;
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
        low_vram: !args.no_low_vram,
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
    if !matches!(resolution, 512 | 1024 | 1536) {
        anyhow::bail!(
            "`--texture-resolution` must be 512, 1024 or 1536 (got {resolution}); 1024 and \
             1536 share the 1024 conditioning/tex-dit configuration"
        );
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
    let incompatible: [(&str, bool); 13] = [
        ("--pipeline-type", args.pipeline_type.is_some()),
        ("--no-low-vram", args.no_low_vram),
        ("--max-num-tokens", args.max_num_tokens.is_some()),
        ("--return-latent", args.return_latent.is_some()),
        ("--num-samples", args.num_samples.is_some()),
        ("--no-texture-export", args.no_texture_export),
        ("--no-uv-unwrap", args.no_uv_unwrap),
        ("--no-postprocess", args.no_postprocess),
        ("--no-remesh", args.no_remesh),
        ("--remesh-band", args.remesh_band.is_some()),
        ("--remesh-project", args.remesh_project.is_some()),
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

/// Build the TripoSG-scribble run options from the reference defaults
/// (`scripts/inference_triposg_scribble.py:32-43`: 16 steps, 512 tokens,
/// dense 8 / hierarchical 8, scribble confidence 0.4, prompt confidence 1.0)
/// plus the shared CLI overrides. `--guidance-scale`, `--flash-decoder` and
/// `--flash-octree-depth` are rejected by the scribble-mode guard because the
/// distilled model runs a single pass with the dense/hierarchical decoder.
fn triposg_scribble_run_options_from_args(
    args: &RunArgs,
) -> anyhow::Result<TripoSgScribbleRunOptions> {
    let defaults = TripoSgScribbleRunOptions::default();
    let options = TripoSgScribbleRunOptions {
        steps: args.steps.unwrap_or(defaults.steps),
        seed: args.seed.unwrap_or(defaults.seed),
        num_tokens: args.num_tokens.unwrap_or(defaults.num_tokens),
        dense_octree_depth: args
            .dense_octree_depth
            .unwrap_or(defaults.dense_octree_depth),
        hierarchical_octree_depth: args
            .hierarchical_octree_depth
            .unwrap_or(defaults.hierarchical_octree_depth),
        query_chunk_size: args.chunk_size.unwrap_or(defaults.query_chunk_size),
        timesteps: args
            .timesteps
            .as_deref()
            .map(parse_timesteps_arg)
            .transpose()?,
        bounds: args.bounds.as_deref().map(parse_bounds_arg).transpose()?,
        faces: args.faces.filter(|&faces| faces > 0),
        scribble_confidence: args.scribble_conf.unwrap_or(defaults.scribble_confidence),
        prompt_confidence: args.prompt_conf.unwrap_or(defaults.prompt_confidence),
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

/// Map the track-mode COLMAP flags onto [`VggtTrackColmapOptions`].
fn vggt_track_colmap_options_from_args(vggt: &VggtRunArgs) -> VggtTrackColmapOptions {
    VggtTrackColmapOptions {
        vis_thresh: vggt.vis_thresh,
        max_reproj_error: vggt.max_reproj_error,
        shared_camera: vggt.shared_camera,
        min_inlier_per_frame: vggt.colmap_min_inlier_per_frame,
        ..VggtTrackColmapOptions::default()
    }
}

/// Sparse-model files replaced by `--colmap-ba`. Every other file in the
/// export directory (for example `points.ply`) is preserved.
const COLMAP_BA_MODEL_FILES: [&str; 3] = ["cameras.bin", "images.bin", "points3D.bin"];

/// Newer COLMAP releases (`Reconstruction::WriteBinary`) also write the rig
/// and frame tables. They are optional: older releases do not, and the
/// three-file model remains readable, but dropping them would discard part of
/// the adjusted reconstruction.
const COLMAP_BA_OPTIONAL_MODEL_FILES: [&str; 2] = ["rigs.bin", "frames.bin"];

/// Command line for the external bundle adjustment (`colmap bundle_adjuster`
/// CLI spelling). Arguments are passed directly to `Command`, never through a
/// shell, so paths with spaces or shell metacharacters stay single arguments.
fn colmap_ba_args(input_dir: &Path, output_dir: &Path) -> Vec<std::ffi::OsString> {
    vec![
        "bundle_adjuster".into(),
        "--input_path".into(),
        input_dir.as_os_str().to_owned(),
        "--output_path".into(),
        output_dir.as_os_str().to_owned(),
    ]
}

/// External BA binary: `LUX3D_COLMAP_BIN` when set and non-empty, else
/// `colmap` resolved through `PATH`.
fn resolve_colmap_program() -> PathBuf {
    std::env::var_os("LUX3D_COLMAP_BIN")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("colmap"))
}

/// Fail-fast availability check for `--colmap-ba`, run before weights are
/// loaded. An explicit path must exist; a bare name is searched on `PATH`
/// (plus the current directory and `.exe` on Windows, matching
/// `Command::new` process search).
fn ensure_colmap_available(program: &Path) -> anyhow::Result<()> {
    if program.is_absolute() || program.components().count() > 1 {
        if program.is_file() {
            return Ok(());
        }
        anyhow::bail!(
            "COLMAP binary `{}` does not exist or is not a file; set `LUX3D_COLMAP_BIN` to the \
             COLMAP executable or add `colmap` to PATH (required by `--colmap-ba`)",
            program.display()
        );
    }
    let mut names = vec![program.to_path_buf()];
    #[cfg(windows)]
    {
        names.push(program.with_extension("exe"));
        // CreateProcess also searches the current directory.
        if program.is_file() {
            return Ok(());
        }
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            for name in &names {
                if dir.join(name).is_file() {
                    return Ok(());
                }
            }
        }
    }
    anyhow::bail!(
        "COLMAP binary `{}` was not found on PATH; set `LUX3D_COLMAP_BIN` to the COLMAP \
         executable or add `colmap` to PATH (required by `--colmap-ba`)",
        program.display()
    )
}

struct ColmapBaStagingDir(PathBuf);

impl Drop for ColmapBaStagingDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Tail of captured process output for error messages.
fn colmap_output_tail(bytes: &[u8]) -> String {
    const MAX_TAIL_BYTES: usize = 4096;
    let start = bytes.len().saturating_sub(MAX_TAIL_BYTES);
    let tail = String::from_utf8_lossy(&bytes[start..]).trim().to_string();
    if tail.is_empty() {
        "<no stderr output>".to_string()
    } else if start > 0 {
        format!("...{tail}")
    } else {
        tail
    }
}

/// Run `colmap bundle_adjuster` on the track export and atomically (per file)
/// replace the sparse model.
///
/// The adjusted model is written to a staging directory inside `colmap_dir`
/// (same filesystem, so the renames are atomic). A failed run leaves the
/// original `cameras.bin` / `images.bin` / `points3D.bin` untouched; the
/// staging directory is always removed. `rigs.bin` / `frames.bin` are copied
/// too when the COLMAP release writes them.
fn run_colmap_bundle_adjustment(colmap_dir: &Path, program: &Path) -> anyhow::Result<()> {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let staging = colmap_dir.join(format!(".lux3d-colmap-ba-{}-{unique}", std::process::id()));
    std::fs::create_dir_all(&staging).map_err(|error| {
        anyhow::anyhow!(
            "failed to create the COLMAP BA staging directory `{}`: {error}",
            staging.display()
        )
    })?;
    let _cleanup = ColmapBaStagingDir(staging.clone());

    let args = colmap_ba_args(colmap_dir, &staging);
    let rendered = args
        .iter()
        .map(|argument| argument.to_string_lossy())
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!("[colmap:ba] {} {rendered}", program.display());

    let output = ProcessCommand::new(program)
        .args(&args)
        .output()
        .map_err(|error| {
            anyhow::anyhow!(
                "failed to launch COLMAP bundle adjuster `{}`: {error}; set `LUX3D_COLMAP_BIN` \
                 to the COLMAP executable or add `colmap` to PATH",
                program.display()
            )
        })?;
    if !output.status.success() {
        anyhow::bail!(
            "`{} {rendered}` failed with status {}; stderr tail:\n{}\n\
             the unadjusted model in `{}` was left untouched",
            program.display(),
            output.status,
            colmap_output_tail(&output.stderr),
            colmap_dir.display()
        );
    }
    for name in COLMAP_BA_MODEL_FILES {
        let path = staging.join(name);
        if !path.is_file() {
            anyhow::bail!(
                "`{} {rendered}` exited successfully but did not write `{}`; \
                 the unadjusted model in `{}` was left untouched",
                program.display(),
                path.display(),
                colmap_dir.display()
            );
        }
    }
    install_colmap_ba_output(&staging, colmap_dir)?;
    eprintln!(
        "[colmap:ba] adjusted {} -> {}",
        COLMAP_BA_MODEL_FILES.join("/"),
        colmap_dir.display()
    );
    Ok(())
}

/// Replace the sparse model with the staged bundle-adjusted files.
///
/// All three required adjusted files are verified before the first rename, so
/// an incomplete output leaves the original model untouched. Each rename is
/// atomic on the export directory's filesystem; optional rig/frame tables are
/// installed when the COLMAP release wrote them.
fn install_colmap_ba_output(staging: &Path, colmap_dir: &Path) -> anyhow::Result<()> {
    for name in COLMAP_BA_MODEL_FILES {
        let source = staging.join(name);
        if !source.is_file() {
            anyhow::bail!(
                "bundle-adjusted model is missing `{}`; the model in `{}` was left untouched",
                source.display(),
                colmap_dir.display()
            );
        }
    }
    for name in COLMAP_BA_MODEL_FILES
        .into_iter()
        .chain(COLMAP_BA_OPTIONAL_MODEL_FILES)
    {
        let source = staging.join(name);
        if !source.is_file() {
            continue;
        }
        let target = colmap_dir.join(name);
        std::fs::rename(&source, &target).map_err(|error| {
            anyhow::anyhow!(
                "failed to replace `{}` with the bundle-adjusted model: {error}",
                target.display()
            )
        })?;
    }
    Ok(())
}

/// Map the GLB export flags onto [`VggtGlbOptions`], validating the
/// `--glb-*` modifiers fail-fast before any weights are loaded.
fn vggt_glb_options_from_args(vggt: &VggtRunArgs) -> anyhow::Result<VggtGlbOptions> {
    let percentile = vggt.glb_conf_percentile;
    if !percentile.is_finite() || !(0.0..=100.0).contains(&percentile) {
        anyhow::bail!(
            "`--glb-conf-percentile` must be a finite percentage in [0, 100], got {percentile}"
        );
    }
    if vggt.glb.is_none() {
        if vggt.glb_show_cam {
            anyhow::bail!("`--glb-show-cam` requires `--glb`");
        }
        if vggt.glb_point_source != VggtGlbPointSourceKind::Depth {
            anyhow::bail!("`--glb-point-source` requires `--glb`");
        }
        if percentile != DEFAULT_GLB_CONF_PERCENTILE {
            anyhow::bail!("`--glb-conf-percentile` requires `--glb`");
        }
    }
    Ok(VggtGlbOptions {
        conf_percentile: percentile,
        show_cam: vggt.glb_show_cam,
        // The demo's optional background masks are ported in the core scene
        // builder but stay off here: the CLI surface is the documented
        // `--glb` / `--glb-conf-percentile` / `--glb-show-cam` set.
        mask_black_bg: false,
        mask_white_bg: false,
    })
}

/// CPU copies of the GLB prediction arrays in the batch-squeezed demo layouts.
struct VggtGlbCpuTensors {
    points: Vec<f32>,
    confidences: Vec<f32>,
    images: Vec<f32>,
    extrinsics: Vec<f32>,
    frames: usize,
    height: usize,
    width: usize,
}

/// Stage the selected VGGT prediction branch (`depth` = the official
/// `world_points_from_depth` + `depth_conf`, `pointmap` = `world_points` +
/// `world_points_conf`, `visual_util.py:68-80`) as CPU f32 arrays.
fn vggt_glb_cpu_tensors(
    output: &VggtInferenceOutput,
    source: VggtGlbPointSourceKind,
) -> anyhow::Result<VggtGlbCpuTensors> {
    let (points_tensor, confidences_tensor) = match source {
        VggtGlbPointSourceKind::Depth => (
            output
                .world_points_from_depth()
                .map_err(anyhow::Error::from)?,
            output.depth_conf.clone(),
        ),
        VggtGlbPointSourceKind::Pointmap => (
            output.world_points.clone(),
            output.world_points_conf.clone(),
        ),
    };
    let point_dims = points_tensor.dims().to_vec();
    let (frames, height, width) = match point_dims.as_slice() {
        [1, frames, height, width, 3] => (*frames, *height, *width),
        [frames, height, width, 3] => (*frames, *height, *width),
        _ => anyhow::bail!(
            "VGGT GLB world points shape {point_dims:?} must be [1,S,H,W,3] or [S,H,W,3]"
        ),
    };
    let confidences_tensor = vggt_squeeze_batch(&confidences_tensor, "point confidences")?;
    let confidence_dims = confidences_tensor.dims().to_vec();
    if confidence_dims != [frames, height, width] {
        anyhow::bail!(
            "VGGT GLB confidence shape {confidence_dims:?} does not match the {frames}x{height}x{width} point grid"
        );
    }
    let extrinsics_tensor = vggt_squeeze_batch(&output.extrinsics, "extrinsics")?;
    let extrinsic_dims = extrinsics_tensor.dims().to_vec();
    if extrinsic_dims != [frames, 3, 4] {
        anyhow::bail!(
            "VGGT GLB extrinsics shape {extrinsic_dims:?} does not match {frames} frames as [S,3,4]"
        );
    }
    let image_dims = output.rgb_frames.dims().to_vec();
    if image_dims != [frames, 3, height, width] {
        anyhow::bail!(
            "VGGT GLB images shape {image_dims:?} does not match [S,3,H,W] = [{frames},3,{height},{width}]"
        );
    }
    Ok(VggtGlbCpuTensors {
        points: vggt_cpu_f32(&points_tensor, "world points")?,
        confidences: vggt_cpu_f32(&confidences_tensor, "world point confidences")?,
        images: vggt_cpu_f32(&output.rgb_frames, "rgb frames")?,
        extrinsics: vggt_cpu_f32(&extrinsics_tensor, "extrinsics")?,
        frames,
        height,
        width,
    })
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
/// model-resolution pixel coordinates with `coords[0] == queries`). Automatic
/// tracking adds `"query_frames": [Q]` (the source frame of each query point)
/// and `"query_points": [Q][2]`.
fn vggt_track_result_to_json(
    track: &VggtTrackResult,
    query_frames: Option<&[usize]>,
    query_points: Option<&[[f32; 2]]>,
) -> anyhow::Result<serde_json::Value> {
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
    let mut value = serde_json::json!({
        "coords": coords,
        "visibility": visibility,
        "confidence": confidence,
    });
    if let Some(query_frames) = query_frames {
        value["query_frames"] = serde_json::json!(query_frames);
    }
    if let Some(query_points) = query_points {
        value["query_points"] = serde_json::json!(query_points);
    }
    Ok(value)
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

/// `--colmap-ba` is track-only: the feedforward export has no multi-view
/// reconstruction, and the track export requires an explicit output directory.
fn validate_vggt_colmap_ba(vggt: &VggtRunArgs) -> anyhow::Result<()> {
    if !vggt.colmap_ba {
        return Ok(());
    }
    if vggt.colmap_mode != VggtColmapModeKind::Track {
        anyhow::bail!(
            "`--colmap-ba` requires `--colmap-mode track`: the feedforward export has no \
             multi-view reconstruction to bundle-adjust"
        );
    }
    if vggt.colmap_dir.is_none() {
        anyhow::bail!("`--colmap-ba` requires `--colmap-dir`");
    }
    Ok(())
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
    let glb_options = vggt_glb_options_from_args(vggt)?;
    let track_queries = match vggt.track_queries.as_deref() {
        Some(json) => Some(parse_vggt_track_queries(json)?),
        None => None,
    };
    if track_queries.is_some() && vggt.track_auto {
        anyhow::bail!(
            "`--track-queries` and `--track-auto` are mutually exclusive: pass explicit queries or let stage A select them"
        );
    }
    if vggt.track_auto {
        if vggt.query_frame_num == 0 {
            anyhow::bail!("`--query-frame-num` must be at least 1");
        }
        if vggt.max_query_pts == 0 {
            anyhow::bail!("`--max-query-pts` must be at least 1");
        }
    } else {
        if vggt.query_frame_num != DEFAULT_QUERY_FRAME_NUM {
            anyhow::bail!("`--query-frame-num` requires `--track-auto`");
        }
        if vggt.max_query_pts != DEFAULT_MAX_QUERY_PTS {
            anyhow::bail!("`--max-query-pts` requires `--track-auto`");
        }
        if vggt.track_keypoints != VggtKeypointSourceKind::AlikedSp {
            anyhow::bail!("`--track-keypoints` requires `--track-auto`");
        }
    }
    let tracking = track_queries.is_some() || vggt.track_auto;
    if tracking && vggt.track_iters == 0 {
        anyhow::bail!("`--track-iters` must be at least 1");
    }
    if tracking && vggt.batch_size > 1 {
        anyhow::bail!(
            "tracking runs a single pass over all frames (the track head requires the query frame first); `--batch-size` chunking greater than 1 cannot be combined with tracking"
        );
    }
    if vggt.track_output.is_some() && !tracking {
        anyhow::bail!("`--track-output` requires `--track-queries` or `--track-auto`");
    }
    if vggt.track_points_ply.is_some() && !tracking {
        anyhow::bail!("`--track-points-ply` requires `--track-queries` or `--track-auto`");
    }
    validate_vggt_colmap_ba(vggt)?;
    if vggt.colmap_mode == VggtColmapModeKind::Track {
        if vggt.colmap_dir.is_none() {
            anyhow::bail!("`--colmap-mode track` requires `--colmap-dir`");
        }
        if !vggt.track_auto {
            anyhow::bail!(
                "`--colmap-mode track` requires `--track-auto` (the export consumes the stage-A tracks)"
            );
        }
    }
    let track_colmap_flags = vggt.vis_thresh != DEFAULT_VIS_THRESHOLD
        || vggt.max_reproj_error != DEFAULT_MAX_REPROJ_ERROR
        || vggt.shared_camera
        || vggt.colmap_min_inlier_per_frame != DEFAULT_MIN_INLIER_PER_FRAME;
    if track_colmap_flags && vggt.colmap_mode != VggtColmapModeKind::Track {
        anyhow::bail!(
            "`--vis-thresh`, `--max-reproj-error`, `--shared-camera` and \
             `--colmap-min-inlier-per-frame` require `--colmap-mode track`"
        );
    }
    let track_output = match &vggt.track_output {
        Some(path) => Some(path.clone()),
        None if tracking => Some(vggt_default_track_output(source)),
        None => None,
    };
    // Fail fast on a missing external binary before weights/inference run.
    let colmap_ba_program = if vggt.colmap_ba {
        let program = resolve_colmap_program();
        ensure_colmap_available(&program)?;
        Some(program)
    } else {
        None
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
    let (inference, track, auto_track) = match &track_queries {
        Some(queries) => {
            let (inference, track) =
                pipeline.run_with_tracks(&prepared, queries, Some(vggt.track_iters))?;
            (inference, Some(track), None)
        }
        None if vggt.track_auto => {
            let auto_options = VggtAutoQueryOptions {
                query_frame_num: vggt.query_frame_num,
                max_query_pts: vggt.max_query_pts,
                track_iters: vggt.track_iters,
                dino_weights: None,
                keypoint_source: vggt.track_keypoints.into(),
                aliked_weights: None,
                superpoint_weights: None,
            };
            let (inference, auto) = pipeline.run_with_auto_tracks(&prepared, &auto_options)?;
            eprintln!(
                "[track-auto] query frames={:?} points={:?} pruned={:?}",
                auto.ranked_query_frames, auto.queries_per_frame, auto.pruned_per_frame
            );
            let track = auto.track.clone();
            (inference, Some(track), Some(auto))
        }
        None => (
            pipeline.infer_prepared_with_options(&prepared, &options)?,
            None,
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
        match vggt.colmap_mode {
            VggtColmapModeKind::Feedforward => {
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
            VggtColmapModeKind::Track => {
                let auto = auto_track.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("`--colmap-mode track` requires `--track-auto`")
                })?;
                let stats = write_track_colmap_from_auto(
                    colmap_dir,
                    &inference,
                    &prepared,
                    &image_names,
                    auto,
                    vggt.colmap_point_source.into(),
                    &vggt_track_colmap_options_from_args(vggt),
                )
                .map_err(anyhow::Error::from)?;
                eprintln!(
                    "[colmap:track] tracks={} cameras={} images={} points3d={} point2d={} min_inliers={} -> {}",
                    stats.track_count,
                    stats.camera_count,
                    stats.image_count,
                    stats.point3d_count,
                    stats.point2d_count,
                    stats.min_inliers_per_frame,
                    colmap_dir.display()
                );
                if let Some(program) = colmap_ba_program.as_deref() {
                    run_colmap_bundle_adjustment(colmap_dir, program)?;
                }
            }
        }
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
        let json = vggt_track_result_to_json(
            track,
            auto_track.as_ref().map(|auto| auto.query_frames.as_slice()),
            auto_track.as_ref().map(|auto| auto.query_points.as_slice()),
        )?;
        std::fs::write(track_path, serde_json::to_string_pretty(&json)?).map_err(|source| {
            anyhow::anyhow!(
                "failed to write VGGT tracks to {}: {source}",
                track_path.display()
            )
        })?;
        eprintln!(
            "[tracks] {} queries x {} frames, iters={} -> {}",
            track.coords.dim(1).unwrap_or(0),
            track.coords.dims().first().copied().unwrap_or(0),
            vggt.track_iters,
            track_path.display()
        );
    }
    if let Some(query_ply) = vggt.track_points_ply.as_deref() {
        let queries = match (&track_queries, &auto_track) {
            (Some(queries), _) => queries.clone(),
            (None, Some(auto)) => auto
                .query_frames
                .iter()
                .zip(&auto.query_points)
                .map(|(frame, xy)| VggtTrackQuery {
                    frame: *frame,
                    xy: *xy,
                })
                .collect(),
            (None, None) => unreachable!("--track-points-ply requires tracking"),
        };
        let point_cloud = vggt_query_point_cloud(&inference, &queries)?;
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
    if let Some(glb_path) = vggt.glb.as_deref() {
        let tensors = vggt_glb_cpu_tensors(&inference, vggt.glb_point_source)?;
        let predictions = VggtGlbPredictions {
            world_points: &tensors.points,
            world_points_conf: &tensors.confidences,
            images: &tensors.images,
            extrinsics: &tensors.extrinsics,
            frames: tensors.frames,
            height: tensors.height,
            width: tensors.width,
        };
        let scene =
            build_vggt_glb_scene(&predictions, &glb_options).map_err(anyhow::Error::from)?;
        write_vggt_glb(&scene, glb_path).map_err(anyhow::Error::from)?;
        eprintln!(
            "[glb] points={} cameras={} -> {}",
            scene.point_count(),
            scene.camera_count(),
            glb_path.display()
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
    // Vulkan mesh decode is dispatch-bound: the triplane sampler and the
    // density MLP issue ~50 tiny ops per chunk, and at 128^3 there are 256
    // chunks at the old 8192 default. Measured on Vulkan: mesh 8.50 s at
    // 8192, 4.18 s at 16384, 2.41 s at 32768 (CUDA 1.96 s); 65536 regressed
    // to 15.1 s total through allocator/memory churn. 32768 is the measured
    // sweet spot. The env knob is for measurement/experiments.
    let query_chunk = std::env::var("LUX3D_TRIPOSR_QUERY_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(32768);
    let mesh = pipeline.extract_mesh(
        &inference.scene_codes,
        mc_resolution,
        mc_threshold,
        query_chunk,
    )?;
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

/// TripoSG input selection: the image-to-3D path (`--source`, repeatable for
/// a batched run) or the reference VAE surface encode path (`--surface-input`).
enum TripoSgRunInput<'a> {
    Image(&'a [PathBuf]),
    Surface(&'a Path),
}

#[allow(clippy::too_many_arguments)]
fn run_triposg(
    model_assets: ModelAssetOptions,
    rmbg_model_path: Option<PathBuf>,
    input: TripoSgRunInput<'_>,
    options: TripoSgRunOptions,
    num_shapes: usize,
    output: &Path,
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> anyhow::Result<()> {
    options.validate().map_err(anyhow::Error::from)?;
    if num_shapes == 0 {
        anyhow::bail!("`--num-shapes` must be >= 1");
    }
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
            // The surface-encode path produces ONE latent; `--num-shapes` is
            // rejected in `reject_unsupported_triposg_options` (defensive
            // guard here as well).
            if num_shapes != 1 {
                anyhow::bail!("`--num-shapes` is not supported with `--surface-input`");
            }
            pipeline.encode_surface_from_path(surface_path, device, &options)?
        }
        TripoSgRunInput::Image(sources) => {
            let inference =
                pipeline.infer_batch_from_paths(sources, device, num_shapes, &options)?;
            if stage_time {
                eprintln!(
                    "[conditioning] triposg: {} image(s), num_shapes {}, batch {}",
                    sources.len(),
                    num_shapes,
                    inference.latents.dims()[0]
                );
            }
            inference.latents
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
    // `num_shapes = 1` + one source (and every surface-input run) uses the
    // same single-sample `extract_mesh` call as the pre-batch pipeline
    // (byte-identical mesh); a real batch decodes per sample via
    // `extract_meshes`.
    let meshes = if latents.dims()[0] == 1 {
        vec![pipeline.extract_mesh(&latents, &options)?]
    } else {
        pipeline.extract_meshes(&latents, &options)?
    };
    if stage_time {
        eprintln!("[stage] mesh: {:.2}s", t2.elapsed().as_secs_f64());
    }
    // `extract_mesh`/`extract_meshes` already returned the post-simplified
    // buffers when `--faces` triggered the merge-close + quadric pass, so
    // these counts (and the meshes written below) reflect the
    // simplification.
    let num_outputs = meshes.len();
    for (index, mesh) in meshes.iter().enumerate() {
        let path = sample_output_path(output, index, num_outputs);
        let faces_note = match options.faces {
            Some(faces) => format!(" (target {faces})"),
            None => String::new(),
        };
        if num_outputs == 1 {
            // Preserve the historical single-output log line verbatim
            // (`--num-shapes 1` + one source).
            eprintln!(
                "[mesh] triposg: {} vertices, {} faces{}",
                mesh.vertices.dims()[0],
                mesh.faces.dims()[0],
                faces_note
            );
        } else {
            eprintln!(
                "[mesh] triposg {}/{}: {} vertices, {} faces{} -> {}",
                index + 1,
                num_outputs,
                mesh.vertices.dims()[0],
                mesh.faces.dims()[0],
                faces_note,
                path.display()
            );
        }
        let t3 = std::time::Instant::now();
        pipeline.export_mesh(mesh, &path)?;
        if stage_time {
            eprintln!("[stage] export: {:.2}s", t3.elapsed().as_secs_f64());
        }
    }
    Ok(())
}

/// TripoSG-scribble run: raw `VAST-AI/TripoSG-scribble` checkpoint -> CLIP
/// text embeds + DINOv2 scribble embeds -> 16-step dual-conditioned Euler
/// denoise (no CFG) -> dense/hierarchical SDF decode -> GLB export.
fn run_triposg_scribble(
    model_dir: PathBuf,
    dit_gguf: Option<PathBuf>,
    source: &Path,
    prompt: &str,
    options: TripoSgScribbleRunOptions,
    output: &Path,
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> anyhow::Result<()> {
    options.validate().map_err(anyhow::Error::from)?;
    let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
    let t0 = std::time::Instant::now();
    let pipeline =
        TripoSgScribblePipeline::load_with_dit_gguf_and_dtype(model_dir, dit_gguf, dtype)?;
    if stage_time {
        eprintln!("[stage] load: {:.2}s", t0.elapsed().as_secs_f64());
    }
    let t1 = std::time::Instant::now();
    let inference = pipeline.infer_from_path(source, prompt, device, &options)?;
    if stage_time {
        eprintln!("[stage] infer: {:.2}s", t1.elapsed().as_secs_f64());
    }
    eprintln!(
        "[conditioning] triposg-scribble: prompt `{}`, text {:?}, image {:?}, \
         scribble_conf {}, prompt_conf {}",
        inference.prompt,
        inference.text_embeds.dims(),
        inference.image_embeds.dims(),
        options.scribble_confidence,
        options.prompt_confidence
    );
    let t2 = std::time::Instant::now();
    let mesh = pipeline.extract_mesh(&inference.latents, &options)?;
    if stage_time {
        eprintln!("[stage] mesh: {:.2}s", t2.elapsed().as_secs_f64());
    }
    eprintln!(
        "[mesh] triposg-scribble: {} vertices, {} faces{}",
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
        .texture_mesh_from_path(
            &imported.vertices,
            &imported.faces,
            imported.uvs.as_deref(),
            source,
            options,
            device,
        )
        .map_err(anyhow::Error::from)?;
    if stage_time {
        eprintln!("[stage] texture: {:.2}s", t1.elapsed().as_secs_f64());
    }
    let t2 = std::time::Instant::now();
    pipeline
        .export
        .write_textured_glb(&textured.mesh, output, true)
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
        vec![
            pipeline
                .run_from_path_with_stages(source, device, &options)
                .map_err(anyhow::Error::from)?,
        ]
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
            // Upstream `to_glb`: doubleSided = false on the remeshed branch
            // (postprocess.py:303); true on the cleanup/pass-through paths.
            let double_sided = !(options.postprocess.enabled && options.postprocess.remesh);
            export
                .write_textured_glb(textured, &sample_output, double_sided)
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

    use super::{Cli, Command, RunArgs, inspect_model, normalize_weights, run_model};
    use lux3d_core::{
        ModelFamily,
        test_support::{
            GpuTestLock, canonical_package_dir, resolve_raw_model_dir_for_tests, runtime_root,
            workspace_root,
        },
    };

    fn repo_root() -> PathBuf {
        runtime_root()
    }

    /// Standalone GGUF package used by the local smoke tests. The canonical
    /// Pi3/Pi3X/TripoSR packages are optional on a development machine, while
    /// `3d/standalone/<family>-q8_0` is the published local form.
    fn smoke_model_dir(name: &str) -> PathBuf {
        repo_root().join("3d").join("standalone").join(name)
    }

    fn test_asset(relative: &str) -> PathBuf {
        workspace_root().join("test-assets").join(relative)
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

    fn colmap_ba_temp_dir(tag: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("current time")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "lux3d-colmap-ba-{tag}-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
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

    /// `--conf-threshold` and `--rmbg-model-path` are never read by `run_vggt`
    /// (VGGT uses `--depth-conf-threshold` / `--colmap-conf-threshold` and raw
    /// images), so they must fail fast instead of being silently ignored.
    #[test]
    fn run_vggt_rejects_unread_conf_threshold_and_rmbg() {
        let flag_options: [(&str, &[&str]); 2] = [
            ("--conf-threshold", &["--conf-threshold", "0.9"]),
            ("--rmbg-model-path", &["--rmbg-model-path", "rmbg"]),
        ];
        for (flag, extra) in flag_options {
            let mut argv = vec![
                "lux3d", "run", "vggt", "--source", "frames", "--output", "out.ply",
            ];
            argv.extend_from_slice(extra);
            let cli = Cli::parse_from(argv);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let error = super::reject_unsupported_vggt_options(&args)
                .expect_err(&format!("{flag} must be rejected for vggt"));
            let message = error.to_string();
            assert!(
                message.contains(flag) && message.contains("not supported for `vggt`"),
                "unexpected message for {flag}: `{message}`"
            );
        }
    }

    #[test]
    fn vggt_colmap_ba_defaults_off_and_parses() {
        let cli = Cli::parse_from([
            "lux3d", "run", "vggt", "--source", "frames", "--output", "out.ply",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert!(
            !args.vggt.colmap_ba,
            "bundle adjustment must stay opt-in (upstream --use_ba default false)"
        );

        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "vggt",
            "--source",
            "frames",
            "--track-auto",
            "--colmap-mode",
            "track",
            "--colmap-dir",
            "sparse",
            "--colmap-ba",
            "--output",
            "out.ply",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert!(args.vggt.colmap_ba);
        super::validate_vggt_colmap_ba(&args.vggt).expect("track mode with a COLMAP directory");
    }

    #[test]
    fn vggt_colmap_ba_requires_track_mode_and_colmap_dir() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "vggt",
            "--source",
            "frames",
            "--colmap-ba",
            "--output",
            "out.ply",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let error = super::validate_vggt_colmap_ba(&args.vggt)
            .expect_err("feedforward mode must reject --colmap-ba");
        assert!(
            error.to_string().contains("`--colmap-mode track`"),
            "unexpected message: `{error}`"
        );

        // Track mode implies an output directory, but keep the dedicated
        // message testable independently of the rest of run_vggt.
        let vggt = super::VggtRunArgs {
            colmap_ba: true,
            colmap_mode: super::VggtColmapModeKind::Track,
            colmap_dir: None,
            ..super::VggtRunArgs::default()
        };
        let error = super::validate_vggt_colmap_ba(&vggt)
            .expect_err("track mode without --colmap-dir must reject --colmap-ba");
        assert!(
            error.to_string().contains("`--colmap-dir`"),
            "unexpected message: `{error}`"
        );
    }

    #[test]
    fn colmap_ba_argument_construction_uses_the_colmap_cli() {
        let arguments = super::colmap_ba_args(Path::new("sparse"), Path::new("sparse/tmp-ba"));
        let rendered: Vec<String> = arguments
            .iter()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            vec![
                "bundle_adjuster",
                "--input_path",
                "sparse",
                "--output_path",
                "sparse/tmp-ba",
            ],
            rendered
        );
    }

    #[test]
    fn colmap_ba_missing_binary_fails_fast() {
        let missing =
            std::env::temp_dir().join(format!("lux3d-missing-colmap-{}", std::process::id()));
        let error = super::ensure_colmap_available(&missing)
            .expect_err("an explicit missing binary path must fail fast");
        let message = error.to_string();
        assert!(
            message.contains("does not exist"),
            "unexpected message: `{message}`"
        );
        assert!(
            message.contains("LUX3D_COLMAP_BIN"),
            "the error must name the override: `{message}`"
        );
    }

    #[test]
    fn colmap_ba_install_replaces_the_model_and_preserves_other_files() {
        let dir = colmap_ba_temp_dir("install");
        let staging = dir.join("staging");
        fs::create_dir_all(&staging).expect("create staging");
        for name in super::COLMAP_BA_MODEL_FILES {
            fs::write(dir.join(name), b"original").expect("write original model");
            fs::write(staging.join(name), b"adjusted").expect("write adjusted model");
        }
        // Newer COLMAP releases also write the rig/frame tables.
        for name in super::COLMAP_BA_OPTIONAL_MODEL_FILES {
            fs::write(dir.join(name), b"original").expect("write original rig/frame table");
            fs::write(staging.join(name), b"adjusted").expect("write adjusted rig/frame table");
        }
        fs::write(dir.join("points.ply"), b"keep").expect("write sibling file");

        super::install_colmap_ba_output(&staging, &dir).expect("install adjusted model");
        for name in super::COLMAP_BA_MODEL_FILES
            .into_iter()
            .chain(super::COLMAP_BA_OPTIONAL_MODEL_FILES)
        {
            assert_eq!(
                b"adjusted".as_slice(),
                fs::read(dir.join(name)).expect("read adjusted model"),
                "`{name}` must be replaced"
            );
        }
        assert_eq!(
            b"keep".as_slice(),
            fs::read(dir.join("points.ply")).expect("read sibling file"),
            "other files in the export directory must be preserved"
        );
        cleanup_dir(&dir);
    }

    #[test]
    fn colmap_ba_install_refuses_an_incomplete_adjusted_model() {
        let dir = colmap_ba_temp_dir("incomplete");
        let staging = dir.join("staging");
        fs::create_dir_all(&staging).expect("create staging");
        for name in super::COLMAP_BA_MODEL_FILES {
            fs::write(dir.join(name), b"original").expect("write original model");
        }
        fs::write(staging.join("cameras.bin"), b"adjusted").expect("write partial output");

        let error = super::install_colmap_ba_output(&staging, &dir)
            .expect_err("a missing adjusted file must fail");
        assert!(
            error.to_string().contains("cameras.bin")
                || error.to_string().contains("images.bin")
                || error.to_string().contains("points3D.bin"),
            "unexpected message: `{error}`"
        );
        for name in super::COLMAP_BA_MODEL_FILES {
            assert_eq!(
                b"original".as_slice(),
                fs::read(dir.join(name)).expect("read original model"),
                "`{name}` must be untouched"
            );
        }
        cleanup_dir(&dir);
    }

    #[test]
    fn colmap_ba_failed_command_leaves_the_model_untouched_and_cleans_staging() {
        let dir = colmap_ba_temp_dir("failure");
        for name in super::COLMAP_BA_MODEL_FILES {
            fs::write(dir.join(name), b"original").expect("write original model");
        }
        fs::write(dir.join("points.ply"), b"keep").expect("write sibling file");

        // The test harness rejects the `bundle_adjuster` arguments with a
        // non-zero status and a stderr message, exercising the failure path
        // without requiring COLMAP.
        let program = std::env::current_exe().expect("test executable");
        let error = super::run_colmap_bundle_adjustment(&dir, &program)
            .expect_err("the fake bundle adjuster must fail");
        let message = error.to_string();
        assert!(
            message.contains("failed with status"),
            "unexpected message: `{message}`"
        );
        assert!(
            message.contains("stderr tail"),
            "the error must include the captured stderr: `{message}`"
        );
        assert!(
            message.contains("left untouched"),
            "unexpected message: `{message}`"
        );

        for name in super::COLMAP_BA_MODEL_FILES {
            assert_eq!(
                b"original".as_slice(),
                fs::read(dir.join(name)).expect("read original model"),
                "`{name}` must be untouched after a failed BA"
            );
        }
        assert_eq!(
            b"keep".as_slice(),
            fs::read(dir.join("points.ply")).expect("read sibling file")
        );
        let staging_leftovers: Vec<_> = fs::read_dir(&dir)
            .expect("list export directory")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.starts_with(".lux3d-colmap-ba-"))
            .collect();
        assert!(
            staging_leftovers.is_empty(),
            "staging directories must be cleaned up: {staging_leftovers:?}"
        );
        cleanup_dir(&dir);
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

    /// `--preprocess-mode square` parses, maps to the core square mode and is
    /// rejected for the other families (it differs from the crop default).
    #[test]
    fn vggt_square_preprocess_mode_parses_and_maps() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "vggt",
            "--source",
            "frames",
            "--preprocess-mode",
            "square",
            "--output",
            "out.ply",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(super::VggtPreprocessKind::Square, args.vggt.preprocess_mode);
        super::reject_unsupported_vggt_options(&args).expect("vggt accepts square");
        let options = super::vggt_run_options_from_args(&args).expect("run options");
        assert_eq!(
            lux3d_core::preprocess::VggtPreprocessMode::Square,
            options.preprocess_mode
        );

        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "pi3",
            "--source",
            "input",
            "--preprocess-mode",
            "square",
            "--output",
            "out",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let error = super::reject_vggt_only_options(&args, args.family)
            .expect_err("square must be rejected for pi3");
        assert!(error.to_string().contains("--preprocess-mode"));
    }

    /// `--track-auto` parses with the official demo defaults, and combining it
    /// with explicit `--track-queries` is rejected before any model load.
    #[test]
    fn vggt_auto_track_flags_parse_and_conflict() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "vggt",
            "--source",
            "frames",
            "--output",
            "out.ply",
            "--track-auto",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert!(args.vggt.track_auto);
        assert_eq!(8, args.vggt.query_frame_num);
        assert_eq!(4096, args.vggt.max_query_pts);
        assert_eq!(
            super::VggtKeypointSourceKind::AlikedSp,
            args.vggt.track_keypoints,
            "the official aliked+sp source is the default"
        );

        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "vggt",
            "--source",
            "frames",
            "--output",
            "out.ply",
            "--track-auto",
            "--track-keypoints",
            "fps",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(
            super::VggtKeypointSourceKind::Fps,
            args.vggt.track_keypoints
        );

        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "vggt",
            "--source",
            "frames",
            "--output",
            "out.ply",
            "--track-auto",
            "--query-frame-num",
            "3",
            "--max-query-pts",
            "128",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(3, args.vggt.query_frame_num);
        assert_eq!(128, args.vggt.max_query_pts);

        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "vggt",
            "--source",
            "frames",
            "--output",
            "out.ply",
            "--track-auto",
            "--track-queries",
            r#"[{"frame":0,"x":1.0,"y":2.0}]"#,
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let error = super::run_vggt(
            super::ModelAssetOptions::default(),
            &args,
            &candle_core::Device::Cpu,
        )
        .expect_err("--track-auto and --track-queries must conflict");
        assert!(
            error.to_string().contains("mutually exclusive"),
            "unexpected error: {error}"
        );

        // A non-default point cap without --track-auto is rejected too.
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "vggt",
            "--source",
            "frames",
            "--output",
            "out.ply",
            "--max-query-pts",
            "64",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let error = super::run_vggt(
            super::ModelAssetOptions::default(),
            &args,
            &candle_core::Device::Cpu,
        )
        .expect_err("--max-query-pts requires --track-auto");
        assert!(error.to_string().contains("requires `--track-auto`"));

        // The non-default FPS query source requires --track-auto as well.
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "vggt",
            "--source",
            "frames",
            "--output",
            "out.ply",
            "--track-keypoints",
            "fps",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let error = super::run_vggt(
            super::ModelAssetOptions::default(),
            &args,
            &candle_core::Device::Cpu,
        )
        .expect_err("--track-keypoints fps requires --track-auto");
        assert!(error.to_string().contains("requires `--track-auto`"));
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
        assert_eq!(
            super::VggtKeypointSourceKind::AlikedSp,
            args.vggt.track_keypoints
        );
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

        // GLB export is opt-in; the confidence percentile keeps the official
        // demo default and the camera cones stay off unless requested.
        assert_eq!(None, args.vggt.glb);
        assert_eq!(
            lux3d_core::export::DEFAULT_GLB_CONF_PERCENTILE,
            args.vggt.glb_conf_percentile
        );
        assert!(!args.vggt.glb_show_cam);
        assert_eq!(
            super::VggtGlbPointSourceKind::Depth,
            args.vggt.glb_point_source
        );
        let glb = super::vggt_glb_options_from_args(&args.vggt).expect("glb options");
        assert_eq!(50.0, glb.conf_percentile);
        assert!(!glb.show_cam);
    }

    /// The GLB flags parse and map onto [`VggtGlbOptions`].
    #[test]
    fn vggt_glb_flags_parse_and_map() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "vggt",
            "--source",
            "frames",
            "--glb",
            "scene.glb",
            "--glb-conf-percentile",
            "30",
            "--glb-show-cam",
            "--glb-point-source",
            "pointmap",
            "--output",
            "out.ply",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(Some(PathBuf::from("scene.glb")), args.vggt.glb);
        assert_eq!(30.0, args.vggt.glb_conf_percentile);
        assert!(args.vggt.glb_show_cam);
        assert_eq!(
            super::VggtGlbPointSourceKind::Pointmap,
            args.vggt.glb_point_source
        );
        let glb = super::vggt_glb_options_from_args(&args.vggt).expect("glb options");
        assert_eq!(30.0, glb.conf_percentile);
        assert!(glb.show_cam);
        assert!(!glb.mask_black_bg);
        assert!(!glb.mask_white_bg);
    }

    /// The confidence percentile must stay in `[0, 100]`, and the `--glb-*`
    /// modifiers require `--glb` rather than being silent no-ops.
    #[test]
    fn vggt_glb_flags_validate_ranges_and_requirements() {
        let parse = |extra: &[&str]| {
            let mut argv = vec![
                "lux3d", "run", "vggt", "--source", "frames", "--output", "out.ply",
            ];
            argv.extend_from_slice(extra);
            let cli = Cli::parse_from(argv);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            args
        };

        for argument in ["--glb-conf-percentile=100.5", "--glb-conf-percentile=-1"] {
            let args = parse(&["--glb", "scene.glb", argument]);
            let error = super::vggt_glb_options_from_args(&args.vggt)
                .expect_err("out-of-range percentile must be rejected");
            assert!(
                error.to_string().contains("in [0, 100]"),
                "unexpected message for {argument}: {error}"
            );
        }

        for (extra, expected) in [
            (vec!["--glb-show-cam"], "`--glb-show-cam` requires `--glb`"),
            (
                vec!["--glb-point-source", "pointmap"],
                "`--glb-point-source` requires `--glb`",
            ),
            (
                vec!["--glb-conf-percentile", "30"],
                "`--glb-conf-percentile` requires `--glb`",
            ),
        ] {
            let args = parse(&extra);
            let error = super::vggt_glb_options_from_args(&args.vggt)
                .expect_err("GLB modifiers without --glb must be rejected");
            assert!(
                error.to_string().contains(expected),
                "unexpected message for {extra:?}: {error}"
            );
        }
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

        let with_conf = super::vggt_track_result_to_json(
            &super::VggtTrackResult {
                coords: coords.clone(),
                visibility: visibility.clone(),
                confidence: Some(confidence),
            },
            None,
            None,
        )
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

        let without_conf = super::vggt_track_result_to_json(
            &super::VggtTrackResult {
                coords,
                visibility,
                confidence: None,
            },
            Some(&[0, 2]),
            Some(&[[1.0f32, 2.0], [3.0, 4.0]]),
        )
        .expect("json");
        assert_eq!(serde_json::Value::Null, without_conf["confidence"]);
        assert_eq!(2, without_conf["coords"].as_array().expect("frames").len());
        // Automatic tracking records the source frame and pixel of every
        // query point.
        assert_eq!(serde_json::json!([0, 2]), without_conf["query_frames"]);
        assert_eq!(
            serde_json::json!([[1.0f32, 2.0], [3.0, 4.0]]),
            without_conf["query_points"]
        );
    }

    /// Every VGGT-only flag (when it differs from its default) is rejected for
    /// the other families instead of being silently ignored.
    #[test]
    fn vggt_only_flags_rejected_for_other_families() {
        let flag_options: [(&str, &[&str]); 22] = [
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
            ("--track-auto", &["--track-auto"]),
            ("--query-frame-num", &["--query-frame-num", "3"]),
            ("--max-query-pts", &["--max-query-pts", "64"]),
            ("--track-output", &["--track-output", "tracks.json"]),
            ("--track-iters", &["--track-iters", "6"]),
            ("--track-points-ply", &["--track-points-ply", "queries.ply"]),
            ("--glb", &["--glb", "scene.glb"]),
            ("--glb-conf-percentile", &["--glb-conf-percentile", "30"]),
            ("--glb-show-cam", &["--glb-show-cam"]),
            ("--glb-point-source", &["--glb-point-source", "pointmap"]),
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
            (
                50usize, 7.0f32, 42u64, 2048usize, 8u32, 9u32, true, 9u32, 32768usize
            ),
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

    /// `--mode scribble` is triposg-only and keeps the reference scribble
    /// defaults: 16 steps, 512 tokens, dense 8 / hierarchical 8, no flash
    /// decoder, scribble confidence 0.4 and prompt confidence 1.0.
    #[test]
    fn triposg_scribble_mode_maps_reference_defaults() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "triposg",
            "--mode",
            "scribble",
            "--prompt",
            "a photo of a cat",
            "--source",
            "scribble.png",
            "--output",
            "out.glb",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(super::TriposgMode::Scribble, args.mode);
        assert_eq!(Some("a photo of a cat"), args.prompt.as_deref());
        let options = super::triposg_scribble_run_options_from_args(&args)
            .expect("scribble reference defaults");
        assert_eq!(super::TripoSgScribbleRunOptions::default(), options);
        assert_eq!(16, options.steps);
        assert_eq!(512, options.num_tokens);
        assert_eq!(8, options.dense_octree_depth);
        assert_eq!(8, options.hierarchical_octree_depth);
        assert_eq!(0.4, options.scribble_confidence);
        assert_eq!(1.0, options.prompt_confidence);
    }

    #[test]
    fn triposg_scribble_mode_maps_cli_overrides() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "triposg",
            "--mode",
            "scribble",
            "--prompt",
            "low poly dragon",
            "--source",
            "scribble.png",
            "--output",
            "out.glb",
            "--steps",
            "8",
            "--seed",
            "7",
            "--num-tokens",
            "256",
            "--dense-octree-depth",
            "6",
            "--hierarchical-octree-depth",
            "7",
            "--chunk-size",
            "2048",
            "--faces",
            "50000",
            "--scribble-conf",
            "0.25",
            "--prompt-conf",
            "0.75",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let options = super::triposg_scribble_run_options_from_args(&args).expect("options");
        assert_eq!(8, options.steps);
        assert_eq!(7, options.seed);
        assert_eq!(256, options.num_tokens);
        assert_eq!(6, options.dense_octree_depth);
        assert_eq!(7, options.hierarchical_octree_depth);
        assert_eq!(2048, options.query_chunk_size);
        assert_eq!(Some(50000), options.faces);
        assert_eq!(0.25, options.scribble_confidence);
        assert_eq!(0.75, options.prompt_confidence);
    }

    /// The image-to-3D mode must not silently ignore scribble flags.
    #[test]
    fn triposg_image_mode_rejects_scribble_flags() {
        for (flag, extra) in [
            ("--prompt", &["--prompt", "cat"][..]),
            ("--scribble-conf", &["--scribble-conf", "0.4"][..]),
            ("--prompt-conf", &["--prompt-conf", "1.0"][..]),
        ] {
            let mut argv = vec![
                "lux3d",
                "run",
                "triposg",
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
            let error = super::reject_unsupported_triposg_options(&args)
                .expect_err(&format!("{flag} must be rejected in image mode"));
            let message = error.to_string();
            assert!(
                message.contains(flag) && message.contains("--mode scribble"),
                "unexpected message for {flag}: `{message}`"
            );
        }
    }

    /// The scribble pipeline is single-pass with the dense decoder and a raw
    /// checkpoint directory; the flags that would contradict those semantics
    /// are rejected before any weights are touched.
    #[test]
    fn triposg_scribble_mode_rejects_conflicting_flags() {
        for (flag, extra) in [
            ("--guidance-scale", &["--guidance-scale", "1.0"][..]),
            ("--flash-decoder", &["--flash-decoder", "true"][..]),
            ("--flash-octree-depth", &["--flash-octree-depth", "8"][..]),
            ("--cache-dir", &["--cache-dir", "hf-cache"][..]),
            ("--rmbg-model-path", &["--rmbg-model-path", "rmbg"][..]),
        ] {
            let mut argv = vec![
                "lux3d",
                "run",
                "triposg",
                "--mode",
                "scribble",
                "--prompt",
                "cat",
                "--source",
                "scribble.png",
                "--output",
                "out.glb",
            ];
            argv.extend_from_slice(extra);
            let cli = Cli::parse_from(argv);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let error = super::reject_scribble_mode_conflicts(&args)
                .expect_err(&format!("{flag} must be rejected in scribble mode"));
            assert!(
                error.to_string().contains(flag),
                "unexpected message for {flag}: `{error}`"
            );
        }
        // `--surface-input` conflicts with `--source` at the clap layer, so
        // exercise the guard with the surface input only.
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "triposg",
            "--mode",
            "scribble",
            "--prompt",
            "cat",
            "--surface-input",
            "surface.npy",
            "--output",
            "out.glb",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let error = super::reject_scribble_mode_conflicts(&args)
            .expect_err("--surface-input must be rejected in scribble mode");
        assert!(error.to_string().contains("--surface-input"));
    }

    /// `--mode scribble`/`--prompt`/the confidence flags are TripoSG-only.
    #[test]
    fn scribble_flags_rejected_for_other_families() {
        for family in ["pi3", "pi3x", "triposr", "vggt", "trellis2"] {
            for (flag, extra) in [
                ("--prompt", &["--prompt", "cat"][..]),
                ("--scribble-conf", &["--scribble-conf", "0.4"][..]),
                ("--prompt-conf", &["--prompt-conf", "1.0"][..]),
                ("--mode", &["--mode", "scribble"][..]),
            ] {
                let mut argv = vec![
                    "lux3d", "run", family, "--source", "input", "--output", "out",
                ];
                argv.extend_from_slice(extra);
                let cli = Cli::parse_from(argv);
                let Command::Run(args) = cli.command else {
                    panic!("expected run command");
                };
                let error = super::reject_triposg_only_options(&args, args.family)
                    .expect_err(&format!("{flag} must be rejected for {family}"));
                assert!(
                    error.to_string().contains(flag),
                    "unexpected message for {flag} on {family}: `{error}`"
                );
            }
        }
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
        assert!(defaults.low_vram, "reference low_vram default is true");
    }

    /// `--no-low-vram` flips the core `low_vram` switch (the `false` value
    /// keeps every loaded stage resident for the run); the flag is
    /// TRELLIS.2-only and rejected with `--texture-mesh`.
    #[test]
    fn trellis2_no_low_vram_maps_to_the_core_option() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--output",
            "out.glb",
            "--no-low-vram",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert!(args.no_low_vram);
        let options = super::trellis2_run_options_from_args(&args).expect("run options");
        assert!(!options.low_vram);

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
        assert!(!args.no_low_vram);
        let options = super::trellis2_run_options_from_args(&args).expect("run options");
        assert!(options.low_vram);

        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--texture-mesh",
            "input.glb",
            "--output",
            "out.glb",
            "--no-low-vram",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let error = super::validate_trellis2_texturing_mode(&args)
            .expect_err("--no-low-vram has no texturing-pipeline meaning");
        assert!(error.to_string().contains("--no-low-vram"));
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

    /// Mesh post-process flags default to the shipped entry-point values
    /// (`example.py:43` / `app.py:503`: remesh on, band 1, project 0) with
    /// the pass enabled.
    #[test]
    fn trellis2_postprocess_flags_default_to_shipped_entry_points() {
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
        assert!(!args.no_remesh);
        assert_eq!(None, args.remesh_band);
        assert_eq!(None, args.remesh_project);
        assert_eq!(None, args.decimation_target);
        assert_eq!(None, args.fill_hole_perimeter);
        assert_eq!(None, args.min_component_area);
        let options = super::trellis2_run_options_from_args(&args).expect("run options");
        assert_eq!(
            super::Trellis2MeshPostprocessOptions::default(),
            options.postprocess
        );
        assert!(options.postprocess.enabled);
        assert!(options.postprocess.remesh);
        assert_eq!(1.0, options.postprocess.remesh_band);
        assert_eq!(0.0, options.postprocess.remesh_project);
        assert_eq!(3e-2, options.postprocess.fill_hole_perimeter);
        assert_eq!(1_000_000, options.postprocess.decimation_target);
        assert_eq!(1e-5, options.postprocess.min_component_area);
    }

    /// `--no-postprocess` / `--no-remesh` / `--remesh-band` /
    /// `--remesh-project` / `--decimation-target` / `--fill-hole-perimeter` /
    /// `--min-component-area` map onto the run options; a target of 0 means
    /// "skip simplification" and is valid.
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

    /// `--no-remesh` restores the cleanup chain; `--remesh-band` /
    /// `--remesh-project` override the shipped remesh values; combining
    /// `--no-remesh` with either remesh knob fails fast.
    #[test]
    fn trellis2_remesh_flags_map_and_no_remesh_conflicts() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--output",
            "out.glb",
            "--no-remesh",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert!(args.no_remesh);
        let options = super::trellis2_run_options_from_args(&args).expect("run options");
        assert!(options.postprocess.enabled, "post-process stays enabled");
        assert!(!options.postprocess.remesh, "--no-remesh flips the branch");

        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--output",
            "out.glb",
            "--remesh-band",
            "2.5",
            "--remesh-project",
            "0.25",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(Some(2.5), args.remesh_band);
        assert_eq!(Some(0.25), args.remesh_project);
        let options = super::trellis2_run_options_from_args(&args).expect("run options");
        assert!(options.postprocess.remesh);
        assert_eq!(2.5, options.postprocess.remesh_band);
        assert_eq!(0.25, options.postprocess.remesh_project);

        for flag in ["--remesh-band", "--remesh-project"] {
            let error = super::trellis2_run_options_from_args(&{
                let Cli {
                    command: Command::Run(args),
                    ..
                } = Cli::parse_from([
                    "lux3d",
                    "run",
                    "trellis2",
                    "--source",
                    "image.png",
                    "--output",
                    "out.glb",
                    "--no-remesh",
                    flag,
                    "1",
                ])
                else {
                    panic!("expected run command");
                };
                args
            })
            .expect_err("conflicting remesh flag must fail");
            assert!(
                error.to_string().contains("conflicts with `--no-remesh`"),
                "unexpected message for {flag}: `{error}`"
            );
        }
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
        let flag_options: [(&str, &[&str]); 28] = [
            ("--pipeline-type", &["--pipeline-type", "512"]),
            ("--no-low-vram", &["--no-low-vram"]),
            ("--max-num-tokens", &["--max-num-tokens", "8192"]),
            ("--texture", &["--texture", "false"]),
            ("--texture-size", &["--texture-size", "1024"]),
            ("--no-texture-export", &["--no-texture-export"]),
            ("--no-uv-unwrap", &["--no-uv-unwrap"]),
            ("--no-postprocess", &["--no-postprocess"]),
            ("--no-remesh", &["--no-remesh"]),
            ("--remesh-band", &["--remesh-band", "2"]),
            ("--remesh-project", &["--remesh-project", "0.5"]),
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
    fn run_triposg_source_is_repeatable_with_num_shapes() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "triposg",
            "--source",
            "first.png",
            "--source",
            "second.png",
            "--num-shapes",
            "2",
            "--output",
            "out.glb",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(
            args.source,
            vec![PathBuf::from("first.png"), PathBuf::from("second.png")]
        );
        assert_eq!(args.num_shapes, Some(2));
        // The options builder is source-agnostic; the batch size is passed to
        // the runner separately (`num_shapes` defaults to 1).
        let options = super::triposg_run_options_from_args(&args).expect("options");
        assert_eq!(options.num_tokens, 2048);
    }

    #[test]
    fn run_triposg_num_shapes_rejected_for_scribble_and_surface_input() {
        // `--mode scribble`: the official scribble script runs a single shape.
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "triposg",
            "--mode",
            "scribble",
            "--source",
            "scribble.png",
            "--prompt",
            "a chair",
            "--num-shapes",
            "2",
            "--output",
            "out.glb",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let error = super::reject_scribble_mode_conflicts(&args)
            .expect_err("--num-shapes must be rejected for scribble mode");
        assert!(
            error.to_string().contains("--num-shapes"),
            "unexpected message: `{error}`"
        );

        // `--surface-input` produces exactly one latent.
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "triposg",
            "--surface-input",
            "surface.npy",
            "--num-shapes",
            "2",
            "--output",
            "out.glb",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let error = super::reject_unsupported_triposg_options(&args)
            .expect_err("--num-shapes must be rejected with --surface-input");
        assert!(
            error.to_string().contains("--num-shapes"),
            "unexpected message: `{error}`"
        );
    }

    #[test]
    fn multiple_sources_rejected_for_single_source_families() {
        for family in ["pi3", "pi3x", "triposr", "trellis2", "vggt"] {
            let cli = Cli::parse_from([
                "lux3d", "run", family, "--source", "first", "--source", "second", "--output",
                "out",
            ]);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let error = super::required_source(&args, family)
                .expect_err(&format!("multiple sources must be rejected for {family}"));
            assert!(
                error.to_string().contains("exactly one `--source`"),
                "unexpected message for {family}: `{error}`"
            );
        }
    }

    #[test]
    fn num_shapes_rejected_for_other_families() {
        for family in ["pi3", "pi3x", "triposr", "vggt"] {
            let cli = Cli::parse_from([
                "lux3d",
                "run",
                family,
                "--source",
                "input",
                "--num-shapes",
                "2",
                "--output",
                "out",
            ]);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let error = super::reject_triposg_only_options(&args, args.family)
                .expect_err(&format!("--num-shapes must be rejected for {family}"));
            assert!(
                error.to_string().contains("--num-shapes"),
                "unexpected message for {family}: `{error}`"
            );
        }

        // `trellis2` does not route through `reject_triposg_only_options`;
        // its own guard must reject the flag (use `--num-samples` instead).
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "input",
            "--num-shapes",
            "2",
            "--output",
            "out",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let error = super::reject_unsupported_trellis2_options(&args)
            .expect_err("--num-shapes must be rejected for trellis2");
        assert!(
            error.to_string().contains("--num-shapes"),
            "unexpected message for trellis2: `{error}`"
        );
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
        assert!(args.source.is_empty());
        let options = super::triposg_run_options_from_args(&args).expect("surface options");
        // `scripts/inference_vae.py:50-58` decodes with
        // `hierarchical_extract_geometry` (no flash decoder, depths 8/9).
        assert!(!options.use_flash_decoder);
        assert_eq!(options.dense_octree_depth, 8);
        assert_eq!(options.hierarchical_octree_depth, 9);
        assert_eq!(options.num_tokens, 2048);
        assert_eq!(options.seed, 42);
    }

    /// The surface encode branch skips the denoise loop, so the denoise
    /// schedule flags would be silent no-ops and must be rejected.
    #[test]
    fn triposg_surface_input_rejects_denoise_flags() {
        let flag_options: [(&str, &[&str]); 3] = [
            ("--steps", &["--steps", "20"]),
            ("--guidance-scale", &["--guidance-scale", "5"]),
            ("--timesteps", &["--timesteps", "999,500,0"]),
        ];
        for (flag, extra) in flag_options {
            let mut argv = vec![
                "lux3d",
                "run",
                "triposg",
                "--surface-input",
                "surface.npy",
                "--output",
                "out.glb",
            ];
            argv.extend_from_slice(extra);
            let cli = Cli::parse_from(argv);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let error = super::reject_unsupported_triposg_options(&args)
                .expect_err(&format!("{flag} with --surface-input must be rejected"));
            let message = error.to_string();
            assert!(
                message.contains(flag) && message.contains("no effect with `--surface-input`"),
                "unexpected message for {flag}: `{message}`"
            );
        }

        // The same flags remain valid for image-to-3D denoising.
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "triposg",
            "--source",
            "image.png",
            "--steps",
            "20",
            "--guidance-scale",
            "5",
            "--output",
            "out.glb",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        super::reject_unsupported_triposg_options(&args)
            .expect("denoise flags are consumed by the image path");
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

    /// Pi3 accepts `--interval` (frame sampling) and `--conf-threshold` (export
    /// gate) but must reject the Pi3X conditioning, TripoSR meshing and RMBG
    /// options it would otherwise silently ignore.
    #[test]
    fn run_pi3_rejects_foreign_family_options() {
        let flag_options: [(&str, &[&str]); 6] = [
            ("--vo", &["--vo"]),
            ("--conditions", &["--conditions", "conditions.npz"]),
            ("--chunk-size", &["--chunk-size", "8"]),
            ("--overlap", &["--overlap", "4"]),
            ("--inject-condition", &["--inject-condition", "pose"]),
            ("--rmbg-model-path", &["--rmbg-model-path", "rmbg"]),
        ];
        for (flag, extra) in flag_options {
            let mut argv = vec![
                "lux3d", "run", "pi3", "--source", "frames", "--output", "out.ply",
            ];
            argv.extend_from_slice(extra);
            let cli = Cli::parse_from(argv);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let error = super::reject_unsupported_pi3_options(&args)
                .expect_err(&format!("{flag} must be rejected for pi3"));
            let message = error.to_string();
            assert!(
                message.contains("is not supported for `pi3`"),
                "unexpected message for {flag}: `{message}`"
            );
            assert!(message.contains(flag), "`{flag}` missing in `{message}`");
        }
    }

    #[test]
    fn pi3_keeps_interval_and_conf_threshold() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "pi3",
            "--source",
            "frames",
            "--interval",
            "2",
            "--conf-threshold",
            "0.5",
            "--output",
            "out.ply",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(Some(2), args.interval);
        assert_eq!(Some(0.5), args.conf_threshold);
        super::reject_unsupported_pi3_options(&args)
            .expect("--interval and --conf-threshold are supported for pi3");
    }

    /// `--mc-resolution` / `--mc-threshold` stay TripoSR inputs; every other
    /// shared flag the TripoSR runtime never reads must fail fast.
    #[test]
    fn run_triposr_rejects_foreign_family_options() {
        let flag_options: [(&str, &[&str]); 8] = [
            ("--vo", &["--vo"]),
            ("--conditions", &["--conditions", "conditions.npz"]),
            ("--chunk-size", &["--chunk-size", "8"]),
            ("--overlap", &["--overlap", "4"]),
            ("--inject-condition", &["--inject-condition", "pose"]),
            ("--interval", &["--interval", "2"]),
            ("--rmbg-model-path", &["--rmbg-model-path", "rmbg"]),
            ("--conf-threshold", &["--conf-threshold", "0.5"]),
        ];
        for (flag, extra) in flag_options {
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
            let error = super::reject_unsupported_triposr_options(&args)
                .expect_err(&format!("{flag} must be rejected for triposr"));
            let message = error.to_string();
            assert!(
                message.contains("is not supported for `triposr`"),
                "unexpected message for {flag}: `{message}`"
            );
            assert!(message.contains(flag), "`{flag}` missing in `{message}`");
        }
    }

    #[test]
    fn triposr_keeps_marching_cubes_flags() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "triposr",
            "--source",
            "horse.png",
            "--mc-resolution",
            "512",
            "--mc-threshold",
            "30",
            "--output",
            "out.obj",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        super::reject_unsupported_triposr_options(&args)
            .expect("--mc-resolution/--mc-threshold are supported for triposr");
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
    fn parses_scribble_standalone_quantize_flags() {
        let cli = Cli::parse_from([
            "lux3d",
            "weights",
            "quantize",
            "triposg",
            "--source",
            "raw-TripoSG-scribble",
            "--output",
            "triposg-scribble-q8_0.gguf",
            "--dtype",
            "q8_0",
            "--fallback-dtype",
            "f16",
            "--scribble",
            "--standalone",
        ]);
        let Command::Weights(args) = cli.command else {
            panic!("expected weights command");
        };
        let super::WeightsCommand::Quantize(args) = args.command else {
            panic!("expected weights quantize command");
        };
        assert!(args.scribble, "--scribble must be set");
        assert!(args.standalone, "--standalone must be set");
    }

    #[test]
    fn standalone_quantize_requires_scribble() {
        let error = Cli::try_parse_from([
            "lux3d",
            "weights",
            "quantize",
            "triposg",
            "--source",
            "raw-TripoSG-scribble",
            "--output",
            "triposg-scribble-q8_0.gguf",
            "--standalone",
        ])
        .expect_err("--standalone without --scribble must be rejected");
        assert!(
            error.to_string().contains("--scribble"),
            "unexpected clap error: {error}"
        );
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
    fn pi3x_rejects_vo_only_flags_without_vo() {
        let flag_options: [(&str, &[&str]); 4] = [
            ("--chunk-size", &["--chunk-size", "8"]),
            ("--overlap", &["--overlap", "4"]),
            ("--conf-threshold", &["--conf-threshold", "0.5"]),
            ("--inject-condition", &["--inject-condition", "pose"]),
        ];
        for (flag, extra) in flag_options {
            let mut argv = vec![
                "lux3d", "run", "pi3x", "--source", "frames", "--output", "out.ply",
            ];
            argv.extend_from_slice(extra);
            let cli = Cli::parse_from(argv);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let error = super::reject_unsupported_pi3x_options(&args)
                .expect_err(&format!("{flag} without --vo must be rejected"));
            let message = error.to_string();
            assert!(
                message.contains(flag) && message.contains("requires `--vo`"),
                "unexpected message for {flag}: `{message}`"
            );
        }

        // With `--vo` the same flags feed the VO pipeline and stay accepted.
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "pi3x",
            "--source",
            "frames",
            "--vo",
            "--chunk-size",
            "8",
            "--overlap",
            "4",
            "--conf-threshold",
            "0.5",
            "--inject-condition",
            "pose",
            "--output",
            "out.ply",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        super::reject_unsupported_pi3x_options(&args)
            .expect("VO flags are consumed when --vo is set");
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
            source: vec![test_asset("pi3-frames")],
            surface_input: None,
            mode: super::TriposgMode::Image,
            prompt: None,
            scribble_conf: None,
            prompt_conf: None,
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
            num_shapes: None,
            dense_octree_depth: None,
            hierarchical_octree_depth: None,
            flash_decoder: None,
            flash_octree_depth: None,
            faces: None,
            dtype: None,
            bounds: None,
            timesteps: None,
            model_path: Some(smoke_model_dir("pi3-q8_0")),
            rmbg_model_path: None,
            cache_dir: None,
            output: output.clone(),
            dump_npz: None,
            vggt: Default::default(),
            pipeline_type: None,
            no_low_vram: false,
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
            no_remesh: false,
            remesh_band: None,
            remesh_project: None,
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
            source: vec![test_asset("mesh-input.png")],
            surface_input: None,
            mode: super::TriposgMode::Image,
            prompt: None,
            scribble_conf: None,
            prompt_conf: None,
            interval: None,
            conditions: None,
            vo: false,
            chunk_size: None,
            overlap: None,
            conf_threshold: None,
            inject_condition: Vec::new(),
            mc_resolution: Some(128),
            mc_threshold: Some(25.0),
            steps: None,
            guidance_scale: None,
            seed: None,
            num_tokens: None,
            num_shapes: None,
            dense_octree_depth: None,
            hierarchical_octree_depth: None,
            flash_decoder: None,
            flash_octree_depth: None,
            faces: None,
            dtype: None,
            bounds: None,
            timesteps: None,
            model_path: Some(smoke_model_dir("triposr-q8_0")),
            rmbg_model_path: None,
            cache_dir: None,
            output: output.clone(),
            dump_npz: None,
            vggt: Default::default(),
            pipeline_type: None,
            no_low_vram: false,
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
            no_remesh: false,
            remesh_band: None,
            remesh_project: None,
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
            source: vec![test_asset("mesh-input.png")],
            surface_input: None,
            mode: super::TriposgMode::Image,
            prompt: None,
            scribble_conf: None,
            prompt_conf: None,
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
            num_shapes: None,
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
            no_low_vram: false,
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
            no_remesh: false,
            remesh_band: None,
            remesh_project: None,
            decimation_target: None,
            fill_hole_perimeter: None,
            min_component_area: None,
        };

        let err = run_model(args).expect_err("invalid marching-cubes resolution must fail");
        assert!(
            err.to_string()
                .contains("--mc-resolution must be at least 2")
        );
    }

    #[test]
    fn run_pi3x_smoke_emits_ply_output() {
        let _guard = GpuTestLock::acquire().expect("gpu test lock");
        let output =
            std::env::temp_dir().join(format!("lux3d-cli-pi3x-{}.ply", std::process::id()));
        let args = RunArgs {
            family: super::Family::Pi3x,
            device: super::DeviceBackend::Cuda,
            source: vec![test_asset("pi3-frames")],
            surface_input: None,
            mode: super::TriposgMode::Image,
            prompt: None,
            scribble_conf: None,
            prompt_conf: None,
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
            num_shapes: None,
            dense_octree_depth: None,
            hierarchical_octree_depth: None,
            flash_decoder: None,
            flash_octree_depth: None,
            faces: None,
            dtype: None,
            bounds: None,
            timesteps: None,
            model_path: Some(smoke_model_dir("pi3x-q8_0")),
            rmbg_model_path: None,
            cache_dir: None,
            output: output.clone(),
            dump_npz: None,
            vggt: Default::default(),
            pipeline_type: None,
            no_low_vram: false,
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
            no_remesh: false,
            remesh_band: None,
            remesh_project: None,
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
            source: vec![test_asset("pi3-frames")],
            surface_input: None,
            mode: super::TriposgMode::Image,
            prompt: None,
            scribble_conf: None,
            prompt_conf: None,
            interval: None,
            conditions: None,
            vo: true,
            chunk_size: Some(2),
            overlap: Some(1),
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
            num_shapes: None,
            dense_octree_depth: None,
            hierarchical_octree_depth: None,
            flash_decoder: None,
            flash_octree_depth: None,
            faces: None,
            dtype: None,
            bounds: None,
            timesteps: None,
            model_path: Some(smoke_model_dir("pi3x-q8_0")),
            rmbg_model_path: None,
            cache_dir: None,
            output: output.clone(),
            dump_npz: None,
            vggt: Default::default(),
            pipeline_type: None,
            no_low_vram: false,
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
            no_remesh: false,
            remesh_band: None,
            remesh_project: None,
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

        // 1536 is an upstream app choice (`app_texturing.py:101`): accepted,
        // voxelizes/encodes at 1536, keeps the 1024 conditioning config.
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--texture-mesh",
            "input.glb",
            "--texture-resolution",
            "1536",
            "--output",
            "out.glb",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let options = super::trellis2_texturing_options_from_args(&args).expect("options");
        assert_eq!(1536, options.resolution);
        assert_eq!(1024, options.cond_resolution());

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

    /// `--voxel-size` without an explicit `--grid-size` must leave the grid
    /// unset so the core derives it; an explicit or default grid still maps.
    #[test]
    fn mesh2ovox_grid_size_mapping_auto_derives_from_voxel_size() {
        assert_eq!(
            Some([256, 256, 256]),
            super::resolve_mesh2ovox_grid_size(Some(&[256]), &[0.01]).expect("explicit grid")
        );
        assert_eq!(
            Some([64, 128, 256]),
            super::resolve_mesh2ovox_grid_size(Some(&[64, 128, 256]), &[]).expect("per-axis grid")
        );

        let cli = Cli::parse_from([
            "lux3d",
            "ovox",
            "mesh2ovox",
            "in.glb",
            "out.vxz",
            "--voxel-size",
            "0.01",
        ]);
        let Command::Ovox(args) = cli.command else {
            panic!("expected ovox command");
        };
        let super::OvoxCommand::Mesh2Ovox(a) = args.command else {
            panic!("expected mesh2ovox");
        };
        assert_eq!(None, a.grid_size, "--grid-size must stay unset");
        assert_eq!(
            Some([0.01, 0.01, 0.01]),
            super::resolve_voxel_size(&a.voxel_size).expect("voxel size")
        );
        assert_eq!(
            None,
            super::resolve_mesh2ovox_grid_size(a.grid_size.as_deref(), &a.voxel_size)
                .expect("auto-derived grid")
        );

        let cli = Cli::parse_from(["lux3d", "ovox", "mesh2ovox", "in.glb", "out.vxz"]);
        let Command::Ovox(args) = cli.command else {
            panic!("expected ovox command");
        };
        let super::OvoxCommand::Mesh2Ovox(a) = args.command else {
            panic!("expected mesh2ovox");
        };
        assert_eq!(
            Some([512, 512, 512]),
            super::resolve_mesh2ovox_grid_size(a.grid_size.as_deref(), &a.voxel_size)
                .expect("default grid")
        );

        let cli = Cli::parse_from([
            "lux3d",
            "ovox",
            "mesh2ovox",
            "in.glb",
            "out.vxz",
            "--grid-size",
            "128",
            "--voxel-size",
            "0.02",
        ]);
        let Command::Ovox(args) = cli.command else {
            panic!("expected ovox command");
        };
        let super::OvoxCommand::Mesh2Ovox(a) = args.command else {
            panic!("expected mesh2ovox");
        };
        assert_eq!(
            Some([128, 128, 128]),
            super::resolve_mesh2ovox_grid_size(a.grid_size.as_deref(), &a.voxel_size)
                .expect("explicit grid wins")
        );
    }
}
