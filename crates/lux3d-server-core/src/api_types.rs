//! OpenAI-style HTTP API types for Lux3D generation endpoints.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum GenerationStatus {
    Queued,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

impl GenerationStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct GenerationError {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct GenerationObject {
    pub id: String,
    pub object: &'static str,
    pub status: String,
    pub model: String,
    pub created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
    pub progress: u8,
    pub output_format: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<GenerationError>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct GenerationListResponse {
    pub object: &'static str,
    pub data: Vec<GenerationObject>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
    pub kind: String,
    pub output_format: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelListResponse {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ServerInfoResponse {
    pub object: &'static str,
    pub name: String,
    pub version: String,
    pub routes: Vec<RouteSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RouteSummary {
    pub path: String,
    pub methods: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, ToSchema)]
pub struct PointCloudGenerationOptions {
    /// Point-cloud artifact format: `ply` (default) or `xyz`.
    #[serde(default)]
    pub output_format: Option<String>,
    pub interval: Option<usize>,
    pub vo: Option<bool>,
    pub chunk_size: Option<usize>,
    pub overlap: Option<usize>,
    pub conf_threshold: Option<f32>,
    #[serde(default)]
    pub inject_condition: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct MeshGenerationOptions {
    /// Mesh artifact format: `glb` (default), `obj`, `ply` or `stl`.
    #[serde(default)]
    pub output_format: Option<String>,
    #[serde(default = "default_mc_resolution")]
    pub mc_resolution: u32,
    #[serde(default = "default_mc_threshold")]
    pub mc_threshold: f32,
    // TripoSG-only options (`Option`-typed so omitted fields stay `None` and
    // triposr keeps working unchanged).
    #[serde(default)]
    pub steps: Option<usize>,
    #[serde(default)]
    pub guidance_scale: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub num_tokens: Option<usize>,
    #[serde(default)]
    pub dense_octree_depth: Option<u32>,
    #[serde(default)]
    pub hierarchical_octree_depth: Option<u32>,
    #[serde(default)]
    pub use_flash_decoder: Option<bool>,
    #[serde(default)]
    pub flash_octree_depth: Option<u32>,
    #[serde(default)]
    pub query_chunk_size: Option<usize>,
    /// Explicit rectified-flow timestep schedule (upstream
    /// `TripoSGPipeline.__call__(timesteps=...)`); when set, `steps` is
    /// ignored and the denoise loop runs exactly these timesteps.
    #[serde(default)]
    pub timesteps: Option<Vec<f32>>,
    /// SDF query bounds override `[min_x, min_y, min_z, max_x, max_y, max_z]`
    /// (default `[-1.005, -1.005, -1.005, 1.005, 1.005, 1.005]`).
    #[serde(default)]
    pub bounds: Option<[f32; 6]>,
    /// Target face count after mesh extraction (`0` disables simplification;
    /// at or above the current face count is a no-op).
    #[serde(default)]
    pub faces: Option<usize>,
    // TRELLIS.2-only options. `steps` and `seed` are shared with TripoSG and
    // feed every TRELLIS.2 Euler sampler; the per-stage `ss_*`/`shape_*`/
    // `tex_*` fields override them for one stage (the per-stage override wins
    // when both set a value, matching `resolve_stage_params`).
    /// TRELLIS.2 pipeline variant: `512`, `1024`, `1024_cascade` (reference
    /// default), or `1536_cascade` (HR cascade targeting the 1536 grid).
    #[serde(default)]
    pub pipeline_type: Option<String>,
    /// TRELLIS.2 cascade token budget (default 49152).
    #[serde(default)]
    pub max_num_tokens: Option<usize>,
    /// Run the TRELLIS.2 texture stages (texture SLAT + guided VAE + PBR
    /// vertex colors, default true); `false` exports the geometry-only mesh.
    #[serde(default)]
    pub texture: Option<bool>,
    /// Sparse-structure Euler step count (defaults to `steps`).
    #[serde(default)]
    pub ss_steps: Option<usize>,
    /// Sparse-structure classifier-free guidance strength (`1.0` disables).
    #[serde(default)]
    pub ss_guidance_strength: Option<f32>,
    /// Sparse-structure CFG rescale (`0` disables).
    #[serde(default)]
    pub ss_guidance_rescale: Option<f32>,
    /// Sparse-structure inclusive t interval `[lo, hi]` where CFG applies.
    #[serde(default)]
    pub ss_guidance_interval: Option<[f32; 2]>,
    /// Sparse-structure t-schedule rescale factor.
    #[serde(default)]
    pub ss_rescale_t: Option<f32>,
    /// Shape SLAT Euler step count (defaults to `steps`; applies to both
    /// cascade shape stages).
    #[serde(default)]
    pub shape_steps: Option<usize>,
    /// Shape SLAT classifier-free guidance strength (`1.0` disables).
    #[serde(default)]
    pub shape_guidance_strength: Option<f32>,
    /// Shape SLAT CFG rescale (`0` disables).
    #[serde(default)]
    pub shape_guidance_rescale: Option<f32>,
    /// Shape SLAT inclusive t interval `[lo, hi]` where CFG applies.
    #[serde(default)]
    pub shape_guidance_interval: Option<[f32; 2]>,
    /// Shape SLAT t-schedule rescale factor.
    #[serde(default)]
    pub shape_rescale_t: Option<f32>,
    /// Texture SLAT Euler step count (defaults to `steps`).
    #[serde(default)]
    pub tex_steps: Option<usize>,
    /// Texture SLAT classifier-free guidance strength (`1.0` disables).
    #[serde(default)]
    pub tex_guidance_strength: Option<f32>,
    /// Texture SLAT CFG rescale (`0` disables).
    #[serde(default)]
    pub tex_guidance_rescale: Option<f32>,
    /// Texture SLAT inclusive t interval `[lo, hi]` where CFG applies.
    #[serde(default)]
    pub tex_guidance_interval: Option<[f32; 2]>,
    /// Texture SLAT t-schedule rescale factor.
    #[serde(default)]
    pub tex_rescale_t: Option<f32>,
    /// Sample count (default 1). Values above 1 run the sequential
    /// seed-offset emulation and export one artifact per sample with an
    /// `_i` suffix; sample 0 is served through `/content`.
    #[serde(default)]
    pub num_samples: Option<usize>,
    /// Baked texture edge length (default 2048; multiple of 8 in `8..=8192`).
    #[serde(default)]
    pub texture_size: Option<usize>,
    /// Run the geometry post-process (default true); `false` keeps the
    /// decoded mesh untouched.
    #[serde(default)]
    pub postprocess: Option<bool>,
    /// Final triangle-count target (default 1_000_000; `0` skips
    /// simplification).
    #[serde(default)]
    pub decimation_target: Option<usize>,
    /// Boundary loops whose total perimeter is below this are closed
    /// (default `3e-2`; `0` disables hole filling).
    #[serde(default)]
    pub fill_hole_perimeter: Option<f32>,
    /// Connected components with total triangle area below this are removed
    /// (default `1e-5`; `0` keeps every component).
    #[serde(default)]
    pub min_component_area: Option<f32>,
}

fn default_mc_resolution() -> u32 {
    256
}

fn default_mc_threshold() -> f32 {
    25.0
}

impl Default for MeshGenerationOptions {
    fn default() -> Self {
        Self {
            output_format: None,
            mc_resolution: default_mc_resolution(),
            mc_threshold: default_mc_threshold(),
            steps: None,
            guidance_scale: None,
            seed: None,
            num_tokens: None,
            dense_octree_depth: None,
            hierarchical_octree_depth: None,
            use_flash_decoder: None,
            flash_octree_depth: None,
            query_chunk_size: None,
            timesteps: None,
            bounds: None,
            faces: None,
            pipeline_type: None,
            max_num_tokens: None,
            texture: None,
            ss_steps: None,
            ss_guidance_strength: None,
            ss_guidance_rescale: None,
            ss_guidance_interval: None,
            ss_rescale_t: None,
            shape_steps: None,
            shape_guidance_strength: None,
            shape_guidance_rescale: None,
            shape_guidance_interval: None,
            shape_rescale_t: None,
            tex_steps: None,
            tex_guidance_strength: None,
            tex_guidance_rescale: None,
            tex_guidance_interval: None,
            tex_rescale_t: None,
            num_samples: None,
            texture_size: None,
            postprocess: None,
            decimation_target: None,
            fill_hole_perimeter: None,
            min_component_area: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct PointCloudGenerationRequest {
    pub model: String,
    #[serde(default)]
    pub options: PointCloudGenerationOptions,
    #[serde(default)]
    pub webhook_url: Option<String>,
    pub source_url: Option<String>,
    pub source_file_id: Option<String>,
    pub source_base64: Option<String>,
    pub source_filename: Option<String>,
    pub conditions_file_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct MeshGenerationRequest {
    #[serde(default = "default_mesh_model")]
    pub model: String,
    #[serde(default)]
    pub options: MeshGenerationOptions,
    #[serde(default)]
    pub webhook_url: Option<String>,
    pub source_url: Option<String>,
    pub source_file_id: Option<String>,
    pub source_base64: Option<String>,
    pub source_filename: Option<String>,
}

fn default_mesh_model() -> String {
    "triposr".to_string()
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, ToSchema)]
pub struct MeshTexturingOptions {
    /// Textured artifact format; only `glb` is supported (the baked PBR
    /// atlas is embedded in the GLB).
    #[serde(default)]
    pub output_format: Option<String>,
    /// RNG seed for the texture SLAT noise (default 42).
    #[serde(default)]
    pub seed: Option<u64>,
    /// Voxelization / shape-encode / texture-sampling resolution: `512`,
    /// `1024` (default) or `1536`.
    #[serde(default)]
    pub texture_resolution: Option<u32>,
    /// Baked texture edge length (default 2048; multiple of 8 in `8..=8192`).
    #[serde(default)]
    pub texture_size: Option<usize>,
    /// Texture SLAT Euler step count (defaults to the pipeline config, 12).
    #[serde(default)]
    pub tex_steps: Option<usize>,
    /// Texture SLAT classifier-free guidance strength (`1.0` disables).
    #[serde(default)]
    pub tex_guidance_strength: Option<f32>,
    /// Texture SLAT CFG rescale (`0` disables).
    #[serde(default)]
    pub tex_guidance_rescale: Option<f32>,
    /// Texture SLAT inclusive t interval `[lo, hi]` where CFG applies.
    #[serde(default)]
    pub tex_guidance_interval: Option<[f32; 2]>,
    /// Texture SLAT t-schedule rescale factor.
    #[serde(default)]
    pub tex_rescale_t: Option<f32>,
    /// Run the reference condition-image preprocessing (foreground square
    /// crop + premultiplied RGB, RMBG-2.0 for opaque inputs; default true).
    #[serde(default)]
    pub preprocess_image: Option<bool>,
    /// Run `preprocess_mesh` (center/scale/y-z swap; default true).
    #[serde(default)]
    pub preprocess_mesh: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct MeshTexturingRequest {
    #[serde(default = "default_texturing_model")]
    pub model: String,
    #[serde(default)]
    pub options: MeshTexturingOptions,
    #[serde(default)]
    pub webhook_url: Option<String>,
    /// Condition image (`source`) URL.
    pub source_url: Option<String>,
    /// Condition image uploaded through the Files API.
    pub source_file_id: Option<String>,
    /// Condition image as a base64 payload.
    pub source_base64: Option<String>,
    /// Condition image filename override.
    pub source_filename: Option<String>,
    /// Mesh to texture (.glb/.obj/.ply/.stl) URL.
    pub mesh_url: Option<String>,
    /// Mesh to texture uploaded through the Files API.
    pub mesh_file_id: Option<String>,
    /// Mesh to texture as a base64 payload.
    pub mesh_base64: Option<String>,
    /// Mesh filename override (drives the reader extension).
    pub mesh_filename: Option<String>,
}

fn default_texturing_model() -> String {
    "trellis2".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ErrorBody {
    pub message: String,
}
