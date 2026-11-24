//! TRELLIS.2 image-to-3D pipeline orchestration (T5 geometry slice + T6
//! 1024 cascade + upstream-parity variants).
//!
//! Pipeline variants (`Trellis2RunOptions::pipeline_type`, reference
//! `run(...)`, `trellis2_image_to_3d.py:489-595`; `ss_res` map
//! `{'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}`):
//!
//! 512 path (`pipeline_type = "512"`):
//!
//! 1. preprocess: RGBA foreground bbox -> square crop -> premultiplied RGB
//!    at 512 (`preprocess::Trellis2Preprocess`).
//! 2. conditioning: DINOv3 ViT-L/16 -> `cond [1, 1029, 1024]`,
//!    `neg_cond = zeros_like(cond)`.
//! 3. sparse structure: seeded noise `[1, 8, 16, 16, 16]` -> 12-step Euler
//!    CFG sampler (strength 7.5, rescale 0.7, interval [0.6, 1.0],
//!    rescale_t 5.0) -> `ss_dec` decoder -> occupancy at 32^3.
//! 4. shape SLAT: seeded noise `[N, 32]` on the active coords -> 12-step
//!    Euler CFG sampler (strength 7.5, rescale 0.5, interval [0.6, 1.0],
//!    rescale_t 3.0) -> de-normalize `slat * std + mean`.
//! 5. shape VAE decode at resolution 512 -> dual-grid head split ->
//!    O-Voxel mesh extraction -> geometry post-process (upstream `to_glb`
//!    branches: shared hole filling, then the narrow-band dual-contouring
//!    remesh + single simplify pass by default, or the cleanup/decimation
//!    chain with `remesh = false`; [`Trellis2MeshPostprocessOptions`],
//!    enabled by default) -> export.
//!
//! 1024 path (`pipeline_type = "1024"`): the same single-resolution flow at
//! SS resolution 64, conditioning at 1024 (`shape_dit_1024` /
//! `tex_dit_1024`), decode at 1024, no cascade.
//!
//! Cascade paths (`1024_cascade` target 1024, `1536_cascade` target 1536;
//! reference `sample_shape_slat_cascade`, `trellis2_image_to_3d.py:277-364`):
//! the SS stage is unchanged (32^3 occupancy, `cond_512`); the shape stages
//! become
//!
//! 4a. LR shape SLAT on the 32^3 coords with `shape_dit_512` + `cond_512`
//!     -> de-normalize.
//! 4b. shape VAE `upsample(slat, 4)` (`Trellis2ShapeVaeDecoder::
//!     upsample_coords`): `from_latent` + stages 0..3 exactly like the full
//!     decode, returning the children of the stage-3 C2S — the active coords
//!     on the 512 grid.
//! 4c. quantize onto the `hr_resolution // 16` grid:
//!     `((coord + 0.5) / 512 * (hr_resolution // 16)).int()`, deduplicate
//!     (lexicographically sorted rows, `torch.unique(dim=0)` semantics) and
//!     enforce the token budget: the loop starts at `hr_resolution = target`
//!     and breaks when `rows < max_num_tokens || hr_resolution == 1024`,
//!     else `hr_resolution -= 128`. The stop resolution is 1024 for BOTH
//!     cascade variants (the reference hardcodes it — the 1024-DiT latent
//!     grid ceiling `hr_resolution // 16 <= 64`... the elastic models accept
//!     larger grids, so the 1536 cascade decodes at 1536 unless the token
//!     budget reduces it to 1024). The 1024 cascade always breaks on its
//!     first pass. Multi-sample runs concatenate the per-sample quantized
//!     rows (distinct across samples, like the reference batch column) and
//!     run ONE loop, so the budget and the returned resolution are global
//!     (`trellis2_image_to_3d.py:327-339`).
//! 4d. HR shape SLAT on the quantized coords with `shape_dit_1024` +
//!     `cond_1024` (DINOv3 at 1024, `[1, 4101, 1024]`) -> de-normalize.
//! 5. shape VAE decode at the returned `hr_resolution` -> dual-grid head
//!    split -> O-Voxel mesh extraction.
//!
//! Texture (all paths, reference `sample_tex_slat` / `decode_tex_slat`,
//! enabled with `Trellis2RunOptions::with_texture`):
//!
//! 6. texture SLAT on the FINAL shape latent coords (the HR quantized
//!    coords in the cascades): noise `[N, 32]` re-normalized shape SLAT
//!    `(shape_slat - mean) / std` concatenated along channels -> `[N, 64]`
//!    -> `tex_dit_512` (512 path, `cond_512`) or `tex_dit_1024` (1024/1536
//!    paths, `cond_1024`) -> 12-step Euler sampler (guidance 1.0, rescale
//!    0.0, interval [0.6, 0.9], R=3.0) -> de-normalize with the texture
//!    vectors.
//! 7. guided texture VAE decode (`tex_dec`, `pred_subdiv = false`): the
//!    shape decoder's raw C2S subdivision logits steer the tex decode, so
//!    the tex output coords equal the shape decode's final coords; the
//!    decoded 6 channels are `raw * 0.5 + 0.5` clamped to [0, 1]
//!    (base_color 0:3, metallic 3, roughness 4, alpha 5).
//! 8. per-vertex base colors: flex_gemm sparse trilinear sampling of the
//!    voxel attributes at the mesh vertices (`MeshWithVoxel.
//!    query_vertex_attrs`) — sampled on the POST-PROCESSED mesh vertices
//!    when the geometry post-process ran (step 5), so no vertex-color
//!    remap is needed; carried on the mesh buffers for PLY/GLB export
//!    alongside the `<stem>.pbrvox.bin` sidecar payload.
//!
//! Textured export (all paths, reference `postprocess.py:190-330`
//! `to_glb`, enabled with [`Trellis2TextureExportOptions::enabled`]
//! inside [`Trellis2RunOptions::texture_export`], default on when
//! texture ran):
//!
//! 9. UV unwrap + PBR bake: xatlas charting/packing at
//!    `texture_size` (default 2048, see [`crate::geometry::uv_unwrap`]),
//!    area-weighted vertex normals, conservative rasterization of the
//!    UV triangles, sparse-trilinear sampling of the same voxel
//!    attributes per texel, and TELEA-style frontier inpainting of the
//!    uncovered texels (see [`crate::geometry::bake_pbr_texture`]). The
//!    result ([`Trellis2Stages::textured`], [`Trellis2TexturedMesh`])
//!    carries atlas UVs + indices, the vertex normals, and both baked
//!    textures; [`Trellis2ExportStage::write_textured_glb`] emits the
//!    textured GLB (embedded PNGs, axis conversion `(x, y, z) -> (x, z,
//!    -y)`, UV V flip). Skipped cleanly (`textured = None`) when texture
//!    export is disabled, `uv_unwrap` is disabled, the voxel attributes
//!    are not the 6-channel PBR layout, or the mesh has no faces.
//!    CPU cost is material (xatlas charting + ~4M texel samples at
//!    2048); see [`crate::geometry::BakedPbrTexture`] for details.
//!    The to_glb path runs the upstream snap-back
//!    ([`build_trellis2_textured_mesh_snapped`]): every covered texel's 3D
//!    position is projected onto the pre-postprocess (original) surface
//!    through a triangle BVH before the attribute volume is sampled
//!    (`postprocess.py:249-266, 254-256`). The standalone texturing
//!    pipeline (`runtime::trellis2_texture`) has no snap-back
//!    (`trellis2_texturing.py:316-332`) and keeps calling the plain
//!    [`build_trellis2_textured_mesh`].
//!
//! Stage weights load sequentially from the mmap-ed canonical package and
//! drop after their stage in the default `low_vram = true` mode (reference
//! `trellis2_image_to_3d.py:55, 107, 119-125`); no more than one DiT is
//! resident (the cascade also reloads the shape VAE between its upsample
//! and decode passes). With [`Trellis2RunOptions::low_vram`] = false (CLI
//! `--no-low-vram`) every stage the run loads is kept resident on the
//! device until the run ends — the reference `to(device)` pinning branch
//! (`trellis2_image_to_3d.py:119-125`). This is a speed-only switch: the
//! sampler math and RNG streams are unchanged, so the outputs are identical
//! for the same seed. A stage load that does not fit in device memory fails
//! with an actionable error naming the flag and the device memory limit; it
//! never silently falls back to per-stage loading. The RNG is an internal
//! CPU-seeded normal generator (locked decision 4): reproducible across
//! Lux3D runs, NOT bit-identical to `torch.randn` — parity tests inject
//! oracle latents instead of comparing RNG streams.
//!
//! Denormalization vectors and sampler defaults are hard-coded from
//! `pipeline.json` (mirrored in the canonical package's
//! `resolved_config.json`) and can be overridden via
//! [`Trellis2PipelineConfig::from_resolved_config`].
//!
//! Per-stage sampler overrides ([`Trellis2SamplerOverrides`] in
//! [`Trellis2RunOptions::ss_sampler`]/`shape_sampler`/`tex_sampler`, the
//! upstream `run(...)` `*_sampler_params` arguments): each field replaces
//! the pipeline-config default (`steps`, `guidance_strength`,
//! `guidance_rescale`, `rescale_t`, `guidance_interval`). The legacy
//! `ss_steps`/`shape_steps`/`tex_steps` options stay the base step count for
//! all three stages; a stage override's `steps` wins over the legacy field
//! when both are set (`resolve_stage_params`).
//!
//! `num_samples` (upstream `run(num_samples)` batching): `num_samples = 1`
//! keeps the single-output path and is bit-identical to the
//! pre-`num_samples` behavior. `num_samples > 1` follows the upstream draw
//! structure (`trellis2_image_to_3d.py:208, 253`): the sparse structure
//! draws ONE joint noise tensor `[num_samples, 8, R, R, R]` per stage
//! (`torch.randn(num_samples, in_channels, reso, reso, reso)`) and every
//! SLAT stage draws ONE joint noise tensor over the concatenated active
//! coords of all samples (`torch.randn(coords.shape[0], in_channels)`), all
//! samples share one conditioning pass, and decoding/export run per sample.
//! Cascade runs quantize/upsample every sample first and run ONE token-budget
//! loop over the concatenated batch coords, so all samples share a single
//! `hr_resolution` and the global token cap is `max_num_tokens`, not
//! `num_samples * max_num_tokens` (`trellis2_image_to_3d.py:327-339,
//! 586-591`; per-sample rows stay distinct exactly like the reference's
//! leading batch column). The Lux3D RNG is not bit-identical to
//! `torch.randn` (locked decision 4),
//! so the draws are distribution-equivalent, not value-identical, to the
//! reference; only the draw structure matches. The single-run entry points
//! error on `num_samples != 1`.
//!
//! `Trellis2Stages::write_latent_npz` is the upstream `return_latent=True`
//! surface (`(out_mesh, (shape_slat, tex_slat, res))`): an `.npz` with
//! `shape_latent [N, 32]` F32, `tex_latent [N, 32]` F32 (empty when texture
//! is off), `coords [N, 3]` I32, `resolution` scalar, cascade
//! `lr_shape_latent [N_lr, 32]` F32 (empty for non-cascade runs) and a
//! UTF-8 JSON `meta` blob.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use candle_core::{DType, Device, Tensor};
use ndarray::Array2;

use super::trellis2_dino::Trellis2Dino;
use super::trellis2_dit::{Trellis2SlatConfig, Trellis2SlatDit, Trellis2SsConfig, Trellis2SsDit};
use super::trellis2_mesh::DualGridMesh;
use super::trellis2_sampler::{sample_dense, sample_sparse, Trellis2SamplerParams};
use super::trellis2_ss_dec::{occupancy_coords, Trellis2SsDecoder, Trellis2SsDecoderConfig};
use super::trellis2_tex::{
    pbr_attrs_from_raw, query_vertex_colors, Trellis2TexDecoder, Trellis2TexDit,
};
use super::trellis2_vae::{decode_shape_mesh, Trellis2ShapeVaeConfig, Trellis2ShapeVaeDecoder};
use crate::contracts::Trellis2PipelineType;
use crate::error::Lux3dError;
use crate::export::Trellis2ExportStage;
use crate::geometry::{
    bake_pbr_texture, bake_pbr_texture_with_snap, fill_holes_with_centroids, uv_unwrap,
    welded_vertex_normals, MeshBvh, Trellis2GeometryStage,
};
use crate::preprocess::{Trellis2BackgroundRemover, Trellis2Preprocess};
use crate::runtime::Rmbg2Model;
use crate::weights::{
    load_canonical_weights, CanonicalWeightSetPaths, ModelAssetOptions, VarBuilder,
};
use crate::ModelFamily;
use crate::Result;

/// The voxel margin of the FlexiDualGrid head (`fdg_vae.py` default).
pub const TRELLIS2_VOXEL_MARGIN: f32 = 0.5;

/// Cascade LR grid resolution (`sample_shape_slat_cascade(lr_resolution=512)`
/// in `run(...)`, `trellis2_image_to_3d.py:566-573`).
pub const TRELLIS2_CASCADE_LR_RESOLUTION: u32 = 512;

/// Cascade stop/conditioning resolution (1024): the reference token-budget
/// loop hardcodes `hr_resolution == 1024` as an unconditional break, and the
/// HR conditioning is `get_cond([image], 1024)` for the `1024`/cascade
/// variants alike. The 1024 cascade's target equals the stop resolution, so
/// it always decodes at 1024 regardless of the token count; the 1536 cascade
/// can only reduce 1536 -> 1408 -> ... -> 1024.
pub const TRELLIS2_CASCADE_HR_RESOLUTION: u32 = 1024;

/// Single-resolution `1024` path: SS occupancy resolution (upstream
/// `ss_res['1024'] = 64`; the SS decoder's 64^3 occupancy is used as-is) and
/// the mesh extraction resolution.
pub const TRELLIS2_SINGLE_1024_SS_RESOLUTION: u32 = 64;
pub const TRELLIS2_SINGLE_1024_SHAPE_RESOLUTION: u32 = 1024;

/// Cascade target resolutions (`resolution` argument of the reference
/// `sample_shape_slat_cascade`): 1024 for `1024_cascade`, 1536 for
/// `1536_cascade`.
pub const TRELLIS2_CASCADE_1536_TARGET_RESOLUTION: u32 = 1536;

/// Cascade C2S upsample stages (`upsample(slat, upsample_times=4)`): 32^3 LR
/// coords double four times to the 512 grid.
pub const TRELLIS2_CASCADE_UPSAMPLE_TIMES: usize = 4;

/// RNG salt for the texture SLAT noise stream (`Trellis2Rng::new(seed ^
/// SALT)`); the shape stages use their own salts so the streams differ.
pub const TRELLIS2_TEX_NOISE_SALT: u64 = 0x534C_4154_5F54_4558;

/// RNG salt for the sparse-structure noise stream. The upstream draws it
/// from the globally seeded torch RNG (`torch.manual_seed(seed)` +
/// `torch.randn(...)`, `trellis2_image_to_3d.py:208`); the Lux3D port keeps
/// a distinct per-stage stream (documented RNG deviation).
const TRELLIS2_SS_NOISE_SALT: u64 = 0x5353_5F4E_4F49_5345;

/// RNG salt for the base (LR in the cascades) shape SLAT noise stream.
const TRELLIS2_SHAPE_NOISE_SALT: u64 = 0x534C_4154_5F4E_4F49;

/// RNG salt for the cascade HR shape SLAT noise stream.
const TRELLIS2_HR_SHAPE_NOISE_SALT: u64 = 0x534C_4154_5F48_5231;

/// `shape_slat_normalization.mean` (`pipeline.json`, 32 channels).
pub const TRELLIS2_SHAPE_SLAT_MEAN: [f32; 32] = [
    0.781296, 0.018091, -0.495192, -0.558457, 1.06053, 0.093252, 1.518149, -0.933218, -0.732996,
    2.604095, -0.118341, -2.143904, 0.495076, -2.179512, -2.130751, -0.996944, 0.261421, -2.217463,
    1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667, -0.059621, 2.22078, 1.621212,
    0.87723, 0.567247, -3.175944, -3.186688, 1.578665,
];

/// `shape_slat_normalization.std` (`pipeline.json`, 32 channels).
pub const TRELLIS2_SHAPE_SLAT_STD: [f32; 32] = [
    5.972266, 4.706852, 5.44501, 5.209927, 5.32022, 4.547237, 5.020802, 5.444004, 5.226681,
    5.683095, 4.831436, 5.286469, 5.652043, 5.367606, 5.525084, 4.730578, 4.805265, 5.124013,
    5.530808, 5.619001, 5.10393, 5.41767, 5.269677, 5.547194, 5.634698, 5.235274, 6.110351,
    5.511298, 6.237273, 4.879207, 5.347008, 5.405691,
];

/// `tex_slat_normalization.mean` (`pipeline.json`, 32 channels).
pub const TRELLIS2_TEX_SLAT_MEAN: [f32; 32] = [
    3.501659, 2.212398, 2.226094, 0.251093, -0.026248, -0.687364, 0.439898, -0.928075, 0.029398,
    -0.339596, -0.869527, 1.038479, -0.972385, 0.126042, -1.129303, 0.455149, -1.209521, 2.069067,
    0.544735, 2.569128, -0.323407, 2.293, -1.925608, -1.217717, 1.213905, 0.971588, -0.023631,
    0.10675, 2.021786, 0.250524, -0.662387, -0.768862,
];

/// `tex_slat_normalization.std` (`pipeline.json`, 32 channels).
pub const TRELLIS2_TEX_SLAT_STD: [f32; 32] = [
    2.665652, 2.743913, 2.765121, 2.595319, 3.037293, 2.291316, 2.144656, 2.911822, 2.969419,
    2.501689, 2.154811, 3.163343, 2.621215, 2.381943, 3.186697, 3.021588, 2.295916, 3.234985,
    3.233086, 2.26014, 2.874801, 2.810596, 3.29272, 2.674999, 2.680878, 2.372054, 2.451546,
    2.353556, 2.995195, 2.379849, 2.786195, 2.77519,
];

/// Deterministic CPU normal RNG: splitmix64-scrambled xorshift64* state with
/// Box-Muller normals (one-value carry). Reproducible across runs; streams
/// intentionally differ from `torch.randn` (locked decision 4).
pub struct Trellis2Rng {
    state: u64,
    spare: Option<f32>,
}

impl Trellis2Rng {
    /// Seed the generator (any u64, including 0 — the state is scrambled
    /// through a splitmix64 round first).
    pub fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let state = z ^ (z >> 31);
        Self {
            state: if state == 0 { 1 } else { state },
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// One standard-normal sample (Box-Muller with a cached spare).
    pub fn normal_f32(&mut self) -> f32 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let scale = (1u64 << 53) as f64;
        let u1 = ((self.next_u64() >> 11) as f64 / scale).max(1e-300);
        let u2 = (self.next_u64() >> 11) as f64 / scale;
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = core::f64::consts::TAU * u2;
        self.spare = Some((radius * angle.sin()) as f32);
        (radius * angle.cos()) as f32
    }

    /// `len` standard-normal samples.
    pub fn randn(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.normal_f32()).collect()
    }
}

/// Draw the sparse-structure noise for `samples` joint samples as ONE tensor
/// `[samples, 8, 16, 16, 16]` from a single RNG stream (upstream
/// `torch.randn(num_samples, in_channels, reso, reso, reso)`,
/// `trellis2_image_to_3d.py:208`). For `samples = 1` this consumes the exact
/// same stream values in the same order as the legacy single-sample draw.
fn draw_ss_noise(samples: usize, seed: u64, device: &Device) -> Result<Tensor> {
    let ss_config = Trellis2SsConfig::trellis_2_4b();
    // The SS flow latent lives on the DiT's dense grid
    // (`ss_flow_img_dit_1_3B_64` -> resolution 16); the occupancy decode
    // then upsamples to `ss_resolution`.
    let latent_res = ss_config.resolution;
    let mut rng = Trellis2Rng::new(seed ^ TRELLIS2_SS_NOISE_SALT);
    let values = rng.randn(samples * ss_config.token_count() * ss_config.in_channels);
    Tensor::from_vec(
        values,
        (
            samples,
            ss_config.in_channels,
            latent_res,
            latent_res,
            latent_res,
        ),
        device,
    )
    .map_err(|source| Lux3dError::CanonicalWeightsValidation {
        message: format!("Trellis2 SS noise allocation failed: {source}"),
    })
}

/// Draw one joint SLAT noise tensor `[rows, 32]` from a single RNG stream
/// (upstream `torch.randn(coords.shape[0], in_channels)`,
/// `trellis2_image_to_3d.py:253`; for batched runs `rows` is the total over
/// all samples' active coords). For `rows = coords.len()` this consumes the
/// exact same stream values in the same order as the legacy single-sample
/// draw.
fn draw_slat_noise(rows: usize, seed: u64, salt: u64, device: &Device) -> Result<Tensor> {
    let mut rng = Trellis2Rng::new(seed ^ salt);
    let values = rng.randn(rows * 32);
    Tensor::from_vec(values, (rows, 32), device).map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!("Trellis2 SLAT noise allocation failed: {source}"),
        }
    })
}

/// A `[rows, 32]` contiguous slice of a joint SLAT noise draw.
fn noise_rows(noise: &Tensor, offset: usize, rows: usize) -> Result<Tensor> {
    noise
        .narrow(0, offset, rows)
        .and_then(|slice| slice.contiguous())
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("Trellis2 joint noise slice [{offset}, {rows}] failed: {source}"),
        })
}

/// Tile a `[1, L, C]` conditioning tensor to `[samples, L, C]` (the upstream
/// cross-attention broadcasts `cond` over the batch; the candle DiTs require
/// matching batch sizes). `[samples, L, C]` passes through unchanged.
fn repeat_cond_to_batch(cond: &Tensor, samples: usize) -> Result<Tensor> {
    let (batch, tokens, channels) =
        cond.dims3()
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 conditioning must be [B, L, C]: {source}"),
            })?;
    if batch == samples {
        return Ok(cond.clone());
    }
    if batch == 1 {
        return cond.repeat((samples, 1, 1)).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "Trellis2 conditioning batch repeat to {samples} (L {tokens}, C {channels}) \
                     failed: {source}"
                ),
            }
        });
    }
    Err(Lux3dError::ModelAssetResolution {
        message: format!(
            "Trellis2 conditioning batch {batch} cannot be shared by {samples} samples"
        ),
    })
}

/// Stage-resolved pipeline parameters. Defaults mirror `pipeline.json`
/// (`resolved_config.json` in the canonical package); `from_resolved_config`
/// applies package overrides.
#[derive(Debug, Clone, PartialEq)]
pub struct Trellis2PipelineConfig {
    /// Shape SLAT de-normalization mean (32).
    pub shape_slat_mean: Vec<f32>,
    /// Shape SLAT de-normalization std (32).
    pub shape_slat_std: Vec<f32>,
    /// Texture SLAT de-normalization mean (32).
    pub tex_slat_mean: Vec<f32>,
    /// Texture SLAT de-normalization std (32).
    pub tex_slat_std: Vec<f32>,
    /// Sparse-structure sampler defaults (steps overridden per run).
    pub ss_sampler: Trellis2SamplerParams,
    /// Shape SLAT sampler defaults (steps overridden per run).
    pub shape_sampler: Trellis2SamplerParams,
    /// Texture SLAT sampler defaults (steps overridden per run).
    pub tex_sampler: Trellis2SamplerParams,
    /// Sparse-structure occupancy resolution for the 512 path (32).
    pub ss_resolution: u32,
    /// Mesh extraction resolution for the 512 path (512).
    pub shape_resolution: u32,
    /// FlexiDualGrid voxel margin (0.5).
    pub voxel_margin: f32,
}

impl Default for Trellis2PipelineConfig {
    fn default() -> Self {
        Self {
            shape_slat_mean: TRELLIS2_SHAPE_SLAT_MEAN.to_vec(),
            shape_slat_std: TRELLIS2_SHAPE_SLAT_STD.to_vec(),
            tex_slat_mean: TRELLIS2_TEX_SLAT_MEAN.to_vec(),
            tex_slat_std: TRELLIS2_TEX_SLAT_STD.to_vec(),
            ss_sampler: Trellis2SamplerParams::default(),
            shape_sampler: Trellis2SamplerParams {
                steps: 12,
                sigma_min: 1e-5,
                guidance_strength: 7.5,
                guidance_rescale: 0.5,
                guidance_interval: [0.6, 1.0],
                rescale_t: 3.0,
            },
            // `tex_slat_sampler` (pipeline.json): guidance is cond-only
            // (1.0), no rescale, interval [0.6, 0.9].
            tex_sampler: Trellis2SamplerParams {
                steps: 12,
                sigma_min: 1e-5,
                guidance_strength: 1.0,
                guidance_rescale: 0.0,
                guidance_interval: [0.6, 0.9],
                rescale_t: 3.0,
            },
            ss_resolution: 32,
            shape_resolution: 512,
            voxel_margin: TRELLIS2_VOXEL_MARGIN,
        }
    }
}

fn read_f32_vec(value: Option<&serde_json::Value>) -> Option<Vec<f32>> {
    let array = value?.as_array()?;
    Some(
        array
            .iter()
            .filter_map(|v| v.as_f64().map(|v| v as f32))
            .collect(),
    )
}

fn read_sampler(value: Option<&serde_json::Value>) -> Option<Trellis2SamplerParams> {
    let value = value?;
    let params = value.get("params")?;
    let mut out = Trellis2SamplerParams {
        steps: params.get("steps")?.as_u64()? as usize,
        ..Trellis2SamplerParams::default()
    };
    if let Some(sigma) = value
        .get("args")
        .and_then(|args| args.get("sigma_min"))
        .and_then(|v| v.as_f64())
    {
        out.sigma_min = sigma;
    }
    if let Some(strength) = params.get("guidance_strength").and_then(|v| v.as_f64()) {
        out.guidance_strength = strength;
    }
    if let Some(rescale) = params.get("guidance_rescale").and_then(|v| v.as_f64()) {
        out.guidance_rescale = rescale;
    }
    if let Some(interval) = params.get("guidance_interval").and_then(|v| v.as_array()) {
        if interval.len() == 2 {
            if let (Some(low), Some(high)) = (interval[0].as_f64(), interval[1].as_f64()) {
                out.guidance_interval = [low, high];
            }
        }
    }
    if let Some(rescale_t) = params.get("rescale_t").and_then(|v| v.as_f64()) {
        out.rescale_t = rescale_t;
    }
    Some(out)
}

impl Trellis2PipelineConfig {
    /// Apply overrides from a `resolved_config.json` / raw `pipeline.json`
    /// document (`pipeline_args` nesting tolerated, as is the raw
    /// `pipeline.json` `{name, args}` wrapper). Missing sections keep the
    /// hard-coded defaults.
    pub fn from_resolved_config(value: &serde_json::Value) -> Self {
        let mut config = Self::default();
        // `resolved_config.json` nests under `pipeline_args`; a raw
        // `pipeline.json` carries the same keys under `args` (or at the
        // root).
        let root = value
            .get("pipeline_args")
            .or_else(|| value.get("args"))
            .unwrap_or(value);
        if let Some(mean) = read_f32_vec(
            root.get("shape_slat_normalization")
                .and_then(|n| n.get("mean")),
        ) {
            if mean.len() == 32 {
                config.shape_slat_mean = mean;
            }
        }
        if let Some(std) = read_f32_vec(
            root.get("shape_slat_normalization")
                .and_then(|n| n.get("std")),
        ) {
            if std.len() == 32 {
                config.shape_slat_std = std;
            }
        }
        if let Some(mean) = read_f32_vec(
            root.get("tex_slat_normalization")
                .and_then(|n| n.get("mean")),
        ) {
            if mean.len() == 32 {
                config.tex_slat_mean = mean;
            }
        }
        if let Some(std) = read_f32_vec(
            root.get("tex_slat_normalization")
                .and_then(|n| n.get("std")),
        ) {
            if std.len() == 32 {
                config.tex_slat_std = std;
            }
        }
        if let Some(sampler) = read_sampler(root.get("sparse_structure_sampler")) {
            config.ss_sampler = sampler;
        }
        if let Some(sampler) = read_sampler(root.get("shape_slat_sampler")) {
            config.shape_sampler = sampler;
        }
        if let Some(sampler) = read_sampler(root.get("tex_slat_sampler")) {
            config.tex_sampler = sampler;
        }
        config
    }
}

/// Per-stage sampler overrides (the upstream `run(...)`
/// `sparse_structure_sampler_params` / `shape_slat_sampler_params` /
/// `tex_slat_sampler_params` arguments, each merged over the pipeline
/// defaults). Field names follow the local [`Trellis2SamplerParams`]
/// config, which matches the upstream `params` keys (`steps`,
/// `guidance_strength`, `guidance_rescale`, `guidance_interval`,
/// `rescale_t`).
///
/// `None` fields keep the pipeline default. When both a stage override's
/// `steps` and the legacy `Trellis2RunOptions::ss_steps`/`shape_steps`/
/// `tex_steps` are set, the override wins (`resolve_stage_params`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Trellis2SamplerOverrides {
    /// Euler step count (upstream `steps`).
    pub steps: Option<usize>,
    /// Classifier-free guidance strength (upstream `guidance_strength`;
    /// 1.0 disables CFG).
    pub guidance_strength: Option<f32>,
    /// CFG rescale (upstream `guidance_rescale`; 0 disables).
    pub guidance_rescale: Option<f32>,
    /// Inclusive t interval `[lo, hi]` where CFG applies (upstream
    /// `guidance_interval`).
    pub guidance_interval: Option<(f32, f32)>,
    /// t-schedule rescale factor (upstream `rescale_t`).
    pub rescale_t: Option<f32>,
}

/// Merge a stage's pipeline-config sampler defaults with the run options:
/// the legacy `steps` field is the base step count, an override's `steps`
/// wins over it, and every other set override field replaces the config
/// default. Pure function so the precedence contract is unit-testable
/// without weights.
pub fn resolve_stage_params(
    base: &Trellis2SamplerParams,
    overrides: Option<&Trellis2SamplerOverrides>,
    steps: usize,
) -> Trellis2SamplerParams {
    let mut out = base.clone();
    out.steps = steps;
    if let Some(overrides) = overrides {
        if let Some(value) = overrides.steps {
            out.steps = value;
        }
        if let Some(value) = overrides.guidance_strength {
            out.guidance_strength = value as f64;
        }
        if let Some(value) = overrides.guidance_rescale {
            out.guidance_rescale = value as f64;
        }
        if let Some(value) = overrides.rescale_t {
            out.rescale_t = value as f64;
        }
        if let Some((low, high)) = overrides.guidance_interval {
            out.guidance_interval = [low as f64, high as f64];
        }
    }
    out
}

/// Cascade HR coordinate quantization + token-budget loop (reference
/// `sample_shape_slat_cascade`, `trellis2_image_to_3d.py:327-339`).
///
/// Each pass quantizes the upsampled 512-grid coords onto the
/// `hr_resolution // 16` grid with the exact reference expression
/// `((coord + 0.5) / lr_resolution * (hr_resolution // 16)).int()`
/// (F32 arithmetic, truncate toward zero — the upsampled coords are
/// non-negative, so this is a floor), deduplicates the rows and breaks when
/// `unique_rows < max_num_tokens || hr_resolution == 1024`, otherwise
/// `hr_resolution -= 128`. `torch.unique(dim=0)` returns lexicographically
/// sorted rows; the returned coords are sorted the same way.
///
/// `target_resolution` is the cascade target (1024 or 1536); the stop
/// resolution stays 1024 for both variants.
///
/// Returns `(unique_sorted_coords, hr_resolution)`.
pub fn quantize_hr_coords(
    hr_coords: &[[i32; 3]],
    lr_resolution: u32,
    target_resolution: u32,
    max_num_tokens: usize,
) -> (Vec<[i32; 3]>, u32) {
    quantize_hr_coords_with_stop(
        hr_coords,
        lr_resolution,
        target_resolution,
        max_num_tokens,
        TRELLIS2_CASCADE_HR_RESOLUTION,
    )
}

/// [`quantize_hr_coords`] with the loop's stop resolution explicit. The
/// reference hardcodes 1024 (the 1024-DiT latent grid ceiling:
/// `hr_resolution // 16 <= 64`), so production runs always stop at 1024 and
/// the 1024 cascade can never reduce its resolution regardless of the token
/// count. The parameter is exposed so unit tests can drive the decrement
/// path (1024 -> 896 -> 768 ...). `resolution - stop_resolution` must be a
/// multiple of 128 for the loop to terminate (true for the reference's 1024
/// and 1536 cascade targets).
pub fn quantize_hr_coords_with_stop(
    hr_coords: &[[i32; 3]],
    lr_resolution: u32,
    resolution: u32,
    max_num_tokens: usize,
    stop_resolution: u32,
) -> (Vec<[i32; 3]>, u32) {
    let lr = lr_resolution as f32;
    let mut hr_resolution = resolution;
    let coords = loop {
        let scale = (hr_resolution / 16) as f32;
        let quant = quantize_hr_sample(hr_coords, lr, scale);
        if quant.len() < max_num_tokens || hr_resolution == stop_resolution {
            break quant;
        }
        hr_resolution -= 128;
    };
    (coords, hr_resolution)
}

/// Quantize one sample's upsampled HR coords onto the `hr_resolution // 16`
/// grid with the exact reference expression
/// `((coord + 0.5) / lr_resolution * (hr_resolution // 16)).int()` (F32
/// arithmetic, truncate toward zero — the upsampled coords are non-negative,
/// so this is a floor), then deduplicate the rows.
///
/// `torch.unique(dim=0)` returns lexicographically sorted rows; the returned
/// coords are sorted the same way.
fn quantize_hr_sample(hr_coords: &[[i32; 3]], lr: f32, scale: f32) -> Vec<[i32; 3]> {
    let mut quant: Vec<[i32; 3]> = hr_coords
        .iter()
        .map(|c| {
            [
                ((c[0] as f32 + 0.5) / lr * scale) as i32,
                ((c[1] as f32 + 0.5) / lr * scale) as i32,
                ((c[2] as f32 + 0.5) / lr * scale) as i32,
            ]
        })
        .collect();
    quant.sort_unstable();
    quant.dedup();
    quant
}

/// Multi-sample [`quantize_hr_coords`]: the reference quantizes the WHOLE
/// batch before the unique/budget loop and combines the rows with a leading
/// batch column (`torch.cat([hr_coords[:, :1], quantized_xyz], dim=1)` then
/// `unique(dim=0)`, `trellis2_image_to_3d.py:327-339`), so identical xyz
/// rows from different samples stay distinct and the token count is the SUM
/// of the per-sample unique counts. One `hr_resolution` is chosen for the
/// whole batch and returned; the caller must decode every sample at it.
///
/// The per-sample rows are deduplicated and sorted exactly like
/// [`quantize_hr_coords`] (the batch marker is implicit in the sample
/// index), and a single sample is bit-identical to the single-sample
/// helper.
pub fn quantize_hr_coords_batched(
    samples: &[Vec<[i32; 3]>],
    lr_resolution: u32,
    target_resolution: u32,
    max_num_tokens: usize,
) -> (Vec<Vec<[i32; 3]>>, u32) {
    quantize_hr_coords_batched_with_stop(
        samples,
        lr_resolution,
        target_resolution,
        max_num_tokens,
        TRELLIS2_CASCADE_HR_RESOLUTION,
    )
}

/// [`quantize_hr_coords_batched`] with the loop's stop resolution explicit
/// (test hook mirroring [`quantize_hr_coords_with_stop`]).
pub fn quantize_hr_coords_batched_with_stop(
    samples: &[Vec<[i32; 3]>],
    lr_resolution: u32,
    resolution: u32,
    max_num_tokens: usize,
    stop_resolution: u32,
) -> (Vec<Vec<[i32; 3]>>, u32) {
    let lr = lr_resolution as f32;
    let mut hr_resolution = resolution;
    let coords = loop {
        let scale = (hr_resolution / 16) as f32;
        let quant: Vec<Vec<[i32; 3]>> = samples
            .iter()
            .map(|sample| quantize_hr_sample(sample, lr, scale))
            .collect();
        let total_rows: usize = quant.iter().map(Vec::len).sum();
        if total_rows < max_num_tokens || hr_resolution == stop_resolution {
            break quant;
        }
        hr_resolution -= 128;
    };
    (coords, hr_resolution)
}

/// Geometry post-processing options for the assembled TRELLIS.2 dual-grid
/// mesh (upstream `to_glb` branches; default = the shipped entry points
/// `example.py:43-45` / `app.py:503-505`, which run `remesh=True`,
/// `remesh_band=1`, `remesh_project=0`; the reference demo additionally caps
/// the texture budget at 4096 but keeps the 1e6 decimation target).
///
/// When [`Trellis2MeshPostprocessOptions::enabled`], the assembled mesh runs
/// BEFORE the texture stage attaches per-vertex colors, so the final
/// (possibly re-topologized) vertices are the ones the PBR color sampling
/// sees and no vertex-color remap is needed. The pass mirrors the upstream
/// order:
///
/// 0. `fill_holes(fill_hole_perimeter)` — upstream's shared pre-branch
///    cleaning (`postprocess.py:108-113`); also covers the reference
///    decode-time `Mesh.fill_holes()` call inside `decode_latent` (same
///    3e-2 default threshold);
/// 1. `remesh = true` (default): build the triangle BVH, run
///    [`crate::geometry::remesh_narrow_band_dc`] on the canonical
///    `[-0.5, 0.5]^3` frame with `resolution = mesh.resolution`,
///    `center = 0`, `scale = (resolution + 3 * remesh_band) / resolution`,
///    `band = remesh_band` and `project_back = remesh_project`, then the
///    single `simplify(decimation_target)` pass. The upstream remesh branch
///    skips the duplicate-face / non-manifold / small-component /
///    orientation-unification chain (`postprocess.py:171-186`);
/// 2. `remesh = false`: the reference cleanup chain — aggressive simplify
///    toward `decimation_target * 3` triangles (`0` skips BOTH simplify
///    passes — cleanup-only post-processing), `remove_duplicate_faces` /
///    `repair_non_manifold_edges` /
///    `remove_small_connected_components(min_component_area)` /
///    `fill_holes`, second simplify toward `decimation_target` + the same
///    cleanup, final `fill_holes` + `unify_face_orientations` + vertex
///    compaction.
///
/// `remesh = false` is byte-identical to the pre-remesh-port pipeline.
/// UV unwrapping / texture baking are a separate workstream and are NOT part
/// of this pass.
#[derive(Debug, Clone, PartialEq)]
pub struct Trellis2MeshPostprocessOptions {
    /// Run the geometry post-process (true). With `false` the pipeline keeps
    /// the decoded mesh byte-identical to the pre-postprocess behavior.
    pub enabled: bool,
    /// Boundary loops whose total perimeter is `<` this threshold are
    /// closed (upstream `fill_holes(max_hole_perimeter=3e-2)` uses a strict
    /// `<` comparison); `0` legitimately disables hole filling.
    pub fill_hole_perimeter: f32,
    /// Final triangle-count target (upstream `to_glb(decimation_target)`,
    /// default 1_000_000). In the remesh branch this is the single
    /// post-remesh simplify pass; in the cleanup branch the reference first
    /// simplifies aggressively to `3 * decimation_target`, then to this
    /// target. `0` skips simplification entirely (topology-only pass).
    pub decimation_target: usize,
    /// Connected components with total triangle area `<` this threshold are
    /// removed (upstream `remove_small_connected_components(1e-5)` keeps
    /// components with `area >= min_area`); `0` keeps every non-degenerate
    /// component. Unused in the remesh branch (upstream skips the cleanup
    /// chain there).
    pub min_component_area: f32,
    /// Rebuild the topology with upstream's narrow-band dual-contouring
    /// remesh (true, matching the shipped entry points
    /// `example.py:43`/`app.py:503`) instead of the cleanup/simplify chain.
    /// `false` restores the previous non-remesh pipeline byte-for-byte.
    pub remesh: bool,
    /// Narrow-band half-width in grid cells (upstream `remesh_band`, shipped
    /// value `1`). Only used when `remesh` is set.
    pub remesh_band: f32,
    /// Project-back lerp factor toward the original surface (upstream
    /// `remesh_project`; applied only when `> 0` like
    /// `remeshing.py:238`, `0` disables, the shipped value; the reference
    /// signature default `0.9` is not used by either shipped entry point).
    /// Negative values are rejected by [`Self::validate`].
    pub remesh_project: f32,
}

impl Default for Trellis2MeshPostprocessOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            fill_hole_perimeter: 3e-2,
            decimation_target: 1_000_000,
            min_component_area: 1e-5,
            remesh: true,
            remesh_band: 1.0,
            remesh_project: 0.0,
        }
    }
}

impl Trellis2MeshPostprocessOptions {
    /// Validate the options. Disabled post-processing skips the threshold
    /// checks (the values are unused). Enabled runs require finite,
    /// non-negative thresholds — `0` is a meaningful "off" for the hole /
    /// component thresholds — and accept any decimation target (`0` = skip
    /// simplification, topology-only). The remesh branch additionally
    /// requires a finite non-negative `remesh_band` and a finite
    /// non-negative `remesh_project` (the remesher itself validates the
    /// derived grid; `0` disables project-back like the official
    /// `project_back > 0` guard).
    pub fn validate(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        if !self.fill_hole_perimeter.is_finite() || self.fill_hole_perimeter < 0.0 {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 postprocess fill_hole_perimeter must be a finite non-negative \
                 threshold (0 disables hole filling)",
            ));
        }
        if !self.min_component_area.is_finite() || self.min_component_area < 0.0 {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 postprocess min_component_area must be a finite non-negative \
                 threshold (0 keeps every component)",
            ));
        }
        if self.remesh {
            if !self.remesh_band.is_finite() || self.remesh_band < 0.0 {
                return Err(Lux3dError::InvalidInput(
                    "Trellis2 postprocess remesh_band must be a finite non-negative \
                     value in grid cells (the shipped value is 1)",
                ));
            }
            if !self.remesh_project.is_finite() || self.remesh_project < 0.0 {
                return Err(Lux3dError::InvalidInput(
                    "Trellis2 postprocess remesh_project must be a finite non-negative \
                     factor (0 disables project-back; the official guard applies it only \
                     when > 0)",
                ));
            }
        }
        Ok(())
    }
}

/// Textured-GLB export options (upstream `to_glb` UV unwrap + bake,
/// `postprocess.py:195-315`).
///
/// When enabled AND the mesh carries the 6-channel PBR payload (the texture
/// stage ran), the pipeline computes the xatlas UV atlas, area-weighted
/// vertex normals and the baked PBR textures after `attach_texture`, and
/// stores the product in `Trellis2Stages::textured`. Skipped cleanly (the
/// field stays `None`, no error) when `with_texture = false` (no PBR
/// payload), `voxel_attr_channels != 6`, `uv_unwrap = false` (baking needs
/// UVs) or `enabled = false`.
#[derive(Debug, Clone, PartialEq)]
pub struct Trellis2TextureExportOptions {
    /// Compute and store the textured atlas mesh (`true`, matching the
    /// reference `to_glb` default).
    pub enabled: bool,
    /// Bake texture edge length (upstream `texture_size`, default 2048).
    /// Must be a multiple of 8 in `8..=8192` (nvdiffrast-compatible
    /// resolution granularity).
    pub texture_size: usize,
    /// Run the xatlas UV unwrap (`true`). With `false` the textured mesh is
    /// skipped entirely — texture baking needs UV coordinates.
    pub uv_unwrap: bool,
}

impl Default for Trellis2TextureExportOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            texture_size: 2048,
            uv_unwrap: true,
        }
    }
}

impl Trellis2TextureExportOptions {
    /// Validate the options. Disabled texture export skips the checks (the
    /// values are unused); enabled runs require a multiple-of-8 texture size
    /// in `8..=8192` and accept either `uv_unwrap` value (a disabled unwrap
    /// skips the textured mesh cleanly instead of failing).
    pub fn validate(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        if self.texture_size < 8 || self.texture_size > 8192 || self.texture_size % 8 != 0 {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 texture export texture_size must be a multiple of 8 in 8..=8192 \
                 (upstream nvdiffrast parity)",
            ));
        }
        Ok(())
    }
}

/// Run options (defaults match the reference `run(...)` signature).
///
/// Sampler parameter precedence (`resolve_stage_params`): the per-stage
/// overrides (`ss_sampler`/`shape_sampler`/`tex_sampler`, `None` = keep the
/// pipeline defaults) win over the legacy `ss_steps`/`shape_steps`/
/// `tex_steps` fields when both set a value; the legacy fields stay the
/// base step count for all three stages otherwise.
#[derive(Debug, Clone, PartialEq)]
pub struct Trellis2RunOptions {
    /// RNG seed (42). Multi-sample runs (`run_many_*`, `num_samples > 1`)
    /// draw ONE joint noise tensor per stage from this seed, matching the
    /// upstream `torch.manual_seed(seed)` + `torch.randn(num_samples, ...)` /
    /// `torch.randn(coords.shape[0], ...)` structure
    /// (`trellis2_image_to_3d.py:208, 253`); Lux3D RNG streams are not
    /// bit-identical to torch (documented deviation).
    pub seed: u64,
    /// Pipeline variant (Cascade1024 by contract).
    pub pipeline_type: Trellis2PipelineType,
    /// Sparse-structure Euler steps (12); the base step count when
    /// `ss_sampler` does not set `steps`.
    pub ss_steps: usize,
    /// Shape SLAT Euler steps (12); the base step count for both cascade
    /// shape stages when `shape_sampler` does not set `steps`.
    pub shape_steps: usize,
    /// Texture SLAT Euler steps (12); the base step count when
    /// `tex_sampler` does not set `steps`.
    pub tex_steps: usize,
    /// Sparse-structure sampler overrides (upstream
    /// `sparse_structure_sampler_params`; `None` keeps the defaults).
    pub ss_sampler: Option<Trellis2SamplerOverrides>,
    /// Shape SLAT sampler overrides (upstream `shape_slat_sampler_params`;
    /// applied to BOTH the LR and HR cascade shape stages).
    pub shape_sampler: Option<Trellis2SamplerOverrides>,
    /// Texture SLAT sampler overrides (upstream `tex_slat_sampler_params`).
    pub tex_sampler: Option<Trellis2SamplerOverrides>,
    /// Run the texture stages (texture SLAT + guided texture VAE + PBR
    /// vertex colors, true — the reference pipeline always textures). With
    /// `false` the pipeline stays geometry-only (the mesh buffers carry the
    /// 7-channel shape attributes and no PBR payload).
    pub with_texture: bool,
    /// Per-stage load/drop mode (the reference `low_vram` switch,
    /// `trellis2_image_to_3d.py:55, 107, 119-125`): `true` (default) loads a
    /// stage, runs it and drops it, exactly the historical Lux3D behavior;
    /// `false` keeps every stage the run loads resident on the device until
    /// the run ends (the reference `to(device)` pinning branch,
    /// `:119-125`). Speed-only: the sampler math and RNG streams are
    /// unchanged, so the outputs are identical for the same seed. When the
    /// resident stage set does not fit in device memory the run fails with
    /// an actionable error naming this flag and the device memory limit —
    /// there is no silent fallback to per-stage loading.
    pub low_vram: bool,
    /// Cascade token budget (49152). The reference loop breaks when the
    /// quantized HR coord count drops below this value — but also whenever
    /// `hr_resolution == 1024`, so the 1024 cascade decodes at 1024 even
    /// when the budget would be exceeded. Multi-sample runs budget the
    /// concatenated batch, so this is a GLOBAL cap shared by every sample
    /// (not per sample), exactly like `sample_shape_slat_cascade`.
    pub max_num_tokens: usize,
    /// Run image preprocessing inside `run_from_path` (true).
    pub preprocess: bool,
    /// Sample count (1). `num_samples = 1` keeps the single-output path
    /// (bit-identical to the pre-`num_samples` behavior) and the single-run
    /// entry points error on `num_samples != 1`. `run_many_from_path` /
    /// `run_many_with_latents` implement `num_samples > 1` with the upstream
    /// `run(num_samples)` draw structure: ONE joint sparse-structure noise
    /// tensor `[N, 8, R, R, R]` and ONE joint SLAT noise tensor per stage
    /// over all samples' coords (`trellis2_image_to_3d.py:208, 253`), one
    /// shared conditioning pass, per-sample SLAT sampling / decode / export.
    pub num_samples: usize,
    /// Geometry post-process of the assembled mesh, applied BEFORE the
    /// texture stage attaches per-vertex colors (upstream `to_glb` mesh
    /// cleaning / remesh; defaults mirror the shipped entry points:
    /// remesh on, band 1, project 0). With `enabled = false` the decoded
    /// mesh is passed through untouched.
    pub postprocess: Trellis2MeshPostprocessOptions,
    /// Textured-GLB export stage (upstream `to_glb` UV unwrap + PBR bake):
    /// UV atlas, area-weighted vertex normals and the baked base-color /
    /// metallic-roughness textures on `Trellis2Stages::textured`. Defaults
    /// mirror the reference (enabled, 2048, unwrap on); skipped cleanly when
    /// the run is geometry-only.
    pub texture_export: Trellis2TextureExportOptions,
}

impl Default for Trellis2RunOptions {
    fn default() -> Self {
        Self {
            seed: 42,
            pipeline_type: Trellis2PipelineType::Cascade1024,
            ss_steps: 12,
            shape_steps: 12,
            tex_steps: 12,
            ss_sampler: None,
            shape_sampler: None,
            tex_sampler: None,
            with_texture: true,
            low_vram: true,
            max_num_tokens: 49152,
            preprocess: true,
            num_samples: 1,
            postprocess: Trellis2MeshPostprocessOptions::default(),
            texture_export: Trellis2TextureExportOptions::default(),
        }
    }
}

/// Per-stage outputs exposed for tests and parity taps.
///
/// `shape_coords`/`shape_latent`/`decoded_feats` always describe the FINAL
/// shape stage: in the 512 path the latent lives on `ss_coords`, in the
/// cascade it is the HR latent on `quant_coords`. The cascade inputs are
/// kept in the additive `lr_*`/`hr_*` fields.
#[derive(Debug, Clone)]
pub struct Trellis2Stages {
    /// DINOv3 conditioning `[1, 1029, 1024]` (or the injected cond).
    pub cond: Tensor,
    /// Negative conditioning (zeros).
    pub neg_cond: Tensor,
    /// Sparse-structure flow latent `[1, 8, R, R, R]` (`None` when the
    /// coordinates were injected directly).
    pub z_s: Option<Tensor>,
    /// Active voxel coordinates at the SS resolution (batch dim dropped).
    pub ss_coords: Vec<[i32; 3]>,
    /// Shape SLAT coordinates (identical to `ss_coords` in the 512 path; the
    /// quantized HR coords in the cascade).
    pub shape_coords: Vec<[i32; 3]>,
    /// De-normalized shape SLAT `[N, 32]` (the HR slat in the cascade).
    pub shape_latent: Tensor,
    /// Decoded 7-channel shape attributes `[N, 7]`.
    pub decoded_feats: Tensor,
    /// Extracted mesh buffers.
    pub mesh: Option<crate::geometry::Trellis2MeshBuffers>,
    /// Cascade only: de-normalized LR (32^3) shape SLAT before the upsample
    /// (`None` in the 512 path).
    pub lr_shape_latent: Option<Tensor>,
    /// Cascade only: shape VAE `upsample(slat, 4)` coords on the 512 grid,
    /// before quantization (`None` in the 512 path).
    pub hr_upsampled_coords: Option<Vec<[i32; 3]>>,
    /// Cascade only: quantized + deduplicated HR coords the HR stage ran on
    /// (identical to `shape_coords`; kept for stage-tap symmetry).
    pub quant_coords: Option<Vec<[i32; 3]>>,
    /// Cascade only: the token-budget resolution the HR stage/decode ran at
    /// (1024 for the 1024 cascade — the loop always stops there; the 1536
    /// cascade decodes at its target unless the budget reduced it; `None`
    /// in the single-resolution paths).
    pub hr_resolution: Option<u32>,
    /// Shape decoder raw C2S subdivision logits (one `[N_parents, 8]` F32
    /// tensor per C2S stage, stage order) — the texture decoder's
    /// `guide_subs`. Empty only for degenerate inputs.
    pub subdiv_logits: Vec<Tensor>,
    /// De-normalized texture SLAT `[N, 32]` on the shape latent coords
    /// (`None` for geometry-only runs).
    pub tex_latent: Option<Tensor>,
    /// Per-voxel PBR attributes `[N, 6]` in [0, 1] (base_color 0:3,
    /// metallic 3, roughness 4, alpha 5) on the shape decode's final coords
    /// (`None` for geometry-only runs).
    pub tex_attrs: Option<Vec<f32>>,
    /// Textured atlas mesh (upstream `to_glb` textured product: xatlas UV
    /// unwrap + area-weighted vertex normals + baked PBR textures).
    /// `None` for geometry-only runs, when
    /// `Trellis2RunOptions::texture_export` is disabled, when
    /// `uv_unwrap = false`, or when the mesh carries no 6-channel PBR
    /// payload (see [`Trellis2TextureExportOptions`]).
    pub textured: Option<Trellis2TexturedMesh>,
}

/// Textured atlas mesh produced by the texture-export stage (upstream
/// `to_glb` `textured_mesh`, `postprocess.py:307-323`).
///
/// All geometry lives in the decoded REFERENCE `[-0.5, 0.5]^3` frame (no
/// axis remap, like `Trellis2MeshBuffers`); the glTF axis conversion
/// `(x, y, z) -> (x, z, -y)` and the UV V flip (`v' = 1 - v`) are applied by
/// the textured GLB writer at export time only.
#[derive(Debug, Clone, PartialEq)]
pub struct Trellis2TexturedMesh {
    /// Atlas mesh positions, flat `[A, 3]` f32 (reference frame). Atlas
    /// vertices duplicate the mesh vertices at UV seams.
    pub vertices: Vec<f32>,
    /// Flat triangle indices into `vertices` (u32).
    pub faces: Vec<u32>,
    /// Area-weighted vertex normals, flat `[A, 3]` f32 (the upstream
    /// `mesh.compute_vertex_normals()` product, gathered through the atlas
    /// vertex map).
    pub normals: Vec<f32>,
    /// Atlas UVs per atlas vertex, flat `[A, 2]` f32 in [0, 1] (pre V-flip;
    /// the baked textures use the nvdiffrast row-0 = v = 1 layout, the GLB
    /// writer flips at export).
    pub uvs: Vec<f32>,
    /// Baked PBR textures (base color RGBA + metallic/roughness) with the
    /// pre-inpaint coverage mask and bake statistics.
    pub texture: crate::geometry::BakedPbrTexture,
    /// Number of charts the unwrap produced (`> 0`).
    pub chart_count: u32,
    /// Number of packed atlases (UVs are merged into a single [0, 1] frame).
    pub atlas_count: u32,
    /// Atlas edge length in texels the packing targeted.
    pub atlas_size: u32,
}

/// Texture-export stage (upstream `to_glb` UV unwrap + bake,
/// `postprocess.py:195-315`): xatlas UV atlas over the (post-processed)
/// mesh, area-weighted vertex normals and the sparse-trilinear PBR bake at
/// `texture_size`.
///
/// This is the projection-free entry point used by the standalone texturing
/// pipeline (no snap-back upstream either, `trellis2_texturing.py:316-332`).
/// The image-to-3D to_glb path uses
/// [`build_trellis2_textured_mesh_snapped`], which projects every covered
/// texel onto the pre-postprocess surface first.
///
/// Skips cleanly (returns `None`, no error) when the options are disabled,
/// `uv_unwrap` is off, the mesh carries no 6-channel PBR payload / vertex
/// colors (geometry-only runs) or the mesh has no triangles. Real failures
/// error with descriptive messages (no panics). Public so exports can build
/// the textured mesh from standalone `Trellis2MeshBuffers` (the same buffers
/// the pipeline produces) without running the GPU pipeline.
pub fn build_trellis2_textured_mesh(
    mesh: &crate::geometry::Trellis2MeshBuffers,
    options: &Trellis2TextureExportOptions,
) -> Result<Option<Trellis2TexturedMesh>> {
    build_trellis2_textured_mesh_impl(mesh, options, None)
}

/// [`build_trellis2_textured_mesh`] with the upstream `to_glb` bake
/// snap-back (`postprocess.py:249-266`): the UV rasterization and 3D position
/// interpolation run on the export mesh, then every covered texel position is
/// projected onto the ORIGINAL pre-postprocess surface through a triangle
/// BVH (`unsigned_distance(..., return_uvw=True)` and the barycentric
/// reconstruction, `postprocess.py:254-256`) before the PBR attribute volume
/// is sampled trilinearly.
///
/// `original_vertices` / `original_faces` are the decoded mesh as it enters
/// [`Trellis2GeometryStage::postprocess_decoded_mesh`] — before remesh /
/// simplify, exactly the surface the official BVH is built from
/// (`postprocess.py:122`). The BVH is built here (median split, CPU); the
/// exported atlas mesh keeps its own post-processed geometry.
///
/// Use this on the to_glb export path ONLY. The texturing pipeline never
/// snaps back (`trellis2_texturing.py:316-332`).
pub fn build_trellis2_textured_mesh_snapped(
    mesh: &crate::geometry::Trellis2MeshBuffers,
    options: &Trellis2TextureExportOptions,
    original_vertices: &[f32],
    original_faces: &[u32],
) -> Result<Option<Trellis2TexturedMesh>> {
    if !textured_export_will_run(mesh, options) {
        return Ok(None);
    }
    let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
    let t_bvh = std::time::Instant::now();
    let original_surface = MeshBvh::build(original_vertices, original_faces)?;
    if stage_time {
        eprintln!(
            "[stage]       snap bvh: {:.2}s ({} faces)",
            t_bvh.elapsed().as_secs_f64(),
            original_faces.len() / 3
        );
    }
    build_trellis2_textured_mesh_impl(mesh, options, Some(&original_surface))
}

/// Shared clean-skip gate of the textured-export entry points: the options
/// are on, the mesh carries the 6-channel PBR payload with per-vertex colors
/// and at least one triangle.
fn textured_export_will_run(
    mesh: &crate::geometry::Trellis2MeshBuffers,
    options: &Trellis2TextureExportOptions,
) -> bool {
    options.enabled
        && options.uv_unwrap
        && mesh.voxel_attr_channels == super::trellis2_tex::Trellis2PbrLayout::CHANNELS
        && mesh.vertex_colors.is_some()
        && mesh.face_count() > 0
}

/// `original_surface`: `Some` = upstream to_glb snap-back; `None` = plain
/// bake (texturing pipeline / standalone export).
fn build_trellis2_textured_mesh_impl(
    mesh: &crate::geometry::Trellis2MeshBuffers,
    options: &Trellis2TextureExportOptions,
    original_surface: Option<&MeshBvh<'_>>,
) -> Result<Option<Trellis2TexturedMesh>> {
    if !textured_export_will_run(mesh, options) {
        return Ok(None);
    }
    mesh.validate()?;
    let vertices: Vec<[f32; 3]> = mesh
        .vertices
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect();
    // `validate()` already checked range and non-negativity.
    let faces: Vec<u32> = mesh.faces.iter().map(|&index| index as u32).collect();
    let voxel_coords: Vec<[i32; 3]> = mesh
        .voxel_coords
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect();
    let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
    let t_uv = std::time::Instant::now();
    let atlas = uv_unwrap(&vertices, &faces, options.texture_size as u32).map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!("Trellis2 UV unwrap failed: {source}"),
        }
    })?;
    if stage_time {
        eprintln!(
            "[stage]       uv unwrap: {:.2}s ({} charts, {} atlases)",
            t_uv.elapsed().as_secs_f64(),
            atlas.chart_count,
            atlas.atlas_count
        );
    }
    // Normals belong to the pre-unwrap surface: compute them on the input
    // vertices/faces, then gather through the atlas `vmapping`. Recomputing
    // on atlas vertices would give duplicated seam vertices chart-local
    // normals, unlike the reference's `vmaps` gather (`postprocess.py:208-216`).
    //
    // The normals are computed on POSITION-WELDED topology: the atlas welds
    // identical positions for charting, and `repair_non_manifold_edges` can
    // leave several index copies per position. A per-copy normal would attach
    // an arbitrary sheet's normal to every chart vertex at that position
    // (measured ~70 deg off the local surface on decoded meshes).
    let base_normals = welded_vertex_normals(&vertices, &faces);
    let normals: Vec<[f32; 3]> = atlas
        .vmapping
        .iter()
        .map(|&original| base_normals[original as usize])
        .collect();
    let t_bake = std::time::Instant::now();
    let texture = match original_surface {
        Some(original_surface) => bake_pbr_texture_with_snap(
            &atlas.vertices,
            &atlas.uvs,
            &atlas.indices,
            original_surface,
            &voxel_coords,
            &mesh.voxel_attrs,
            mesh.resolution,
            options.texture_size,
        ),
        None => bake_pbr_texture(
            &atlas.vertices,
            &atlas.uvs,
            &atlas.indices,
            &voxel_coords,
            &mesh.voxel_attrs,
            mesh.resolution,
            options.texture_size,
        ),
    }
    .map_err(|source| Lux3dError::CanonicalWeightsValidation {
        message: format!("Trellis2 PBR texture bake failed: {source}"),
    })?;
    if stage_time {
        eprintln!(
            "[stage]       pbr bake: {:.2}s ({} atlas faces, {} voxels)",
            t_bake.elapsed().as_secs_f64(),
            atlas.indices.len() / 3,
            voxel_coords.len()
        );
    }
    Ok(Some(Trellis2TexturedMesh {
        vertices: atlas.vertices.iter().flatten().copied().collect(),
        faces: atlas.indices,
        normals: normals.iter().flatten().copied().collect(),
        uvs: atlas.uvs.iter().flatten().copied().collect(),
        texture,
        chart_count: atlas.chart_count,
        atlas_count: atlas.atlas_count,
        atlas_size: atlas.atlas_size,
    }))
}

impl Trellis2Stages {
    fn tensor_to_f32_rows(tensor: &Tensor, rows: usize, channels: usize) -> Result<Array2<f32>> {
        let values = tensor
            .to_device(&Device::Cpu)
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 latent npz download failed: {source}"),
            })?;
        Ok(
            Array2::from_shape_vec((rows, channels), values).map_err(|error| {
                Lux3dError::ContractValidation {
                    message: format!("Trellis2 latent npz shape mismatch: {error}"),
                }
            })?,
        )
    }

    /// Upstream `return_latent=True` surface: serialize the final latents to
    /// an `.npz` with
    ///
    /// * `shape_latent` — `[N, 32]` F32 (de-normalized; the HR slat for the
    ///   cascades),
    /// * `tex_latent` — `[N, 32]` F32 (empty `[0, 32]` for geometry-only
    ///   runs),
    /// * `coords` — `[N, 3]` I32 (the shape latent coords),
    /// * `resolution` — `()` scalar I32 (the decode resolution; the cascade
    ///   `hr_resolution` when present),
    /// * `lr_shape_latent` — `[N_lr, 32]` F32 (empty `[0, 32]` for
    ///   non-cascade runs),
    /// * `meta` — UTF-8 JSON `Array1<u8>`: `{"pipeline_type", "resolution",
    ///   "num_coords", "cascade", "hr_resolution"}`.
    pub fn write_latent_npz(&self, path: &Path, pipeline_type: Trellis2PipelineType) -> Result<()> {
        let shape_rows = self.shape_coords.len();
        let shape_latent = Self::tensor_to_f32_rows(&self.shape_latent, shape_rows, 32)?;
        let tex_latent = match &self.tex_latent {
            Some(tex) => Self::tensor_to_f32_rows(tex, shape_rows, 32)?,
            None => Array2::zeros((0, 32)),
        };
        let mut coords = Array2::<i32>::zeros((shape_rows, 3));
        for (index, coord) in self.shape_coords.iter().enumerate() {
            coords[[index, 0]] = coord[0];
            coords[[index, 1]] = coord[1];
            coords[[index, 2]] = coord[2];
        }
        let resolution = self
            .hr_resolution
            .unwrap_or(self.mesh.as_ref().map(|m| m.resolution).unwrap_or(512));
        let lr_shape_latent = match &self.lr_shape_latent {
            Some(lr) => {
                let rows = lr
                    .dim(0)
                    .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                        message: format!("Trellis2 latent npz download failed: {source}"),
                    })?;
                Self::tensor_to_f32_rows(lr, rows, 32)?
            }
            None => Array2::zeros((0, 32)),
        };
        let meta = serde_json::json!({
            "pipeline_type": pipeline_type.as_str(),
            "resolution": resolution,
            "num_coords": shape_rows,
            "cascade": self.lr_shape_latent.is_some(),
            "hr_resolution": self.hr_resolution,
        });
        let meta_bytes =
            serde_json::to_vec(&meta).map_err(|source| Lux3dError::ContractValidation {
                message: format!("Trellis2 latent npz meta serialization failed: {source}"),
            })?;
        let file = std::fs::File::create(path).map_err(|source| Lux3dError::ModelAssetIo {
            path: path.to_path_buf(),
            source,
        })?;
        let mut npz = ndarray_npy::NpzWriter::new(file);
        let npz_error = |source: ndarray_npy::WriteNpzError| Lux3dError::ContractValidation {
            message: format!(
                "Trellis2 latent npz write to {} failed: {source}",
                path.display()
            ),
        };
        npz.add_array("shape_latent", &shape_latent)
            .map_err(npz_error)?;
        npz.add_array("tex_latent", &tex_latent)
            .map_err(npz_error)?;
        npz.add_array("coords", &coords).map_err(npz_error)?;
        npz.add_array(
            "resolution",
            &ndarray::Array0::from_elem((), resolution as i32),
        )
        .map_err(npz_error)?;
        npz.add_array("lr_shape_latent", &lr_shape_latent)
            .map_err(npz_error)?;
        npz.add_array("meta", &ndarray::Array1::from_vec(meta_bytes))
            .map_err(npz_error)?;
        npz.finish().map_err(npz_error)?;
        Ok(())
    }
}

/// Shape-stage products of [`Trellis2Pipeline::sample_shape_slat_cascade`]
/// (all tensors de-normalized, F32).
#[derive(Debug, Clone)]
pub struct Trellis2CascadeShape {
    /// De-normalized LR (32^3) shape SLAT `[N_lr, 32]`.
    pub lr_slat: Tensor,
    /// Shape VAE `upsample(slat, 4)` coords on the 512 grid
    /// (pre-quantization; `hr_resolution`-grid parents would be
    /// `hr_resolution / 16` if the budget loop ever reduced the target).
    pub hr_upsampled_coords: Vec<[i32; 3]>,
    /// Quantized + deduplicated coords the HR stage ran on (lexicographically
    /// sorted, `torch.unique(dim=0)` semantics).
    pub quant_coords: Vec<[i32; 3]>,
    /// The token-budget resolution (1024 for the 1024 cascade — the loop
    /// always breaks at 1024; 1536 for the 1536 cascade unless the budget
    /// reduced it toward the 1024 stop).
    pub hr_resolution: u32,
    /// De-normalized HR shape SLAT `[N_hr, 32]` on `quant_coords`.
    pub hr_slat: Tensor,
}

/// Per-sample cascade products held between the batched LR/upsample phase
/// and the joint HR / texture draws in
/// [`Trellis2Pipeline::run_batched_core`]. Unlike [`Trellis2CascadeShape`]
/// the HR SLAT is not part of the struct: the joint HR noise draw needs every
/// sample's quantized coords first. The batch-level `hr_resolution` is shared
/// by every part and is NOT stored here (the official cascade returns one
/// `res` for the whole batch, `trellis2_image_to_3d.py:339,586-591`).
struct Trellis2BatchedCascadePart {
    ss_coords: Vec<[i32; 3]>,
    lr_slat: Tensor,
    hr_upsampled_coords: Vec<[i32; 3]>,
    quant_coords: Vec<[i32; 3]>,
}

/// Raw TRELLIS.2 checkpoint layout (`models/TRELLIS.2-4B`), stage prefix ->
/// file relative path. The 1024 shape/texture DiTs serve the cascade; the
/// texture decoder produces the per-voxel PBR attributes.
const RAW_STAGE_FILES: [(&str, &str); 9] = [
    ("image_cond", "external/dinov3/model.safetensors"),
    ("ss_flow", "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"),
    (
        "ss_dec",
        "external/ss_dec/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
    ),
    (
        "shape_dit_512",
        "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
    ),
    (
        "shape_dit_1024",
        "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
    ),
    (
        "shape_dec",
        "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
    ),
    (
        "tex_dit_512",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
    ),
    (
        "tex_dit_1024",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
    ),
    ("tex_dec", "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors"),
];

/// Stage models pinned on the device by a `low_vram = false` run (the
/// reference `Trellis2ImageTo3DPipeline.to(device)` pinning branch,
/// `trellis2_image_to_3d.py:119-125`).
///
/// One cache serves a whole run: the first load of each stage prefix is kept
/// as an `Arc` and every later load for the same prefix (the cascade loads
/// the shape VAE twice, the 1024-conditioned variants run DINOv3 twice)
/// reuses it. The cache is stage-agnostic (`dyn Any`); each lookup downcasts
/// back to the concrete model type the stage loader produced.
#[derive(Default)]
struct ResidentStages {
    /// Nesting depth of active `low_vram = false` runs; only the outermost
    /// [`ResidentStagesGuard`] drop clears the cache, so nested entry points
    /// (`run_from_path` -> `run_from_path_with_stages` -> `run_prepared`)
    /// share one residency window.
    active_runs: usize,
    /// Loaded stage models keyed by their stage prefix (`ss_flow`, ...); the
    /// elastic 1024/1536 DiTs append their config resolution so a batched
    /// run can pin one instance per target grid.
    models: HashMap<String, Arc<dyn std::any::Any + Send + Sync>>,
}

impl std::fmt::Debug for ResidentStages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidentStages")
            .field("active_runs", &self.active_runs)
            .field("models", &self.models.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl ResidentStages {
    /// Downcast the pinned model for `stage`, if any. Returns an owned `Arc`
    /// so callers can run the stage without holding the cache lock.
    fn cached<T: std::any::Any + Send + Sync>(&self, stage: &str) -> Option<Arc<T>> {
        self.models
            .get(stage)
            .and_then(|model| Arc::clone(model).downcast::<T>().ok())
    }
}

/// Run-scoped guard for [`Trellis2RunOptions::low_vram`] = false. The cache
/// nesting depth is bumped on entry and decremented on drop; the outermost
/// drop (success or error path alike) releases every pinned stage, so a
/// failed run never leaves weights resident.
struct ResidentStagesGuard<'a> {
    resident: &'a Mutex<ResidentStages>,
    armed: bool,
}

impl Drop for ResidentStagesGuard<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut resident = lock_resident(self.resident);
        resident.active_runs = resident.active_runs.saturating_sub(1);
        if resident.active_runs == 0 {
            resident.models.clear();
        }
    }
}

fn lock_resident(resident: &Mutex<ResidentStages>) -> MutexGuard<'_, ResidentStages> {
    resident
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Total device memory in bytes when the backend exposes it (CUDA:
/// `cudarc::driver::CudaContext::total_mem`). `None` on CPU and on backends
/// without the query; only used to make a resident-mode load failure
/// actionable.
fn device_memory_limit(device: &Device) -> Option<u64> {
    #[cfg(feature = "cuda")]
    {
        if let Ok(cuda) = device.as_cuda_device() {
            if let Ok(bytes) = cuda.cuda_stream().context().total_mem() {
                return Some(bytes as u64);
            }
        }
    }
    let _ = device;
    None
}

/// Actionable failure for a stage load inside a `low_vram = false` run:
/// names the option/CLI flag and the device memory limit instead of
/// silently falling back to per-stage loading.
fn resident_stage_load_error(stage: &str, device: &Device, source: Lux3dError) -> Lux3dError {
    let limit = match device_memory_limit(device) {
        Some(bytes) => format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0)),
        None => "unknown (the backend does not expose a memory limit)".to_string(),
    };
    Lux3dError::CanonicalWeightsValidation {
        message: format!(
            "TRELLIS.2 stage `{stage}` failed to load while `low_vram = false` keeps every \
             stage resident on the device (CLI `--no-low-vram`; device memory limit: {limit}). \
             The resident stage set must fit in device memory at load time; refusing to fall \
             back to per-stage loading. Re-run without `--no-low-vram`, or use a device with \
             enough memory. Underlying error: {source}"
        ),
    }
}

/// TRELLIS.2 image-to-3D pipeline (geometry slice).
#[derive(Debug)]
pub struct Trellis2Pipeline {
    /// Canonical package paths (`Some` when loaded through
    /// [`Self::load`]).
    pub weights: Option<CanonicalWeightSetPaths>,
    pub preprocess: Trellis2Preprocess,
    pub geometry: Trellis2GeometryStage,
    pub export: Trellis2ExportStage,
    pub config: Trellis2PipelineConfig,
    /// RMBG-2.0/BiRefNet background remover used for inputs without a usable
    /// alpha channel. `None` makes RGB inputs fail in preprocessing (the
    /// reference always runs RMBG; there is no opaque fallback).
    pub rmbg: Option<Rmbg2Model>,
    /// Raw checkpoint dir + device when loaded through
    /// [`Self::load_from_raw_dir`]. `pub(super)`: the texturing core
    /// (`runtime::trellis2_texture`) resolves the shape-encoder stage file
    /// through it.
    pub(super) raw_dir: Option<(PathBuf, Device)>,
    /// Stages pinned by a `low_vram = false` run; empty and unused in the
    /// default per-stage load/drop mode.
    resident: Mutex<ResidentStages>,
}

impl Trellis2Pipeline {
    /// Load from the canonical weights package (no HF access).
    pub fn load(options: ModelAssetOptions) -> Result<Self> {
        Self::load_with_rmbg(options, None, &Device::Cpu)
    }

    /// Canonical TRELLIS.2 load with an optional RMBG-2.0 package/GGUF/raw
    /// directory for RGB inputs.
    pub fn load_with_rmbg(
        options: ModelAssetOptions,
        rmbg_path: Option<&Path>,
        rmbg_device: &Device,
    ) -> Result<Self> {
        let weights = load_canonical_weights(ModelFamily::Trellis2, options)?;
        let config = weights
            .read_resolved_config_json()
            .map(|value| Trellis2PipelineConfig::from_resolved_config(&value))
            .unwrap_or_default();
        let rmbg = match rmbg_path {
            Some(path) => Some(Rmbg2Model::load_from_path(path, rmbg_device)?),
            None => None,
        };
        Ok(Self {
            weights: Some(weights),
            preprocess: Trellis2Preprocess::new(config.shape_resolution)?,
            geometry: Trellis2GeometryStage,
            export: Trellis2ExportStage,
            config,
            rmbg,
            raw_dir: None,
            resident: Mutex::new(ResidentStages::default()),
        })
    }

    /// Local-development loader over the raw TRELLIS.2 checkpoint directory
    /// (`models/TRELLIS.2-4B` layout). Each stage opens its own checkpoint
    /// file with the canonical stage prefix applied.
    pub fn load_from_raw_dir(path: &Path, device: &Device) -> Result<Self> {
        Self::load_from_raw_dir_with_rmbg(path, device, None)
    }

    /// Raw-checkpoint load with an optional RMBG-2.0 package/GGUF/raw
    /// directory for RGB inputs.
    pub fn load_from_raw_dir_with_rmbg(
        path: &Path,
        device: &Device,
        rmbg_path: Option<&Path>,
    ) -> Result<Self> {
        let mut config = Trellis2PipelineConfig::default();
        let pipeline_json = path.join("pipeline.json");
        if pipeline_json.is_file() {
            let body = std::fs::read_to_string(&pipeline_json).map_err(|source| {
                Lux3dError::ModelAssetIo {
                    path: pipeline_json.clone(),
                    source,
                }
            })?;
            let value: serde_json::Value = serde_json::from_str(&body).map_err(|source| {
                Lux3dError::CanonicalManifestJson {
                    path: pipeline_json.clone(),
                    source,
                }
            })?;
            config = Trellis2PipelineConfig::from_resolved_config(&value);
        }
        for (prefix, relative) in RAW_STAGE_FILES {
            let file = path.join(relative);
            if !file.is_file() {
                return Err(Lux3dError::MissingWeightFile { path: file });
            }
            let _ = prefix;
        }
        let rmbg = match rmbg_path {
            Some(path) => Some(Rmbg2Model::load_from_path(path, device)?),
            None => None,
        };
        Ok(Self {
            weights: None,
            preprocess: Trellis2Preprocess::new(config.shape_resolution)?,
            geometry: Trellis2GeometryStage,
            export: Trellis2ExportStage,
            config,
            rmbg,
            raw_dir: Some((path.to_path_buf(), device.clone())),
            resident: Mutex::new(ResidentStages::default()),
        })
    }

    pub(super) fn background_remover(&self) -> Option<&dyn Trellis2BackgroundRemover> {
        self.rmbg
            .as_ref()
            .map(|model| model as &dyn Trellis2BackgroundRemover)
    }

    /// VarBuilder rooted at one stage's namespace: the returned builder
    /// resolves the stage's bare tensor names (e.g. `from_latent.weight`),
    /// matching the stage loaders' convention. For the canonical package the
    /// merged file is namespaced with `.pp(prefix)`; for a raw dir the stage
    /// file already carries bare keys, so it is mmap-ed directly.
    fn stage_var_builder(&self, prefix: &str, device: &Device) -> Result<VarBuilder<'static>> {
        match (&self.raw_dir, &self.weights) {
            (Some((dir, raw_device)), _) => {
                if raw_device.location() != device.location() {
                    return Err(Lux3dError::InvalidInput(
                        "Trellis2 raw-dir pipeline is bound to a different device",
                    ));
                }
                let (_, relative) = RAW_STAGE_FILES
                    .iter()
                    .find(|(name, _)| *name == prefix)
                    .ok_or_else(|| Lux3dError::ModelAssetResolution {
                        message: format!(
                            "no raw TRELLIS.2 checkpoint mapped for stage prefix `{prefix}`"
                        ),
                    })?;
                let file = dir.join(relative);
                // SAFETY: the mmap-backed builder keeps the mapping alive for
                // every tensor created from it; the file is not mutated.
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        std::slice::from_ref(&file),
                        DType::F32,
                        device,
                    )
                }
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to open raw TRELLIS.2 stage `{prefix}`: {source}"),
                })?;
                Ok(vb)
            }
            (None, Some(weights)) => {
                // SAFETY: mmap-backed; the canonical file outlives the tensor
                // set produced from it and is not mutated.
                let vb = unsafe { weights.var_builder(DType::F32, device) }.map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("failed to open canonical TRELLIS.2 weights: {source}"),
                    }
                })?;
                Ok(vb.pp(prefix))
            }
            (None, None) => Err(Lux3dError::InvalidInput(
                "Trellis2Pipeline has no weights loaded",
            )),
        }
    }

    /// Open the run-scoped residency window when
    /// [`Trellis2RunOptions::low_vram`] is false (reference
    /// `Trellis2ImageTo3DPipeline.to(device)` with `low_vram = False`,
    /// `trellis2_image_to_3d.py:119-125`). The guard bumps a nesting depth so
    /// nested public entry points share one window; the outermost drop
    /// releases the pinned stages on both the success and the error path.
    /// `low_vram = true` returns a disarmed guard and keeps the exact
    /// historical per-stage load/drop behavior.
    fn enter_run_mode(&self, options: &Trellis2RunOptions) -> ResidentStagesGuard<'_> {
        if options.low_vram {
            return ResidentStagesGuard {
                resident: &self.resident,
                armed: false,
            };
        }
        let mut resident = lock_resident(&self.resident);
        if resident.active_runs == 0 {
            // Defensive: a previous window must have released its stages.
            resident.models.clear();
        }
        resident.active_runs += 1;
        drop(resident);
        ResidentStagesGuard {
            resident: &self.resident,
            armed: true,
        }
    }

    /// Load or reuse one stage model.
    ///
    /// With no active residency window (`low_vram = true`, the default) this
    /// loads a private `Arc` that the existing `drop` sites release exactly
    /// like before. Inside a `low_vram = false` window the first load is
    /// pinned in [`Self::resident`] and later acquisitions of the same stage
    /// prefix reuse it, so the weights stay on the device for the whole run
    /// (the reference pinning branch). A load failure inside the window
    /// returns [`resident_stage_load_error`] instead of silently falling
    /// back to per-stage loading.
    fn acquire_stage_model<T>(
        &self,
        stage: &str,
        cache_key: &str,
        device: &Device,
        load: impl FnOnce() -> Result<T>,
    ) -> Result<Arc<T>>
    where
        T: std::any::Any + Send + Sync,
    {
        let resident = lock_resident(&self.resident).active_runs > 0;
        if resident {
            if let Some(model) = lock_resident(&self.resident).cached::<T>(cache_key) {
                return Ok(model);
            }
        }
        let model = match load() {
            Ok(model) => model,
            Err(source) => {
                return Err(if resident {
                    resident_stage_load_error(stage, device, source)
                } else {
                    source
                });
            }
        };
        let model = Arc::new(model);
        if !resident {
            return Ok(model);
        }
        let mut cache = lock_resident(&self.resident);
        if cache.active_runs == 0 {
            // The run window closed while the stage loaded; hand the caller
            // its private copy (the cache is already released).
            return Ok(model);
        }
        if let Some(existing) = cache.cached::<T>(cache_key) {
            return Ok(existing);
        }
        cache.models.insert(cache_key.to_string(), model.clone());
        Ok(model)
    }

    /// DINOv3 image-conditioning stage (`image_cond`). The reference moves
    /// `image_cond_model` to the device once when `low_vram = False`
    /// (`trellis2_image_to_3d.py:119-125`); the 1024-conditioned variants
    /// call it twice (`cond_512` + `cond_1024`), so the resident window
    /// makes the second pass a cache hit.
    fn acquire_image_cond(&self, device: &Device) -> Result<Arc<Trellis2Dino>> {
        self.acquire_stage_model("image_cond", "image_cond", device, || {
            let vb = self.stage_var_builder("image_cond", device)?;
            Trellis2Dino::load(vb, &super::trellis2_dino::Trellis2DinoConfig::vitl16()).map_err(
                |source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to load TRELLIS.2 DINOv3: {source}"),
                },
            )
        })
    }

    /// Sparse-structure flow DiT (`ss_flow`).
    fn acquire_ss_flow(&self, device: &Device) -> Result<Arc<Trellis2SsDit>> {
        self.acquire_stage_model("ss_flow", "ss_flow", device, || {
            let vb = self.stage_var_builder("ss_flow", device)?;
            Trellis2SsDit::load(vb, &Trellis2SsConfig::trellis_2_4b()).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to load TRELLIS.2 ss_flow: {source}"),
                }
            })
        })
    }

    /// Sparse-structure decoder (`ss_dec`).
    fn acquire_ss_dec(&self, device: &Device) -> Result<Arc<Trellis2SsDecoder>> {
        self.acquire_stage_model("ss_dec", "ss_dec", device, || {
            let vb = self.stage_var_builder("ss_dec", device)?;
            Trellis2SsDecoder::load(vb, &Trellis2SsDecoderConfig::ss_dec_16l8()).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to load TRELLIS.2 ss_dec: {source}"),
                }
            })
        })
    }

    /// Shape or texture SLAT flow DiT (`shape_dit_512` / `shape_dit_1024` /
    /// `tex_dit_512` / `tex_dit_1024`). The elastic 1024/1536 checkpoints
    /// take a different coordinate bound per target grid, so the cache key
    /// carries the config resolution; a batched 1536 cascade can legitimately
    /// run different samples at 1536 and at the 1024 stop resolution.
    fn acquire_slat_dit(
        &self,
        stage: &str,
        config: Trellis2SlatConfig,
        device: &Device,
    ) -> Result<Arc<Trellis2SlatDit>> {
        let config_resolution = config.resolution;
        let cache_key = format!("{stage}@{}", config_resolution);
        self.acquire_stage_model(stage, &cache_key, device, || {
            let vb = self.stage_var_builder(stage, device)?;
            Trellis2SlatDit::load(vb, &config).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to load TRELLIS.2 {stage}: {source}"),
                }
            })
        })
    }

    /// Texture SLAT flow DiT (`tex_dit_512` / `tex_dit_1024`); the cache key
    /// carries the elastic config resolution (see [`Self::acquire_slat_dit`]).
    fn acquire_tex_dit(
        &self,
        stage: &str,
        config: Trellis2SlatConfig,
        device: &Device,
    ) -> Result<Arc<Trellis2TexDit>> {
        let config_resolution = config.resolution;
        let cache_key = format!("{stage}@{}", config_resolution);
        self.acquire_stage_model(stage, &cache_key, device, || {
            let vb = self.stage_var_builder(stage, device)?;
            Trellis2TexDit::load(vb, &config).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to load TRELLIS.2 {stage}: {source}"),
                }
            })
        })
    }

    /// Shape VAE decoder (`shape_dec`), used for the cascade upsample and the
    /// mesh decode.
    fn acquire_shape_vae(&self, device: &Device) -> Result<Arc<Trellis2ShapeVaeDecoder>> {
        self.acquire_stage_model("shape_dec", "shape_dec", device, || {
            let vb = self.stage_var_builder("shape_dec", device)?;
            Trellis2ShapeVaeDecoder::load(vb, &Trellis2ShapeVaeConfig::flexi_dual_shape()).map_err(
                |source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to load TRELLIS.2 shape_dec: {source}"),
                },
            )
        })
    }

    /// Guided texture VAE decoder (`tex_dec`).
    fn acquire_tex_dec(&self, device: &Device) -> Result<Arc<Trellis2TexDecoder>> {
        self.acquire_stage_model("tex_dec", "tex_dec", device, || {
            let vb = self.stage_var_builder("tex_dec", device)?;
            Trellis2TexDecoder::load(vb, &Trellis2ShapeVaeConfig::tex_decoder()).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to load TRELLIS.2 tex_dec: {source}"),
                }
            })
        })
    }

    /// `(ss_resolution, decode_resolution)` for the selected variant (the
    /// upstream `ss_res` map and the `res` decode resolution): `512` ->
    /// `(32, 512)` (package config), `1024` -> `(64, 1024)`, `1024_cascade`
    /// -> `(32, 1024)`, `1536_cascade` -> `(32, 1536 target)`. All four
    /// variants are implemented; this cannot fail for known variants but
    /// keeps the `Result` surface for future additions.
    pub fn resolutions(&self, options: &Trellis2RunOptions) -> Result<(u32, u32)> {
        match options.pipeline_type {
            Trellis2PipelineType::R512 => {
                Ok((self.config.ss_resolution, self.config.shape_resolution))
            }
            Trellis2PipelineType::R1024 => Ok((
                TRELLIS2_SINGLE_1024_SS_RESOLUTION,
                TRELLIS2_SINGLE_1024_SHAPE_RESOLUTION,
            )),
            Trellis2PipelineType::Cascade1024 => {
                Ok((self.config.ss_resolution, TRELLIS2_CASCADE_HR_RESOLUTION))
            }
            Trellis2PipelineType::Cascade1536 => Ok((
                self.config.ss_resolution,
                TRELLIS2_CASCADE_1536_TARGET_RESOLUTION,
            )),
        }
    }

    /// DINOv3 conditioning: `cond [1, 1029, 1024]` + zero negative cond.
    pub fn image_conditioning(
        &self,
        prepared: &crate::contracts::Trellis2PreparedImage,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let pixel_values = self
            .preprocess
            .normalized_pixel_values(prepared, device)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 DINOv3 input normalization failed: {source}"),
            })?;
        self.dino_conditioning_pixels(&pixel_values, device)
    }

    /// Cascade conditioning at the HR resolution (1024): DINOv3 over
    /// pre-built pixel values (`[1, 3, 1024, 1024]`), producing
    /// `[1, 4101, 1024]` + zero negative cond. `pub(super)`: the texturing
    /// pipeline (`runtime::trellis2_texture`) shares the DINOv3 load for
    /// its 512/1024 `get_cond` passes.
    pub(super) fn dino_conditioning_pixels(
        &self,
        pixel_values: &Tensor,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let dino = self.acquire_image_cond(device)?;
        let cond = dino.forward(pixel_values).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 DINOv3 forward failed: {source}"),
            }
        })?;
        let neg_cond =
            Tensor::zeros_like(&cond).map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 negative cond allocation failed: {source}"),
            })?;
        Ok((cond, neg_cond))
    }

    /// The preprocess stage at the cascade HR resolution. It only owns the
    /// resolution-specific resize (`resize_crop`); the crop itself is shared
    /// with the 512 stage.
    fn cascade_preprocess_hr(&self) -> Result<Trellis2Preprocess> {
        Trellis2Preprocess::new(TRELLIS2_CASCADE_HR_RESOLUTION)
    }

    /// Cascade `cond_1024` for an already-prepared 512 image
    /// ([`Self::run_prepared`], which no longer holds the source pixels):
    /// the prepared pixels are the square crop resized to 512, so resizing
    /// them to 1024 (LANCZOS3, the preprocess resampler) approximates the
    /// reference's direct crop -> 1024 resize (a double resample).
    /// [`Self::run_from_path_with_stages`] re-prepares from the source and
    /// is exact.
    fn cascade_cond_1024_from_prepared(
        &self,
        prepared: &crate::contracts::Trellis2PreparedImage,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let src = prepared.target_size.width as usize;
        let dst = TRELLIS2_CASCADE_HR_RESOLUTION as usize;
        if prepared.target_size.width != prepared.target_size.height
            || prepared.uint8_rgb.len() != src * src * 3
        {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 cascade 1024 conditioning needs a square 512 prepared image",
            ));
        }
        let resized =
            crate::preprocess::resize_lanczos_u8::<3>(&prepared.uint8_rgb, src, src, dst, dst);
        let pixel_values = crate::preprocess::normalized_pixel_values_from_u8(
            &resized,
            TRELLIS2_CASCADE_HR_RESOLUTION,
            device,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("Trellis2 DINOv3 input normalization failed: {source}"),
        })?;
        self.dino_conditioning_pixels(&pixel_values, device)
    }

    /// Sparse-structure stage: sample `z_s` from `noise` (or the seeded RNG
    /// when `noise` is `None`) and decode the occupancy coords at
    /// `ss_resolution`. The SS flow DiT and decoder are dropped afterwards.
    /// `params` are the resolved SS sampler parameters
    /// (`resolve_stage_params(&config.ss_sampler, options.ss_sampler,
    /// options.ss_steps)`).
    pub fn sample_sparse_structure(
        &self,
        cond: &Tensor,
        neg_cond: &Tensor,
        noise: Option<&Tensor>,
        seed: u64,
        params: &Trellis2SamplerParams,
        device: &Device,
        ss_resolution: u32,
    ) -> Result<(Tensor, Vec<[i32; 3]>)> {
        let (z_s, mut batch) = self.sample_sparse_structure_many(
            cond,
            neg_cond,
            noise,
            1,
            seed,
            params,
            device,
            ss_resolution,
        )?;
        Ok((
            z_s,
            batch
                .pop()
                .expect("sample_sparse_structure_many(1) yields one coord set"),
        ))
    }

    /// Batched sparse-structure stage (`num_samples > 1`): ONE joint noise
    /// draw `[samples, 8, R, R, R]` (upstream `torch.randn(num_samples, ...)`,
    /// `trellis2_image_to_3d.py:208`), one batched flow pass (the `[1, L, C]`
    /// conditioning is tiled over the batch) and one batched occupancy
    /// decode; the active coords are returned per sample. `samples = 1`
    /// consumes exactly the legacy single-sample stream values in the same
    /// order.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sample_sparse_structure_many(
        &self,
        cond: &Tensor,
        neg_cond: &Tensor,
        noise: Option<&Tensor>,
        samples: usize,
        seed: u64,
        params: &Trellis2SamplerParams,
        device: &Device,
        ss_resolution: u32,
    ) -> Result<(Tensor, Vec<Vec<[i32; 3]>>)> {
        if samples == 0 {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 sparse structure needs at least one sample",
            ));
        }
        let ss_config = Trellis2SsConfig::trellis_2_4b();
        let noise = match noise {
            Some(noise) => {
                let dims = noise.dims();
                if dims.len() != 5
                    || dims[1] != ss_config.in_channels
                    || (dims[2], dims[3], dims[4])
                        != (
                            ss_config.resolution,
                            ss_config.resolution,
                            ss_config.resolution,
                        )
                {
                    return Err(Lux3dError::ModelAssetResolution {
                        message: format!(
                            "Trellis2 SS init noise must be [N, {}, {}, {}, {}], got {dims:?}",
                            ss_config.in_channels,
                            ss_config.resolution,
                            ss_config.resolution,
                            ss_config.resolution
                        ),
                    });
                }
                match dims[0] {
                    batch if batch == samples => noise.clone(),
                    1 => noise.repeat((samples, 1, 1, 1, 1)).map_err(|source| {
                        Lux3dError::CanonicalWeightsValidation {
                            message: format!("Trellis2 SS noise batch repeat failed: {source}"),
                        }
                    })?,
                    batch => {
                        return Err(Lux3dError::ModelAssetResolution {
                            message: format!(
                                "Trellis2 SS init noise batch {batch} cannot seed {samples} samples"
                            ),
                        })
                    }
                }
            }
            None => draw_ss_noise(samples, seed, device)?,
        };
        let cond_batch = repeat_cond_to_batch(cond, samples)?;
        let neg_cond_batch = repeat_cond_to_batch(neg_cond, samples)?;
        let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
        let mut stage_start = std::time::Instant::now();
        let ss_dit = self.acquire_ss_flow(device)?;
        if stage_time {
            eprintln!(
                "[stage] trellis2 ss dit load: {:.2}s",
                stage_start.elapsed().as_secs_f64()
            );
            stage_start = std::time::Instant::now();
        }
        let (z_s, _trace) = sample_dense(
            &|x: &Tensor, t_model: f32, c: &Tensor| ss_dit.forward(x, t_model, c),
            &noise,
            &cond_batch,
            &neg_cond_batch,
            params,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("Trellis2 SS sampling failed: {source}"),
        })?;
        drop(ss_dit);
        if stage_time {
            eprintln!(
                "[stage] trellis2 ss sample: {:.2}s",
                stage_start.elapsed().as_secs_f64()
            );
            stage_start = std::time::Instant::now();
        }

        let decoder = self.acquire_ss_dec(device)?;
        if stage_time {
            eprintln!(
                "[stage] trellis2 ss dec load: {:.2}s",
                stage_start.elapsed().as_secs_f64()
            );
            stage_start = std::time::Instant::now();
        }
        let logits =
            decoder
                .forward(&z_s)
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("Trellis2 SS decode failed: {source}"),
                })?;
        drop(decoder);
        if stage_time {
            eprintln!(
                "[stage] trellis2 ss decode: {:.2}s",
                stage_start.elapsed().as_secs_f64()
            );
            stage_start = std::time::Instant::now();
        }
        let coords4 = occupancy_coords(&logits, ss_resolution).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 occupancy extraction failed: {source}"),
            }
        })?;
        if stage_time {
            eprintln!(
                "[stage] trellis2 ss occupancy: {:.2}s",
                stage_start.elapsed().as_secs_f64()
            );
        }
        let mut coords_by_sample = vec![Vec::new(); samples];
        for coord in coords4 {
            let batch = coord[0];
            if batch < 0 || batch as usize >= samples {
                return Err(Lux3dError::ModelAssetResolution {
                    message: format!(
                        "Trellis2 occupancy coord {coord:?} is outside the {samples}-sample batch"
                    ),
                });
            }
            coords_by_sample[batch as usize].push([coord[1], coord[2], coord[3]]);
        }
        Ok((z_s, coords_by_sample))
    }

    /// Shape SLAT stage (512): sample the (normalized) latent on `coords`
    /// and de-normalize. The flow DiT is dropped afterwards. `params` are
    /// the resolved shape sampler parameters (`resolve_stage_params`).
    pub fn sample_shape_slat(
        &self,
        cond: &Tensor,
        neg_cond: &Tensor,
        coords: &[[i32; 3]],
        noise: Option<&Tensor>,
        seed: u64,
        params: &Trellis2SamplerParams,
        device: &Device,
    ) -> Result<Tensor> {
        self.sample_shape_slat_stage(
            "shape_dit_512",
            Trellis2SlatConfig::shape_512(),
            TRELLIS2_SHAPE_NOISE_SALT,
            cond,
            neg_cond,
            coords,
            noise,
            seed,
            params,
            device,
        )
    }

    /// Cascade HR shape SLAT stage (`shape_dit_1024`): same sampler params,
    /// a distinct RNG salt so the HR noise stream differs from the LR one,
    /// and the elastic validation bound for the HR grid. Negative cond is
    /// `zeros_like(cond)` exactly like the reference. `noise` allows an
    /// injected `[N, 32]` start (batched joint draw; `None` = seeded RNG).
    #[allow(clippy::too_many_arguments)]
    fn sample_shape_slat_hr(
        &self,
        cond: &Tensor,
        coords: &[[i32; 3]],
        noise: Option<&Tensor>,
        seed: u64,
        params: &Trellis2SamplerParams,
        device: &Device,
        hr_grid_resolution: usize,
    ) -> Result<Tensor> {
        let neg_cond =
            Tensor::zeros_like(cond).map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 negative cond allocation failed: {source}"),
            })?;
        let config = if hr_grid_resolution > 64 {
            // 1536 cascade (or a budget-reduced 1536 pass): the elastic DiT
            // accepts the larger grid; raise the validation bound to 96.
            Trellis2SlatConfig::shape_1536()
        } else {
            Trellis2SlatConfig::shape_1024()
        };
        self.sample_shape_slat_stage(
            "shape_dit_1024",
            config,
            TRELLIS2_HR_SHAPE_NOISE_SALT,
            cond,
            &neg_cond,
            coords,
            noise,
            seed,
            params,
            device,
        )
    }

    /// One shape SLAT sampling pass over a stage DiT: seeded (or injected)
    /// noise `[N, 32]` on `coords` -> Euler/CFG -> de-normalize. The DiT is
    /// dropped before the de-normalization so only one flow model is ever
    /// resident in the default `low_vram = true` mode (12 GiB budget); with
    /// `low_vram = false` the explicit drop releases only this call's `Arc`
    /// clone and the run cache keeps the model pinned.
    #[allow(clippy::too_many_arguments)]
    fn sample_shape_slat_stage(
        &self,
        stage: &str,
        config: Trellis2SlatConfig,
        noise_salt: u64,
        cond: &Tensor,
        neg_cond: &Tensor,
        coords: &[[i32; 3]],
        noise: Option<&Tensor>,
        seed: u64,
        params: &Trellis2SamplerParams,
        device: &Device,
    ) -> Result<Tensor> {
        let noise = match noise {
            Some(noise) => noise.clone(),
            None => draw_slat_noise(coords.len(), seed, noise_salt, device)?,
        };
        let dit = self.acquire_slat_dit(stage, config, device)?;
        let (slat_norm, _trace) = sample_sparse(
            &|x: &Tensor, t_model: f32, c: &Tensor, cs: &[[i32; 3]]| dit.forward(x, cs, t_model, c),
            coords,
            &noise,
            cond,
            neg_cond,
            params,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("Trellis2 shape SLAT sampling failed: {source}"),
        })?;
        drop(dit);
        self.denormalize_shape_slat(&slat_norm)
    }

    /// `slat * std + mean` with the package shape-normalization vectors.
    pub fn denormalize_shape_slat(&self, slat_norm: &Tensor) -> Result<Tensor> {
        let device = slat_norm.device();
        let mean = Tensor::from_vec(self.config.shape_slat_mean.clone(), (1, 32), device).map_err(
            |source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 mean vector allocation failed: {source}"),
            },
        )?;
        let std = Tensor::from_vec(self.config.shape_slat_std.clone(), (1, 32), device).map_err(
            |source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 std vector allocation failed: {source}"),
            },
        )?;
        slat_norm
            .broadcast_mul(&std)
            .and_then(|t| t.broadcast_add(&mean))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 shape SLAT de-normalization failed: {source}"),
            })
    }

    /// Shape VAE `upsample(slat, 4)` only (the reference
    /// `sample_shape_slat_cascade` upsample step, `trellis2_image_to_3d.py:319-326`):
    /// `from_latent` + stages 0..3 exactly like the full decode, returning
    /// the HR coords on the 512 grid. The batched cascade runs this per
    /// sample BEFORE the shared token-budget loop so the whole batch is
    /// quantized and budgeted together.
    pub fn upsample_shape_slat(
        &self,
        lr_coords: &[[i32; 3]],
        lr_slat_denorm: &Tensor,
        device: &Device,
    ) -> Result<Vec<[i32; 3]>> {
        let profile = std::env::var_os("LUX3D_STAGE_TIME").is_some();
        let load_mark = std::time::Instant::now();
        let decoder = self.acquire_shape_vae(device)?;
        if profile {
            eprintln!(
                "[stage] trellis2 shape vae load: {:.2}s",
                load_mark.elapsed().as_secs_f64()
            );
        }
        let hr_coords = decoder
            .upsample_coords(lr_coords, lr_slat_denorm, TRELLIS2_CASCADE_UPSAMPLE_TIMES)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 shape VAE upsample failed: {source}"),
            })?;
        drop(decoder);
        Ok(hr_coords)
    }

    /// Cascade upsample + quantize steps (the middle of the reference
    /// `sample_shape_slat_cascade`, `trellis2_image_to_3d.py:319-339`):
    ///
    /// 1. load the shape VAE and run `upsample(slat, 4)` — `from_latent` +
    ///    stages 0..3 exactly like the full decode — to obtain the HR coords
    ///    on the 512 grid ([`Self::upsample_shape_slat`]);
    /// 2. drop the VAE, then quantize onto the `hr_resolution // 16` grid,
    ///    deduplicate and run the token-budget loop
    ///    ([`quantize_hr_coords`]) starting at `target_resolution` (1024 or
    ///    1536; the loop's stop resolution stays 1024).
    ///
    /// Returns `(hr_upsampled_coords, quant_coords, hr_resolution)`. The
    /// 1024 cascade's loop always breaks at `hr_resolution == 1024`, so
    /// `hr_resolution` is 1024 there regardless of the token count; the
    /// 1536 cascade decodes at 1536 unless the token budget reduces it.
    /// Multi-sample runs use [`quantize_hr_coords_batched`] on the whole
    /// batch instead of this per-sample helper.
    pub fn upsample_and_quantize_shape_slat(
        &self,
        lr_coords: &[[i32; 3]],
        lr_slat_denorm: &Tensor,
        max_num_tokens: usize,
        target_resolution: u32,
        device: &Device,
    ) -> Result<(Vec<[i32; 3]>, Vec<[i32; 3]>, u32)> {
        let hr_coords = self.upsample_shape_slat(lr_coords, lr_slat_denorm, device)?;
        let (quant_coords, hr_resolution) = quantize_hr_coords(
            &hr_coords,
            TRELLIS2_CASCADE_LR_RESOLUTION,
            target_resolution,
            max_num_tokens,
        );
        Ok((hr_coords, quant_coords, hr_resolution))
    }

    /// Full cascade shape stage (reference `sample_shape_slat_cascade`):
    /// LR SLAT on `coords` (32^3) with `shape_dit_512` + `cond_lr`, shape
    /// VAE `upsample(slat, 4)`, HR quantization + token budget starting at
    /// `target_resolution` (1024 or 1536), HR SLAT on the quantized coords
    /// with `shape_dit_1024` + `cond_hr`. Both stages use the resolved
    /// `shape_slat_sampler` params (defaults: 12 steps, guidance 7.5,
    /// rescale 0.5, interval [0.6, 1.0], R=3.0); the DiTs and the VAE are
    /// loaded and dropped sequentially.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_shape_slat_cascade(
        &self,
        cond_lr: &Tensor,
        cond_hr: &Tensor,
        coords: &[[i32; 3]],
        lr_noise: Option<&Tensor>,
        seed: u64,
        params: &Trellis2SamplerParams,
        max_num_tokens: usize,
        target_resolution: u32,
        device: &Device,
    ) -> Result<Trellis2CascadeShape> {
        let neg_lr = Tensor::zeros_like(cond_lr).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 negative cond allocation failed: {source}"),
            }
        })?;
        let lr_slat =
            self.sample_shape_slat(cond_lr, &neg_lr, coords, lr_noise, seed, params, device)?;
        let (hr_upsampled_coords, quant_coords, hr_resolution) = self
            .upsample_and_quantize_shape_slat(
                coords,
                &lr_slat,
                max_num_tokens,
                target_resolution,
                device,
            )?;
        if quant_coords.is_empty() {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 cascade quantized the upsampled coords to zero HR tokens",
            ));
        }
        let hr_slat = self.sample_shape_slat_hr(
            cond_hr,
            &quant_coords,
            None,
            seed,
            params,
            device,
            (hr_resolution / 16) as usize,
        )?;
        Ok(Trellis2CascadeShape {
            lr_slat,
            hr_upsampled_coords,
            quant_coords,
            hr_resolution,
            hr_slat,
        })
    }

    /// Shape VAE decode + dual-grid mesh extraction from a de-normalized
    /// shape latent on `coords` (no flow DiT involved). `cond`/`z_s` are
    /// placeholders; the stage struct carries the decode products.
    /// `resolution` is the mesh extraction resolution (512 production;
    /// oracle fixtures replay at their own resolution).
    pub fn decode_shape_slat(
        &self,
        coords: &[[i32; 3]],
        slat_denorm: &Tensor,
        resolution: u32,
        device: &Device,
    ) -> Result<Trellis2Stages> {
        let shape_resolution = resolution;
        let (decoded, mesh) =
            self.decode_shape_vae(coords, slat_denorm, device, shape_resolution)?;
        let mesh = self.assemble_mesh(shape_resolution, mesh, &decoded.coords, &decoded.feats)?;
        Ok(Trellis2Stages {
            cond: Tensor::zeros((1,), DType::F32, device).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("placeholder cond allocation failed: {source}"),
                }
            })?,
            neg_cond: Tensor::zeros((1,), DType::F32, device).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("placeholder neg cond allocation failed: {source}"),
                }
            })?,
            z_s: None,
            ss_coords: coords.to_vec(),
            shape_coords: coords.to_vec(),
            shape_latent: slat_denorm.clone(),
            decoded_feats: decoded.feats,
            mesh: Some(mesh),
            lr_shape_latent: None,
            hr_upsampled_coords: None,
            quant_coords: None,
            hr_resolution: None,
            subdiv_logits: decoded.subdiv_logits,
            tex_latent: None,
            tex_attrs: None,
            textured: None,
        })
    }

    fn decode_shape_vae(
        &self,
        coords: &[[i32; 3]],
        slat_denorm: &Tensor,
        device: &Device,
        shape_resolution: u32,
    ) -> Result<(super::trellis2_vae::ShapeVaeDecodeOutput, DualGridMesh)> {
        let profile = std::env::var_os("LUX3D_STAGE_TIME").is_some();
        let load_mark = std::time::Instant::now();
        let decoder = self.acquire_shape_vae(device)?;
        if profile {
            eprintln!(
                "[stage] trellis2 shape vae load: {:.2}s",
                load_mark.elapsed().as_secs_f64()
            );
        }
        decode_shape_mesh(
            &decoder,
            coords,
            slat_denorm,
            shape_resolution as usize,
            self.config.voxel_margin,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("Trellis2 shape VAE decode failed: {source}"),
        })
    }

    /// Re-normalize the de-normalized shape SLAT for the texture stage:
    /// `shape_slat_norm = (shape_slat - mean) / std` with the shape
    /// normalization vectors (`sample_tex_slat`, pipeline 407-409).
    pub fn normalize_shape_slat(&self, shape_slat_denorm: &Tensor) -> Result<Tensor> {
        let device = shape_slat_denorm.device();
        let mean = Tensor::from_vec(self.config.shape_slat_mean.clone(), (1, 32), device).map_err(
            |source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 mean vector allocation failed: {source}"),
            },
        )?;
        let std = Tensor::from_vec(self.config.shape_slat_std.clone(), (1, 32), device).map_err(
            |source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 std vector allocation failed: {source}"),
            },
        )?;
        shape_slat_denorm
            .broadcast_sub(&mean)
            .and_then(|t| t.broadcast_div(&std))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 shape SLAT normalization failed: {source}"),
            })
    }

    /// Texture SLAT stage (512): re-normalize the shape SLAT, sample the
    /// (normalized) texture latent with `tex_dit_512` on the shape latent
    /// coords, de-normalize with the texture vectors. The flow DiT is
    /// dropped before the de-normalization. `params` are the resolved
    /// texture sampler parameters (`resolve_stage_params`).
    pub fn sample_tex_slat(
        &self,
        cond: &Tensor,
        shape_slat_denorm: &Tensor,
        coords: &[[i32; 3]],
        noise: Option<&Tensor>,
        seed: u64,
        params: &Trellis2SamplerParams,
        device: &Device,
    ) -> Result<Tensor> {
        self.sample_tex_slat_stage(
            "tex_dit_512",
            Trellis2SlatConfig::tex_512(),
            cond,
            shape_slat_denorm,
            coords,
            noise,
            seed,
            params,
            device,
        )
    }

    /// Texture SLAT stage (cascade/1024 + `1536_cascade`): same sampler
    /// params with the `tex_dit_1024` checkpoint (reference conditions the
    /// texture stage on `cond_1024`); `hr_grid_resolution` picks the
    /// elastic validation bound for the HR coord grid. `noise` allows an
    /// injected `[N, 32]` start (test surface; `None` = seeded RNG).
    #[allow(clippy::too_many_arguments)]
    pub fn sample_tex_slat_hr(
        &self,
        cond: &Tensor,
        shape_slat_denorm: &Tensor,
        coords: &[[i32; 3]],
        noise: Option<&Tensor>,
        seed: u64,
        params: &Trellis2SamplerParams,
        device: &Device,
        hr_grid_resolution: usize,
    ) -> Result<Tensor> {
        let config = if hr_grid_resolution > 64 {
            Trellis2SlatConfig::tex_1536()
        } else {
            Trellis2SlatConfig::tex_1024()
        };
        self.sample_tex_slat_stage(
            "tex_dit_1024",
            config,
            cond,
            shape_slat_denorm,
            coords,
            noise,
            seed,
            params,
            device,
        )
    }

    /// One texture SLAT sampling pass: seeded (or injected) noise `[N, 32]`
    /// on `coords`, concatenated with the re-normalized shape SLAT
    /// (`noise || shape_norm`, reference concat order) -> Euler sampler ->
    /// `slat * tex_std + tex_mean`.
    #[allow(clippy::too_many_arguments)]
    fn sample_tex_slat_stage(
        &self,
        stage: &str,
        config: Trellis2SlatConfig,
        cond: &Tensor,
        shape_slat_denorm: &Tensor,
        coords: &[[i32; 3]],
        noise: Option<&Tensor>,
        seed: u64,
        params: &Trellis2SamplerParams,
        device: &Device,
    ) -> Result<Tensor> {
        if shape_slat_denorm
            .dim(0)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 shape latent has no rows: {source}"),
            })?
            != coords.len()
        {
            return Err(Lux3dError::ModelAssetResolution {
                message: format!(
                    "shape latent row count does not match the {} texture coords",
                    coords.len()
                ),
            });
        }
        let noise = match noise {
            Some(noise) => noise.clone(),
            None => draw_slat_noise(coords.len(), seed, TRELLIS2_TEX_NOISE_SALT, device)?,
        };
        let shape_slat_norm = self.normalize_shape_slat(shape_slat_denorm)?;
        let dit = self.acquire_tex_dit(stage, config, device)?;
        let (slat_norm, _trace) = sample_sparse(
            &|x: &Tensor, t_model: f32, c: &Tensor, cs: &[[i32; 3]]| {
                dit.forward(x, &shape_slat_norm, cs, t_model, c)
            },
            coords,
            &noise,
            cond,
            cond,
            params,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("Trellis2 texture SLAT sampling failed: {source}"),
        })?;
        drop(dit);
        self.denormalize_tex_slat(&slat_norm)
    }

    /// `slat * std + mean` with the package texture-normalization vectors.
    pub fn denormalize_tex_slat(&self, slat_norm: &Tensor) -> Result<Tensor> {
        let device = slat_norm.device();
        let mean = Tensor::from_vec(self.config.tex_slat_mean.clone(), (1, 32), device).map_err(
            |source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 mean vector allocation failed: {source}"),
            },
        )?;
        let std = Tensor::from_vec(self.config.tex_slat_std.clone(), (1, 32), device).map_err(
            |source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 std vector allocation failed: {source}"),
            },
        )?;
        slat_norm
            .broadcast_mul(&std)
            .and_then(|t| t.broadcast_add(&mean))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 texture SLAT de-normalization failed: {source}"),
            })
    }

    /// Guided texture VAE decode (`decode_tex_slat`): `tex_dec` on the
    /// de-normalized texture latent with the shape decoder's raw C2S
    /// subdivision logits as guides; the decoded 6 channels are
    /// `raw * 0.5 + 0.5` clamped to [0, 1]. Returns the final voxel coords
    /// (equal to the shape decode's final coords) and the flat `[N, 6]`
    /// PBR attributes.
    pub fn decode_tex_attrs(
        &self,
        coords: &[[i32; 3]],
        tex_latent_denorm: &Tensor,
        guide_subs: &[Tensor],
        device: &Device,
    ) -> Result<(Vec<[i32; 3]>, Vec<f32>)> {
        let decoder = self.acquire_tex_dec(device)?;
        let decoded = decoder
            .decode(coords, tex_latent_denorm, guide_subs)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 texture VAE decode failed: {source}"),
            })?;
        drop(decoder);
        let attrs = pbr_attrs_from_raw(&decoded.feats).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 PBR postprocess failed: {source}"),
            }
        })?;
        Ok((decoded.coords, attrs))
    }

    /// Attach the texture stage products to assembled mesh buffers: the
    /// voxel payload switches to the 6-channel PBR attributes on the shape
    /// decode's final coords and the per-vertex base colors are sampled
    /// with the reference sparse trilinear (`query_vertex_attrs`).
    fn attach_texture(
        &self,
        mesh: &mut crate::geometry::Trellis2MeshBuffers,
        coords: &[[i32; 3]],
        attrs: &[f32],
        resolution: u32,
    ) -> Result<()> {
        let vertices: Vec<[f32; 3]> = mesh
            .vertices
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let colors =
            query_vertex_colors(&vertices, coords, attrs, resolution).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("Trellis2 vertex color sampling failed: {source}"),
                }
            })?;
        mesh.voxel_attrs = attrs.to_vec();
        mesh.voxel_attr_channels = super::trellis2_tex::Trellis2PbrLayout::CHANNELS;
        mesh.vertex_colors = Some(colors);
        mesh.validate()
    }

    /// Texture-export stage (upstream `to_glb` UV unwrap + bake); see
    /// [`build_trellis2_textured_mesh`] for the behavior and the clean-skip
    /// conditions. `original_surface` is the pre-postprocess snapshot that
    /// enables the upstream texel snap-back
    /// ([`build_trellis2_textured_mesh_snapped`]); `None` keeps the plain
    /// texturing-pipeline bake.
    fn build_textured_mesh(
        &self,
        mesh: &crate::geometry::Trellis2MeshBuffers,
        options: &Trellis2TextureExportOptions,
        original_surface: Option<&(Vec<f32>, Vec<u32>)>,
    ) -> Result<Option<Trellis2TexturedMesh>> {
        match original_surface {
            Some((vertices, faces)) => {
                build_trellis2_textured_mesh_snapped(mesh, options, vertices, faces)
            }
            None => build_trellis2_textured_mesh(mesh, options),
        }
    }

    /// Snapshot the pre-postprocess surface for the bake snap-back.
    ///
    /// Upstream `to_glb` fills holes (`mesh.fill_holes(max_hole_perimeter =
    /// 3e-2)`, `postprocess.py:108-113`) and then builds its BVH from the
    /// cleaned decoded mesh right before the remesh/simplify branches
    /// (`postprocess.py:122`), keeping it for the texel projection
    /// (`postprocess.py:254-256`). Our [`Self::postprocess_decoded_mesh`]
    /// owns that whole chain and starts with the same fill, so the snapshot
    /// applies the SAME `fill_holes(fill_hole_perimeter)` pass here before
    /// returning; the snap BVH is thus built from the filled surface, not the
    /// raw decoded one. Already-filled meshes pass through byte-identically
    /// (the fill only appends centroid fan patches for closed loops under the
    /// threshold).
    ///
    /// Returns `None` when no textured mesh will be baked (`with_texture`
    /// off, texture export disabled / unwrap off, empty mesh) or when the
    /// geometry post-process is disabled — then the bake mesh IS the
    /// original surface and the projection is the identity, so the BVH is
    /// not built.
    fn capture_bake_snap_surface(
        mesh: &crate::geometry::Trellis2MeshBuffers,
        options: &Trellis2RunOptions,
    ) -> Result<Option<(Vec<f32>, Vec<u32>)>> {
        if !options.postprocess.enabled
            || !options.with_texture
            || !options.texture_export.enabled
            || !options.texture_export.uv_unwrap
            || mesh.face_count() == 0
        {
            return Ok(None);
        }
        let vertex_count = mesh.vertex_count();
        let mut faces = Vec::with_capacity(mesh.faces.len());
        for &index in &mesh.faces {
            let index = usize::try_from(index).map_err(|_| {
                Lux3dError::InvalidInput("Trellis2 bake snap-back rejected a negative face index")
            })?;
            if index >= vertex_count {
                return Err(Lux3dError::InvalidInput(
                    "Trellis2 bake snap-back rejected a face index outside the vertex buffer",
                ));
            }
            faces.push(index as u32);
        }
        // Same shared pre-branch fill the geometry post-process applies
        // (`postprocess.py:108-113`), so the snap BVH and the remesh BVH see
        // the same filled surface.
        let (vertices, faces) = fill_holes_with_centroids(
            &mesh.vertices,
            &faces,
            options.postprocess.fill_hole_perimeter,
        );
        Ok(Some((vertices, faces)))
    }

    fn assemble_mesh(
        &self,
        resolution: u32,
        mesh: DualGridMesh,
        coords: &[[i32; 3]],
        feats: &Tensor,
    ) -> Result<crate::geometry::Trellis2MeshBuffers> {
        let mut voxel_coords = Vec::with_capacity(coords.len() * 3);
        for coord in coords {
            voxel_coords.extend_from_slice(&[coord[0], coord[1], coord[2]]);
        }
        let attrs = feats
            .to_device(&Device::Cpu)
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 decoded feats download failed: {source}"),
            })?;
        let attr_channels = if coords.is_empty() {
            7
        } else {
            attrs.len() / coords.len()
        };
        self.geometry.assemble(
            resolution,
            mesh.vertices,
            mesh.faces,
            voxel_coords,
            attrs,
            attr_channels,
        )
    }

    /// Geometry post-process of the assembled dual-grid mesh (upstream
    /// `decode_latent` `fill_holes` + `to_glb` remesh/cleanup branches,
    /// implemented by `Trellis2GeometryStage::postprocess_decoded_mesh`). Runs
    /// BEFORE `attach_texture` so the final (possibly re-topologized) vertices
    /// are the ones the texture stage samples PBR colors for — no
    /// vertex-color remap is needed on the production path, because
    /// `assemble_mesh` always produces `vertex_colors = None`.
    ///
    /// With `options.enabled = false` the input buffers are returned
    /// untouched (byte-identical to the pre-postprocess pipeline). The
    /// `voxel_coords` / `voxel_attrs` payload, `resolution` and channel count
    /// pass through unchanged; only `vertices` / `faces` (and, defensively,
    /// a pre-attached `vertex_colors` buffer) can change.
    fn postprocess_decoded_mesh(
        &self,
        mesh: crate::geometry::Trellis2MeshBuffers,
        options: &Trellis2MeshPostprocessOptions,
    ) -> Result<crate::geometry::Trellis2MeshBuffers> {
        self.geometry.postprocess_decoded_mesh(mesh, options)
    }

    /// Core execution dispatch: the 512 path or the 1024 cascade. `cond_hr`
    /// is the cascade's 1024 conditioning (`None` -> reuse `cond`, the
    /// documented `run_with_latents` behavior). `tex_latent_override` is an
    /// injected de-normalized texture latent (the `run_with_latents`
    /// `tex_init` surface; skipped when `with_texture` is false).
    #[allow(clippy::too_many_arguments)]
    fn run_core(
        &self,
        cond: &Tensor,
        neg_cond: &Tensor,
        cond_hr: Option<&Tensor>,
        ss_noise: Option<&Tensor>,
        coords_override: Option<Vec<[i32; 3]>>,
        shape_noise: Option<&Tensor>,
        shape_latent_override: Option<&Tensor>,
        tex_latent_override: Option<&Tensor>,
        device: &Device,
        options: &Trellis2RunOptions,
    ) -> Result<Trellis2Stages> {
        // Resolved per-stage sampler params: stage overrides > legacy steps
        // fields > pipeline-config defaults (`resolve_stage_params`).
        let ss_params = resolve_stage_params(
            &self.config.ss_sampler,
            options.ss_sampler.as_ref(),
            options.ss_steps,
        );
        let shape_params = resolve_stage_params(
            &self.config.shape_sampler,
            options.shape_sampler.as_ref(),
            options.shape_steps,
        );
        let tex_params = resolve_stage_params(
            &self.config.tex_sampler,
            options.tex_sampler.as_ref(),
            options.tex_steps,
        );
        // Fail fast on invalid mesh post-process / texture-export options
        // (no-ops when disabled).
        options.postprocess.validate()?;
        options.texture_export.validate()?;
        match options.pipeline_type {
            Trellis2PipelineType::R512 => self.run_core_single(
                cond,
                neg_cond,
                cond,
                "shape_dit_512",
                Trellis2SlatConfig::shape_512(),
                "tex_dit_512",
                Trellis2SlatConfig::tex_512(),
                ss_noise,
                coords_override,
                shape_noise,
                shape_latent_override,
                tex_latent_override,
                device,
                options,
                &ss_params,
                &shape_params,
                &tex_params,
            ),
            Trellis2PipelineType::R1024 => self.run_core_single(
                cond,
                neg_cond,
                cond_hr.unwrap_or(cond),
                "shape_dit_1024",
                Trellis2SlatConfig::shape_1024(),
                "tex_dit_1024",
                Trellis2SlatConfig::tex_1024(),
                ss_noise,
                coords_override,
                shape_noise,
                shape_latent_override,
                tex_latent_override,
                device,
                options,
                &ss_params,
                &shape_params,
                &tex_params,
            ),
            Trellis2PipelineType::Cascade1024 => self.run_core_cascade(
                cond,
                neg_cond,
                cond_hr.unwrap_or(cond),
                TRELLIS2_CASCADE_HR_RESOLUTION,
                ss_noise,
                coords_override,
                shape_noise,
                shape_latent_override,
                tex_latent_override,
                device,
                options,
                &ss_params,
                &shape_params,
                &tex_params,
            ),
            Trellis2PipelineType::Cascade1536 => self.run_core_cascade(
                cond,
                neg_cond,
                cond_hr.unwrap_or(cond),
                TRELLIS2_CASCADE_1536_TARGET_RESOLUTION,
                ss_noise,
                coords_override,
                shape_noise,
                shape_latent_override,
                tex_latent_override,
                device,
                options,
                &ss_params,
                &shape_params,
                &tex_params,
            ),
        }
    }

    /// Batched multi-sample core (`num_samples > 1`, the upstream
    /// `run(num_samples)` draw structure): ONE joint sparse-structure noise
    /// draw `[N, 8, R, R, R]` (`trellis2_image_to_3d.py:208`), ONE joint
    /// SLAT noise draw per SLAT stage over the concatenated per-sample coords
    /// (`torch.randn(coords.shape[0], in_channels)`, `:253`), one shared
    /// conditioning pass, the flow passes batched where the candle model
    /// supports it (the dense SS DiT), and per-sample SLAT sampling /
    /// VAE decode / post-process / export. `num_samples = 1` never reaches
    /// this function: the `run_many_*` entry points keep the unbatched
    /// single-pass path so its noise consumption and output stay
    /// bit-identical.
    #[allow(clippy::too_many_arguments, unused_assignments)]
    fn run_batched_core(
        &self,
        cond: &Tensor,
        neg_cond: &Tensor,
        cond_hr: Option<&Tensor>,
        ss_noise: Option<&Tensor>,
        shape_init: Option<&Tensor>,
        tex_init: Option<&Tensor>,
        device: &Device,
        options: &Trellis2RunOptions,
    ) -> Result<Vec<Trellis2Stages>> {
        let count = options.num_samples;
        debug_assert!(count > 1, "run_batched_core requires num_samples > 1");
        let (ss_resolution, decode_resolution) = self.resolutions(options)?;
        let ss_params = resolve_stage_params(
            &self.config.ss_sampler,
            options.ss_sampler.as_ref(),
            options.ss_steps,
        );
        let shape_params = resolve_stage_params(
            &self.config.shape_sampler,
            options.shape_sampler.as_ref(),
            options.shape_steps,
        );
        let tex_params = resolve_stage_params(
            &self.config.tex_sampler,
            options.tex_sampler.as_ref(),
            options.tex_steps,
        );
        options.postprocess.validate()?;
        options.texture_export.validate()?;
        let cond_shape = cond_hr.unwrap_or(cond);
        let neg_cond_shape = Tensor::zeros_like(cond_shape).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 negative cond allocation failed: {source}"),
            }
        })?;
        let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
        let mut stage_start = std::time::Instant::now();
        macro_rules! stage_done {
            ($name:expr) => {
                if stage_time {
                    eprintln!(
                        "[stage] trellis2 {}: {:.2}s",
                        $name,
                        stage_start.elapsed().as_secs_f64()
                    );
                    stage_start = std::time::Instant::now();
                }
            };
        }
        if stage_time {
            eprintln!(
                "[stage] trellis2 batched num_samples: {count} (one conditioning pass, one joint \
                 noise draw per stage)"
            );
        }
        let (z_s, coords_by_sample) = self.sample_sparse_structure_many(
            cond,
            neg_cond,
            ss_noise,
            count,
            options.seed,
            &ss_params,
            device,
            ss_resolution,
        )?;
        for coords in &coords_by_sample {
            if coords.is_empty() {
                return Err(Lux3dError::InvalidInput(
                    "Trellis2 sparse structure decoded to zero active voxels for a sample",
                ));
            }
        }
        stage_done!("ss");
        let mut stages = Vec::with_capacity(count);
        match options.pipeline_type {
            Trellis2PipelineType::R512 | Trellis2PipelineType::R1024 => {
                let single = matches!(options.pipeline_type, Trellis2PipelineType::R512);
                let (shape_stage, shape_config, tex_stage, tex_config) = if single {
                    (
                        "shape_dit_512",
                        Trellis2SlatConfig::shape_512(),
                        "tex_dit_512",
                        Trellis2SlatConfig::tex_512(),
                    )
                } else {
                    (
                        "shape_dit_1024",
                        Trellis2SlatConfig::shape_1024(),
                        "tex_dit_1024",
                        Trellis2SlatConfig::tex_1024(),
                    )
                };
                let total_rows: usize = coords_by_sample.iter().map(Vec::len).sum();
                // ONE joint shape SLAT noise draw over every sample's active
                // coords (the upstream draw shape has the concatenated
                // coords in dim 0; `trellis2_image_to_3d.py:253`).
                let shape_joint = match shape_init {
                    Some(_) => None,
                    None => Some(draw_slat_noise(
                        total_rows,
                        options.seed,
                        TRELLIS2_SHAPE_NOISE_SALT,
                        device,
                    )?),
                };
                // ONE joint texture SLAT noise draw (same rows: the texture
                // stage runs on the shape coords).
                let tex_joint = if options.with_texture && tex_init.is_none() {
                    Some(draw_slat_noise(
                        total_rows,
                        options.seed,
                        TRELLIS2_TEX_NOISE_SALT,
                        device,
                    )?)
                } else {
                    None
                };
                let mut shape_latents = Vec::with_capacity(count);
                let mut row_offset = 0_usize;
                for coords in &coords_by_sample {
                    let shape_latent = match shape_init {
                        Some(latent) => {
                            let rows = latent.dim(0).map_err(|source| {
                                Lux3dError::CanonicalWeightsValidation {
                                    message: format!("Trellis2 shape latent has no rows: {source}"),
                                }
                            })?;
                            if rows != coords.len() {
                                return Err(Lux3dError::ModelAssetResolution {
                                    message: format!(
                                        "shape latent row count {rows} does not match the active \
                                         voxel count {}",
                                        coords.len()
                                    ),
                                });
                            }
                            latent.clone()
                        }
                        None => {
                            let joint = shape_joint.as_ref().expect("joint draw present");
                            let slice = noise_rows(joint, row_offset, coords.len())?;
                            self.sample_shape_slat_stage(
                                shape_stage,
                                shape_config.clone(),
                                TRELLIS2_SHAPE_NOISE_SALT,
                                cond_shape,
                                &neg_cond_shape,
                                coords,
                                Some(&slice),
                                options.seed,
                                &shape_params,
                                device,
                            )?
                        }
                    };
                    row_offset += coords.len();
                    shape_latents.push(shape_latent);
                }
                stage_done!("shape spatial");
                let mut tex_latents = Vec::with_capacity(count);
                let mut tex_offset = 0_usize;
                for (index, coords) in coords_by_sample.iter().enumerate() {
                    let tex_latent = if !options.with_texture {
                        None
                    } else {
                        match tex_init {
                            Some(latent) => {
                                let rows = latent.dim(0).map_err(|source| {
                                    Lux3dError::CanonicalWeightsValidation {
                                        message: format!(
                                            "Trellis2 texture latent has no rows: {source}"
                                        ),
                                    }
                                })?;
                                if rows != coords.len() {
                                    return Err(Lux3dError::ModelAssetResolution {
                                        message: format!(
                                            "texture latent row count {rows} does not match the \
                                             shape latent coord count {}",
                                            coords.len()
                                        ),
                                    });
                                }
                                Some(latent.clone())
                            }
                            None => {
                                let joint = tex_joint.as_ref().expect("joint draw present");
                                let slice = noise_rows(joint, tex_offset, coords.len())?;
                                Some(self.sample_tex_slat_stage(
                                    tex_stage,
                                    tex_config.clone(),
                                    cond_shape,
                                    &shape_latents[index],
                                    coords,
                                    Some(&slice),
                                    options.seed,
                                    &tex_params,
                                    device,
                                )?)
                            }
                        }
                    };
                    tex_offset += coords.len();
                    tex_latents.push(tex_latent);
                }
                stage_done!("texture spatial");
                for (index, coords) in coords_by_sample.iter().enumerate() {
                    let shape_latent = &shape_latents[index];
                    let (decoded, mesh) =
                        self.decode_shape_vae(coords, shape_latent, device, decode_resolution)?;
                    let mut mesh = self.assemble_mesh(
                        decode_resolution,
                        mesh,
                        &decoded.coords,
                        &decoded.feats,
                    )?;
                    let snap_surface = Self::capture_bake_snap_surface(&mesh, options)?;
                    mesh = self.postprocess_decoded_mesh(mesh, &options.postprocess)?;
                    let tex_attrs = match &tex_latents[index] {
                        Some(tex_latent) => {
                            let (tex_coords, attrs) = self.decode_tex_attrs(
                                coords,
                                tex_latent,
                                &decoded.subdiv_logits,
                                device,
                            )?;
                            assert_eq!(
                                tex_coords, decoded.coords,
                                "guided texture decode coords must equal the shape decode coords"
                            );
                            self.attach_texture(
                                &mut mesh,
                                &decoded.coords,
                                &attrs,
                                decode_resolution,
                            )?;
                            Some(attrs)
                        }
                        None => None,
                    };
                    let textured = self.build_textured_mesh(
                        &mesh,
                        &options.texture_export,
                        snap_surface.as_ref(),
                    )?;
                    stages.push(Trellis2Stages {
                        cond: cond.clone(),
                        neg_cond: neg_cond.clone(),
                        z_s: Some(noise_rows(&z_s, index, 1)?),
                        ss_coords: coords.clone(),
                        shape_coords: coords.clone(),
                        shape_latent: shape_latent.clone(),
                        decoded_feats: decoded.feats,
                        mesh: Some(mesh),
                        lr_shape_latent: None,
                        hr_upsampled_coords: None,
                        quant_coords: None,
                        hr_resolution: None,
                        subdiv_logits: decoded.subdiv_logits,
                        tex_latent: tex_latents[index].clone(),
                        tex_attrs,
                        textured,
                    });
                }
                stage_done!("decode+postprocess+export");
            }
            Trellis2PipelineType::Cascade1024 | Trellis2PipelineType::Cascade1536 => {
                let lr_noise = match shape_init {
                    Some(_) => None,
                    None => Some(draw_slat_noise(
                        coords_by_sample.iter().map(Vec::len).sum(),
                        options.seed,
                        TRELLIS2_SHAPE_NOISE_SALT,
                        device,
                    )?),
                };
                // LR shape SLAT per sample (the candle sparse DiT is single
                // sample) + deterministic upsample. The LR noise rows come
                // from the ONE joint draw. The token budget runs ONCE over
                // the whole batch below (the official `unique`/budget loop
                // sees the concatenated batch coords, so every sample shares
                // one `hr_resolution`, `trellis2_image_to_3d.py:327-339`).
                let mut lr_parts: Vec<(Vec<[i32; 3]>, Tensor, Vec<[i32; 3]>)> =
                    Vec::with_capacity(count);
                let mut row_offset = 0_usize;
                for coords in &coords_by_sample {
                    let lr_slat = match shape_init {
                        Some(latent) => {
                            let rows = latent.dim(0).map_err(|source| {
                                Lux3dError::CanonicalWeightsValidation {
                                    message: format!("Trellis2 shape latent has no rows: {source}"),
                                }
                            })?;
                            if rows != coords.len() {
                                return Err(Lux3dError::ModelAssetResolution {
                                    message: format!(
                                        "LR shape latent row count {rows} does not match the \
                                         active voxel count {}",
                                        coords.len()
                                    ),
                                });
                            }
                            latent.clone()
                        }
                        None => {
                            let joint = lr_noise.as_ref().expect("joint draw present");
                            let slice = noise_rows(joint, row_offset, coords.len())?;
                            self.sample_shape_slat_stage(
                                "shape_dit_512",
                                Trellis2SlatConfig::shape_512(),
                                TRELLIS2_SHAPE_NOISE_SALT,
                                cond,
                                neg_cond,
                                coords,
                                Some(&slice),
                                options.seed,
                                &shape_params,
                                device,
                            )?
                        }
                    };
                    row_offset += coords.len();
                    let hr_upsampled_coords = self.upsample_shape_slat(coords, &lr_slat, device)?;
                    lr_parts.push((coords.clone(), lr_slat, hr_upsampled_coords));
                }
                let hr_upsampled_by_sample: Vec<Vec<[i32; 3]>> = lr_parts
                    .iter()
                    .map(|(_, _, hr_upsampled)| hr_upsampled.clone())
                    .collect();
                let (quant_by_sample, hr_resolution) = quantize_hr_coords_batched(
                    &hr_upsampled_by_sample,
                    TRELLIS2_CASCADE_LR_RESOLUTION,
                    decode_resolution,
                    options.max_num_tokens,
                );
                let mut parts: Vec<Trellis2BatchedCascadePart> = Vec::with_capacity(count);
                for ((ss_coords, lr_slat, hr_upsampled_coords), quant_coords) in
                    lr_parts.into_iter().zip(quant_by_sample)
                {
                    if quant_coords.is_empty() {
                        return Err(Lux3dError::InvalidInput(
                            "Trellis2 cascade quantized the upsampled coords to zero HR tokens",
                        ));
                    }
                    parts.push(Trellis2BatchedCascadePart {
                        ss_coords,
                        lr_slat,
                        hr_upsampled_coords,
                        quant_coords,
                    });
                }
                stage_done!("shape spatial (LR + upsample)");
                // ONE joint HR noise draw over every sample's quantized
                // coords (`torch.randn(coords.shape[0], in_channels)` after
                // the unique/budget loop, `trellis2_image_to_3d.py:343`).
                let total_hr_rows: usize = parts.iter().map(|part| part.quant_coords.len()).sum();
                let hr_joint = draw_slat_noise(
                    total_hr_rows,
                    options.seed,
                    TRELLIS2_HR_SHAPE_NOISE_SALT,
                    device,
                )?;
                let mut hr_latents = Vec::with_capacity(count);
                let mut hr_offset = 0_usize;
                // ONE shared latent-grid bound for every sample (the batch
                // budget returned a single `hr_resolution`).
                let hr_grid = (hr_resolution / 16) as usize;
                for part in &parts {
                    let slice = noise_rows(&hr_joint, hr_offset, part.quant_coords.len())?;
                    hr_offset += part.quant_coords.len();
                    hr_latents.push(self.sample_shape_slat_hr(
                        cond_shape,
                        &part.quant_coords,
                        Some(&slice),
                        options.seed,
                        &shape_params,
                        device,
                        hr_grid,
                    )?);
                }
                stage_done!("shape spatial (HR)");
                // ONE joint texture noise draw over the HR shape coords,
                // drawn after the shape SLAT stages like the reference
                // (`sample_tex_slat`, `trellis2_image_to_3d.py:412`).
                let tex_joint = if options.with_texture && tex_init.is_none() {
                    Some(draw_slat_noise(
                        total_hr_rows,
                        options.seed,
                        TRELLIS2_TEX_NOISE_SALT,
                        device,
                    )?)
                } else {
                    None
                };
                let mut tex_latents = Vec::with_capacity(count);
                let mut tex_offset = 0_usize;
                // One elastic texture config for the whole batch: the shared
                // cascade resolution decides the latent grid bound.
                let tex_config = if hr_grid > 64 {
                    Trellis2SlatConfig::tex_1536()
                } else {
                    Trellis2SlatConfig::tex_1024()
                };
                for (index, part) in parts.iter().enumerate() {
                    let tex_latent = if !options.with_texture {
                        None
                    } else {
                        match tex_init {
                            Some(latent) => {
                                let rows = latent.dim(0).map_err(|source| {
                                    Lux3dError::CanonicalWeightsValidation {
                                        message: format!(
                                            "Trellis2 texture latent has no rows: {source}"
                                        ),
                                    }
                                })?;
                                if rows != part.quant_coords.len() {
                                    return Err(Lux3dError::ModelAssetResolution {
                                        message: format!(
                                            "texture latent row count {rows} does not match the \
                                             HR coord count {}",
                                            part.quant_coords.len()
                                        ),
                                    });
                                }
                                Some(latent.clone())
                            }
                            None => {
                                let joint = tex_joint.as_ref().expect("joint draw present");
                                let slice = noise_rows(joint, tex_offset, part.quant_coords.len())?;
                                Some(self.sample_tex_slat_stage(
                                    "tex_dit_1024",
                                    tex_config.clone(),
                                    cond_shape,
                                    &hr_latents[index],
                                    &part.quant_coords,
                                    Some(&slice),
                                    options.seed,
                                    &tex_params,
                                    device,
                                )?)
                            }
                        }
                    };
                    tex_offset += part.quant_coords.len();
                    tex_latents.push(tex_latent);
                }
                stage_done!("texture spatial");
                for (index, part) in parts.iter().enumerate() {
                    let (decoded, mesh) = self.decode_shape_vae(
                        &part.quant_coords,
                        &hr_latents[index],
                        device,
                        hr_resolution,
                    )?;
                    let mut mesh =
                        self.assemble_mesh(hr_resolution, mesh, &decoded.coords, &decoded.feats)?;
                    let snap_surface = Self::capture_bake_snap_surface(&mesh, options)?;
                    mesh = self.postprocess_decoded_mesh(mesh, &options.postprocess)?;
                    let tex_attrs = match &tex_latents[index] {
                        Some(tex_latent) => {
                            let (tex_coords, attrs) = self.decode_tex_attrs(
                                &part.quant_coords,
                                tex_latent,
                                &decoded.subdiv_logits,
                                device,
                            )?;
                            assert_eq!(
                                tex_coords, decoded.coords,
                                "guided texture decode coords must equal the shape decode coords"
                            );
                            self.attach_texture(&mut mesh, &decoded.coords, &attrs, hr_resolution)?;
                            Some(attrs)
                        }
                        None => None,
                    };
                    let textured = self.build_textured_mesh(
                        &mesh,
                        &options.texture_export,
                        snap_surface.as_ref(),
                    )?;
                    stages.push(Trellis2Stages {
                        cond: cond.clone(),
                        neg_cond: neg_cond.clone(),
                        z_s: Some(noise_rows(&z_s, index, 1)?),
                        ss_coords: part.ss_coords.clone(),
                        shape_coords: part.quant_coords.clone(),
                        shape_latent: hr_latents[index].clone(),
                        decoded_feats: decoded.feats,
                        mesh: Some(mesh),
                        lr_shape_latent: Some(part.lr_slat.clone()),
                        hr_upsampled_coords: Some(part.hr_upsampled_coords.clone()),
                        quant_coords: Some(part.quant_coords.clone()),
                        hr_resolution: Some(hr_resolution),
                        subdiv_logits: decoded.subdiv_logits,
                        tex_latent: tex_latents[index].clone(),
                        tex_attrs,
                        textured,
                    });
                }
                stage_done!("decode+postprocess+export");
            }
        }
        Ok(stages)
    }

    /// Single-resolution core for `512` and `1024` (shared; the reference
    /// duplicates the two branches in `run(...)`): SS stage (unless
    /// `coords_override`) on `cond_ss`, shape SLAT stage (unless
    /// `shape_latent_override`) and texture SLAT stage (unless
    /// `tex_latent_override` or `with_texture = false`) on `cond_shape`,
    /// VAE decode + mesh at the variant's resolution. `neg_cond` belongs to
    /// `cond_ss`; the shape/tex negative cond is `zeros_like(cond_shape)`
    /// exactly like the reference `get_cond`.
    #[allow(clippy::too_many_arguments, unused_assignments)]
    fn run_core_single(
        &self,
        cond_ss: &Tensor,
        neg_cond: &Tensor,
        cond_shape: &Tensor,
        shape_stage: &str,
        shape_config: Trellis2SlatConfig,
        tex_stage: &str,
        tex_config: Trellis2SlatConfig,
        ss_noise: Option<&Tensor>,
        coords_override: Option<Vec<[i32; 3]>>,
        shape_noise: Option<&Tensor>,
        shape_latent_override: Option<&Tensor>,
        tex_latent_override: Option<&Tensor>,
        device: &Device,
        options: &Trellis2RunOptions,
        ss_params: &Trellis2SamplerParams,
        shape_params: &Trellis2SamplerParams,
        tex_params: &Trellis2SamplerParams,
    ) -> Result<Trellis2Stages> {
        let (ss_resolution, shape_resolution) = self.resolutions(options)?;
        let neg_cond_shape = Tensor::zeros_like(cond_shape).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 negative cond allocation failed: {source}"),
            }
        })?;
        let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
        let mut stage_start = std::time::Instant::now();
        macro_rules! stage_done {
            ($name:expr) => {
                if stage_time {
                    eprintln!(
                        "[stage] trellis2 {}: {:.2}s",
                        $name,
                        stage_start.elapsed().as_secs_f64()
                    );
                    stage_start = std::time::Instant::now();
                }
            };
        }
        let (z_s, ss_coords) = match coords_override {
            Some(coords) => (None, coords),
            None => {
                let (z_s, coords) = self.sample_sparse_structure(
                    cond_ss,
                    neg_cond,
                    ss_noise,
                    options.seed,
                    ss_params,
                    device,
                    ss_resolution,
                )?;
                (Some(z_s), coords)
            }
        };
        stage_done!("ss");
        if ss_coords.is_empty() {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 sparse structure decoded to zero active voxels",
            ));
        }
        let shape_latent = match shape_latent_override {
            Some(latent) => {
                let rows =
                    latent
                        .dim(0)
                        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                            message: format!("Trellis2 shape latent has no rows: {source}"),
                        })?;
                if rows != ss_coords.len() {
                    return Err(Lux3dError::ModelAssetResolution {
                        message: format!(
                            "shape latent row count {rows} does not match the active voxel count {}",
                            ss_coords.len()
                        ),
                    });
                }
                latent.clone()
            }
            None => self.sample_shape_slat_stage(
                shape_stage,
                shape_config,
                TRELLIS2_SHAPE_NOISE_SALT,
                cond_shape,
                &neg_cond_shape,
                &ss_coords,
                shape_noise,
                options.seed,
                shape_params,
                device,
            )?,
        };
        stage_done!("shape spatial");
        // Texture SLAT on the shape latent coords (reference order: after
        // the shape stage, before the decodes).
        let tex_latent = self.texture_latent_single(
            cond_shape,
            &shape_latent,
            &ss_coords,
            tex_stage,
            tex_config,
            tex_latent_override,
            device,
            options,
            tex_params,
        )?;
        stage_done!("texture spatial");
        let (decoded, mesh) =
            self.decode_shape_vae(&ss_coords, &shape_latent, device, shape_resolution)?;
        stage_done!("shape vae");
        let mut mesh =
            self.assemble_mesh(shape_resolution, mesh, &decoded.coords, &decoded.feats)?;
        // Bake snap-back source: the decoded mesh BEFORE the geometry
        // post-process (upstream builds its BVH before remesh/simplify,
        // `postprocess.py:122`, and projects texels onto it,
        // `postprocess.py:254-256`).
        let snap_surface = Self::capture_bake_snap_surface(&mesh, options)?;
        // Geometry post-process (upstream `to_glb` cleaning) BEFORE the
        // texture stage: the re-topologized vertices are the ones the
        // per-vertex PBR color sampling sees.
        mesh = self.postprocess_decoded_mesh(mesh, &options.postprocess)?;
        stage_done!("postprocess");
        let tex_attrs = match &tex_latent {
            Some(tex_latent) => {
                let (tex_coords, attrs) =
                    self.decode_tex_attrs(&ss_coords, tex_latent, &decoded.subdiv_logits, device)?;
                assert_eq!(
                    tex_coords, decoded.coords,
                    "guided texture decode coords must equal the shape decode coords"
                );
                self.attach_texture(&mut mesh, &decoded.coords, &attrs, shape_resolution)?;
                stage_done!("texture vae");
                Some(attrs)
            }
            None => None,
        };
        // Texture export (upstream `to_glb` UV unwrap + bake). Skips cleanly
        // for geometry-only runs (see `build_textured_mesh`); the snap-back
        // projects onto the pre-postprocess surface captured above.
        let textured =
            self.build_textured_mesh(&mesh, &options.texture_export, snap_surface.as_ref())?;
        stage_done!("texture export");
        Ok(Trellis2Stages {
            cond: cond_ss.clone(),
            neg_cond: neg_cond.clone(),
            z_s,
            ss_coords: ss_coords.clone(),
            shape_coords: ss_coords,
            shape_latent,
            decoded_feats: decoded.feats,
            mesh: Some(mesh),
            lr_shape_latent: None,
            hr_upsampled_coords: None,
            quant_coords: None,
            hr_resolution: None,
            subdiv_logits: decoded.subdiv_logits,
            tex_latent,
            tex_attrs,
            textured,
        })
    }

    /// Shared texture-latent step for the single-resolution core: `None`
    /// when `with_texture` is off; the injected de-normalized latent when
    /// provided (validated against the coord count); sampled otherwise.
    #[allow(clippy::too_many_arguments)]
    fn texture_latent_single(
        &self,
        cond: &Tensor,
        shape_latent: &Tensor,
        coords: &[[i32; 3]],
        tex_stage: &str,
        tex_config: Trellis2SlatConfig,
        tex_latent_override: Option<&Tensor>,
        device: &Device,
        options: &Trellis2RunOptions,
        tex_params: &Trellis2SamplerParams,
    ) -> Result<Option<Tensor>> {
        if !options.with_texture {
            return Ok(None);
        }
        match tex_latent_override {
            Some(latent) => {
                let rows =
                    latent
                        .dim(0)
                        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                            message: format!("Trellis2 texture latent has no rows: {source}"),
                        })?;
                if rows != coords.len() {
                    return Err(Lux3dError::ModelAssetResolution {
                        message: format!(
                            "texture latent row count {rows} does not match the shape latent \
                             coord count {}",
                            coords.len()
                        ),
                    });
                }
                Ok(Some(latent.clone()))
            }
            None => Ok(Some(self.sample_tex_slat_stage(
                tex_stage,
                tex_config,
                cond,
                shape_latent,
                coords,
                None,
                options.seed,
                tex_params,
                device,
            )?)),
        }
    }

    /// Cascade core shared by `1024_cascade` and `1536_cascade` (reference
    /// `run(...)`, `sample_shape_slat_cascade(resolution = target)`): SS
    /// stage at 32^3 on `cond` (the reference conditions the SS stage on
    /// `cond_512`), LR shape SLAT (unless `shape_latent_override` — the
    /// injected de-normalized LR slat), shape VAE `upsample(slat, 4)` -> HR
    /// quantization + token budget starting at `target_resolution` (the
    /// loop's stop resolution stays 1024), HR shape SLAT on `cond_hr` with
    /// `shape_dit_1024`, texture SLAT on `cond_hr` with `tex_dit_1024` on
    /// the HR quantized coords, decode at the returned `hr_resolution` +
    /// mesh. Model residency stays sequential in the default mode: LR DiT ->
    /// drop, shape VAE -> drop, HR DiT -> drop, tex DiT -> drop, shape VAE
    /// (decode) -> drop, tex VAE. With `low_vram = false` those stages are
    /// pinned on the device for the whole run instead.
    #[allow(clippy::too_many_arguments)]
    fn run_core_cascade(
        &self,
        cond: &Tensor,
        neg_cond: &Tensor,
        cond_hr: &Tensor,
        target_resolution: u32,
        ss_noise: Option<&Tensor>,
        coords_override: Option<Vec<[i32; 3]>>,
        lr_noise: Option<&Tensor>,
        lr_slat_override: Option<&Tensor>,
        tex_latent_override: Option<&Tensor>,
        device: &Device,
        options: &Trellis2RunOptions,
        ss_params: &Trellis2SamplerParams,
        shape_params: &Trellis2SamplerParams,
        tex_params: &Trellis2SamplerParams,
    ) -> Result<Trellis2Stages> {
        let (ss_resolution, _) = self.resolutions(options)?;
        let (z_s, ss_coords) = match coords_override {
            Some(coords) => (None, coords),
            None => {
                let (z_s, coords) = self.sample_sparse_structure(
                    cond,
                    neg_cond,
                    ss_noise,
                    options.seed,
                    ss_params,
                    device,
                    ss_resolution,
                )?;
                (Some(z_s), coords)
            }
        };
        if ss_coords.is_empty() {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 sparse structure decoded to zero active voxels",
            ));
        }
        // LR shape SLAT: injected de-normalized latent or sampled at 32^3.
        let lr_slat = match lr_slat_override {
            Some(latent) => {
                let rows =
                    latent
                        .dim(0)
                        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                            message: format!("Trellis2 shape latent has no rows: {source}"),
                        })?;
                if rows != ss_coords.len() {
                    return Err(Lux3dError::ModelAssetResolution {
                        message: format!(
                            "LR shape latent row count {rows} does not match the active voxel count {}",
                            ss_coords.len()
                        ),
                    });
                }
                latent.clone()
            }
            None => self.sample_shape_slat(
                cond,
                neg_cond,
                &ss_coords,
                lr_noise,
                options.seed,
                shape_params,
                device,
            )?,
        };
        let (hr_upsampled_coords, quant_coords, hr_resolution) = self
            .upsample_and_quantize_shape_slat(
                &ss_coords,
                &lr_slat,
                options.max_num_tokens,
                target_resolution,
                device,
            )?;
        if quant_coords.is_empty() {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 cascade quantized the upsampled coords to zero HR tokens",
            ));
        }
        let hr_grid = (hr_resolution / 16) as usize;
        let hr_slat = self.sample_shape_slat_hr(
            cond_hr,
            &quant_coords,
            None,
            options.seed,
            shape_params,
            device,
            hr_grid,
        )?;
        // Texture SLAT on the HR quantized coords, conditioned on `cond_hr`
        // (reference texture stage uses the 1024 conditioning + DiT).
        let tex_latent = self.texture_latent_cascade(
            cond_hr,
            &hr_slat,
            &quant_coords,
            tex_latent_override,
            device,
            options,
            tex_params,
            hr_grid,
        )?;
        let (decoded, mesh) =
            self.decode_shape_vae(&quant_coords, &hr_slat, device, hr_resolution)?;
        let mut mesh = self.assemble_mesh(hr_resolution, mesh, &decoded.coords, &decoded.feats)?;
        // Bake snap-back source: the decoded mesh BEFORE the geometry
        // post-process (upstream builds its BVH before remesh/simplify,
        // `postprocess.py:122`, and projects texels onto it,
        // `postprocess.py:254-256`).
        let snap_surface = Self::capture_bake_snap_surface(&mesh, options)?;
        // Geometry post-process (upstream `to_glb` cleaning) BEFORE the
        // texture stage: the re-topologized vertices are the ones the
        // per-vertex PBR color sampling sees.
        mesh = self.postprocess_decoded_mesh(mesh, &options.postprocess)?;
        let tex_attrs = match &tex_latent {
            Some(tex_latent) => {
                let (tex_coords, attrs) = self.decode_tex_attrs(
                    &quant_coords,
                    tex_latent,
                    &decoded.subdiv_logits,
                    device,
                )?;
                assert_eq!(
                    tex_coords, decoded.coords,
                    "guided texture decode coords must equal the shape decode coords"
                );
                self.attach_texture(&mut mesh, &decoded.coords, &attrs, hr_resolution)?;
                Some(attrs)
            }
            None => None,
        };
        // Texture export (upstream `to_glb` UV unwrap + bake). Skips cleanly
        // for geometry-only runs (see `build_textured_mesh`); the snap-back
        // projects onto the pre-postprocess surface captured above.
        let textured =
            self.build_textured_mesh(&mesh, &options.texture_export, snap_surface.as_ref())?;
        Ok(Trellis2Stages {
            cond: cond.clone(),
            neg_cond: neg_cond.clone(),
            z_s,
            ss_coords,
            shape_coords: quant_coords.clone(),
            shape_latent: hr_slat,
            decoded_feats: decoded.feats,
            mesh: Some(mesh),
            lr_shape_latent: Some(lr_slat),
            hr_upsampled_coords: Some(hr_upsampled_coords),
            quant_coords: Some(quant_coords),
            hr_resolution: Some(hr_resolution),
            subdiv_logits: decoded.subdiv_logits,
            tex_latent,
            tex_attrs,
            textured,
        })
    }

    /// Shared texture-latent step for the cascade core: identical guards to
    /// [`Self::texture_latent_single`] with the 1024 texture stage and the
    /// elastic validation bound for the HR grid.
    #[allow(clippy::too_many_arguments)]
    fn texture_latent_cascade(
        &self,
        cond_hr: &Tensor,
        hr_slat: &Tensor,
        quant_coords: &[[i32; 3]],
        tex_latent_override: Option<&Tensor>,
        device: &Device,
        options: &Trellis2RunOptions,
        tex_params: &Trellis2SamplerParams,
        hr_grid_resolution: usize,
    ) -> Result<Option<Tensor>> {
        if !options.with_texture {
            return Ok(None);
        }
        match tex_latent_override {
            Some(latent) => {
                let rows =
                    latent
                        .dim(0)
                        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                            message: format!("Trellis2 texture latent has no rows: {source}"),
                        })?;
                if rows != quant_coords.len() {
                    return Err(Lux3dError::ModelAssetResolution {
                        message: format!(
                            "texture latent row count {rows} does not match the HR coord \
                             count {}",
                            quant_coords.len()
                        ),
                    });
                }
                Ok(Some(latent.clone()))
            }
            None => Ok(Some(self.sample_tex_slat_hr(
                cond_hr,
                hr_slat,
                quant_coords,
                None,
                options.seed,
                tex_params,
                device,
                hr_grid_resolution,
            )?)),
        }
    }

    /// Guard the single-output entry points against the multi-sample mode:
    /// the upstream `run(num_samples)` batch is implemented by
    /// [`Self::run_many_from_path`] / [`Self::run_many_with_latents`] (joint
    /// per-stage noise draws + per-sample decode), so `num_samples != 1`
    /// must not silently run once.
    fn check_single_sample(options: &Trellis2RunOptions) -> Result<()> {
        if options.num_samples != 1 {
            return Err(Lux3dError::ContractValidation {
                message: format!(
                    "Trellis2 num_samples = {} requires the run_many_* entry points \
                     (joint per-stage noise draws); the single-output runs accept \
                     num_samples = 1 only",
                    options.num_samples
                ),
            });
        }
        Ok(())
    }

    /// Does the variant need a second conditioning pass at 1024 (the SS
    /// stage is always conditioned on `cond_512`)?
    fn needs_cond_1024(pipeline_type: Trellis2PipelineType) -> bool {
        !matches!(pipeline_type, Trellis2PipelineType::R512)
    }

    /// Run the pipeline on an image path: preprocess -> DINOv3 -> SS -> SLAT
    /// -> VAE decode -> mesh.
    pub fn run_from_path(
        &self,
        path: &Path,
        device: &Device,
        options: &Trellis2RunOptions,
    ) -> Result<crate::contracts::Trellis2Mesh> {
        Self::check_single_sample(options)?;
        let stages = self.run_from_path_with_stages(path, device, options)?;
        let mesh = stages.mesh.expect("run_from_path always produces a mesh");
        mesh.to_contract()
    }

    /// [`Self::run_from_path`] returning the full per-stage outputs. The
    /// 1024-conditioned branches crop the source once (after at most one
    /// RMBG-2.0 run) and resize that crop to both 512 and 1024, matching the
    /// reference `preprocess_image` -> `get_cond(512)`/`get_cond(1024)` flow.
    pub fn run_from_path_with_stages(
        &self,
        path: &Path,
        device: &Device,
        options: &Trellis2RunOptions,
    ) -> Result<Trellis2Stages> {
        Self::check_single_sample(options)?;
        // Residency window covers preprocessing + conditioning + all stages
        // (`low_vram = false` keeps the DINOv3 conditioning model resident
        // across the 512/1024 cond passes too).
        let _resident = self.enter_run_mode(options);
        if !options.preprocess {
            return Err(Lux3dError::InvalidInput(
                "run_from_path requires options.preprocess = true; use run_prepared for \
                 pre-prepared images",
            ));
        }
        // Reference `run()` calls `preprocess_image` once and then resizes
        // that same crop for `cond_512` and `cond_1024`; never run RMBG twice.
        let crop = self
            .preprocess
            .prepare_crop_from_path(path, self.background_remover())?;
        if Self::needs_cond_1024(options.pipeline_type) {
            let (cond, neg_cond, cond_hr) = self.conditioning_512_and_1024(&crop, device)?;
            return self.run_core(
                &cond,
                &neg_cond,
                Some(&cond_hr),
                None,
                None,
                None,
                None,
                None,
                device,
                options,
            );
        }
        let prepared = self.preprocess.resize_crop(&crop)?;
        self.run_prepared(&prepared, device, options)
    }

    /// `cond_512` + `cond_1024` from one source crop (the reference
    /// `preprocess_image` -> `get_cond([image], 512)` +
    /// `get_cond([image], 1024)`): the crop is resized to each DINOv3 input
    /// resolution exactly once.
    fn conditioning_512_and_1024(
        &self,
        crop: &crate::preprocess::Trellis2PreparedCrop,
        device: &Device,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let preprocess_hr = self.cascade_preprocess_hr()?;
        let prepared = self.preprocess.resize_crop(crop)?;
        let prepared_hr = preprocess_hr.resize_crop(crop)?;
        let (cond, neg_cond) = self.image_conditioning(&prepared, device)?;
        let pixel_values_hr = preprocess_hr
            .normalized_pixel_values(&prepared_hr, device)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 DINOv3 input normalization failed: {source}"),
            })?;
        let (cond_hr, _neg_hr) = self.dino_conditioning_pixels(&pixel_values_hr, device)?;
        Ok((cond, neg_cond, cond_hr))
    }

    /// Run the pipeline on an already-preprocessed image.
    ///
    /// 1024-conditioned variants note: the prepared image only carries the
    /// 512 crop, so `cond_1024` is built by LANCZOS-resizing the prepared
    /// pixels to 1024 (a documented double-resample approximation —
    /// [`Self::run_from_path_with_stages`] re-prepares from the source and
    /// is exact).
    pub fn run_prepared(
        &self,
        prepared: &crate::contracts::Trellis2PreparedImage,
        device: &Device,
        options: &Trellis2RunOptions,
    ) -> Result<Trellis2Stages> {
        Self::check_single_sample(options)?;
        let _resident = self.enter_run_mode(options);
        let (cond, neg_cond) = self.image_conditioning(prepared, device)?;
        let cond_hr = match Self::needs_cond_1024(options.pipeline_type) {
            true => {
                let (cond_hr, _neg) = self.cascade_cond_1024_from_prepared(prepared, device)?;
                Some(cond_hr)
            }
            false => None,
        };
        self.run_core(
            &cond,
            &neg_cond,
            cond_hr.as_ref(),
            None,
            None,
            None,
            None,
            None,
            device,
            options,
        )
    }

    /// Run with injected stage tensors (frozen test surface):
    ///
    /// * `ss_init`: sparse-structure flow noise `[1, 8, R, R, R]`.
    /// * `shape_init`: de-normalized shape latent `[N, 32]` on the SS coords
    ///   — when provided the LR shape SLAT sampler is skipped. `512` decodes
    ///   it directly; the cascades treat it as the LR slat (upsample ->
    ///   quantize -> HR sample -> decode; the row count must match the
    ///   decoded coords).
    /// * `tex_init`: de-normalized texture latent `[N, 32]` on the shape
    ///   latent coords — when provided the texture SLAT sampler is skipped
    ///   (`512` decodes it on the SS coords; `1024`/cascade paths on the HR
    ///   quantized coords). Requires `with_texture`; must be `None` when
    ///   texture is off.
    ///
    /// Cascade note: the frozen surface carries a single `cond`, so the
    /// injected cond feeds both the LR (`cond_512`) and HR (`cond_1024`)
    /// stages — cross-attention accepts any context length and the injected
    /// parity runs supply the cond they compare against.
    pub fn run_with_latents(
        &self,
        cond: &Tensor,
        ss_init: &Tensor,
        shape_init: Option<&Tensor>,
        tex_init: Option<&Tensor>,
        device: &Device,
        options: &Trellis2RunOptions,
    ) -> Result<crate::contracts::Trellis2Mesh> {
        Self::check_single_sample(options)?;
        let _resident = self.enter_run_mode(options);
        self.run_with_latents_many_inner(
            cond,
            ss_init,
            shape_init,
            tex_init,
            device,
            &options.clone(),
        )
        .map(|mut stages| {
            stages
                .pop()
                .expect("run_with_latents_many_inner yields num_samples stages")
        })
        .and_then(|stages| {
            let mesh = stages
                .mesh
                .expect("run_with_latents always produces a mesh");
            mesh.to_contract()
        })
    }

    /// Multi-sample surface (`num_samples > 1`): the upstream
    /// `run(num_samples)` draw structure — one joint sparse-structure noise
    /// draw `[N, 8, R, R, R]`, one joint SLAT noise draw per stage over all
    /// samples' coords, one shared conditioning pass — with per-sample SLAT
    /// sampling / decode / export. The injected `ss_init` seeds the batch
    /// (a `[1, ...]` tensor is tiled over the samples, the legacy behavior);
    /// `shape_init`/`tex_init` are validated per sample. `num_samples = 1`
    /// keeps the unbatched single pass and is bit-identical to
    /// [`Self::run_with_latents`].
    pub fn run_many_with_latents(
        &self,
        cond: &Tensor,
        ss_init: &Tensor,
        shape_init: Option<&Tensor>,
        tex_init: Option<&Tensor>,
        device: &Device,
        options: &Trellis2RunOptions,
    ) -> Result<Vec<Trellis2Stages>> {
        let _resident = self.enter_run_mode(options);
        if tex_init.is_some() && !options.with_texture {
            return Err(Lux3dError::InvalidInput(
                "tex_init requires options.with_texture = true (texture is disabled)",
            ));
        }
        if !matches!(options.pipeline_type, Trellis2PipelineType::R512) {
            self.resolutions(options)?;
        }
        self.run_with_latents_many_inner(cond, ss_init, shape_init, tex_init, device, options)
    }

    fn run_with_latents_many_inner(
        &self,
        cond: &Tensor,
        ss_init: &Tensor,
        shape_init: Option<&Tensor>,
        tex_init: Option<&Tensor>,
        device: &Device,
        options: &Trellis2RunOptions,
    ) -> Result<Vec<Trellis2Stages>> {
        let count = if options.num_samples == 0 {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 num_samples must be >= 1",
            ));
        } else {
            options.num_samples
        };
        let neg_cond =
            Tensor::zeros_like(cond).map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 negative cond allocation failed: {source}"),
            })?;
        if count == 1 {
            // Keep the single-sample pass byte-for-byte: same call, same
            // noise consumption order.
            return Ok(vec![self.run_core(
                cond,
                &neg_cond,
                None,
                Some(ss_init),
                None,
                None,
                shape_init,
                tex_init,
                device,
                options,
            )?]);
        }
        self.run_batched_core(
            cond,
            &neg_cond,
            None,
            Some(ss_init),
            shape_init,
            tex_init,
            device,
            options,
        )
    }

    /// Multi-sample surface from an image path (`num_samples > 1`): the
    /// upstream `run(num_samples)` draw structure (see
    /// [`Self::run_batched_core`]) over ONE preprocessing + DINOv3
    /// conditioning pass. `num_samples = 1` returns exactly one element and
    /// is bit-identical to [`Self::run_from_path_with_stages`].
    pub fn run_many_from_path(
        &self,
        path: &Path,
        device: &Device,
        options: &Trellis2RunOptions,
    ) -> Result<Vec<Trellis2Stages>> {
        if options.num_samples == 0 {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 num_samples must be >= 1",
            ));
        }
        let _resident = self.enter_run_mode(options);
        if !options.preprocess {
            return Err(Lux3dError::InvalidInput(
                "run_many_from_path requires options.preprocess = true; use \
                 run_many_with_latents for pre-conditioned runs",
            ));
        }
        let crop = self
            .preprocess
            .prepare_crop_from_path(path, self.background_remover())?;
        let (cond, neg_cond, cond_hr) = if Self::needs_cond_1024(options.pipeline_type) {
            let (cond, neg_cond, cond_hr) = self.conditioning_512_and_1024(&crop, device)?;
            (cond, neg_cond, Some(cond_hr))
        } else {
            let prepared = self.preprocess.resize_crop(&crop)?;
            let (cond, neg_cond) = self.image_conditioning(&prepared, device)?;
            (cond, neg_cond, None)
        };
        if options.num_samples == 1 {
            // Keep the single-sample pass byte-for-byte: same call, same
            // noise consumption order.
            return Ok(vec![self.run_core(
                &cond,
                &neg_cond,
                cond_hr.as_ref(),
                None,
                None,
                None,
                None,
                None,
                device,
                options,
            )?]);
        }
        self.run_batched_core(
            &cond,
            &neg_cond,
            cond_hr.as_ref(),
            None,
            None,
            None,
            device,
            options,
        )
    }

    /// Write the mesh buffers with the TRELLIS.2 export stage.
    pub fn export_mesh(
        &self,
        mesh: &crate::geometry::Trellis2MeshBuffers,
        path: &Path,
    ) -> Result<crate::export::MeshFormat> {
        self.export.write_mesh(mesh, path)
    }

    /// Write the mesh plus the PBR voxel sidecar pair
    /// (`<stem>.pbrvox.bin` + `<stem>.pbr.json`); returns the binary
    /// sidecar path. Fails for geometry-only meshes (7-channel shape
    /// attrs) — run with `with_texture` first.
    pub fn export_mesh_with_pbr(
        &self,
        mesh: &crate::geometry::Trellis2MeshBuffers,
        path: &Path,
    ) -> Result<std::path::PathBuf> {
        self.export.write_mesh_with_pbr(mesh, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Trellis2MeshBuffers;

    fn tensor_values(tensor: &Tensor) -> Vec<f32> {
        tensor
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to_vec1")
    }

    /// Weightless pipeline shell for the residency-window tests (same shape
    /// as the real loaders, no stage files).
    fn residency_test_pipeline() -> Trellis2Pipeline {
        let config = Trellis2PipelineConfig::default();
        Trellis2Pipeline {
            weights: None,
            preprocess: Trellis2Preprocess::new(config.shape_resolution).expect("preprocess"),
            geometry: Trellis2GeometryStage,
            export: Trellis2ExportStage,
            config,
            rmbg: None,
            raw_dir: None,
            resident: Mutex::new(ResidentStages::default()),
        }
    }

    fn counting_load<'a>(
        loads: &'a std::sync::atomic::AtomicUsize,
    ) -> impl Fn() -> Result<String> + 'a {
        move || {
            loads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok("stage".to_string())
        }
    }

    /// `low_vram` defaults to true (the reference per-stage load/drop mode).
    #[test]
    fn low_vram_defaults_to_per_stage_load_drop() {
        assert!(Trellis2RunOptions::default().low_vram);
    }

    /// `low_vram = false` opens one run-scoped residency window: the first
    /// load of a stage is pinned and later acquisitions reuse it, nested
    /// entry points share the window, and the outermost drop releases
    /// everything. The default `low_vram = true` never pins a stage.
    #[test]
    fn low_vram_false_pins_stages_for_the_whole_run() {
        let pipeline = residency_test_pipeline();
        let default_options = Trellis2RunOptions::default();
        assert!(default_options.low_vram);

        // Default mode: every acquisition is a private load, nothing pinned.
        let loads = std::sync::atomic::AtomicUsize::new(0);
        let first = pipeline
            .acquire_stage_model("dummy", "dummy", &Device::Cpu, counting_load(&loads))
            .expect("default load");
        let second = pipeline
            .acquire_stage_model("dummy", "dummy", &Device::Cpu, counting_load(&loads))
            .expect("default load");
        assert_eq!(2, loads.load(std::sync::atomic::Ordering::SeqCst));
        assert!(!Arc::ptr_eq(&first, &second));
        assert!(lock_resident(&pipeline.resident).models.is_empty());
        drop(first);
        drop(second);

        // Resident mode: nested guards share the window; the outer drop
        // releases the pinned stage.
        let options = Trellis2RunOptions {
            low_vram: false,
            ..Trellis2RunOptions::default()
        };
        let loads = std::sync::atomic::AtomicUsize::new(0);
        {
            let outer = pipeline.enter_run_mode(&options);
            {
                let inner = pipeline.enter_run_mode(&options);
                let first = pipeline
                    .acquire_stage_model("dummy", "dummy", &Device::Cpu, counting_load(&loads))
                    .expect("resident load");
                let second = pipeline
                    .acquire_stage_model("dummy", "dummy", &Device::Cpu, counting_load(&loads))
                    .expect("resident reuse");
                assert_eq!(1, loads.load(std::sync::atomic::Ordering::SeqCst));
                assert!(Arc::ptr_eq(&first, &second));
                assert_eq!(
                    2,
                    lock_resident(&pipeline.resident).active_runs,
                    "nested guards share one window"
                );
                drop(inner);
                assert_eq!(
                    1,
                    lock_resident(&pipeline.resident).active_runs,
                    "the inner drop keeps the outer window alive"
                );
                assert!(lock_resident(&pipeline.resident)
                    .cached::<String>("dummy")
                    .is_some());
            }
            drop(outer);
            assert!(lock_resident(&pipeline.resident).models.is_empty());
        }
        assert_eq!(0, lock_resident(&pipeline.resident).active_runs);
        assert!(lock_resident(&pipeline.resident).models.is_empty());
    }

    /// A stage load failure inside the resident window names the CLI flag
    /// and the device memory limit instead of falling back.
    #[test]
    fn resident_load_failure_names_flag_and_device_memory_limit() {
        let error = resident_stage_load_error(
            "shape_dit_512",
            &Device::Cpu,
            Lux3dError::InvalidInput("simulated out of memory"),
        );
        let message = error.to_string();
        assert!(message.contains("--no-low-vram"), "message: {message}");
        assert!(message.contains("shape_dit_512"), "message: {message}");
        assert!(
            message.contains("device memory limit"),
            "message: {message}"
        );
        assert!(
            message.contains("simulated out of memory"),
            "underlying cause must be kept: {message}"
        );
    }

    /// `num_samples > 1` draws ONE joint sparse-structure noise tensor
    /// `[N, 8, 16, 16, 16]` from a single stream — the upstream
    /// `torch.randn(num_samples, in_channels, reso, reso, reso)`
    /// (`trellis2_image_to_3d.py:208`) — not the old per-sample
    /// `seed + i` sequential emulation. The first sample slice is exactly
    /// the `num_samples = 1` draw.
    #[test]
    fn batched_ss_noise_is_one_joint_draw() {
        let device = Device::Cpu;
        let seed = 1234_u64;
        let config = Trellis2SsConfig::trellis_2_4b();
        let samples = 2_usize;

        let joint = draw_ss_noise(samples, seed, &device).expect("joint SS draw");
        assert_eq!(
            joint.dims(),
            &[
                samples,
                config.in_channels,
                config.resolution,
                config.resolution,
                config.resolution
            ]
        );
        let values = tensor_values(&joint);
        let expected = Trellis2Rng::new(seed ^ TRELLIS2_SS_NOISE_SALT)
            .randn(samples * config.token_count() * config.in_channels);
        assert_eq!(values, expected, "the batch must come from ONE draw");

        let single = draw_ss_noise(1, seed, &device).expect("single SS draw");
        let single_values = tensor_values(&single);
        assert_eq!(
            &values[..single_values.len()],
            &single_values[..],
            "sample 0 must be the num_samples = 1 draw"
        );
        let offset_draw = draw_ss_noise(1, seed + 1, &device).expect("seed + 1 draw");
        assert_ne!(
            &values[single_values.len()..],
            &tensor_values(&offset_draw)[..],
            "sample 1 must continue the joint stream, not use seed + 1"
        );
    }

    /// One joint SLAT noise draw `[rows, 32]` over the concatenated
    /// per-sample coords (`torch.randn(coords.shape[0], in_channels)`,
    /// `trellis2_image_to_3d.py:253`): a prefix of a larger joint draw is
    /// the smaller draw from the same seed/salt.
    #[test]
    fn batched_slat_noise_is_one_joint_draw() {
        let device = Device::Cpu;
        let seed = 99_u64;
        let joint =
            draw_slat_noise(13, seed, TRELLIS2_SHAPE_NOISE_SALT, &device).expect("joint SLAT draw");
        assert_eq!(joint.dims(), &[13, 32]);
        assert_eq!(
            tensor_values(&joint),
            Trellis2Rng::new(seed ^ TRELLIS2_SHAPE_NOISE_SALT).randn(13 * 32)
        );

        let prefix =
            draw_slat_noise(5, seed, TRELLIS2_SHAPE_NOISE_SALT, &device).expect("prefix SLAT draw");
        assert_eq!(
            &tensor_values(&joint)[..5 * 32],
            &tensor_values(&prefix)[..],
            "the joint stream prefix must match the smaller draw"
        );

        // The per-sample slices are contiguous [rows_i, 32] views of the one
        // joint tensor in row order.
        let slice = noise_rows(&joint, 5, 4).expect("slice");
        assert_eq!(slice.dims(), &[4, 32]);
        assert_eq!(
            tensor_values(&slice),
            tensor_values(&joint)[5 * 32..9 * 32].to_vec()
        );
    }

    /// Multi-sample cascade token budgeting is GLOBAL: the reference
    /// concatenates the batch coords before the `unique`/budget loop
    /// (`trellis2_image_to_3d.py:327-339`), so two samples that each fit a
    /// per-sample cap but exceed it together must share ONE reduced
    /// `hr_resolution`, and the global row count must stay under the cap.
    #[test]
    fn batched_cascade_budget_is_global_and_shares_one_resolution() {
        // 40 distinct LR x-coords spaced 4 apart, identical per sample; the
        // reference's leading batch column keeps equal xyz rows from
        // different samples distinct under `unique(dim=0)`.
        let sample: Vec<[i32; 3]> = (0..40).map(|k| [4 * k, 0, 0]).collect();
        let samples = vec![sample.clone(), sample.clone()];
        let cap = 45_usize;

        // Per sample the 1536-grid pass yields 30 unique rows (< 45), so a
        // per-sample budget would keep 1536 for both samples.
        let (single, single_resolution) =
            quantize_hr_coords(&sample, TRELLIS2_CASCADE_LR_RESOLUTION, 1536, cap);
        assert_eq!(30, single.len());
        assert_eq!(1536, single_resolution);

        // The batch total at 1536 is 60 >= 45, so the shared loop reduces
        // the resolution until the GLOBAL count fits (1024 here).
        let (batched, hr_resolution) =
            quantize_hr_coords_batched(&samples, TRELLIS2_CASCADE_LR_RESOLUTION, 1536, cap);
        assert_eq!(
            1024, hr_resolution,
            "the batch must share one reduced resolution"
        );
        let total: usize = batched.iter().map(Vec::len).sum();
        assert!(
            total < cap,
            "global token count {total} must stay below {cap}"
        );
        for sample in &batched {
            assert_eq!(20, sample.len());
            let mut sorted = sample.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(&sorted, sample, "per-sample rows stay sorted and unique");
        }

        // A batch of one is bit-identical to the single-sample helper.
        let (one, one_resolution) = quantize_hr_coords_batched(
            std::slice::from_ref(&samples[0]),
            TRELLIS2_CASCADE_LR_RESOLUTION,
            1536,
            cap,
        );
        assert_eq!(single_resolution, one_resolution);
        assert_eq!(single, one[0]);
    }

    /// Regression guard for `num_samples = 1`: the joint-draw helpers consume
    /// the exact legacy single-sample stream values in the same order, so
    /// the single-output path (and `run_many_*` with N = 1, which keeps the
    /// unbatched call) is bit-identical to the pre-`num_samples` behavior.
    #[test]
    fn single_sample_noise_matches_the_legacy_single_pass() {
        let device = Device::Cpu;
        let seed = 42_u64;
        let config = Trellis2SsConfig::trellis_2_4b();
        assert_eq!(
            tensor_values(&draw_ss_noise(1, seed, &device).expect("SS draw")),
            Trellis2Rng::new(seed ^ TRELLIS2_SS_NOISE_SALT)
                .randn(config.token_count() * config.in_channels),
            "single-sample SS draw order changed"
        );
        let rows = 7_usize;
        assert_eq!(
            tensor_values(
                &draw_slat_noise(rows, seed, TRELLIS2_SHAPE_NOISE_SALT, &device)
                    .expect("shape draw")
            ),
            Trellis2Rng::new(seed ^ TRELLIS2_SHAPE_NOISE_SALT).randn(rows * 32),
            "single-sample shape SLAT draw order changed"
        );
        assert_eq!(
            tensor_values(
                &draw_slat_noise(rows, seed, TRELLIS2_TEX_NOISE_SALT, &device)
                    .expect("texture draw")
            ),
            Trellis2Rng::new(seed ^ TRELLIS2_TEX_NOISE_SALT).randn(rows * 32),
            "single-sample texture SLAT draw order changed"
        );
    }

    /// The `[1, L, C]` conditioning pass is tiled over the batch, and a
    /// `[N, L, C]` tensor passes through unchanged.
    #[test]
    fn conditioning_batch_tiling_shares_one_pass() {
        let device = Device::Cpu;
        let cond = Tensor::from_vec(
            (0..24).map(|v| v as f32).collect::<Vec<f32>>(),
            (1, 6, 4),
            &device,
        )
        .expect("cond");
        let tiled = repeat_cond_to_batch(&cond, 3).expect("tile");
        assert_eq!(tiled.dims(), &[3, 6, 4]);
        let all = tensor_values(&tiled);
        let one = tensor_values(&cond);
        for sample in 0..3 {
            assert_eq!(&all[sample * one.len()..(sample + 1) * one.len()], &one[..]);
        }
        assert_eq!(
            repeat_cond_to_batch(&tiled, 3)
                .expect("already batched")
                .dims(),
            &[3, 6, 4]
        );
    }

    /// Full `resolution^3` PBR grid whose base color ramps along Z (the same
    /// fixture the `pbr_bake` snap tests use): a sample at `z = 0` is the
    /// ramp midpoint 0.5 (127 after quantization), at `z = 0.25` it is
    /// ~0.8333 (212).
    fn z_ramp_grid(resolution: u32) -> (Vec<i32>, Vec<f32>) {
        let resolution_i = resolution as i32;
        let mut voxel_coords =
            Vec::with_capacity((resolution * resolution * resolution) as usize * 3);
        let mut attrs = Vec::with_capacity((resolution * resolution * resolution) as usize * 6);
        for z in 0..resolution_i {
            let value = z as f32 / (resolution_i - 1) as f32;
            for y in 0..resolution_i {
                for x in 0..resolution_i {
                    voxel_coords.extend([x, y, z]);
                    attrs.extend([value, value, value, 0.0, 0.0, 1.0]);
                }
            }
        }
        (voxel_coords, attrs)
    }

    /// Flat quad in the plane `z = plane_z` spanning `[-0.4, 0.4]^2` with
    /// the full `[0, 1]` UV square; returns the flat f32 vertices, `i64`
    /// triangle indices and per-corner UVs.
    fn plane_quad(plane_z: f32) -> (Vec<f32>, Vec<i64>, Vec<[f32; 2]>) {
        let vertices = vec![
            [-0.4_f32, -0.4, plane_z],
            [0.4, -0.4, plane_z],
            [0.4, 0.4, plane_z],
            [-0.4, 0.4, plane_z],
        ];
        (
            vertices.iter().flatten().copied().collect(),
            vec![0, 1, 2, 0, 2, 3],
            vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        )
    }

    /// A texture-export-ready mesh: the drifted quad (`z = 0.25`) with the
    /// 6-channel PBR payload and per-vertex colors present.
    fn drifted_pbr_buffers() -> Trellis2MeshBuffers {
        let resolution = 4_u32;
        let (voxel_coords, voxel_attrs) = z_ramp_grid(resolution);
        let (vertices, faces, _uvs) = plane_quad(0.25);
        Trellis2MeshBuffers {
            resolution,
            vertices,
            faces,
            voxel_coords,
            voxel_attrs,
            voxel_attr_channels: 6,
            vertex_colors: Some(vec![0.0; 12]),
        }
    }

    /// The to_glb texture export snaps every covered texel onto the
    /// pre-postprocess surface before sampling the PBR volume
    /// (`postprocess.py:249-266, 254-256`). The drifted export mesh samples
    /// the z = 0 plateau (127) with the snap-back and the z = 0.25 plateau
    /// (~212) without it.
    #[test]
    fn snapped_texture_export_projects_texels_onto_the_pre_postprocess_surface() {
        let mesh = drifted_pbr_buffers();
        // The ORIGINAL surface (pre-postprocess) is the plane z = 0.
        let (original_vertices, original_faces, _uvs) = plane_quad(0.0);
        let original_faces_u32: Vec<u32> = original_faces.iter().map(|&f| f as u32).collect();
        let options = Trellis2TextureExportOptions {
            enabled: true,
            texture_size: 32,
            uv_unwrap: true,
        };

        let snapped = build_trellis2_textured_mesh_snapped(
            &mesh,
            &options,
            &original_vertices,
            &original_faces_u32,
        )
        .expect("snapped texture export")
        .expect("PBR payload present");
        let plain = build_trellis2_textured_mesh(&mesh, &options)
            .expect("plain texture export")
            .expect("PBR payload present");

        let texture = &snapped.texture;
        let mut covered = 0_usize;
        for texel in 0..texture.valid_mask.len() {
            if !texture.valid_mask[texel] {
                continue;
            }
            covered += 1;
            assert_eq!(
                127,
                texture.base_color_rgba[texel * 4],
                "snapped texel {texel} must sample the original z = 0 surface"
            );
            let plain_red = plain.texture.base_color_rgba[texel * 4];
            assert!(
                (211..=213).contains(&plain_red),
                "unprojected texel {texel} red {plain_red} must sample the drifted z = 0.25 surface"
            );
        }
        assert!(covered > 0, "the atlas must cover texels");
        // Same atlas layout, same coverage: only the sampled surface differs.
        assert_eq!(texture.valid_mask, plain.texture.valid_mask);
        assert_eq!(texture.stats.valid_texels, plain.texture.stats.valid_texels);
    }

    /// The snap-back source is captured only for textured to_glb runs and
    /// only when the geometry post-process will actually move the surface.
    #[test]
    fn bake_snap_surface_capture_matches_the_textured_to_glb_gate() {
        let mesh = drifted_pbr_buffers();
        let options = Trellis2RunOptions::default();
        let captured = Trellis2Pipeline::capture_bake_snap_surface(&mesh, &options)
            .expect("capture")
            .expect("default run captures the decoded surface");
        assert_eq!(captured.0, mesh.vertices);
        assert_eq!(
            captured.1,
            mesh.faces.iter().map(|&f| f as u32).collect::<Vec<u32>>()
        );

        // Geometry-only run: no bake, no BVH.
        let geometry_only = Trellis2RunOptions {
            with_texture: false,
            ..Trellis2RunOptions::default()
        };
        assert!(
            Trellis2Pipeline::capture_bake_snap_surface(&mesh, &geometry_only)
                .expect("capture")
                .is_none()
        );

        // Texture export disabled: no bake, no BVH.
        let no_export = Trellis2RunOptions {
            texture_export: Trellis2TextureExportOptions {
                enabled: false,
                ..Trellis2TextureExportOptions::default()
            },
            ..Trellis2RunOptions::default()
        };
        assert!(
            Trellis2Pipeline::capture_bake_snap_surface(&mesh, &no_export)
                .expect("capture")
                .is_none()
        );

        // Post-process disabled: the bake mesh is the original surface, so
        // the projection would be the identity — skip the snapshot.
        let no_postprocess = Trellis2RunOptions {
            postprocess: Trellis2MeshPostprocessOptions {
                enabled: false,
                ..Trellis2MeshPostprocessOptions::default()
            },
            ..Trellis2RunOptions::default()
        };
        assert!(
            Trellis2Pipeline::capture_bake_snap_surface(&mesh, &no_postprocess)
                .expect("capture")
                .is_none()
        );

        // Invalid face indices fail descriptively (never panic / wrap).
        let mut broken = drifted_pbr_buffers();
        broken.faces.push(99);
        let error = Trellis2Pipeline::capture_bake_snap_surface(&broken, &options)
            .expect_err("out-of-range face rejected");
        assert!(error
            .to_string()
            .contains("face index outside the vertex buffer"));
    }

    /// The captured snap surface runs the SAME shared `fill_holes`
    /// pre-pass as the geometry post-process (`postprocess.py:108-122`), so
    /// the snap BVH sees the filled surface instead of the raw hole
    /// boundary.
    #[test]
    fn bake_snap_surface_capture_applies_the_shared_fill_holes_prepass() {
        let mut mesh = drifted_pbr_buffers();
        // One tiny quad (side 0.005): its boundary loop perimeter (0.02) is
        // under the default 3e-2, so the fill closes it with a centroid fan.
        mesh.vertices = vec![
            0.0, 0.0, 0.0, 0.005, 0.0, 0.0, 0.005, 0.005, 0.0, 0.0, 0.005, 0.0,
        ];
        mesh.faces = vec![0, 1, 2, 0, 2, 3];
        let options = Trellis2RunOptions::default();
        let (vertices, faces) = Trellis2Pipeline::capture_bake_snap_surface(&mesh, &options)
            .expect("capture")
            .expect("snapshot");
        assert_eq!(15, vertices.len(), "the fill adds the loop centroid");
        assert_eq!(18, faces.len(), "the loop is closed with a centroid fan");

        // `fill_hole_perimeter = 0` keeps the raw decoded surface (the
        // shared fill is a no-op), preserving the already-filled behavior.
        let no_fill = Trellis2RunOptions {
            postprocess: Trellis2MeshPostprocessOptions {
                fill_hole_perimeter: 0.0,
                ..Trellis2MeshPostprocessOptions::default()
            },
            ..Trellis2RunOptions::default()
        };
        let (vertices, faces) = Trellis2Pipeline::capture_bake_snap_surface(&mesh, &no_fill)
            .expect("capture")
            .expect("snapshot");
        assert_eq!(mesh.vertices, vertices);
        assert_eq!(vec![0, 1, 2, 0, 2, 3], faces);
    }
}
