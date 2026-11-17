//! TRELLIS.2 texturing pipeline core (reference
//! `trellis2/pipelines/trellis2_texturing.py`): an EXISTING triangle mesh
//! plus one condition image -> a UV-atlas PBR-textured mesh
//! ([`Trellis2TexturedMesh`], GLB-writable via
//! `Trellis2ExportStage::write_textured_glb`).
//!
//! Flow ([`Trellis2Pipeline::texture_mesh`], reference `run(...)`):
//!
//! 1. `preprocess_mesh` (`preprocess_mesh_for_texturing`): center the vertex
//!    bbox at its midpoint, scale by `0.99999 / max_extent`, then swap the
//!    y/z axes with `y' = -z, z' = y` (a proper rotation, winding kept) —
//!    the reference `[-0.5, 0.5]^3` mesh frame. This is the mesh half of a
//!    round trip: the textured-GLB writer's axis conversion `(x, y, z) ->
//!    (x, z, -y)` restores the original orientation at export.
//! 2. `preprocess_image` + `get_cond`: the reference foreground square crop
//!    (premultiplied RGB, RMBG-2.0 for opaque inputs) at `512` when
//!    `resolution == 512`, else at `1024` (so `1024` and `1536` share the
//!    `cond_1024` configuration), then DINOv3 ViT-L/16 conditioning
//!    (`cond` + `neg_cond = zeros_like(cond)`).
//! 3. `encode_shape_slat`: `mesh_to_flexible_dual_grid` at `resolution`
//!    ([`MeshToDualGridOptions::for_shape_slat_encoder`]: explicit
//!    `[-0.5, 0.5]^3` aabb, `boundary_weight = 0.2`,
//!    `regularization_weight = 1e-2`), the 6-channel encoder input
//!    (`concat_dual_grid_input`) and the `shape_enc_next_dc_f16c32_fp16`
//!    network returning the RAW mean latent on the `resolution / 16` grid
//!    (four S2C downsamples).
//! 4. `sample_tex_slat`: the shape latent is re-normalized with
//!    `shape_slat_normalization` inside the stage; noise `[N, 32]` is
//!    concatenated with it (noise first) into the 64-channel
//!    `tex_dit_512` / `tex_dit_1024` input; the sampled latent is
//!    de-normalized with `tex_slat_normalization`. The reference selects
//!    `tex_slat_flow_model_512` only for `resolution == 512` and the 1024
//!    model otherwise (`trellis2_texturing.py:401`), so `1536` rides the
//!    same 1024 checkpoint through the elastic RoPE path on its 96-cell
//!    latent grid. Sampler params:
//!    `resolve_stage_params(&config.tex_sampler, Some(&options.tex_sampler),
//!    12)`.
//! 5. `decode_tex_slat`: guided texture VAE decode (`tex_dec`,
//!    `pred_subdiv = false`) with SYNTHESIZED guide logits
//!    ([`synthesize_tex_guide_subs`]), `raw * 0.5 + 0.5` -> the 6-channel
//!    PBR layout (base_color 0:3, metallic 3, roughness 4, alpha 5).
//! 6. `postprocess_mesh`: when the input mesh carries UVs they are USED
//!    (the atlas is not re-unwrapped) — the reference
//!    `trellis2_texturing.py:299-313` branch, which keeps
//!    `mesh.visual.uv` and only calls `uv_unwrap` when it is absent. With
//!    provided UVs the bake runs on them directly and they are carried into
//!    the textured GLB; otherwise `uv_unwrap` (xatlas) runs at
//!    `texture_size`. Area-weighted vertex normals, per-texel
//!    sparse-trilinear sampling of the PBR voxels and TELEA-style inpainting
//!    follow in both branches ([`crate::geometry::bake_pbr_texture`], via
//!    `build_trellis2_textured_mesh`). Axis conversion and the UV V-flip
//!    happen only at GLB export.
//!
//! # Guide synthesis (the upstream SparseTensor-cache equivalent)
//!
//! The reference texturing pipeline calls the texture decoder WITHOUT
//! `guide_subs` (`decode_tex_slat` -> `tex_decoder(slat)`): with
//! `pred_subdiv = false` each C2S block looks up the `channel2spatial_2`
//! spatial cache that `SparseSpatial2Channel` registered on the ENCODER's
//! downsample outputs. Those caches propagate through `SparseTensor.replace`
//! and the flow's sparse ops into the sampled slat, so the tex decode walks
//! the exact coordinate tree the encoder built from the mesh voxelization.
//!
//! Our SparseTensor model carries only `(coords, feats)` — caches do not
//! survive the sampler — so [`synthesize_tex_guide_subs`] rebuilds the same
//! information from the voxelization occupancy chain and passes it as
//! `guide_subs` (one `[N_parents, 8]` logits tensor per C2S stage, rows
//! aligned with the stage's parent coords in the DECODER's order, `+1`
//! selects / `-1` drops, binarized `logits > 0` inside C2S exactly like the
//! reference). The cache contents are exactly derivable: with `V` the
//! full-resolution voxelization coords, the encoder's S2C chain produces
//! `L_k = unique(V / 2^(4-k))` for `k = 1..4` (each S2C divides by 2 and
//! dedups), and the cache registered at scale `2^k` records children
//! `L_{k-1}` grouped under parents `L_k`. The per-parent 8-bit child mask
//! therefore represents the chain EXACTLY — every child has exactly one
//! parent, so no occupancy pattern is lost — and the synthesized bits equal
//! the upstream cache's `subdivision` bit pattern.
//!
//! Two documented deviations, both immaterial:
//!
//! * The DECODE OUTPUT ROW ORDER differs: the upstream cache yields the
//!   children in the encoder's row order (the voxelization kernel's
//!   insertion order), while the guided C2S emits children grouped by parent
//!   in parent order, ascending `subidx = dx + 2*dy + 4*dz`. Downstream
//!   consumers (flex_gemm-style sparse trilinear sampling, the PBR bake) are
//!   hashmap-based and order-insensitive;
//!   [`Trellis2TexturingOutput::tex_coords`] rows stay aligned with the
//!   returned attribute rows.
//! * The guide logits are occupancy constants (`+1` / `-1`), not the shape
//!   decoder's learned `to_subdiv` outputs — the texturing reference never
//!   runs the shape decoder at all (the encoder caches replace it), so there
//!   is no upstream logit source to match.
//!
//! # Shape-encoder weights
//!
//! The canonical TRELLIS.2 package has no `shape_enc` stage yet, so the
//! encoder loads from the standalone raw checkpoint
//! `ckpts/shape_enc_next_dc_f16c32_fp16.safetensors` under the model dir
//! ([`load_trellis2_shape_encoder_from_model_assets`],
//! [`Trellis2Pipeline::load_shape_encoder`]). A future canonical package
//! that carries `shape_enc.*` keys in its merged file is honored
//! automatically.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use candle_core::{DType, Device, Tensor};

use super::trellis2::{
    build_trellis2_textured_mesh, resolve_stage_params, Trellis2Pipeline, Trellis2SamplerOverrides,
    Trellis2TextureExportOptions, Trellis2TexturedMesh,
};
use super::trellis2_shape_enc::{
    load_trellis2_shape_encoder_from_stage_file, Trellis2ShapeEncoder, Trellis2ShapeEncoderConfig,
};
use super::trellis2_tex::{query_vertex_colors, Trellis2PbrLayout};
use super::trellis2_vae::Trellis2ShapeVaeConfig;
use crate::contracts::{SpatialSize, TensorContract, Trellis2PreparedImage, Trellis2RawInput};
use crate::error::Lux3dError;
use crate::geometry::{
    area_weighted_vertex_normals, bake_pbr_texture, mesh_to_flexible_dual_grid,
    MeshToDualGridOptions, Trellis2MeshBuffers,
};
use crate::preprocess::{resize_lanczos_u8, Trellis2Preprocess};
use crate::weights::{is_gguf_path, load_canonical_weights, ModelAssetOptions, VarBuilder};
use crate::{ModelFamily, Result};

/// Raw shape-SLAT encoder checkpoint path relative to the TRELLIS.2 model
/// dir (the canonical package has no `shape_enc` stage yet).
pub const TRELLIS2_SHAPE_ENC_STAGE_REL: &str = "ckpts/shape_enc_next_dc_f16c32_fp16.safetensors";
/// Shape-SLAT encoder stored beside a merged canonical weight file
/// (`external/shape_enc/model.safetensors`). Quantized TRELLIS.2 GGUFs omit
/// the encoder stage, so the canonical package carries the fp16 encoder as a
/// separate external tensor file.
pub const TRELLIS2_SHAPE_ENC_EXTERNAL_REL: &str = "external/shape_enc/model.safetensors";

/// Options for [`Trellis2Pipeline::texture_mesh`] (the reference `run(...)`
/// signature, defaults matched).
#[derive(Debug, Clone, PartialEq)]
pub struct Trellis2TexturingOptions {
    /// RNG seed for the texture SLAT noise (42).
    pub seed: u64,
    /// Voxelization / shape-encode / texture-sampling resolution (1024).
    /// `512` selects the `tex_dit_512` stage and 512 conditioning; `1024`
    /// and `1536` share the `tex_dit_1024` stage and the 1024 conditioning
    /// configuration — the reference app radio offers 512/1024/1536
    /// (`app_texturing.py:101`) and `run(...)` picks the 512 model only for
    /// `resolution == 512` while voxelizing/encoding at `resolution`
    /// (`trellis2_texturing.py:399-407`).
    pub resolution: u32,
    /// Texture sampler overrides merged over the pipeline-config
    /// `tex_slat_sampler` defaults (`resolve_stage_params`; the base step
    /// count is 12, the reference default). All-`None` keeps the config.
    pub tex_sampler: Trellis2SamplerOverrides,
    /// Baked texture edge length (2048; multiple of 8 in `8..=8192`).
    pub texture_size: usize,
    /// Run the reference image preprocessing (foreground bbox square crop +
    /// premultiplied RGB, RMBG-2.0 for opaque inputs; `true`). With `false`
    /// the input RGB is used as-is (alpha ignored, like the reference DINOv3
    /// feature extractor's `convert("RGB")`) and must be square.
    pub preprocess_image: bool,
    /// Run `preprocess_mesh` (center/scale/y-z swap; `true`). With `false`
    /// the vertices are used as-is (they must already be inside
    /// `[-0.5, 0.5]^3`).
    pub preprocess_mesh: bool,
}

impl Default for Trellis2TexturingOptions {
    fn default() -> Self {
        Self {
            seed: 42,
            resolution: 1024,
            tex_sampler: Trellis2SamplerOverrides::default(),
            texture_size: 2048,
            preprocess_image: true,
            preprocess_mesh: true,
        }
    }
}

impl Trellis2TexturingOptions {
    /// Validate the options: `resolution` in `{512, 1024, 1536}` (the three
    /// upstream app choices; 1024 and 1536 share the 1024 conditioning/flow
    /// configuration) and a `texture_size` matching the texture-export
    /// contract (multiple of 8 in `8..=8192`).
    pub fn validate(&self) -> Result<()> {
        if !matches!(self.resolution, 512 | 1024 | 1536) {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 texturing resolution must be 512, 1024 or 1536 (the reference \
                 app choices; 1024 and 1536 share the 1024 get_cond / tex flow \
                 configuration)",
            ));
        }
        let texture = Trellis2TextureExportOptions {
            enabled: true,
            texture_size: self.texture_size,
            uv_unwrap: true,
        };
        texture.validate()?;
        Ok(())
    }

    /// The DINOv3 conditioning resolution (`get_cond(image, 512 if
    /// resolution == 512 else 1024)`); `1536` texturing uses the 1024
    /// conditioning configuration.
    pub fn cond_resolution(&self) -> u32 {
        if self.resolution == 512 {
            512
        } else {
            1024
        }
    }
}

/// Texturing pipeline output (the reference `run(...)` returns the textured
/// mesh; the stage tensors are exposed for tests and parity taps).
#[derive(Debug, Clone)]
pub struct Trellis2TexturingOutput {
    /// The UV-atlas textured mesh (the upstream `postprocess_mesh` product):
    /// xatlas atlas vertices/indices, area-weighted vertex normals, atlas
    /// UVs and the baked base-color + metallic-roughness textures. All
    /// geometry is in the PREPROCESSED mesh frame (`[-0.5, 0.5]^3`,
    /// y/z-swapped); the GLB writer applies the axis conversion and UV
    /// V-flip at export.
    pub mesh: Trellis2TexturedMesh,
    /// Shape latent coordinates (the encoder output: unique `voxel / 16`,
    /// row-major sorted) on the `resolution / 16` grid.
    pub shape_coords: Vec<[i32; 3]>,
    /// Texture decode output voxel coordinates — EQUAL TO THE ORIGINAL
    /// VOXELIZATION COORDS AS A SET (upstream cache semantics; row order is
    /// the guided C2S order, see the module docs). Aligned with the baked
    /// PBR voxel attributes.
    pub tex_coords: Vec<[i32; 3]>,
    /// The shape SLAT fed to the texture sampler: the RAW encoder mean
    /// latent `[N, 32]` (NOT normalized — `sample_tex_slat` applies
    /// `shape_slat_normalization` internally; this is the tensor a parity
    /// tap compares against the reference `encode_shape_slat` output).
    pub shape_latent: Tensor,
    /// The DE-NORMALIZED texture SLAT `[N, 32]` on `shape_coords` (what the
    /// texture VAE consumed).
    pub tex_latent: Tensor,
}

/// Reference `preprocess_mesh` (`trellis2_texturing.py:106-120`): center the
/// vertex bbox, scale so the largest extent maps to `0.99999`, then swap the
/// y/z axes with `y' = -z`, `z' = y` (a proper rotation — face winding is
/// preserved, so `faces` pass through unchanged). The result is asserted
/// into `[-0.5, 0.5]^3`.
///
/// Returns the transformed flat vertex buffer (face indices are unchanged).
/// Errors on empty/degenerate meshes (zero extent), non-finite vertices, or
/// a post-transform vertex outside `[-0.5, 0.5]` (the reference `assert`).
pub fn preprocess_mesh_for_texturing(vertices: &[f32], faces: &[u32]) -> Result<Vec<f32>> {
    let _ = faces;
    if vertices.is_empty() || vertices.len() % 3 != 0 {
        return Err(Lux3dError::InvalidInput(
            "Trellis2 texturing mesh vertex buffer must be non-empty with a length that \
             is a multiple of 3",
        ));
    }
    if vertices.iter().any(|v| !v.is_finite()) {
        return Err(Lux3dError::InvalidInput(
            "Trellis2 texturing mesh contains a non-finite vertex",
        ));
    }
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for vertex in vertices.chunks_exact(3) {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    let center = [
        (min[0] + max[0]) / 2.0,
        (min[1] + max[1]) / 2.0,
        (min[2] + max[2]) / 2.0,
    ];
    let max_extent = (max[0] - min[0]).max((max[1] - min[1]).max(max[2] - min[2]));
    if !(max_extent.is_finite() && max_extent > 0.0) {
        return Err(Lux3dError::InvalidInput(
            "Trellis2 texturing mesh is degenerate (zero or non-finite extent); the \
             reference 0.99999 / max_extent scale is undefined",
        ));
    }
    let scale = 0.99999_f32 / max_extent;
    let mut out = Vec::with_capacity(vertices.len());
    for vertex in vertices.chunks_exact(3) {
        // (v - center) * scale, then the axis swap: y' = -z, z' = y.
        let x = (vertex[0] - center[0]) * scale;
        let y = (vertex[1] - center[1]) * scale;
        let z = (vertex[2] - center[2]) * scale;
        out.push(x);
        out.push(-z);
        out.push(y);
    }
    // The reference asserts the range; surfaced as an error here.
    if out.iter().any(|&v| v < -0.5 || v > 0.5) {
        return Err(Lux3dError::InvalidInput(
            "Trellis2 preprocessed mesh vertices out of the [-0.5, 0.5] range",
        ));
    }
    Ok(out)
}

/// Per-corner UV layouts hold two floats per entry of `faces` (each entry is
/// one face corner), i.e. `2 * faces.len()` values — not `2/3` of that.
/// Shared by the public [`Trellis2Pipeline::texture_mesh`] validation and the
/// input-UV bake helper so the two cannot diverge.
fn validate_corner_uv_count(faces_len: usize, uv_len: usize) -> Result<()> {
    if uv_len != faces_len * 2 {
        return Err(Lux3dError::ContractValidation {
            message: format!(
                "Trellis2 input UV layout must hold two values per face corner ({} values), got {}",
                faces_len * 2,
                uv_len
            ),
        });
    }
    Ok(())
}

/// Build the textured atlas mesh from INPUT per-corner UVs (the reference
/// `postprocess_mesh` UV-reuse branch, `trellis2_texturing.py:299-313`):
/// no UV unwrap runs, the provided UVs are rasterized/baked directly and
/// carried into the textured mesh (and thus the GLB).
///
/// Corners are split per distinct `(input vertex, UV)` pair so the result is
/// an atlas-style per-vertex UV buffer (duplicates only at UV seams). The
/// area-weighted normals are computed on the INPUT topology and gathered to
/// the split vertices — the reference keeps `mesh.vertex_normals` unchanged
/// in this branch (no `vmap` gather, unlike the unwrap branch).
///
/// `corner_uvs` layout is [`ImportedMesh::uvs`](crate::export::ImportedMesh)
/// (two floats per face corner, trimesh-internal lower-left convention).
/// `chart_count` is `0`: no chart unwrap ran for this mesh.
fn build_textured_mesh_with_input_uvs(
    vertices: &[f32],
    faces: &[u32],
    corner_uvs: &[f32],
    tex_coords: &[[i32; 3]],
    attrs: &[f32],
    resolution: u32,
    texture_size: usize,
) -> Result<Trellis2TexturedMesh> {
    validate_corner_uv_count(faces.len(), corner_uvs.len())?;
    if vertices.len() % 3 != 0 || faces.len() % 3 != 0 {
        return Err(Lux3dError::InvalidInput(
            "Trellis2 input UV layout requires [V, 3] vertices and [F, 3] faces",
        ));
    }
    let input_vertices: Vec<[f32; 3]> = vertices
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect();
    if faces
        .iter()
        .any(|&index| index as usize >= input_vertices.len())
    {
        return Err(Lux3dError::InvalidInput(
            "Trellis2 input UV layout face index outside the vertex buffer",
        ));
    }
    let input_normals = area_weighted_vertex_normals(&input_vertices, faces);

    let mut split_lookup: HashMap<(u32, [u32; 2]), u32> = HashMap::new();
    let mut atlas_vertices: Vec<[f32; 3]> = Vec::new();
    let mut atlas_normals: Vec<[f32; 3]> = Vec::new();
    let mut atlas_uvs: Vec<[f32; 2]> = Vec::new();
    let mut atlas_faces: Vec<u32> = Vec::with_capacity(faces.len());
    for (corner, &vertex_index) in faces.iter().enumerate() {
        let uv = [corner_uvs[corner * 2], corner_uvs[corner * 2 + 1]];
        // Exact-bit UV keys: corners of the same vertex sharing the same UV
        // stay one atlas vertex; distinct UVs split at the seam.
        let key = (vertex_index, [uv[0].to_bits(), uv[1].to_bits()]);
        let split = match split_lookup.get(&key) {
            Some(&index) => index,
            None => {
                let index = u32::try_from(atlas_vertices.len()).map_err(|_| {
                    Lux3dError::ContractValidation {
                        message: "Trellis2 input UV layout exceeds the u32 atlas vertex \
                                      index range"
                            .to_string(),
                    }
                })?;
                let source = vertex_index as usize;
                atlas_vertices.push(input_vertices[source]);
                atlas_normals.push(input_normals[source]);
                atlas_uvs.push(uv);
                split_lookup.insert(key, index);
                index
            }
        };
        atlas_faces.push(split);
    }

    let texture = bake_pbr_texture(
        &atlas_vertices,
        &atlas_uvs,
        &atlas_faces,
        tex_coords,
        attrs,
        resolution,
        texture_size,
    )
    .map_err(|source| Lux3dError::CanonicalWeightsValidation {
        message: format!("Trellis2 PBR texture bake failed: {source}"),
    })?;

    Ok(Trellis2TexturedMesh {
        vertices: atlas_vertices.iter().flatten().copied().collect(),
        faces: atlas_faces,
        normals: atlas_normals.iter().flatten().copied().collect(),
        uvs: atlas_uvs.iter().flatten().copied().collect(),
        texture,
        chart_count: 0,
        atlas_count: 1,
        atlas_size: texture_size as u32,
    })
}

/// Synthesize the texture decoder's `guide_subs` from the mesh voxelization
/// occupancy chain (see the module docs for the upstream-cache equivalence).
///
/// * `voxel_coords` — the full-resolution voxelization coords `V`
///   (`mesh_to_flexible_dual_grid(...,
///   MeshToDualGridOptions::for_shape_slat_encoder(resolution))
///   .voxel_indices`), unique, non-negative.
/// * `latent_coords` — the encoder's latent coords (the texture decode's
///   starting coords); must equal `unique(V / 16)` sorted row-major (the
///   guide rows must align with the decoder's parent coords, which start at
///   the encoder output).
///
/// Returns 4 `[N_parents_stage, 8]` F32 tensors (`+1` child present / `-1`
/// absent), in decoder C2S stage order, on `device`. The simulated decode
/// coord chain is validated to reconstruct `V` exactly.
pub fn synthesize_tex_guide_subs(
    voxel_coords: &[[i32; 3]],
    latent_coords: &[[i32; 3]],
    device: &Device,
) -> Result<Vec<Tensor>> {
    if voxel_coords.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "Trellis2 texturing guide synthesis requires a non-empty voxelization",
        ));
    }
    if latent_coords.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "Trellis2 texturing guide synthesis requires non-empty latent coords",
        ));
    }
    if voxel_coords.iter().any(|c| c.iter().any(|&v| v < 0)) {
        return Err(Lux3dError::InvalidInput(
            "Trellis2 voxelization coords must be non-negative for guide synthesis",
        ));
    }
    let mut v_sorted = voxel_coords.to_vec();
    v_sorted.sort_unstable();
    v_sorted.dedup();
    if v_sorted.len() != voxel_coords.len() {
        return Err(Lux3dError::InvalidInput(
            "Trellis2 voxelization coords contain duplicates (guide synthesis requires \
             the raw voxelization output)",
        ));
    }
    // The encoder's latent grid: four S2C downsamples == floor(coord / 16).
    let mut expected_latent: Vec<[i32; 3]> = v_sorted
        .iter()
        .map(|c| [c[0] / 16, c[1] / 16, c[2] / 16])
        .collect();
    expected_latent.sort_unstable();
    expected_latent.dedup();
    let mut latent_sorted = latent_coords.to_vec();
    latent_sorted.sort_unstable();
    latent_sorted.dedup();
    if latent_sorted != expected_latent {
        return Err(Lux3dError::ModelAssetResolution {
            message: format!(
                "Trellis2 latent coords do not match unique(voxels / 16): {} latent rows \
                 vs {} expected (the guide rows must align with the encoder output)",
                latent_coords.len(),
                expected_latent.len()
            ),
        });
    }

    // Level sets between the latent grid and the full voxelization.
    let level = |div: i32| -> HashSet<[i32; 3]> {
        v_sorted
            .iter()
            .map(|c| [c[0] / div, c[1] / div, c[2] / div])
            .collect()
    };
    let l3 = level(8); // unique(V / 8): the latent parents' children
    let l2 = level(4);
    let l1 = level(2);
    let v_set: HashSet<[i32; 3]> = v_sorted.iter().copied().collect();

    // Walk the decode chain: each stage's guide rows align with the parent
    // coords the DECODER will hold (the C2S expansion order), so the chain
    // is simulated with the same child-selection the guides encode.
    let c2s_stage_count = Trellis2ShapeVaeConfig::tex_decoder().c2s_stage_count();
    if c2s_stage_count != 4 {
        return Err(Lux3dError::ContractValidation {
            message: format!(
                "Trellis2 tex decoder must carry 4 C2S stages for guide synthesis, got \
                 {c2s_stage_count}"
            ),
        });
    }
    let mut parents = latent_coords.to_vec();
    let mut guides = Vec::with_capacity(c2s_stage_count);
    for children in [&l3, &l2, &l1, &v_set] {
        let bits = guide_bits(&parents, children);
        let flat: Vec<f32> = bits.iter().flat_map(|row| row.iter().copied()).collect();
        guides.push(
            Tensor::from_vec(flat, (parents.len(), 8), device).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("Trellis2 guide tensor allocation failed: {source}"),
                }
            })?,
        );
        parents = c2s_expand_coords(&parents, &bits);
    }
    let mut final_sorted = parents;
    final_sorted.sort_unstable();
    final_sorted.dedup();
    if final_sorted != v_sorted {
        return Err(Lux3dError::ContractValidation {
            message: format!(
                "Trellis2 synthesized guide chain does not reconstruct the voxelization \
                 ({} decoded coords vs {} voxel coords)",
                final_sorted.len(),
                v_sorted.len()
            ),
        });
    }
    Ok(guides)
}

/// Per-parent 8-bit child-selection logits: `+1.0` where the child
/// `parent * 2 + (dx, dy, dz)` (subidx `dx + 2*dy + 4*dz`) is in
/// `children_set`, `-1.0` otherwise (C2S binarizes `logits > 0`).
fn guide_bits(parents: &[[i32; 3]], children_set: &HashSet<[i32; 3]>) -> Vec<[f32; 8]> {
    parents
        .iter()
        .map(|parent| {
            let mut row = [-1.0_f32; 8];
            for subidx in 0..8usize {
                let dx = (subidx & 1) as i32;
                let dy = ((subidx >> 1) & 1) as i32;
                let dz = ((subidx >> 2) & 1) as i32;
                let child = [parent[0] * 2 + dx, parent[1] * 2 + dy, parent[2] * 2 + dz];
                if children_set.contains(&child) {
                    row[subidx] = 1.0;
                }
            }
            row
        })
        .collect()
}

/// The `channel_to_spatial` coordinate expansion for a guide bit block
/// (children grouped by parent in parent order, ascending `subidx`): the
/// exact coordinate order the guided decoder produces.
fn c2s_expand_coords(parents: &[[i32; 3]], bits: &[[f32; 8]]) -> Vec<[i32; 3]> {
    let mut out = Vec::new();
    for (parent, row) in parents.iter().zip(bits.iter()) {
        for subidx in 0..8usize {
            if row[subidx] > 0.0 {
                let dx = (subidx & 1) as i32;
                let dy = ((subidx >> 1) & 1) as i32;
                let dz = ((subidx >> 2) & 1) as i32;
                out.push([parent[0] * 2 + dx, parent[1] * 2 + dy, parent[2] * 2 + dz]);
            }
        }
    }
    out
}

/// Read the key names of a safetensors file header (host-side, no tensor
/// data touched). Used to detect whether a merged canonical package carries
/// a `shape_enc.` stage.
fn safetensors_header_keys(path: &Path) -> Result<Vec<String>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|source| Lux3dError::ModelAssetIo {
        path: path.to_path_buf(),
        source,
    })?;
    let mut len_bytes = [0_u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|source| Lux3dError::ModelAssetIo {
            path: path.to_path_buf(),
            source,
        })?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    if header_len == 0 || header_len > 256 * 1024 * 1024 {
        return Err(Lux3dError::ModelAssetResolution {
            message: format!("suspicious safetensors header length at {}", path.display()),
        });
    }
    let mut header_bytes = vec![0_u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|source| Lux3dError::ModelAssetIo {
            path: path.to_path_buf(),
            source,
        })?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes).map_err(|source| {
        Lux3dError::CanonicalManifestJson {
            path: path.to_path_buf(),
            source,
        }
    })?;
    Ok(header
        .as_object()
        .map(|object| {
            object
                .keys()
                .filter(|key| key.as_str() != "__metadata__")
                .cloned()
                .collect()
        })
        .unwrap_or_default())
}

/// Load the shape SLAT encoder from a standalone stage file or a merged
/// canonical package file. `pub(super)` core shared by
/// [`load_trellis2_shape_encoder_from_model_assets`] and
/// [`Trellis2Pipeline::load_shape_encoder`].
pub(super) fn shape_encoder_from_paths(
    stage_file: Option<&Path>,
    canonical_file: Option<&Path>,
    device: &Device,
) -> Result<Trellis2ShapeEncoder> {
    if let Some(file) = stage_file {
        if !file.is_file() {
            return Err(Lux3dError::MissingWeightFile {
                path: file.to_path_buf(),
            });
        }
        // SAFETY: the mmap-backed builder keeps the mapping alive for every
        // tensor created from it; the file is not mutated.
        return unsafe { load_trellis2_shape_encoder_from_stage_file(file, device) }.map_err(
            |source| Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "failed to load the TRELLIS.2 shape encoder from {}: {source}",
                    file.display()
                ),
            },
        );
    }
    if let Some(file) = canonical_file {
        if file.is_file() && !is_gguf_path(file) {
            let keys = safetensors_header_keys(file)?;
            if keys.iter().any(|key| key.starts_with("shape_enc.")) {
                // SAFETY: mmap-backed; the file outlives the builder's
                // tensors and is not mutated.
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        std::slice::from_ref(&file),
                        DType::F32,
                        device,
                    )
                }
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!(
                        "failed to open the canonical TRELLIS.2 weights for the shape \
                         encoder: {source}"
                    ),
                })?;
                return Trellis2ShapeEncoder::load(
                    vb.pp("shape_enc"),
                    &Trellis2ShapeEncoderConfig::flexi_dual_shape(),
                )
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!(
                        "failed to load the TRELLIS.2 shape encoder from the canonical \
                         package: {source}"
                    ),
                });
            }
        }
    }
    Err(Lux3dError::ModelAssetResolution {
        message: format!(
            "the TRELLIS.2 shape encoder checkpoint `{TRELLIS2_SHAPE_ENC_STAGE_REL}` is \
             unavailable: pass a model dir containing it (the canonical package has no \
             `shape_enc` stage yet); searched a standalone stage file and the merged \
             canonical package for `shape_enc.` keys"
        ),
    })
}

/// Load the shape SLAT encoder from the TRELLIS.2 model assets: a
/// standalone `ckpts/shape_enc_next_dc_f16c32_fp16.safetensors` under
/// `options.canonical_dir` when present (the raw `TRELLIS.2-4B` model dir
/// layout), else `shape_enc.*` keys in the resolved canonical package's
/// merged weights file. Errors with a clear message when neither exists
/// (the canonical package has no `shape_enc` stage today).
pub fn load_trellis2_shape_encoder_from_model_assets(
    options: &ModelAssetOptions,
    device: &Device,
) -> Result<Trellis2ShapeEncoder> {
    let (stage_file, canonical_file) = match &options.canonical_dir {
        Some(dir) if dir.is_dir() => {
            let stage = dir.join(TRELLIS2_SHAPE_ENC_STAGE_REL);
            let external = dir.join(TRELLIS2_SHAPE_ENC_EXTERNAL_REL);
            let stage = if stage.is_file() {
                Some(stage)
            } else {
                external.is_file().then_some(external)
            };
            (stage, None)
        }
        Some(_) => (None, None),
        None => {
            // Resolve the canonical package the same way the pipeline does
            // (cache dir / download), then look for the stage in it.
            let weights = load_canonical_weights(ModelFamily::Trellis2, options.clone())?;
            (None, Some(weights.canonical_file.clone()))
        }
    };
    shape_encoder_from_paths(stage_file.as_deref(), canonical_file.as_deref(), device)
}

impl Trellis2Pipeline {
    /// Load the shape SLAT encoder for THIS pipeline's weights: the raw
    /// dir's standalone checkpoint, or `shape_enc.*` keys in the canonical
    /// merged file. The raw-dir device must match `device` (the same
    /// contract as the other stage loads).
    pub fn load_shape_encoder(&self, device: &Device) -> Result<Trellis2ShapeEncoder> {
        match (&self.raw_dir, &self.weights) {
            (Some((dir, raw_device)), _) => {
                if raw_device.location() != device.location() {
                    return Err(Lux3dError::InvalidInput(
                        "Trellis2 raw-dir pipeline is bound to a different device",
                    ));
                }
                let file = dir.join(TRELLIS2_SHAPE_ENC_STAGE_REL);
                shape_encoder_from_paths(Some(&file), None, device)
            }
            (None, Some(weights)) => {
                let external = weights
                    .canonical_file
                    .parent()
                    .map(|dir| dir.join(TRELLIS2_SHAPE_ENC_EXTERNAL_REL))
                    .filter(|file| file.is_file());
                shape_encoder_from_paths(external.as_deref(), Some(&weights.canonical_file), device)
            }
            (None, None) => Err(Lux3dError::InvalidInput(
                "Trellis2Pipeline has no weights loaded",
            )),
        }
    }

    /// TRELLIS.2 texturing pipeline core (reference
    /// `Trellis2TexturingPipeline.run`): an existing triangle mesh plus one
    /// condition image -> [`Trellis2TexturingOutput`] with the baked PBR
    /// atlas mesh. See the module docs for the stage walk and the guide
    /// synthesis contract.
    ///
    /// * `vertices` — flat `[V, 3]` input mesh positions (any frame;
    ///   `preprocess_mesh_for_texturing` normalizes unless
    ///   `options.preprocess_mesh` is false).
    /// * `faces` — flat zero-based triangle indices.
    /// * `corner_uvs` — optional per-corner UVs carried by the imported mesh
    ///   (`ImportedMesh::uvs` layout: two floats per face corner, length
    ///   `2 * faces.len()`, trimesh-internal lower-left convention).
    ///   `Some` takes the reference UV-reuse branch
    ///   (`trellis2_texturing.py:299-313`): the provided layout is baked and
    ///   exported as-is, NO xatlas unwrap runs. `None` takes the unwrap
    ///   branch. Partial or unusable sets are already collapsed to `None` by
    ///   the importer (documented rule); a `Some` buffer that does not cover
    ///   every corner is rejected rather than silently baked.
    /// * `image` — the condition image (reference `preprocess_image` +
    ///   `get_cond` semantics; the pipeline's RMBG-2.0 remover handles
    ///   opaque inputs when `options.preprocess_image` is set).
    pub fn texture_mesh(
        &self,
        vertices: &[f32],
        faces: &[u32],
        corner_uvs: Option<&[f32]>,
        image: &Trellis2RawInput,
        options: &Trellis2TexturingOptions,
        device: &Device,
    ) -> Result<Trellis2TexturingOutput> {
        options.validate()?;
        if vertices.is_empty() || vertices.len() % 3 != 0 {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 texturing mesh vertices must be a non-empty [V, 3] buffer",
            ));
        }
        if faces.is_empty() || faces.len() % 3 != 0 {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 texturing mesh faces must be a non-empty [F, 3] buffer",
            ));
        }
        let vertex_count = vertices.len() / 3;
        if faces.iter().any(|&f| f as usize >= vertex_count) {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 texturing mesh face index outside the vertex buffer",
            ));
        }
        if let Some(uvs) = corner_uvs {
            validate_corner_uv_count(faces.len(), uvs.len())?;
            if uvs.iter().any(|value| !value.is_finite()) {
                return Err(Lux3dError::InvalidInput(
                    "Trellis2 texturing input UVs contain a non-finite value",
                ));
            }
        }

        // 1. preprocess_mesh: center/scale + y/z axis swap into
        //    [-0.5, 0.5]^3.
        let mesh_vertices = if options.preprocess_mesh {
            preprocess_mesh_for_texturing(vertices, faces)?
        } else {
            if vertices.iter().any(|v| !v.is_finite()) {
                return Err(Lux3dError::InvalidInput(
                    "Trellis2 texturing mesh contains a non-finite vertex",
                ));
            }
            if vertices
                .chunks_exact(3)
                .any(|v| v.iter().any(|&c| c < -0.5 || c > 0.5))
            {
                return Err(Lux3dError::InvalidInput(
                    "Trellis2 texturing with preprocess_mesh = false requires vertices \
                     already inside [-0.5, 0.5]^3",
                ));
            }
            vertices.to_vec()
        };

        // 2. conditioning: the reference crops once and feeds the same crop
        //    to `get_cond` at the conditioning resolution (512 only for
        //    `resolution == 512`, else 1024; 1536 texturing uses 1024).
        let cond_resolution = options.cond_resolution();
        let cond_preprocess = Trellis2Preprocess::new(cond_resolution)?;
        let prepared = if options.preprocess_image {
            cond_preprocess.preprocess_with_rembg(image.clone(), self.background_remover())?
        } else {
            prepared_without_preprocess(image, cond_resolution)?
        };
        let pixel_values = cond_preprocess
            .normalized_pixel_values(&prepared, device)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 DINOv3 input normalization failed: {source}"),
            })?;
        let (cond, _neg_cond) = self.dino_conditioning_pixels(&pixel_values, device)?;

        // 3. voxelization + 4. shape SLAT encode (raw mean latent). The
        //    shape-encoder grid size is the full `resolution` (the reference
        //    `encode_shape_slat(mesh, resolution)` voxelizes at 1536 even
        //    when the 1024 conditioning/flow configuration is selected).
        let vox = mesh_to_flexible_dual_grid(
            &mesh_vertices,
            faces,
            &MeshToDualGridOptions::for_shape_slat_encoder(options.resolution),
        )
        .map_err(|source| Lux3dError::ContractValidation {
            message: format!("Trellis2 texturing mesh voxelization failed: {source}"),
        })?;
        if vox.voxel_indices.is_empty() {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 texturing mesh voxelized to zero voxels",
            ));
        }
        let voxel_rows = vox.voxel_indices.len();
        let dual_flat: Vec<f32> = vox
            .dual_vertices
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect();
        let flags_flat: Vec<f32> = vox
            .intersected
            .iter()
            .flat_map(|f| f.iter().map(|&b| b as u8 as f32))
            .collect();
        let dual_tensor =
            Tensor::from_vec(dual_flat, (voxel_rows, 3), &Device::Cpu).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("Trellis2 dual vertex tensor allocation failed: {source}"),
                }
            })?;
        let flags_tensor =
            Tensor::from_vec(flags_flat, (voxel_rows, 3), &Device::Cpu).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("Trellis2 intersected tensor allocation failed: {source}"),
                }
            })?;
        let encoder = self.load_shape_encoder(device)?;
        let slat = encoder
            .encode_shape_slat(
                &vox.voxel_indices,
                &dual_tensor,
                &flags_tensor,
                options.resolution as usize,
                device,
            )
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 shape SLAT encode failed: {source}"),
            })?;
        drop(encoder);
        let shape_latent = slat.feats;
        let latent_coords = slat.coords;

        // 5. texture SLAT sampling (the shape latent is re-normalized inside
        //    the stage; tex params resolved from the config).
        let tex_params = resolve_stage_params(
            &self.config.tex_sampler,
            Some(&options.tex_sampler),
            12, // reference tex_slat_sampler default step count
        );
        let tex_latent = if options.resolution == 512 {
            self.sample_tex_slat(
                &cond,
                &shape_latent,
                &latent_coords,
                None,
                options.seed,
                &tex_params,
                device,
            )?
        } else {
            // 1024 and 1536 both use the reference `tex_slat_flow_model_1024`
            // checkpoint (`run(...)`: `tex_slat_flow_model_512` if resolution
            // == 512 else `tex_slat_flow_model_1024`); the 1536 shape latent
            // lives on the 96-cell grid, so `sample_tex_slat_hr` selects the
            // same checkpoint with the elastic validation bound.
            self.sample_tex_slat_hr(
                &cond,
                &shape_latent,
                &latent_coords,
                None,
                options.seed,
                &tex_params,
                device,
                (options.resolution / 16) as usize,
            )?
        };

        // 6. guided texture VAE decode with the synthesized occupancy guides.
        let guides = synthesize_tex_guide_subs(&vox.voxel_indices, &latent_coords, device)?;
        let (tex_coords, attrs) =
            self.decode_tex_attrs(&latent_coords, &tex_latent, &guides, device)?;
        // Upstream cache semantics: the decoded coords must equal the
        // original voxelization coords as a SET (row order may differ).
        let mut decoded_sorted = tex_coords.clone();
        decoded_sorted.sort_unstable();
        let mut voxel_sorted = vox.voxel_indices.clone();
        voxel_sorted.sort_unstable();
        if decoded_sorted != voxel_sorted {
            return Err(Lux3dError::ContractValidation {
                message: format!(
                    "Trellis2 texture decode coords diverged from the mesh voxelization \
                     ({} decoded vs {} voxel rows); the synthesized guide chain is \
                     inconsistent with the encoder coords",
                    decoded_sorted.len(),
                    voxel_sorted.len()
                ),
            });
        }

        // 7. postprocess_mesh: UV reuse when the input carries UVs (no
        //    unwrap), else xatlas unwrap + bake, over the preprocessed mesh.
        //    The unwrap branch samples per-vertex colors alongside (they
        //    feed the PLY/GLB vertex-color path and satisfy the stage's
        //    PBR-payload guard); the UV-reuse branch only needs the bake,
        //    exactly like the reference postprocess_mesh.
        let textured = if let Some(corner_uvs) = corner_uvs {
            build_textured_mesh_with_input_uvs(
                &mesh_vertices,
                faces,
                corner_uvs,
                &tex_coords,
                &attrs,
                options.resolution,
                options.texture_size,
            )?
        } else {
            let vertices_rows: Vec<[f32; 3]> = mesh_vertices
                .chunks_exact(3)
                .map(|c| [c[0], c[1], c[2]])
                .collect();
            let vertex_colors =
                query_vertex_colors(&vertices_rows, &tex_coords, &attrs, options.resolution)
                    .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                        message: format!("Trellis2 vertex color sampling failed: {source}"),
                    })?;
            let buffers = Trellis2MeshBuffers {
                resolution: options.resolution,
                vertices: mesh_vertices,
                faces: faces.iter().map(|&f| f as i64).collect(),
                voxel_coords: tex_coords.iter().flat_map(|c| [c[0], c[1], c[2]]).collect(),
                voxel_attrs: attrs,
                voxel_attr_channels: Trellis2PbrLayout::CHANNELS,
                vertex_colors: Some(vertex_colors),
            };
            build_trellis2_textured_mesh(
                &buffers,
                &Trellis2TextureExportOptions {
                    enabled: true,
                    texture_size: options.texture_size,
                    uv_unwrap: true,
                },
            )?
            .ok_or_else(|| Lux3dError::ContractValidation {
                message: "Trellis2 texture export skipped cleanly but the texturing \
                              pipeline requires a baked atlas mesh"
                    .to_string(),
            })?
        };

        Ok(Trellis2TexturingOutput {
            mesh: textured,
            shape_coords: latent_coords,
            tex_coords,
            shape_latent,
            tex_latent,
        })
    }

    /// File-path entry point for [`Self::texture_mesh`]: load the condition
    /// image, run the reference `preprocess_image` (foreground square crop +
    /// premultiplied RGB, RMBG-2.0 for opaque inputs) exactly once, then
    /// texture the mesh. The prepared crop is square and already
    /// premultiplied, so the inner call runs with
    /// `preprocess_image = false` (no second RMBG pass).
    ///
    /// This surface has no UV channel and always takes the unwrap branch;
    /// callers holding `ImportedMesh::uvs` should call [`Self::texture_mesh`]
    /// directly to reuse them.
    pub fn texture_mesh_from_path(
        &self,
        vertices: &[f32],
        faces: &[u32],
        image_path: &Path,
        options: &Trellis2TexturingOptions,
        device: &Device,
    ) -> Result<Trellis2TexturingOutput> {
        options.validate()?;
        let (raw, inner) = if options.preprocess_image {
            let crop = self
                .preprocess
                .prepare_crop_from_path(image_path, self.background_remover())?;
            let (width, height) = crop.rgb().dimensions();
            if width != height {
                return Err(Lux3dError::ContractValidation {
                    message: format!("TRELLIS.2 texturing crop is not square ({width}x{height})"),
                });
            }
            let rgb: Vec<[f32; 3]> = crop
                .rgb()
                .pixels()
                .map(|pixel| {
                    [
                        pixel[0] as f32 / 255.0,
                        pixel[1] as f32 / 255.0,
                        pixel[2] as f32 / 255.0,
                    ]
                })
                .collect();
            let raw = Trellis2RawInput {
                original_size: SpatialSize { width, height },
                rgba: None,
                rgb: Some(rgb),
            };
            // The crop above already ran the foreground preprocessing; only
            // the resolution-specific resize remains.
            let mut inner = options.clone();
            inner.preprocess_image = false;
            (raw, inner)
        } else {
            let raw = Trellis2Preprocess::load_raw_input_from_path(image_path)?;
            (raw, options.clone())
        };
        self.texture_mesh(vertices, faces, None, &raw, &inner, device)
    }
}

/// `preprocess_image = false` conditioning input: the reference feeds the
/// untouched PIL image to the DINOv3 feature extractor, which converts to
/// RGB (alpha dropped, no compositing) and resizes to its input side. This
/// port needs a square buffer: the input must be square and is Lanczos
/// resized to `cond_resolution`.
fn prepared_without_preprocess(
    input: &Trellis2RawInput,
    cond_resolution: u32,
) -> Result<Trellis2PreparedImage> {
    if input.original_size.width != input.original_size.height {
        return Err(Lux3dError::InvalidInput(
            "Trellis2 texturing with preprocess_image = false requires a square input \
             image (the reference feature extractor accepts any aspect ratio; this port \
             needs a square buffer)",
        ));
    }
    let side = input.original_size.width as usize;
    let plane = side
        .checked_mul(side)
        .ok_or_else(|| Lux3dError::InvalidInput("TRELLIS.2 input size overflows usize"))?;
    let source: Vec<[f32; 3]> = match (&input.rgba, &input.rgb) {
        (Some(values), _) => {
            if values.len() != plane {
                return Err(Lux3dError::ContractValidation {
                    message: format!(
                        "TRELLIS.2 raw input rgba length {} does not match {}x{}",
                        values.len(),
                        side,
                        side
                    ),
                });
            }
            values.iter().map(|p| [p[0], p[1], p[2]]).collect()
        }
        (None, Some(values)) => {
            if values.len() != plane {
                return Err(Lux3dError::ContractValidation {
                    message: format!(
                        "TRELLIS.2 raw input rgb length {} does not match {}x{}",
                        values.len(),
                        side,
                        side
                    ),
                });
            }
            values.clone()
        }
        (None, None) => {
            return Err(Lux3dError::InvalidInput(
                "TRELLIS.2 raw input requires rgb or rgba values",
            ));
        }
    };
    // [0, 1] f32 -> u8 (the contract-stage quantization convention), then
    // the preprocess-stage Lanczos resize to the conditioning side.
    let mut rgb_u8 = Vec::with_capacity(plane * 3);
    for pixel in &source {
        for channel in 0..3 {
            rgb_u8.push((pixel[channel].clamp(0.0, 1.0) * 255.0) as u8);
        }
    }
    let resized = resize_lanczos_u8::<3>(
        &rgb_u8,
        side,
        side,
        cond_resolution as usize,
        cond_resolution as usize,
    );
    Ok(Trellis2PreparedImage {
        family: ModelFamily::Trellis2,
        original_size: input.original_size,
        target_size: SpatialSize::new(cond_resolution, cond_resolution),
        uint8_rgb: resized,
        point: None,
        pixel_values: TensorContract::new(
            "pixel_values",
            vec![1, 3, cond_resolution as usize, cond_resolution as usize],
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference app radio offers exactly 512/1024/1536
    /// (`app_texturing.py:101`); `validate` must accept those and reject
    /// everything else.
    #[test]
    fn texturing_resolution_validation_matches_app_choices() {
        for resolution in [512_u32, 1024, 1536] {
            let options = Trellis2TexturingOptions {
                resolution,
                ..Default::default()
            };
            options
                .validate()
                .unwrap_or_else(|error| panic!("resolution {resolution} must validate: {error}"));
        }
        for resolution in [0_u32, 256, 768, 1024 + 1, 2048, 3072] {
            let options = Trellis2TexturingOptions {
                resolution,
                ..Default::default()
            };
            assert!(
                options.validate().is_err(),
                "resolution {resolution} must be rejected"
            );
        }
    }

    /// 1536 voxelizes/encodes at 1536 (grid size = resolution, latent grid =
    /// resolution / 16 = 96) while conditioning stays at the 1024
    /// configuration (`trellis2_texturing.py:399-407`).
    #[test]
    fn texturing_1536_uses_full_grid_with_1024_conditioning() {
        let options = Trellis2TexturingOptions {
            resolution: 1536,
            ..Default::default()
        };
        options.validate().expect("1536 must validate");
        assert_eq!(1024, options.cond_resolution());
        assert_eq!(96, options.resolution / 16);

        let grid = MeshToDualGridOptions::for_shape_slat_encoder(options.resolution);
        assert_eq!([1536, 1536, 1536], grid.grid_size);
        assert_eq!(
            Some([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]),
            grid.aabb,
            "the shape encoder keeps the reference [-0.5, 0.5]^3 aabb"
        );
    }

    /// Existing 512/1024 paths keep their dispatch: 512 -> 512 conditioning,
    /// 1024 -> 1024 conditioning.
    #[test]
    fn texturing_existing_resolutions_unchanged() {
        assert_eq!(Trellis2TexturingOptions::default().resolution, 1024);
        for (resolution, expected_cond) in [(512_u32, 512_u32), (1024, 1024), (1536, 1024)] {
            let options = Trellis2TexturingOptions {
                resolution,
                ..Default::default()
            };
            assert_eq!(expected_cond, options.cond_resolution());
        }
    }

    /// Quad fixture for the UV-reuse tests: reference-frame quad in `z = 0`,
    /// two triangles, per-corner UV rect `0.1..0.9` (trimesh-internal
    /// lower-left convention) and a constant red PBR volume at resolution 2.
    fn textured_quad_fixture() -> (Vec<f32>, Vec<u32>, Vec<f32>, Vec<[i32; 3]>, Vec<f32>) {
        let vertices = vec![
            -0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.5, 0.5, 0.0, -0.5, 0.5, 0.0,
        ];
        let faces = vec![0, 1, 2, 0, 2, 3];
        let corner_uvs = vec![0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.9];
        let mut coords = Vec::new();
        let mut attrs = Vec::new();
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    coords.push([x, y, z]);
                    // base color red, metallic 0, roughness 0.5, alpha 1.
                    attrs.extend([1.0, 0.0, 0.0, 0.0, 0.5, 1.0]);
                }
            }
        }
        (vertices, faces, corner_uvs, coords, attrs)
    }

    /// Reference `postprocess_mesh` UV-reuse branch
    /// (`trellis2_texturing.py:299-313`): provided UVs are baked directly
    /// (no xatlas unwrap), and the textured GLB carries them back unchanged
    /// (the writer's V flip is undone by the importer's V flip).
    #[test]
    fn provided_uvs_bypass_unwrap_and_roundtrip_through_the_textured_glb() {
        let (vertices, faces, corner_uvs, coords, attrs) = textured_quad_fixture();
        let textured = build_textured_mesh_with_input_uvs(
            &vertices,
            &faces,
            &corner_uvs,
            &coords,
            &attrs,
            2,
            64,
        )
        .expect("input-UV bake");

        assert_eq!(0, textured.chart_count, "no unwrap runs with input UVs");
        assert_eq!(1, textured.atlas_count);
        assert_eq!(64, textured.atlas_size);
        // Shared corners with identical UVs stay shared: 4 split vertices.
        assert_eq!(vec![0, 1, 2, 0, 2, 3], textured.faces);
        assert_eq!(
            vec![0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
            textured.uvs,
            "per-atlas-vertex UVs are the provided corner UVs, deduplicated"
        );
        assert_eq!(12, textured.vertices.len());
        assert_eq!(12, textured.normals.len());
        assert!(textured.texture.stats.valid_texels > 0);
        for texel in 0..64 * 64 {
            if !textured.texture.valid_mask[texel] {
                continue;
            }
            assert_eq!(
                [255, 0, 0, 255],
                textured.texture.base_color_rgba[texel * 4..texel * 4 + 4]
            );
            assert_eq!(
                [0, 127, 0, 255],
                textured.texture.metallic_roughness[texel * 4..texel * 4 + 4]
            );
        }

        let path = std::env::temp_dir().join(format!(
            "lux3d-trellis2-input-uv-{}.glb",
            std::process::id()
        ));
        crate::export::Trellis2ExportStage
            .write_textured_glb(&textured, &path, true)
            .expect("write textured glb");
        let imported = crate::export::load_mesh(&path).expect("read textured glb");
        let _ = std::fs::remove_file(&path);

        assert_eq!(textured.faces, imported.faces);
        let imported_uvs = imported.uvs.expect("textured GLB must carry TEXCOORD_0");
        assert_eq!(
            corner_uvs.len(),
            imported_uvs.len(),
            "UV count is unchanged through bake + GLB export"
        );
        for (expected, actual) in corner_uvs.iter().zip(imported_uvs.iter()) {
            assert!(
                (expected - actual).abs() <= 1e-6,
                "UV {expected} must survive the GLB round trip, got {actual}"
            );
        }
    }

    /// Meshes without UVs keep the current behavior: xatlas unwrap runs and
    /// still produces an atlas (the byte-identical fallback path).
    #[test]
    fn no_uv_mesh_still_unwraps() {
        let (vertices, faces, _corner_uvs, coords, attrs) = textured_quad_fixture();
        let buffers = Trellis2MeshBuffers {
            resolution: 2,
            vertices,
            faces: faces.iter().map(|&face| face as i64).collect(),
            voxel_coords: coords.iter().flatten().copied().collect(),
            voxel_attrs: attrs,
            voxel_attr_channels: 6,
            vertex_colors: Some(vec![1.0; 4 * 3]),
        };
        let textured = build_trellis2_textured_mesh(
            &buffers,
            &Trellis2TextureExportOptions {
                enabled: true,
                texture_size: 64,
                uv_unwrap: true,
            },
        )
        .expect("build succeeds")
        .expect("PBR payload present");
        assert!(
            textured.chart_count > 0,
            "a mesh without UVs must still run the xatlas unwrap"
        );
        assert!(!textured.uvs.is_empty());
        for uv in textured.uvs.chunks_exact(2) {
            assert!((0.0..=1.0).contains(&uv[0]) && (0.0..=1.0).contains(&uv[1]));
        }
    }

    /// Regression: the entry-point validation must count two floats per
    /// entry of `faces` (per corner), e.g. 12 triangles = 36 corners = 72
    /// floats. An earlier `2 * faces.len() / 3` rule rejected valid layouts
    /// (caught by the 4B e2e cube fixture).
    #[test]
    fn corner_uv_count_is_two_per_face_corner() {
        validate_corner_uv_count(36, 72).expect("12 triangles -> 72 UV floats");
        let error = validate_corner_uv_count(36, 24).expect_err("24 floats are not per-corner");
        assert!(
            error.to_string().contains("(72 values), got 24"),
            "unexpected message: {error}"
        );
    }

    /// A partial UV buffer cannot reach the bake (the importer already
    /// collapses incomplete sets); a direct caller gets a clear error.
    #[test]
    fn partial_uv_layout_is_rejected() {
        let (vertices, faces, corner_uvs, coords, attrs) = textured_quad_fixture();
        let error = build_textured_mesh_with_input_uvs(
            &vertices,
            &faces,
            &corner_uvs[..corner_uvs.len() - 1],
            &coords,
            &attrs,
            2,
            64,
        )
        .expect_err("partial UV layout must fail");
        assert!(
            error.to_string().contains("two values per face corner"),
            "unexpected message: {error}"
        );
    }
}
