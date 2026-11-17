//! PBR texture baking for the TRELLIS.2 textured-GLB workstream
//! (upstream `to_glb` attribute sampling, `postprocess.py:229-304`).
//!
//! The reference pipeline rasterizes the atlas UV triangles with nvdiffrast
//! at `texture_size` (2048), interpolates each covered texel's 3D position
//! from the triangle vertices, trilinearly samples the sparse PBR attribute
//! volume at that position and finally inpaints the uncovered texels with
//! `cv2.inpaint(..., INPAINT_TELEA)` (radius 3 for base color, 1 for
//! metallic / roughness / alpha) so no black seams appear at UV borders.
//!
//! This module reproduces that on CPU:
//!
//! * Rasterization: scanline edge-function test of every pixel center
//!   (`(tx + 0.5, ty + 0.5)`) against each triangle, conservatively dilated
//!   by 0.5 texel ([`COVERAGE_PAD_TEXELS`]) so triangles meeting at a shared
//!   edge overlap instead of leaving crack pixels (nvdiffrast rasterizes
//!   pixel centers too; the dilation plays the role of its conservative
//!   rasterization and matches the inpaint radius scale). UV row 0 is
//!   `v = 1` — the nvdiffrast image layout the upstream texture PNGs use;
//!   the GLB writer flips V (`v' = 1 - v`) at export exactly like the
//!   reference.
//! * Overlap policy: when triangles overlap a texel (the conservative
//!   dilation makes shared-edge texels overlap by design), the HIGHEST
//!   triangle index wins (last write in ascending face order). Deterministic
//!   given the index buffer; the covered set itself is order-independent.
//! * Attribute sampling: the exact sparse trilinear of
//!   `runtime::trellis2_tex::query_vertex_attrs` (the `flex_gemm`
//!   `grid_sample_3d(mode='trilinear')` semantics `attach_texture` uses for
//!   vertex colors) over all 6 PBR channels, in bounded chunks (the texel 3D
//!   position is re-derived per texel from the stored winning face id, so no
//!   huge intermediate buffer is kept).
//! * Quantization: `clamp(v, 0, 1) * 255` floor — the same truncation as the
//!   reference `np.clip(attrs * 255, 0, 255).astype(np.uint8)`.
//! * Snap-back ([`bake_pbr_texture_with_snap`], to_glb export path only):
//!   after UV rasterization and 3D position interpolation, every covered
//!   texel's position is projected onto the ORIGINAL (pre-postprocess)
//!   surface through a [`MeshBvh`] and the attribute volume is sampled at the
//!   projected position — the exact upstream order
//!   (`postprocess.py:249-266`): `bvh.unsigned_distance(valid_pos,
//!   return_uvw=True)` then `valid_pos = (vertices[faces[face_id]] *
//!   uvw).sum(dim=1)` (`postprocess.py:254-256`), correcting the geometric
//!   error introduced by simplification/remeshing. [`bake_pbr_texture`]
//!   itself stays projection-free (the texturing pipeline has no snap-back,
//!   `trellis2_texturing.py:316-332`).
//! * Inpaint: a TELEA-approximation (see [`inpaint_channels`]): iterative
//!   onion-peel nearest-valid propagation with `1 / distance^2` weighting
//!   over a square window of the requested radius, each channel group
//!   starting from the ORIGINAL valid mask (upstream `cv2.inpaint` per
//!   image: base color RGB radius 3, alpha radius 1, roughness radius 1,
//!   metallic radius 1). Valid texels are never touched; the fill grows
//!   smoothly inward from the boundary so no black seams remain.
//!
//! Memory: two RGBA byte textures (`size * size * 4` each), one bool mask
//! and one i32 face-id buffer per texel; sampled positions/attributes live
//! only in bounded chunks ([`SAMPLE_CHUNK_TEXELS`] texels at a time). For a
//! 2048 atlas that is ~40 MiB steady state, no per-pixel allocations.

use crate::error::Lux3dError;
use crate::runtime::{query_vertex_attrs_in_aabb, Trellis2PbrLayout};
use crate::Result;

use super::mesh_bvh::MeshBvh;

/// Conservative rasterization dilation in texels: pixel centers up to this
/// far outside a triangle still count as covered (crack prevention at shared
/// edges; matches the one-texel inpaint radius scale).
pub const COVERAGE_PAD_TEXELS: f32 = 0.5;

/// Texels sampled per `query_vertex_attrs` chunk (bounded memory: the chunk
/// holds `SAMPLE_CHUNK_TEXELS * (3 + 6)` f32 values only).
const SAMPLE_CHUNK_TEXELS: usize = 1 << 16;

/// Baked PBR textures plus the coverage mask and bake statistics.
///
/// * `base_color_rgba`: `[r, g, b, a]` per texel — the glTF
///   `pbrMetallicRoughness.baseColorTexture` (upstream concatenates the
///   inpainted base color with the inpainted alpha).
/// * `metallic_roughness`: `[0, roughness, metallic, 255]` per texel — the
///   glTF `metallicRoughnessTexture` channel packing (G = roughness,
///   B = metallic, R = 0, A = 255), matching the reference
///   `np.concatenate([zeros, roughness, metallic])`. R stays 0 and A stays
///   255 everywhere (the reference does not inpaint those constant planes).
/// * `valid_mask`: texels covered by at least one rasterized triangle BEFORE
///   inpainting (the upstream `mask`); `stats.valid_texels` counts it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BakedPbrTexture {
    /// Texture edge length (`size * size` texels).
    pub size: usize,
    /// RGBA base color, `size * size * 4` bytes.
    pub base_color_rgba: Vec<u8>,
    /// `[0, roughness, metallic, 255]`, `size * size * 4` bytes.
    pub metallic_roughness: Vec<u8>,
    /// Pre-inpaint coverage mask, `size * size` entries.
    pub valid_mask: Vec<bool>,
    /// Coverage / fill statistics.
    pub stats: BakeStats,
}

/// Bake statistics (parity taps and tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BakeStats {
    /// Total texels (`size * size`).
    pub texel_count: usize,
    /// Texels covered by a rasterized triangle (pre-inpaint).
    pub valid_texels: usize,
    /// Texels filled by the inpaint pass (every non-covered texel is filled
    /// when at least one texel is valid — the only mode that reaches here).
    pub inpainted_texels: usize,
    /// Triangles rasterized.
    pub triangle_count: usize,
}

/// Placement of a sparse attribute grid in world space: the reference
/// `grid_sample_3d` call `grid = (position - aabb[0]) / voxel_size` with
/// `voxel_size = (aabb[1] - aabb[0]) / grid_size`
/// (`o-voxel/o_voxel/postprocess.py:259-266`).
///
/// [`PbrSampleGrid::unit`] is the TRELLIS.2 `[-0.5, 0.5]^3` frame; o_voxel
/// assets may carry any box (`.vxz` does not store one).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PbrSampleGrid {
    /// `[[min_x, min_y, min_z], [max_x, max_y, max_z]]`.
    pub aabb: [[f32; 3]; 2],
    /// Per-axis voxel grid resolution.
    pub grid_size: [u32; 3],
}

impl PbrSampleGrid {
    /// The TRELLIS.2 reference frame `[-0.5, 0.5]^3` at a cubic resolution.
    pub const fn unit(resolution: u32) -> Self {
        Self {
            aabb: [[-0.5; 3], [0.5; 3]],
            grid_size: [resolution; 3],
        }
    }
}

/// Bake the 6-channel PBR voxel attributes into two RGBA textures at
/// `texture_size` (upstream 2048).
///
/// * `vertices` / `uvs` / `indices` describe the ATLAS mesh in the reference
///   frame (the product of `geometry::uv_unwrap`).
/// * `voxel_coords` / `voxel_attrs` are the sparse PBR grid `[N, 3]` /
///   `[N, 6]` (`base_color 0:3, metallic 3, roughness 4, alpha 5` — the
///   `Trellis2PbrLayout`); `resolution` is the voxel grid resolution so
///   `(vertex + 0.5) * resolution` maps into the grid exactly like
///   `attach_texture` does. Boundary vertices clamp naturally (the sparse
///   trilinear drops out-of-grid neighbors and renormalizes the rest).
///
/// This entry point does NOT run the upstream texel snap-back: it is the
/// texturing-pipeline bake (`trellis2_texturing.py:316-332` never projects
/// texels). The to_glb export path uses [`bake_pbr_texture_with_snap`].
///
/// Errors are descriptive (no panics): mismatched buffers, empty inputs, or
/// zero rasterized coverage.
pub fn bake_pbr_texture(
    vertices: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
    voxel_coords: &[[i32; 3]],
    voxel_attrs: &[f32],
    resolution: u32,
    texture_size: usize,
) -> Result<BakedPbrTexture> {
    bake_pbr_texture_in_grid(
        vertices,
        uvs,
        indices,
        voxel_coords,
        voxel_attrs,
        PbrSampleGrid::unit(resolution),
        texture_size,
    )
}

/// [`bake_pbr_texture`] with an explicit attribute-volume placement
/// ([`PbrSampleGrid`]) — the o_voxel `ovox2glb` path where the volume was
/// voxelized into a caller-supplied `aabb` / `grid_size`.
#[allow(clippy::too_many_arguments)]
pub fn bake_pbr_texture_in_grid(
    vertices: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
    voxel_coords: &[[i32; 3]],
    voxel_attrs: &[f32],
    grid: PbrSampleGrid,
    texture_size: usize,
) -> Result<BakedPbrTexture> {
    bake_pbr_texture_impl(
        vertices,
        uvs,
        indices,
        voxel_coords,
        voxel_attrs,
        grid,
        texture_size,
        None,
    )
}

/// [`bake_pbr_texture`] with the upstream `to_glb` snap-back onto the
/// ORIGINAL surface (`postprocess.py:249-266`).
///
/// After the UV triangles are rasterized and each covered texel's 3D position
/// is interpolated from the atlas mesh (the post-simplify/remesh export
/// mesh), the position is projected onto `original_surface` — the BVH built
/// from the pre-postprocess mesh (`postprocess.py:122`) — exactly like the
/// reference:
///
/// 1. `bvh.unsigned_distance(valid_pos, return_uvw=True)` returns the
///    nearest triangle and its barycentric `uvw`,
/// 2. `valid_pos = (vertices[faces[face_id]] * uvw).sum(dim=1)` reconstructs
///    the position on the original surface
///    ([`MeshBvh::closest_point`]'s `position` is that same reconstruction),
/// 3. the attribute volume is sampled trilinearly at the projected position.
///
/// The projection is skipped for uncovered texels (they are inpainted from
/// covered ones, identical to the reference order).
#[allow(clippy::too_many_arguments)]
pub fn bake_pbr_texture_with_snap(
    vertices: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
    original_surface: &MeshBvh<'_>,
    voxel_coords: &[[i32; 3]],
    voxel_attrs: &[f32],
    resolution: u32,
    texture_size: usize,
) -> Result<BakedPbrTexture> {
    bake_pbr_texture_impl(
        vertices,
        uvs,
        indices,
        voxel_coords,
        voxel_attrs,
        PbrSampleGrid::unit(resolution),
        texture_size,
        Some(original_surface),
    )
}

/// [`bake_pbr_texture_with_snap`] with an explicit attribute-volume
/// placement ([`PbrSampleGrid`]) — the o_voxel `ovox2glb` path.
#[allow(clippy::too_many_arguments)]
pub fn bake_pbr_texture_with_snap_in_grid(
    vertices: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
    original_surface: &MeshBvh<'_>,
    voxel_coords: &[[i32; 3]],
    voxel_attrs: &[f32],
    grid: PbrSampleGrid,
    texture_size: usize,
) -> Result<BakedPbrTexture> {
    bake_pbr_texture_impl(
        vertices,
        uvs,
        indices,
        voxel_coords,
        voxel_attrs,
        grid,
        texture_size,
        Some(original_surface),
    )
}

#[allow(clippy::too_many_arguments)]
fn bake_pbr_texture_impl(
    vertices: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
    voxel_coords: &[[i32; 3]],
    voxel_attrs: &[f32],
    grid: PbrSampleGrid,
    texture_size: usize,
    original_surface: Option<&MeshBvh<'_>>,
) -> Result<BakedPbrTexture> {
    let channels = Trellis2PbrLayout::CHANNELS;
    if indices.is_empty() || indices.len() % 3 != 0 {
        return Err(Lux3dError::InvalidInput(
            "PBR bake needs a non-empty triangle index buffer (length multiple of 3)",
        ));
    }
    if uvs.len() != vertices.len() {
        return Err(Lux3dError::InvalidInput(
            "PBR bake UV count must match the atlas vertex count",
        ));
    }
    let vertex_count = vertices.len();
    for &index in indices {
        if index as usize >= vertex_count {
            return Err(Lux3dError::InvalidInput(
                "PBR bake rejected a face index outside the atlas vertex buffer",
            ));
        }
    }
    if vertices.iter().any(|v| v.iter().any(|c| !c.is_finite())) {
        return Err(Lux3dError::InvalidInput(
            "PBR bake rejected a non-finite atlas vertex",
        ));
    }
    if uvs.iter().any(|uv| uv.iter().any(|c| !c.is_finite())) {
        return Err(Lux3dError::InvalidInput(
            "PBR bake rejected a non-finite UV",
        ));
    }
    if texture_size == 0 || texture_size > 8192 {
        return Err(Lux3dError::InvalidInput(
            "PBR bake texture_size must be in 1..=8192",
        ));
    }
    if voxel_coords.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "PBR bake needs a non-empty sparse PBR voxel grid",
        ));
    }
    if voxel_attrs.len() != voxel_coords.len() * channels {
        return Err(Lux3dError::InvalidInput(
            "PBR bake voxel attribute buffer must hold 6 channels per voxel coordinate",
        ));
    }
    if grid.grid_size.iter().any(|&size| size == 0) {
        return Err(Lux3dError::InvalidInput(
            "PBR bake voxel grid resolution must be positive on every axis",
        ));
    }
    for axis in 0..3 {
        if !(grid.aabb[0][axis] < grid.aabb[1][axis])
            || !grid.aabb[0][axis].is_finite()
            || !grid.aabb[1][axis].is_finite()
        {
            return Err(Lux3dError::InvalidInput(
                "PBR bake attribute-volume aabb must be finite with min < max on every axis",
            ));
        }
    }

    let texel_count = texture_size * texture_size;
    let mut face_ids = vec![-1_i32; texel_count];
    rasterize_uv_triangles(uvs, indices, texture_size, &mut face_ids);
    // Unique covered texels (overlapping triangles write the same texel; the
    // last (highest-index) triangle wins, per the documented overlap policy).
    let valid_texels = face_ids.iter().filter(|&&face| face >= 0).count();

    if valid_texels == 0 {
        return Err(Lux3dError::InvalidInput(
            "PBR bake covered zero texels: UV triangles do not overlap any pixel center",
        ));
    }

    // The metallic/roughness texture keeps R = 0 / A = 255 on EVERY texel
    // (the reference never inpaints those constant planes).
    let mut metallic_roughness = vec![0_u8; texel_count * 4];
    for chunk in metallic_roughness.chunks_exact_mut(4) {
        chunk[0] = 0;
        chunk[3] = 255;
    }

    let mut baked = BakedPbrTexture {
        size: texture_size,
        base_color_rgba: vec![0_u8; texel_count * 4],
        metallic_roughness,
        valid_mask: vec![false; texel_count],
        stats: BakeStats {
            texel_count,
            valid_texels,
            inpainted_texels: texel_count - valid_texels,
            triangle_count: indices.len() / 3,
        },
    };

    // Per-texel 3D position -> optional snap-back -> sparse trilinear
    // sampling, in bounded chunks. The winning face id is re-evaluated
    // (barycentric interpolation of the triangle vertices) instead of
    // storing per-texel barycentrics.
    let mut chunk_positions: Vec<[f32; 3]> = Vec::with_capacity(SAMPLE_CHUNK_TEXELS);
    let mut chunk_texels: Vec<usize> = Vec::with_capacity(SAMPLE_CHUNK_TEXELS);
    for texel in 0..texel_count {
        let face = face_ids[texel];
        if face < 0 {
            continue;
        }
        let mut position =
            interpolate_texel_position(texel, face as usize, uvs, vertices, indices, texture_size)?;
        if let Some(original_surface) = original_surface {
            // Upstream `bvh.unsigned_distance(valid_pos, return_uvw=True)`
            // then `(vertices[faces[face_id]] * uvw).sum(dim=1)`
            // (`postprocess.py:254-256`); `ClosestPoint::position` is that
            // same barycentric reconstruction on the original triangle.
            position = original_surface.closest_point(position).position;
        }
        chunk_positions.push(position);
        chunk_texels.push(texel);
        if chunk_positions.len() == SAMPLE_CHUNK_TEXELS {
            sample_chunk_into_textures(
                &chunk_positions,
                &chunk_texels,
                voxel_coords,
                voxel_attrs,
                grid,
                &mut baked,
            )?;
            chunk_positions.clear();
            chunk_texels.clear();
        }
    }
    if !chunk_positions.is_empty() {
        sample_chunk_into_textures(
            &chunk_positions,
            &chunk_texels,
            voxel_coords,
            voxel_attrs,
            grid,
            &mut baked,
        )?;
    }

    // The valid mask reflects the pre-inpaint coverage (the upstream `mask`).
    for (texel, mask) in baked.valid_mask.iter_mut().enumerate() {
        *mask = face_ids[texel] >= 0;
    }

    // Inpainting (upstream cv2.inpaint TELEA, each image from the ORIGINAL
    // mask): base color RGB radius 3, alpha radius 1, then the
    // metallic/roughness planes radius 1 (R/A of the MR texture stay
    // constant: 0 and 255, unpainted like the reference).
    let size = texture_size;
    inpaint_channels(
        &mut baked.base_color_rgba,
        &baked.valid_mask,
        size,
        &[0, 1, 2],
        3,
    );
    inpaint_channels(&mut baked.base_color_rgba, &baked.valid_mask, size, &[3], 1);
    inpaint_channels(
        &mut baked.metallic_roughness,
        &baked.valid_mask,
        size,
        &[1, 2],
        1,
    );
    Ok(baked)
}

/// Signed edge function for edge `a -> b` evaluated at `p`
/// (`(b - a) x (p - a)` in 2D pixel space).
#[inline]
fn edge_function(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

/// UV `(u, v)` of a vertex mapped into pixel space: `x = u * size`,
/// `y = (1 - v) * size` — row 0 is `v = 1`, the nvdiffrast image layout.
#[inline]
fn uv_to_pixel(uv: [f32; 2], size_f: f32) -> [f32; 2] {
    [uv[0] * size_f, (1.0 - uv[1]) * size_f]
}

/// Conservative scanline rasterization of the atlas UV triangles into
/// `face_ids` (the winning face index per covered texel, `-1` for uncovered).
///
/// A pixel is covered when its CENTER lies within 0.5 texel of the triangle
/// (signed-distance edge test). Overlaps resolve to the highest triangle
/// index (triangles run in ascending order, so the last write wins).
fn rasterize_uv_triangles(uvs: &[[f32; 2]], indices: &[u32], size: usize, face_ids: &mut [i32]) {
    let size_f = size as f32;
    let pad = COVERAGE_PAD_TEXELS;
    for (face_index, face) in indices.chunks_exact(3).enumerate() {
        let p0 = uv_to_pixel(uvs[face[0] as usize], size_f);
        let mut p1 = uv_to_pixel(uvs[face[1] as usize], size_f);
        let mut p2 = uv_to_pixel(uvs[face[2] as usize], size_f);
        // Signed area (pixel units); flip the winding so "inside" is
        // edge functions >= 0.
        let area = edge_function(p0, p1, p2);
        if !area.is_finite() || area == 0.0 {
            continue;
        }
        if area < 0.0 {
            std::mem::swap(&mut p1, &mut p2);
        }
        // Edges (v0 -> v1), (v1 -> v2), (v2 -> v0); coverage when every edge
        // function >= -pad * edge_length (signed-distance form, so the
        // dilation is a true 0.5 texel offset of the triangle boundary).
        let edges = [(p0, p1), (p1, p2), (p2, p0)];
        let mut dx = [0.0_f32; 3];
        let mut dy = [0.0_f32; 3];
        let mut threshold = [0.0_f32; 3];
        for (i, (a, b)) in edges.iter().enumerate() {
            dx[i] = b[0] - a[0];
            dy[i] = b[1] - a[1];
            let length = (dx[i] * dx[i] + dy[i] * dy[i]).sqrt();
            threshold[i] = -pad * length;
        }
        let min_x = p0[0].min(p1[0]).min(p2[0]);
        let max_x = p0[0].max(p1[0]).max(p2[0]);
        let min_y = p0[1].min(p1[1]).min(p2[1]);
        let max_y = p0[1].max(p1[1]).max(p2[1]);
        let x_start = ((min_x - pad - 1.0).floor() as i64).clamp(0, size as i64 - 1);
        let x_end = ((max_x + pad + 1.0).ceil() as i64).clamp(0, size as i64 - 1);
        let y_start = ((min_y - pad - 1.0).floor() as i64).clamp(0, size as i64 - 1);
        let y_end = ((max_y + pad + 1.0).ceil() as i64).clamp(0, size as i64 - 1);

        for ty in y_start..=y_end {
            let cy = ty as f32 + 0.5;
            // Edge functions at the row start, then incremental x steps
            // (dE/dx = -dy for E(a, b, p) with p stepping in x).
            let cx = x_start as f32 + 0.5;
            let mut e0 = edge_function(edges[0].0, edges[0].1, [cx, cy]);
            let mut e1 = edge_function(edges[1].0, edges[1].1, [cx, cy]);
            let mut e2 = edge_function(edges[2].0, edges[2].1, [cx, cy]);
            for tx in x_start..=x_end {
                if e0 >= threshold[0] && e1 >= threshold[1] && e2 >= threshold[2] {
                    let texel = ty as usize * size + tx as usize;
                    face_ids[texel] = face_index as i32;
                }
                e0 -= dy[0];
                e1 -= dy[1];
                e2 -= dy[2];
            }
        }
    }
}

/// Interpolate the 3D reference-frame position of one covered texel:
/// re-evaluate the barycentric coordinates of the texel's pixel center in
/// the winning triangle and blend the triangle vertices.
fn interpolate_texel_position(
    texel: usize,
    face_index: usize,
    uvs: &[[f32; 2]],
    vertices: &[[f32; 3]],
    indices: &[u32],
    size: usize,
) -> Result<[f32; 3]> {
    let tx = texel % size;
    let ty = texel / size;
    let pixel = [tx as f32 + 0.5, ty as f32 + 0.5];
    let size_f = size as f32;
    let face = [
        indices[face_index * 3] as usize,
        indices[face_index * 3 + 1] as usize,
        indices[face_index * 3 + 2] as usize,
    ];
    let c0 = uv_to_pixel(uvs[face[0]], size_f);
    let c1 = uv_to_pixel(uvs[face[1]], size_f);
    let c2 = uv_to_pixel(uvs[face[2]], size_f);
    let area = edge_function(c0, c1, c2);
    if !area.is_finite() || area == 0.0 {
        return Err(Lux3dError::InvalidInput(
            "PBR bake hit a degenerate UV triangle while interpolating a texel",
        ));
    }
    let w0 = edge_function(c1, c2, pixel) / area;
    let w1 = edge_function(c2, c0, pixel) / area;
    let w2 = edge_function(c0, c1, pixel) / area;
    let v0 = vertices[face[0]];
    let v1 = vertices[face[1]];
    let v2 = vertices[face[2]];
    Ok([
        w0 * v0[0] + w1 * v1[0] + w2 * v2[0],
        w0 * v0[1] + w1 * v1[1] + w2 * v2[1],
        w0 * v0[2] + w1 * v1[2] + w2 * v2[2],
    ])
}

/// Sample the 6 PBR channels at the chunk positions with the reference
/// sparse trilinear and write the quantized RGBA planes into the textures.
fn sample_chunk_into_textures(
    positions: &[[f32; 3]],
    texels: &[usize],
    voxel_coords: &[[i32; 3]],
    voxel_attrs: &[f32],
    grid: PbrSampleGrid,
    baked: &mut BakedPbrTexture,
) -> Result<()> {
    let channels = Trellis2PbrLayout::CHANNELS;
    let sampled = query_vertex_attrs_in_aabb(
        positions,
        voxel_coords,
        voxel_attrs,
        channels,
        grid.aabb,
        grid.grid_size,
    )
    .map_err(|error| Lux3dError::CanonicalWeightsValidation {
        message: format!("Trellis2 PBR texel sampling failed: {error}"),
    })?;
    for (row, &texel) in texels.iter().enumerate() {
        let base = row * channels;
        let dst = texel * 4;
        for channel in 0..3 {
            baked.base_color_rgba[dst + channel] = quantize(sampled[base + channel]);
        }
        baked.base_color_rgba[dst + 3] = quantize(sampled[base + Trellis2PbrLayout::ALPHA]);
        baked.metallic_roughness[dst] = 0;
        baked.metallic_roughness[dst + 1] = quantize(sampled[base + Trellis2PbrLayout::ROUGHNESS]);
        baked.metallic_roughness[dst + 2] = quantize(sampled[base + Trellis2PbrLayout::METALLIC]);
        baked.metallic_roughness[dst + 3] = 255;
    }
    Ok(())
}

/// `clamp(0, 1) * 255` floor — the reference
/// `np.clip(attrs * 255, 0, 255).astype(np.uint8)` truncation.
#[inline]
fn quantize(value: f32) -> u8 {
    if !value.is_finite() {
        return 0;
    }
    let scaled = value.clamp(0.0, 1.0) * 255.0;
    scaled.floor() as u8
}

/// TELEA-approximation inpaint over the given channels of an interleaved
/// RGBA byte buffer.
///
/// Algorithm (documented approximation of `cv2.inpaint(..., INPAINT_TELEA)`):
/// iterative onion-peel propagation. Each pass finds every still-unknown
/// texel adjacent to the known region (8-neighborhood) and fills it with the
/// known-window average weighted by `1 / distance^2` over the square window
/// of `radius` texels (TELEA's geometric weight; the directional weight and
/// disk window are dropped). The newly filled texels join the known region
/// for the next pass, so the fill grows smoothly inward from the boundary
/// and preserves every valid texel. Passes repeat until no unknown texel
/// touches the known region; any stragglers (only possible when zero valid
/// texels exist — rejected by the caller) fall back to the per-channel mean
/// of the known texels. Deterministic: scanline-order frontier processing,
/// fixed arithmetic. Returns the number of texels filled.
fn inpaint_channels(
    buffer: &mut [u8],
    original_known: &[bool],
    size: usize,
    channel_indices: &[usize],
    radius: u32,
) -> usize {
    let texel_count = size * size;
    debug_assert_eq!(buffer.len(), texel_count * 4);
    debug_assert_eq!(original_known.len(), texel_count);
    let radius = radius as i64;
    let mut known = original_known.to_vec();
    let mut filled = 0usize;
    // Remaining unknown texels, scanline order; each pass only walks this
    // shrinking list (never the full texture), splitting it into the frontier
    // (touches the known region) and the still-unreachable rest.
    let mut unknown: Vec<u32> = (0..texel_count as u32)
        .filter(|&texel| !known[texel as usize])
        .collect();
    let mut still_unknown: Vec<u32> = Vec::with_capacity(unknown.len());
    while !unknown.is_empty() {
        let mut frontier: Vec<u32> = Vec::new();
        for &texel in &unknown {
            let texel = texel as usize;
            let tx = texel % size;
            let ty = texel / size;
            let mut touches = false;
            'outer: for dy in [-1_i64, 0, 1] {
                for dx in [-1_i64, 0, 1] {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = tx as i64 + dx;
                    let ny = ty as i64 + dy;
                    if nx < 0 || ny < 0 || nx >= size as i64 || ny >= size as i64 {
                        continue;
                    }
                    if known[ny as usize * size + nx as usize] {
                        touches = true;
                        break 'outer;
                    }
                }
            }
            if touches {
                frontier.push(texel as u32);
            } else {
                still_unknown.push(texel as u32);
            }
        }
        if frontier.is_empty() {
            break;
        }
        for &texel in &frontier {
            let texel = texel as usize;
            let tx = texel % size;
            let ty = texel / size;
            let mut sums = [0.0_f64; 4];
            let mut weights = 0.0_f64;
            let x_min = (tx as i64 - radius).max(0);
            let x_max = (tx as i64 + radius).min(size as i64 - 1);
            let y_min = (ty as i64 - radius).max(0);
            let y_max = (ty as i64 + radius).min(size as i64 - 1);
            for ny in y_min..=y_max {
                for nx in x_min..=x_max {
                    let neighbor = ny as usize * size + nx as usize;
                    if !known[neighbor] {
                        continue;
                    }
                    let dx = (nx - tx as i64) as f64;
                    let dy = (ny - ty as i64) as f64;
                    let weight = 1.0 / (dx * dx + dy * dy + 1e-12);
                    weights += weight;
                    for (slot, &channel) in sums.iter_mut().zip(channel_indices.iter()) {
                        *slot += weight * buffer[neighbor * 4 + channel] as f64;
                    }
                }
            }
            if weights > 0.0 {
                let dst = texel * 4;
                for (slot, &channel) in sums.iter_mut().zip(channel_indices.iter()) {
                    buffer[dst + channel] = (*slot / weights).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        for &texel in &frontier {
            known[texel as usize] = true;
        }
        filled += frontier.len();
        std::mem::swap(&mut unknown, &mut still_unknown);
        still_unknown.clear();
    }
    // Safety net: mean fill for anything unreachable (requires zero valid
    // texels; the caller rejects that case before reaching here).
    let mut unknown = 0usize;
    let mut mean = [0.0_f64; 4];
    let mut known_count = 0usize;
    for texel in 0..texel_count {
        if known[texel] {
            known_count += 1;
            for (slot, &channel) in mean.iter_mut().zip(channel_indices.iter()) {
                *slot += buffer[texel * 4 + channel] as f64;
            }
        } else {
            unknown += 1;
        }
    }
    if unknown > 0 && known_count > 0 {
        for texel in 0..texel_count {
            if known[texel] {
                continue;
            }
            let dst = texel * 4;
            for (slot, &channel) in mean.iter_mut().zip(channel_indices.iter()) {
                buffer[dst + channel] =
                    (*slot / known_count as f64).round().clamp(0.0, 255.0) as u8;
            }
        }
        filled += unknown;
    }
    filled
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::uv_unwrap;

    /// Test-friendly input bundle for [`bake_pbr_texture`].
    struct BakeInputs {
        vertices: Vec<[f32; 3]>,
        uvs: Vec<[f32; 2]>,
        indices: Vec<u32>,
        coords: Vec<[i32; 3]>,
        attrs: Vec<f32>,
        resolution: u32,
    }

    impl BakeInputs {
        fn bake(&self, texture_size: usize) -> Result<BakedPbrTexture> {
            bake_pbr_texture(
                &self.vertices,
                &self.uvs,
                &self.indices,
                &self.coords,
                &self.attrs,
                self.resolution,
                texture_size,
            )
        }
    }

    /// A two-triangle quad covering `uv_rect` in UV space. The atlas
    /// vertices map to positions near the given voxel center so the sparse
    /// trilinear returns the voxel attributes EXACTLY (the lone present
    /// voxel wins after renormalization) for every covered texel.
    fn constant_fixture(
        resolution: u32,
        voxel: [i32; 3],
        attrs: [f32; 6],
        uv_rect: [f32; 2],
    ) -> BakeInputs {
        let center = |axis: usize| (voxel[axis] as f32 + 0.5) / resolution as f32 - 0.5;
        let (u_lo, u_hi) = (uv_rect[0], 1.0 - uv_rect[0]);
        let corners = [[u_lo, u_lo], [u_hi, u_lo], [u_hi, u_hi], [u_lo, u_hi]];
        let vertices: Vec<[f32; 3]> = corners
            .iter()
            .map(|uv| {
                [
                    center(0) + uv[0] * 1e-3,
                    center(1) + uv[1] * 1e-3,
                    center(2),
                ]
            })
            .collect();
        BakeInputs {
            vertices,
            uvs: corners.to_vec(),
            indices: vec![0, 1, 2, 0, 2, 3],
            coords: vec![voxel],
            attrs: attrs.to_vec(),
            resolution,
        }
    }

    /// A 2x2 voxel grid along X with a two-color ramp (voxel (0,0,0) black,
    /// (1,0,0) white): the quad's 3D corners sit exactly at the two voxel
    /// centers so covered texels blend linearly across the boundary.
    fn gradient_fixture() -> BakeInputs {
        let resolution = 2_u32;
        let center = |voxel: i32| (voxel as f32 + 0.5) / resolution as f32 - 0.5;
        let vertices = vec![
            [center(0), center(0), center(0)],
            [center(1), center(0), center(0)],
            [center(1), center(1), center(0)],
            [center(0), center(1), center(0)],
        ];
        let uvs = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let coords = vec![[0, 0, 0], [1, 0, 0]];
        let attrs = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, // voxel 0: black, metallic 0, rough 0, alpha 1
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // voxel 1: white, metallic 1, rough 1, alpha 1
        ];
        BakeInputs {
            vertices,
            uvs,
            indices: vec![0, 1, 2, 0, 2, 3],
            coords,
            attrs,
            resolution,
        }
    }

    #[test]
    fn bake_constant_volume_matches_analytic_colors() {
        // Single voxel: base color red (1, 0, 0), metallic 0.25, roughness
        // 0.75, alpha 1. floor(0.25 * 255) = 63, floor(0.75 * 255) = 191.
        let fixture = constant_fixture(8, [4, 4, 4], [1.0, 0.0, 0.0, 0.25, 0.75, 1.0], [0.1, 0.1]);
        let baked = fixture.bake(64).expect("bake");

        assert_eq!(64, baked.size);
        assert_eq!(64 * 64 * 4, baked.base_color_rgba.len());
        assert_eq!(64 * 64 * 4, baked.metallic_roughness.len());
        assert_eq!(64 * 64, baked.valid_mask.len());
        assert_eq!(2, baked.stats.triangle_count);
        assert!(baked.stats.valid_texels > 0, "quad must cover texels");
        assert!(baked.stats.valid_texels < 64 * 64, "border stays invalid");

        // Every VALID texel matches the analytic quantized values.
        for texel in 0..64 * 64 {
            if !baked.valid_mask[texel] {
                continue;
            }
            assert_eq!(
                [255, 0, 0, 255],
                baked.base_color_rgba[texel * 4..texel * 4 + 4],
                "valid texel {texel} base color"
            );
            assert_eq!(
                [0, 191, 63, 255],
                baked.metallic_roughness[texel * 4..texel * 4 + 4],
                "valid texel {texel} metallic/roughness"
            );
        }
    }

    #[test]
    fn inpaint_fills_invalid_texels_with_boundary_colors() {
        let fixture = constant_fixture(8, [2, 2, 2], [1.0, 0.0, 0.0, 0.25, 0.75, 1.0], [0.1, 0.1]);
        let baked = fixture.bake(64).expect("bake");

        // The texture border (u, v near 0 / 1) is invalid pre-inpaint but
        // must be filled with the same red color — no black seams.
        for texel in [0, 63, 63 * 64, 64 * 64 - 1] {
            assert!(!baked.valid_mask[texel], "border texel {texel} was covered");
            assert_eq!(
                [255, 0, 0, 255],
                baked.base_color_rgba[texel * 4..texel * 4 + 4],
                "inpainted border texel {texel}"
            );
            // MR channels get the radius-1 fill: roughness/metallic restored.
            assert_eq!(
                [0, 191, 63, 255],
                baked.metallic_roughness[texel * 4..texel * 4 + 4],
                "inpainted border texel {texel} mr"
            );
        }
        // Constant volume: inpaint must reach EVERY texel.
        assert_eq!(
            baked.stats.valid_texels + baked.stats.inpainted_texels,
            baked.stats.texel_count
        );
    }

    #[test]
    fn bake_gradient_volume_interpolates_and_layouts_match_upstream() {
        let fixture = gradient_fixture();
        let baked = fixture.bake(32).expect("bake gradient");
        assert!(baked.stats.valid_texels > 0);

        let mut min_r = 255_u8;
        let mut max_r = 0_u8;
        for texel in 0..32 * 32 {
            if !baked.valid_mask[texel] {
                continue;
            }
            let row = &baked.base_color_rgba[texel * 4..texel * 4 + 4];
            let (r, g, b, a) = (row[0], row[1], row[2], row[3]);
            assert_eq!(r, g, "gray ramp: r {r} vs g {g}");
            assert_eq!(g, b, "gray ramp: g {g} vs b {b}");
            assert_eq!(255, a, "alpha is 1.0 across the grid");
            min_r = min_r.min(r);
            max_r = max_r.max(r);
        }
        // Texel centers never hit uv 0/1 exactly, so the ramp ends are the
        // nearest quantized samples, not the pure voxel colors.
        assert!(min_r <= 8, "ramp low end sampled, min r = {min_r}");
        assert!(max_r >= 247, "ramp high end sampled, max r = {max_r}");

        // Color is linear in x (the volume only ramps along X): r is
        // non-decreasing along every row of valid texels.
        for row in 0..32 {
            let mut previous: Option<u8> = None;
            for column in 0..32 {
                let texel = row * 32 + column;
                if !baked.valid_mask[texel] {
                    continue;
                }
                let r = baked.base_color_rgba[texel * 4];
                if let Some(previous_r) = previous {
                    assert!(r >= previous_r, "row {row}: r decreased at column {column}");
                }
                previous = Some(r);
            }
        }

        // MR layout: R = 0 and A = 255 on EVERY texel (valid or inpainted).
        for texel in 0..32 * 32 {
            assert_eq!(0, baked.metallic_roughness[texel * 4]);
            assert_eq!(255, baked.metallic_roughness[texel * 4 + 3]);
        }
    }

    #[test]
    fn bake_is_deterministic() {
        let fixture = gradient_fixture();
        let first = fixture.bake(48).expect("first bake");
        let second = fixture.bake(48).expect("second bake");
        assert_eq!(first, second);
    }

    /// Full `resolution^3` sparse grid whose base color ramps with the voxel
    /// Z index (`0, 1/(R-1), ..., 1`), so the trilinear sample is a Z
    /// plateau independent of x/y: `0.5` at `z = 0`, `~0.8333` at
    /// `z = 0.25` for `R = 4`.
    fn z_ramp_grid(resolution: u32) -> (Vec<[i32; 3]>, Vec<f32>) {
        let resolution_i = resolution as i32;
        let mut coords = Vec::with_capacity((resolution * resolution * resolution) as usize);
        let mut attrs = Vec::with_capacity((resolution * resolution * resolution) as usize * 6);
        for z in 0..resolution_i {
            let value = z as f32 / (resolution_i - 1) as f32;
            for y in 0..resolution_i {
                for x in 0..resolution_i {
                    coords.push([x, y, z]);
                    attrs.extend([value, value, value, 0.0, 0.0, 1.0]);
                }
            }
        }
        (coords, attrs)
    }

    /// Flat quad in the plane `z = plane_z` spanning `[-0.4, 0.4]^2`, with
    /// the full `[0, 1]` UV square.
    fn plane_quad(plane_z: f32) -> (Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<u32>) {
        let vertices = vec![
            [-0.4, -0.4, plane_z],
            [0.4, -0.4, plane_z],
            [0.4, 0.4, plane_z],
            [-0.4, 0.4, plane_z],
        ];
        let uvs = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        (vertices, uvs, vec![0, 1, 2, 0, 2, 3])
    }

    /// Upstream `to_glb` snap-back (`postprocess.py:249-266`): a deliberately
    /// drifted export mesh must sample the volume at the projected
    /// ORIGINAL-surface position, not at its own drifted position.
    #[test]
    fn snap_back_projects_texels_onto_the_original_surface() {
        let resolution = 4_u32;
        let (coords, attrs) = z_ramp_grid(resolution);

        // Original (pre-postprocess) surface: the plane z = 0.
        let (original_vertices, _, original_faces) = plane_quad(0.0);
        let original_flat: Vec<f32> = original_vertices.iter().flatten().copied().collect();
        let bvh = MeshBvh::build(&original_flat, &original_faces).expect("build bvh");

        // Export mesh drifted +0.25 in Z (stand-in for the geometric error
        // introduced by simplification/remeshing).
        let (drifted_vertices, uvs, indices) = plane_quad(0.25);

        // The projection itself moves the sample position toward the
        // original surface: nearest point at z = 0, 0.25 away from the
        // drifted point.
        let hit = bvh.closest_point([0.0, 0.0, 0.25]);
        assert!((hit.unsigned_distance - 0.25).abs() <= 1e-6);
        assert!(
            hit.position[2].abs() <= 1e-6,
            "projection must land on the original plane, got {:?}",
            hit.position
        );

        let unprojected = bake_pbr_texture(
            &drifted_vertices,
            &uvs,
            &indices,
            &coords,
            &attrs,
            resolution,
            16,
        )
        .expect("unprojected bake");
        let snapped = bake_pbr_texture_with_snap(
            &drifted_vertices,
            &uvs,
            &indices,
            &bvh,
            &coords,
            &attrs,
            resolution,
            16,
        )
        .expect("snapped bake");
        let original = bake_pbr_texture(
            &original_vertices,
            &uvs,
            &indices,
            &coords,
            &attrs,
            resolution,
            16,
        )
        .expect("original bake");

        let mut covered = 0_usize;
        for texel in 0..16 * 16 {
            if !snapped.valid_mask[texel] {
                continue;
            }
            covered += 1;
            // z = 0 samples the ramp midpoint 0.5 -> floor(127.5) = 127.
            assert_eq!(
                127,
                snapped.base_color_rgba[texel * 4],
                "snapped texel {texel} must sample the original surface"
            );
            // The unprojected sample sits at z = 0.25 -> ~0.8333 -> >= 211.
            let unprojected_red = unprojected.base_color_rgba[texel * 4];
            assert!(
                (211..=213).contains(&unprojected_red),
                "unprojected texel {texel} red {unprojected_red} must sample the drifted plane"
            );
        }
        assert!(covered > 0, "the quad must cover texels");

        // Projection recovers exactly what baking the undrifted export mesh
        // produces: same coverage, samples and inpaint result.
        assert_eq!(original, snapped);
        // And the drift really is observable without the snap-back.
        assert_ne!(original.base_color_rgba, unprojected.base_color_rgba);
    }

    /// Snapping is a no-op when the export mesh already IS the original
    /// surface (the post-process-disabled path is not the only case: a
    /// pass-through export must not perturb samples).
    #[test]
    fn snap_back_on_the_original_mesh_keeps_the_bake_identical() {
        let resolution = 4_u32;
        let (coords, attrs) = z_ramp_grid(resolution);
        let (vertices, uvs, indices) = plane_quad(0.0);
        let flat: Vec<f32> = vertices.iter().flatten().copied().collect();
        let bvh = MeshBvh::build(&flat, &indices).expect("build bvh");

        let unprojected =
            bake_pbr_texture(&vertices, &uvs, &indices, &coords, &attrs, resolution, 16)
                .expect("unprojected bake");
        let snapped = bake_pbr_texture_with_snap(
            &vertices, &uvs, &indices, &bvh, &coords, &attrs, resolution, 16,
        )
        .expect("snapped bake");
        assert_eq!(unprojected, snapped);
    }

    #[test]
    fn bake_boundary_vertices_clamp_to_single_voxel() {
        // resolution 1, single voxel (0, 0, 0); quad corners EXACTLY on the
        // grid boundary (+-0.5). q = (v + 0.5) * 1 hits the grid edge and the
        // out-of-grid neighbors must be dropped, leaving the lone voxel at
        // full weight after renormalization (no zero/black color).
        let corners = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
        let fixture = BakeInputs {
            vertices: corners.iter().map(|uv| [uv[0], uv[1], 0.5]).collect(),
            uvs: corners
                .iter()
                .map(|uv| [uv[0] + 0.5, uv[1] + 0.5])
                .collect(),
            indices: vec![0, 1, 2, 0, 2, 3],
            coords: vec![[0, 0, 0]],
            attrs: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            resolution: 1,
        };
        let baked = fixture.bake(16).expect("bake");
        assert!(baked.stats.valid_texels > 0);
        for texel in 0..16 * 16 {
            if !baked.valid_mask[texel] {
                continue;
            }
            assert_eq!(
                [255, 0, 0, 255],
                baked.base_color_rgba[texel * 4..texel * 4 + 4]
            );
        }
    }

    #[test]
    fn bake_rejects_invalid_inputs_descriptively() {
        let fixture = gradient_fixture();
        let error = fixture.bake(0).expect_err("zero texture size");
        assert!(error.to_string().contains("texture_size"));

        let mut broken = gradient_fixture();
        broken.indices.push(0);
        let error = broken.bake(16).expect_err("bad index count");
        assert!(error.to_string().contains("multiple of 3"));

        let mut broken = gradient_fixture();
        broken.indices = vec![0, 1, 9];
        let error = broken.bake(16).expect_err("bad face index");
        assert!(error
            .to_string()
            .contains("outside the atlas vertex buffer"));

        let mut broken = gradient_fixture();
        broken.attrs.truncate(6);
        let error = broken.bake(16).expect_err("attr row mismatch");
        assert!(error
            .to_string()
            .contains("6 channels per voxel coordinate"));

        let mut broken = gradient_fixture();
        broken.coords.clear();
        let error = broken.bake(16).expect_err("empty voxel grid");
        assert!(error
            .to_string()
            .contains("non-empty sparse PBR voxel grid"));

        // Degenerate UV triangles cover nothing.
        let mut broken = gradient_fixture();
        broken.uvs = vec![[0.0, 0.0]; 4];
        let error = broken.bake(16).expect_err("zero coverage");
        assert!(error.to_string().contains("zero texels"));
    }

    #[test]
    fn inpaint_channel_fills_hole_from_ring_neighbors() {
        // Direct unit test of the inpaint primitive: a 9x9 plane, all 200
        // except a 3x3 hole in the middle; radius 1 fill must produce values
        // near the ring color (smooth, non-zero, no black seam).
        let size = 9usize;
        let mut buffer = vec![200_u8; size * size * 4];
        let mut mask = vec![true; size * size];
        for y in 3..6 {
            for x in 3..6 {
                let texel = y * size + x;
                mask[texel] = false;
                buffer[texel * 4] = 0;
            }
        }
        let filled = inpaint_channels(&mut buffer, &mask, size, &[0], 1);
        assert_eq!(9, filled);
        for y in 3..6 {
            for x in 3..6 {
                let value = buffer[(y * size + x) * 4];
                assert!(
                    value >= 190 && value <= 210,
                    "hole texel ({x},{y}) = {value}"
                );
            }
        }
        // Untouched valid pixels keep their exact value.
        assert_eq!(200, buffer[0]);
        assert_eq!(200, buffer[(8 * size + 8) * 4]);
    }

    /// Performance tap (NOT run by default — release build only):
    /// `cargo test -p lux3d-core --release --lib perf_unwrap_bake -- --ignored --nocapture`
    ///
    /// Generates an ~80k-triangle UV sphere in the reference frame, a full
    /// 64^3 six-channel voxel grid, and times the full texture-export stage
    /// (xatlas unwrap + bake + inpaint) at a 2048 atlas, printing the stage
    /// timings and bake statistics. Cost notes live in the module docs.
    #[test]
    #[ignore = "perf tap: run with --release --ignored --nocapture"]
    fn perf_unwrap_bake_2048() {
        let rings = 200usize;
        let segments = 200usize;
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        for ring in 0..=rings {
            let theta = std::f32::consts::PI * ring as f32 / rings as f32;
            for segment in 0..segments {
                let phi = 2.0 * std::f32::consts::PI * segment as f32 / segments as f32;
                vertices.push([
                    0.49 * theta.sin() * phi.cos(),
                    0.49 * theta.sin() * phi.sin(),
                    0.49 * theta.cos(),
                ]);
            }
        }
        let index = |ring: usize, segment: usize| {
            (ring % (rings + 1)) * segments + (segment % segments)
        } as u32;
        for ring in 0..rings {
            for segment in 0..segments {
                let a = index(ring, segment);
                let b = index(ring + 1, segment);
                let c = index(ring + 1, segment + 1);
                let d = index(ring, segment + 1);
                faces.extend([a, b, c, a, c, d]);
            }
        }
        let triangle_count = faces.len() / 3;
        assert_eq!(2 * rings * segments, triangle_count);

        // Full 64^3 grid with 6 channels (worst-case sampling cost).
        let resolution = 64_u32;
        let mut coords = Vec::new();
        let mut attrs = Vec::new();
        for z in 0..resolution as i32 {
            for y in 0..resolution as i32 {
                for x in 0..resolution as i32 {
                    coords.push([x, y, z]);
                    attrs.extend([
                        (x as f32 + 0.5) / resolution as f32,
                        (y as f32 + 0.5) / resolution as f32,
                        (z as f32 + 0.5) / resolution as f32,
                        0.5,
                        0.5,
                        1.0,
                    ]);
                }
            }
        }

        let unwrap_start = std::time::Instant::now();
        let atlas = uv_unwrap(&vertices, &faces, 2048).expect("unwrap");
        let unwrap_ms = unwrap_start.elapsed().as_millis();

        let bake_start = std::time::Instant::now();
        let baked = bake_pbr_texture(
            &atlas.vertices,
            &atlas.uvs,
            &atlas.indices,
            &coords,
            &attrs,
            resolution,
            2048,
        )
        .expect("bake");
        let bake_ms = bake_start.elapsed().as_millis();

        println!(
            "perf unwrap+bake 2048: triangles={triangle_count} charts={} atlases={} \
             unwrap_ms={unwrap_ms} bake_ms={bake_ms} valid={} inpainted={} total_ms={}",
            atlas.chart_count,
            atlas.atlas_count,
            baked.stats.valid_texels,
            baked.stats.inpainted_texels,
            unwrap_ms + bake_ms
        );
        assert!(baked.stats.valid_texels > 0);
    }
}
