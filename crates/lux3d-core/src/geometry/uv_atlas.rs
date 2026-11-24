//! UV atlas generation for the TRELLIS.2 textured-GLB workstream
//! (upstream `to_glb` `mesh.uv_unwrap(...)`, `postprocess.py:201-214`).
//!
//! The reference pipeline unwraps the cleaned mesh with cumesh
//! (`uv_unwrap(compute_charts_kwargs={threshold_cone_half_angle_rad: pi/2,
//! refine_iterations: 0, global_iterations: 1, smooth_strength: 1},
//! return_vmaps=True)`) and then rebuilds the mesh vertices with the atlas
//! vertex map: the output mesh duplicates vertices at UV seams and
//! `vmaps[new]` points back at the original vertex. This module produces the
//! same product with `xatlas-rs-v2` (vendored xatlas):
//!
//! * `uv_unwrap` returns [`UvAtlas`] — atlas UVs, the new->original vertex
//!   map (`vmapping`, the upstream `vmaps`), the re-indexed triangle buffer
//!   and the atlas vertex positions (original positions gathered through
//!   `vmapping`, so consumers never need to re-index themselves).
//!   xatlas emits vertex UVs in ATLAS TEXEL units (xatlas.h: "Not
//!   normalized - values are in Atlas width and height range"); they are
//!   normalized by the packed atlas dimensions here so [`UvAtlas::uvs`] is
//!   in `[0, 1]^2` like the upstream atlas UVs.
//! * Output is deterministic: xatlas is single-threaded with a fixed RNG
//!   seed, so identical input produces identical output (covered by a test).
//!
//! * Duplicate-position vertices are welded before charting: the TRELLIS.2
//!   postprocess runs `repair_non_manifold_edges`, which splits vertices
//!   along non-manifold edges and leaves several indices at the SAME
//!   position. xatlas treats those as disconnected islands, so a mesh with
//!   even a few thousand such splits explodes into thousands of one-triangle
//!   charts. Welding by position (relative epsilon `1e-6` of the bounding-box
//!   diagonal) restores the shared topology without moving any vertex; the
//!   atlas `vmapping` is composed with the weld map, so it still points at
//!   the ORIGINAL input vertex indices.
//!
//! Upstream option mapping (cumesh is not xatlas — its kwargs have no
//! one-to-one xatlas counterpart, so the closest settings are used and the
//! deviations are documented here):
//!
//! * `threshold_cone_half_angle_rad = pi/2` (90 deg) — the most permissive
//!   normal cone; xatlas has no explicit cone threshold and grows charts by
//!   its combined cost model. The xatlas default weights
//!   (`normal_deviation_weight = 2`, `max_cost = 2`) are used unchanged.
//! * `refine_iterations = 0`, `global_iterations = 1` — xatlas
//!   `ChartOptions::max_iterations = 1` (one chart-grow/seed pass, no
//!   refinement).
//! * `smooth_strength = 1` — no xatlas counterpart; xatlas' internal LSCM /
//!   piecewise parameterization plays that role. Not mapped.
//! * Packing: `PackOptions::resolution = texture_size` (default 2048, the
//!   upstream bake texture size), `texels_per_unit = 0` on the first pass
//!   (xatlas estimates the density), then a re-pack with that estimate made
//!   explicit so the atlas comes out at EXACTLY `texture_size` (the
//!   estimate pass can overshoot by a few percent); everything else xatlas
//!   defaults (`bilinear = true`, `padding = 0`; the seam gaps are filled by
//!   the TELEA-style inpaint of `geometry::pbr_bake`, radius 3, same as the
//!   reference `cv2.inpaint` pass).
//!
//! If xatlas needs more than one 2048x2048 atlas for very dense meshes, the
//! per-atlas UVs are merged vertically (`v = (v + atlas_index) /
//! atlas_count`) so the result stays a single `[0, 1]^2` atlas; the bake and
//! GLB stages work on the merged frame unchanged.
//!
//! xatlas keeps non-parameterizable (zero-area in 3D) input faces in its
//! output index buffer with sentinel vertices (`atlasIndex = -1`, uv
//! `(0, 0)`); [`uv_unwrap`] drops those faces from [`UvAtlas::indices`] (they
//! are degenerate — no visual coverage is lost — and leaving them in would
//! rasterize the whole texture corner). The atlas vertex array still carries
//! the sentinel vertices (their UVs are `(0, 0)`-mapped) so the index slots
//! stay aligned with the input faces; [`UvAtlas::indices`] can therefore be
//! slightly shorter than `3 * input faces`.

use std::collections::HashMap;

use super::mesh_ops::merge_close_vertices;
use crate::Result;
use crate::error::Lux3dError;
use xatlas_rs_v2::{ChartOptions, IndexData, MeshData, MeshDecl, PackOptions, Xatlas};

/// UV atlas product of [`uv_unwrap`] (the upstream
/// `out_vertices/out_faces/out_uvs/out_vmaps` quartet).
#[derive(Debug, Clone, PartialEq)]
pub struct UvAtlas {
    /// Atlas UVs per ATLAS vertex, flat `[A, 2]` rows in `[0, 1]`. Row order
    /// matches `vmapping` / `vertices`. These are the pre-flip atlas UVs the
    /// texture bake rasterizes (the GLB writer applies the upstream V-flip
    /// `v = 1 - v` at export time).
    pub uvs: Vec<[f32; 2]>,
    /// Atlas vertex -> ORIGINAL input vertex index (`new vertex -> original`,
    /// the upstream `vmaps`). Duplicated at UV seams.
    pub vmapping: Vec<u32>,
    /// Triangle indices into the ATLAS vertex array (flat `[F, 3]`).
    /// Slot-aligned with the input faces except that non-parameterizable
    /// (zero-area) input faces are dropped (see the module docs).
    pub indices: Vec<u32>,
    /// Atlas vertex positions (flat `[A, 3]`, reference frame) — the
    /// original positions gathered through `vmapping`.
    pub vertices: Vec<[f32; 3]>,
    /// Number of charts xatlas produced (parity/stat, `> 0` for a non-empty
    /// mesh).
    pub chart_count: u32,
    /// Number of packed atlases (1 unless the mesh needed more than one
    /// `texture_size` atlas; UVs are merged into a single frame either way).
    pub atlas_count: u32,
    /// Atlas edge length in texels the packing targeted (`texture_size`).
    pub atlas_size: u32,
}

/// Weld distance as a fraction of the mesh bounding-box diagonal. Exact
/// duplicate positions produced by topology repair are at distance 0; the
/// relative epsilon also catches float noise and is far below any feature
/// scale (a 1536^3 TRELLIS.2 grid cell is ~1e-3 of the diagonal).
const UV_WELD_RELATIVE_EPSILON: f32 = 1e-6;

/// Weld duplicate positions for charting.
///
/// Returns `(welded_vertices, welded_faces, weld_origin)` where
/// `weld_origin[welded_index]` is the ORIGINAL input vertex index that
/// supplied the representative position (used to keep [`UvAtlas::vmapping`]
/// in the original index space).
fn weld_uv_input(vertices: &[[f32; 3]], faces: &[u32]) -> (Vec<[f32; 3]>, Vec<u32>, Vec<u32>) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for vertex in vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    let diagonal =
        ((max[0] - min[0]).powi(2) + (max[1] - min[1]).powi(2) + (max[2] - min[2]).powi(2)).sqrt();
    // Huge-but-finite input coordinates can overflow the squared diagonal;
    // fall back to an absolute epsilon instead of merging the whole mesh.
    let epsilon = if diagonal.is_finite() && diagonal > 0.0 {
        (diagonal * UV_WELD_RELATIVE_EPSILON).max(1e-9)
    } else {
        1e-9
    };

    let mut flat = Vec::with_capacity(vertices.len() * 3);
    for vertex in vertices {
        flat.extend_from_slice(vertex);
    }
    let merged = merge_close_vertices(&flat, faces, epsilon);
    let welded = merged
        .vertices
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect();
    (welded, merged.faces, merged.vertex_remap)
}

/// Unwrap `vertices`/`faces` into a texture atlas sized `texture_size`
/// (upstream bake resolution, e.g. 2048). Returns the atlas mesh with
/// per-atlas-vertex UVs and the mapping back to the input vertices.
///
/// Errors (descriptive, no panics): empty face list, out-of-range or
/// non-finite input, or an xatlas indexing failure. Vertices must be in the
/// reference `[-0.5, 0.5]^3` frame (xatlas is frame-agnostic; the value only
/// matters for texel density, which is estimated against `texture_size`).
pub fn uv_unwrap(vertices: &[[f32; 3]], faces: &[u32], texture_size: u32) -> Result<UvAtlas> {
    if faces.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "UV unwrap requires at least one triangle",
        ));
    }
    if vertices.len() > u32::MAX as usize {
        return Err(Lux3dError::InvalidInput(
            "UV unwrap supports at most u32::MAX vertices",
        ));
    }
    if faces.len() % 3 != 0 {
        return Err(Lux3dError::InvalidInput(
            "UV unwrap face buffer length must be a multiple of 3",
        ));
    }
    let vertex_count = vertices.len();
    for &index in faces {
        if index as usize >= vertex_count {
            return Err(Lux3dError::InvalidInput(
                "UV unwrap rejected a face index outside the vertex buffer",
            ));
        }
    }
    if vertices.iter().any(|v| v.iter().any(|c| !c.is_finite())) {
        return Err(Lux3dError::InvalidInput(
            "UV unwrap rejected a non-finite vertex",
        ));
    }
    if texture_size == 0 || texture_size > 8192 {
        return Err(Lux3dError::InvalidInput(
            "UV unwrap texture_size must be in 1..=8192",
        ));
    }

    // Weld duplicate-position vertices before charting (see the module docs):
    // postprocess `repair_non_manifold_edges` leaves several indices at the
    // same position, which xatlas would otherwise treat as separate islands.
    let (welded_vertices, welded_faces, weld_origin) = weld_uv_input(vertices, faces);
    let welded_vertex_count = welded_vertices.len();

    // xatlas consumes the flat position buffer and the u32 index buffer; the
    // mesh declaration must be created BEFORE the atlas handle (the wrapper
    // ties the atlas's lifetime to the declaration borrow).
    let mut flat_positions = Vec::with_capacity(welded_vertices.len() * 3);
    for vertex in &welded_vertices {
        flat_positions.extend_from_slice(vertex);
    }
    let decl = MeshDecl {
        vertex_position_data: MeshData::Contiguous(&flat_positions),
        index_data: Some(IndexData::U32(&welded_faces)),
        face_count: (welded_faces.len() / 3) as u32,
        ..MeshDecl::default()
    };
    let mut atlas = Xatlas::new();
    atlas.add_mesh(&decl).map_err(|error| {
        Lux3dError::InvalidInput(match error {
            xatlas_rs_v2::AddMeshError::IndexOutOfRange => {
                "UV unwrap rejected a face index outside the vertex buffer (xatlas)"
            }
            xatlas_rs_v2::AddMeshError::InvalidIndexCount => {
                "UV unwrap needs a triangle index count divisible by 3 (xatlas)"
            }
            xatlas_rs_v2::AddMeshError::InvalidFaceVertexCount => {
                "UV unwrap rejected an invalid face vertex count (xatlas)"
            }
            xatlas_rs_v2::AddMeshError::Error => "xatlas failed to accept the mesh",
        })
    })?;
    atlas.add_mesh_join();

    // Upstream mapping (see module docs): one chart-grow/seed iteration, no
    // refinement; pack into a texture_size atlas with xatlas defaults.
    let chart_options = ChartOptions {
        max_iterations: 1,
        ..ChartOptions::default()
    };
    atlas.compute_charts(&chart_options);
    // Pack once with xatlas' density estimate (`texels_per_unit = 0`), then
    // re-pack with that estimate made EXPLICIT: with a non-zero density and
    // a non-zero resolution xatlas produces atlases of EXACTLY the requested
    // size (the estimate pass can overshoot by a few percent, while the
    // upstream bake targets exactly `texture_size`).
    atlas.pack_charts(&PackOptions {
        resolution: texture_size,
        texels_per_unit: 0.0,
        ..PackOptions::default()
    });
    let estimated_density = atlas.texels_per_unit();
    if estimated_density > 0.0 {
        atlas.pack_charts(&PackOptions {
            resolution: texture_size,
            texels_per_unit: estimated_density,
            ..PackOptions::default()
        });
    }

    let atlas_count = atlas.atlas_count().max(1);
    let chart_count = atlas.chart_count();
    if chart_count == 0 {
        return Err(Lux3dError::InvalidInput(
            "UV unwrap produced zero charts (degenerate mesh)",
        ));
    }

    let mut out = UvAtlas {
        uvs: Vec::new(),
        vmapping: Vec::new(),
        indices: Vec::new(),
        vertices: Vec::new(),
        chart_count,
        atlas_count,
        atlas_size: atlas.width().max(atlas.height()),
    };
    // Whether each output atlas vertex was placed in the atlas (vertices of
    // non-parameterizable faces carry atlasIndex = -1).
    let mut vertex_placed = Vec::with_capacity(1024);
    let atlas_count_f = atlas_count as f32;
    // xatlas emits vertex UVs in ATLAS TEXEL units (xatlas.h: "Not
    // normalized - values are in Atlas width and height range"), so they are
    // divided by the packed atlas dimensions here.
    let width_f = atlas.width().max(1) as f32;
    let height_f = atlas.height().max(1) as f32;
    for mesh in atlas.meshes() {
        let base = out.vmapping.len() as u32;
        for vertex in &mesh.vertex_array {
            let atlas_index = vertex.atlas_index;
            if atlas_index >= atlas_count as i32 {
                return Err(Lux3dError::InvalidInput(
                    "UV unwrap produced a vertex outside the packed atlas set",
                ));
            }
            let xref = vertex.xref;
            if xref as usize >= welded_vertex_count {
                return Err(Lux3dError::InvalidInput(
                    "UV unwrap produced a vertex map entry outside the welded vertex buffer",
                ));
            }
            // Compose the atlas map with the weld map so `vmapping` keeps the
            // documented atlas -> ORIGINAL input vertex semantics.
            let original = weld_origin[xref as usize];
            // atlasIndex = -1 marks vertices of non-parameterizable (zero
            // area in 3D) faces; xatlas emits them with uv (0, 0). They stay
            // in the vertex array so the index buffer stays aligned, but
            // their faces are dropped below (documented deviation).
            let placed = atlas_index >= 0;
            let (u, v) = if placed {
                (
                    (vertex.uv[0] / width_f).clamp(0.0, 1.0),
                    (vertex.uv[1] / height_f).clamp(0.0, 1.0),
                )
            } else {
                (0.0, 0.0)
            };
            out.uvs
                .push([u, (v + atlas_index.max(0) as f32) / atlas_count_f]);
            out.vmapping.push(original);
            out.vertices.push(welded_vertices[xref as usize]);
            vertex_placed.push(placed);
        }
        // Keep only the parameterized faces (see the atlasIndex note above);
        // xatlas aligns the output index slots with the input faces, so a
        // dropped slot is exactly one degenerate input triangle.
        for face in mesh.index_array.chunks_exact(3) {
            if vertex_placed[base as usize + face[0] as usize]
                && vertex_placed[base as usize + face[1] as usize]
                && vertex_placed[base as usize + face[2] as usize]
            {
                for &index in face {
                    out.indices.push(base + index);
                }
            }
        }
    }

    if out.indices.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "UV unwrap produced an empty atlas mesh",
        ));
    }
    for &index in &out.indices {
        if index as usize >= out.vmapping.len() {
            return Err(Lux3dError::InvalidInput(
                "UV unwrap produced an atlas index outside the atlas vertex buffer",
            ));
        }
    }
    for uv in &out.uvs {
        if uv.iter().any(|c| !c.is_finite()) {
            return Err(Lux3dError::InvalidInput(
                "UV unwrap produced a non-finite UV",
            ));
        }
    }
    Ok(out)
}

/// Area-weighted vertex normals (upstream `mesh.compute_vertex_normals()` +
/// `normals = mesh.read_vertex_normals()[vmaps]`, `postprocess.py:215-216`):
/// each face accumulates its unnormalized cross-product normal
/// (`2 * area * face_normal`) onto its three vertices and the sums are
/// normalized. Degenerate faces contribute zero; a vertex touched only by
/// degenerate faces gets a zero normal. Deterministic (face order
/// summation, same as the reference trimesh/cumesh accumulation).
pub fn area_weighted_vertex_normals(vertices: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0_f32; 3]; vertices.len()];
    for face in indices.chunks_exact(3) {
        let (a, b, c) = (
            vertices[face[0] as usize],
            vertices[face[1] as usize],
            vertices[face[2] as usize],
        );
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let cross = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        for &vertex_index in face {
            let normal = &mut normals[vertex_index as usize];
            normal[0] += cross[0];
            normal[1] += cross[1];
            normal[2] += cross[2];
        }
    }
    for normal in &mut normals {
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if length > 0.0 && length.is_finite() {
            *normal = [normal[0] / length, normal[1] / length, normal[2] / length];
        } else {
            *normal = [0.0; 3];
        }
    }
    normals
}

/// Area-weighted vertex normals computed on POSITION-WELDED topology, then
/// scattered back to every input vertex copy.
///
/// `area_weighted_vertex_normals` averages per input index. After
/// `repair_non_manifold_edges` the same position can exist as several index
/// copies belonging to different sheets; the atlas re-welds those positions
/// for charting (`UvAtlas::vmapping` maps every copy onto one representative),
/// so a per-copy normal would attach the representative copy's sheet normal
/// to every chart at that position. Welding first makes the normal a property
/// of the position, which is what the unwrapped atlas actually samples.
pub fn welded_vertex_normals(vertices: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut position_to_weld: HashMap<[u32; 3], u32> = HashMap::new();
    let mut weld_of: Vec<u32> = Vec::with_capacity(vertices.len());
    let mut weld_positions: Vec<[f32; 3]> = Vec::new();
    for &vertex in vertices {
        let key = [
            vertex[0].to_bits(),
            vertex[1].to_bits(),
            vertex[2].to_bits(),
        ];
        let id = *position_to_weld.entry(key).or_insert_with(|| {
            let id = weld_positions.len() as u32;
            weld_positions.push(vertex);
            id
        });
        weld_of.push(id);
    }
    let mut weld_normals = vec![[0.0_f32; 3]; weld_positions.len()];
    for face in indices.chunks_exact(3) {
        let (a, b, c) = (face[0] as usize, face[1] as usize, face[2] as usize);
        if a >= vertices.len() || b >= vertices.len() || c >= vertices.len() {
            continue;
        }
        let wa = weld_of[a] as usize;
        let wb = weld_of[b] as usize;
        let wc = weld_of[c] as usize;
        let pa = weld_positions[wa];
        let pb = weld_positions[wb];
        let pc = weld_positions[wc];
        let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let cross = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        for welded in [wa, wb, wc] {
            let normal = &mut weld_normals[welded];
            normal[0] += cross[0];
            normal[1] += cross[1];
            normal[2] += cross[2];
        }
    }
    for normal in &mut weld_normals {
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if length > 0.0 && length.is_finite() {
            *normal = [normal[0] / length, normal[1] / length, normal[2] / length];
        } else {
            *normal = [0.0; 3];
        }
    }
    weld_of
        .iter()
        .map(|&id| weld_normals[id as usize])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit cube: 8 vertices, 12 triangles, outward winding.
    fn cube() -> (Vec<[f32; 3]>, Vec<u32>) {
        let vertices: Vec<[f32; 3]> = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
        .into_iter()
        .collect();
        let faces: Vec<u32> = [
            [0, 2, 1],
            [0, 3, 2], // bottom (-z)
            [4, 5, 6],
            [4, 6, 7], // top (+z)
            [0, 1, 5],
            [0, 5, 4], // front (-y)
            [2, 3, 7],
            [2, 7, 6], // back (+y)
            [0, 4, 7],
            [0, 7, 3], // left (-x)
            [1, 2, 6],
            [1, 6, 5], // right (+x)
        ]
        .iter()
        .flatten()
        .copied()
        .collect();
        (vertices, faces)
    }

    #[test]
    fn welded_vertex_normals_share_one_normal_per_position() {
        // Two sheets meeting at the same position (duplicate index copies):
        // one triangle in the XY plane (normal +Z), one in the XZ plane
        // (normal -Y). Every copy at the shared position must receive the
        // SAME welded (normalized sum) normal.
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            // copies of the shared origin and of (1, 0, 0), plus a new point
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let indices = vec![0, 1, 2, 3, 4, 5];
        let normals = welded_vertex_normals(&vertices, &indices);
        assert_eq!(normals.len(), vertices.len());
        let expected = {
            let sum = [0.0_f32, -1.0, 1.0];
            let len = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
            [sum[0] / len, sum[1] / len, sum[2] / len]
        };
        for &(a, b) in &[(0, 3), (1, 4)] {
            assert!((normals[a][0] - expected[0]).abs() < 1e-6);
            assert!((normals[a][1] - expected[1]).abs() < 1e-6);
            assert!((normals[a][2] - expected[2]).abs() < 1e-6);
            assert_eq!(normals[a], normals[b]);
        }
        for normal in &normals {
            let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])
                .sqrt();
            assert!((length - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn uv_unwrap_cube_produces_valid_atlas() {
        let (vertices, faces) = cube();
        let atlas = uv_unwrap(&vertices, &faces, 2048).expect("cube unwrap");

        // Charts were produced and the atlas came out at the requested size
        // (the second pack pass pins the density estimate to texture_size).
        assert!(atlas.chart_count > 0, "cube must produce charts");
        assert_eq!(2048, atlas.atlas_size, "atlas must be exactly texture_size");

        // Atlas UVs are finite and inside [0, 1].
        assert!(!atlas.uvs.is_empty());
        for uv in &atlas.uvs {
            assert!(uv[0] >= 0.0 && uv[0] <= 1.0);
            assert!(uv[1] >= 0.0 && uv[1] <= 1.0);
        }
        // The UVs must SPREAD over the atlas: a collapsed cluster would mean
        // xatlas' texel-unit UVs were not normalized by the atlas size.
        let mut min_uv = [f32::MAX; 2];
        let mut max_uv = [0.0_f32; 2];
        for uv in &atlas.uvs {
            min_uv[0] = min_uv[0].min(uv[0]);
            min_uv[1] = min_uv[1].min(uv[1]);
            max_uv[0] = max_uv[0].max(uv[0]);
            max_uv[1] = max_uv[1].max(uv[1]);
        }
        assert!(
            max_uv[0] - min_uv[0] > 0.25 && max_uv[1] - min_uv[1] > 0.25,
            "cube UVs must cover a meaningful atlas region, got {min_uv:?}..{max_uv:?}"
        );

        // Indices are in range and every input triangle is represented.
        let atlas_vertices = atlas.vmapping.len();
        assert_eq!(0, atlas.indices.len() % 3);
        assert!(atlas.indices.len() >= faces.len(), "atlas keeps all faces");
        for &index in &atlas.indices {
            assert!((index as usize) < atlas_vertices);
        }

        // Every atlas vertex maps back into the input vertex buffer and
        // carries the original position.
        for (atlas_vertex, &original) in atlas.vertices.iter().zip(atlas.vmapping.iter()) {
            assert_eq!(&vertices[original as usize], atlas_vertex);
        }

        // The atlas mesh validates with the shared area-weighted normals.
        let normals = area_weighted_vertex_normals(&atlas.vertices, &atlas.indices);
        assert_eq!(atlas.vmapping.len(), normals.len());
        for normal in &normals {
            let length =
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            assert!(length <= 1.0 + 1e-5);
        }
    }

    #[test]
    fn uv_unwrap_is_deterministic() {
        let (vertices, faces) = cube();
        let first = uv_unwrap(&vertices, &faces, 2048).expect("first unwrap");
        let second = uv_unwrap(&vertices, &faces, 2048).expect("second unwrap");
        assert_eq!(first, second);
    }

    #[test]
    fn uv_unwrap_seam_duplicates_map_to_originals() {
        let (vertices, faces) = cube();
        let atlas = uv_unwrap(&vertices, &faces, 512).expect("cube unwrap");
        // A cube needs seam duplication: the atlas vertex count exceeds the
        // 8 input vertices, and every original vertex is referenced.
        assert!(atlas.vmapping.len() > vertices.len());
        let mut referenced = vec![false; vertices.len()];
        for &original in &atlas.vmapping {
            referenced[original as usize] = true;
        }
        assert!(referenced.iter().all(|&seen| seen), "all corners used");
    }

    /// `repair_non_manifold_edges` emits duplicate indices at the SAME
    /// position; charting must see the welded topology, but `vmapping` must
    /// still reference the original (pre-weld) index space.
    #[test]
    fn uv_unwrap_welds_identical_positions_but_keeps_original_vmapping() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            // Half B duplicates the shared edge vertices at identical
            // positions with new indices.
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let faces = [0, 1, 2, 3, 4, 5];
        let atlas = uv_unwrap(&vertices, &faces, 256).expect("welded unwrap");

        let mut originals: Vec<u32> = atlas.vmapping.clone();
        originals.sort_unstable();
        originals.dedup();
        assert_eq!(
            originals,
            vec![0, 1, 2, 5],
            "duplicate indices 3/4 must map back onto originals 1/2"
        );
        assert_eq!(6, atlas.indices.len(), "both triangles survive");
        for (atlas_vertex, &original) in atlas.vertices.iter().zip(atlas.vmapping.iter()) {
            assert_eq!(&vertices[original as usize], atlas_vertex);
        }
    }

    /// A triangle-soup mesh (every face has its own vertex copies at the SAME
    /// grid positions) is welded back into one connected sheet, so the chart
    /// count stays small instead of one-chart-per-triangle. This is the
    /// failure mode of a mesh that went through non-manifold repair.
    #[test]
    fn uv_unwrap_welds_triangle_soup_into_one_sheet() {
        const SIDE: u32 = 10;
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        let position = |x: u32, y: u32| [x as f32, y as f32, 0.0];
        for y in 0..SIDE {
            for x in 0..SIDE {
                for corners in [
                    [(x, y), (x + 1, y), (x + 1, y + 1)],
                    [(x, y), (x + 1, y + 1), (x, y + 1)],
                ] {
                    let base = vertices.len() as u32;
                    for (cx, cy) in corners {
                        vertices.push(position(cx, cy));
                    }
                    faces.extend_from_slice(&[base, base + 1, base + 2]);
                }
            }
        }
        assert_eq!(SIDE as usize * SIDE as usize * 2, faces.len() / 3);
        assert_eq!(faces.len(), vertices.len());

        let atlas = uv_unwrap(&vertices, &faces, 128).expect("soup unwrap");
        assert!(
            atlas.chart_count <= 4,
            "welded sheet must not fragment into {} charts",
            atlas.chart_count
        );
        assert_eq!(SIDE as usize * SIDE as usize * 2, atlas.indices.len() / 3);
    }

    #[test]
    fn uv_unwrap_rejects_empty_and_invalid_meshes() {
        let (vertices, faces) = cube();
        let error = uv_unwrap(&vertices, &[], 2048).expect_err("empty mesh must fail");
        assert!(error.to_string().contains("at least one triangle"));

        let error = uv_unwrap(&vertices, &[0, 1, 8], 2048).expect_err("out of range");
        assert!(error.to_string().contains("outside the vertex buffer"));

        let mut non_finite = vertices.clone();
        non_finite[0] = [f32::NAN; 3];
        let error = uv_unwrap(&non_finite, &faces, 2048).expect_err("non-finite");
        assert!(error.to_string().contains("non-finite"));

        let error = uv_unwrap(&vertices, &faces, 0).expect_err("zero texture size");
        assert!(error.to_string().contains("texture_size"));
    }

    #[test]
    fn vertex_normals_match_upstream_area_weighting() {
        // Two triangles forming a quad in the XY plane facing +Z: both face
        // normals are +Z, so the shared-edge vertices get +Z normals too.
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = [0, 1, 2, 0, 2, 3];
        let normals = area_weighted_vertex_normals(&vertices, &indices);
        assert_eq!(4, normals.len());
        for normal in &normals {
            assert!(
                (normal[2] - 1.0).abs() < 1e-6,
                "all normals +Z, got {normal:?}"
            );
            assert!(normal[0].abs() < 1e-6 && normal[1].abs() < 1e-6);
        }

        // Degenerate faces contribute nothing and a lone vertex stays zero.
        let vertices = vec![[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let normals = area_weighted_vertex_normals(&vertices, &[0, 1, 2]);
        assert_eq!([[0.0; 3]; 3], normals[..]);
    }
}
