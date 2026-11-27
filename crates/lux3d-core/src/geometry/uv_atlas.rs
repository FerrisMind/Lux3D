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
//!   diagonal, same clustering as `geometry::mesh_ops::merge_close_vertices`)
//!   restores the shared topology without moving any vertex; the atlas
//!   `vmapping` is composed with the weld map, so it still points at the
//!   ORIGINAL input vertex indices. Unlike that utility, welding here keeps
//!   EVERY face: corners that weld together simply make the face degenerate,
//!   which xatlas ignores while keeping its face slot aligned.
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
//! * Packing mirrors the reference `uv_bake.cpp:1138-1183`: ONE stock
//!   `PackOptions` pass (`resolution = 0`, `texels_per_unit = 0`,
//!   `padding = 0`, `bilinear = true`, `block_align = false`). xatlas grows
//!   a single atlas at its own density estimate (it targets a ~1024^2 atlas
//!   and keeps the tight used extents), and the texel-space UVs xatlas emits
//!   are divided by the final atlas width/height (`uv_bake.cpp:1192`), so
//!   the charts always span the full `[0, 1]^2` frame regardless of the
//!   packer's internal resolution. Density/occupancy at the bake size come
//!   from that normalization, not from packing at the bake resolution. Seam
//!   gaps at padding 0 are filled by the TELEA-style inpaint of
//!   `geometry::pbr_bake` (radius 3, same as the reference `cv2.inpaint`
//!   pass). `texture_size` stays in the signature for API compatibility; it
//!   is only the fallback atlas size for a fully degenerate mesh (no charts)
//!   and no longer caps the packing resolution.
//!
//! * Every input face survives: [`UvAtlas::indices`] is EXACTLY `3 * input
//!   face count` long and slot-aligned with the input faces, with
//!   [`UvAtlas::face_ids`] recording which input face each output triangle
//!   came from (a permutation of `0..input_face_count`). Faces xatlas cannot
//!   parameterize (zero 3D area, collapsed by vertex welding, or a failed
//!   chart) are kept POINT-COLLAPSED: three fresh atlas vertices at the
//!   face's original positions share one UV — a charted twin at the same
//!   position when one exists, `(0, 0)` otherwise — so they sample a single
//!   texel of a neighboring chart instead of painting atlas-spanning wedges
//!   (reference `uv_bake.cpp`: "uncharted faces kept with point-collapsed
//!   UVs"). Collapsing them in UV space is why they cannot be dropped:
//!   dropping opens welded-boundary cracks.
//!
//! Fallback detector: if a stock pack ever returns more than one atlas
//! (not expected with `resolution = 0`, which grows an uncapped single
//! atlas), the per-atlas UVs are merged vertically as before
//! (`v = (v + atlas_index) / atlas_count`) so the result stays a single
//! `[0, 1]^2` frame, [`UvAtlas::clusters_fell_back`] is set and a warning is
//! printed. The bake and GLB stages work on the merged frame unchanged.

use std::collections::HashMap;

use crate::Result;
use crate::error::Lux3dError;
use xatlas_rs_v2::{ChartOptions, ChartType, IndexData, MeshData, MeshDecl, PackOptions, Xatlas};

/// UV atlas product of [`uv_unwrap`] (the upstream
/// `out_vertices/out_faces/out_uvs/out_vmaps` quartet, plus face provenance).
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
    ///
    /// EXACTLY one output triangle per input face, in input face order:
    /// `indices.len() == 3 * input_face_count`. Input faces xatlas could not
    /// parameterize (zero 3D area, collapsed by welding, or a failed chart)
    /// are kept point-collapsed onto one UV (a charted twin at the same
    /// position when one exists, `(0, 0)` otherwise) instead of being
    /// dropped, so the mesh topology never loses faces (see the module docs).
    pub indices: Vec<u32>,
    /// Atlas vertex positions (flat `[A, 3]`, reference frame) — the
    /// original positions gathered through `vmapping`.
    pub vertices: Vec<[f32; 3]>,
    /// Number of charts xatlas produced (parity/stat, `> 0` for any
    /// non-degenerate mesh).
    pub chart_count: u32,
    /// Output triangle slot -> input face index. `indices.len() / 3` entries,
    /// a permutation of `0..input_face_count` (input order on the
    /// single-mesh path). Assets that carry per-face data (colors, material
    /// ids) ride through the atlas re-indexing with this map.
    pub face_ids: Vec<u32>,
    /// Number of input faces kept but point-collapsed (no chart / invalid
    /// chart). They occupy real triangles in `indices` with zero UV area.
    pub uncharted_face_count: u32,
    /// Number of chart clusters the atlas was built from. This leaf unpacks
    /// ONE global cluster, so it is always 1; the field exists so the
    /// clustering stage (`precluster`, L4) can report its own count without
    /// an API break.
    pub cluster_count: u32,
    /// True iff the stock single-atlas pack returned more than one atlas and
    /// the per-atlas UVs were merged vertically. `false` on the normal path.
    pub clusters_fell_back: bool,
    /// Number of packed atlases: 1 for the stock single-atlas pack; `> 1`
    /// only on the `clusters_fell_back` fallback path.
    pub atlas_count: u32,
    /// Atlas edge length in texels (`max(width, height)` of the packed
    /// atlas). With the stock pack this is the density-estimated atlas size
    /// (roughly 1024 for a typical mesh), NOT `texture_size`.
    pub atlas_size: u32,
}

/// Weld distance as a fraction of the mesh bounding-box diagonal. Exact
/// duplicate positions produced by topology repair are at distance 0; the
/// relative epsilon also catches float noise and is far below any feature
/// scale (a 1536^3 TRELLIS.2 grid cell is ~1e-3 of the diagonal).
const UV_WELD_RELATIVE_EPSILON: f32 = 1e-6;

/// Weld distance for `vertices` (absolute, world units).
fn weld_epsilon(vertices: &[[f32; 3]]) -> f32 {
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
    if diagonal.is_finite() && diagonal > 0.0 {
        (diagonal * UV_WELD_RELATIVE_EPSILON).max(1e-9)
    } else {
        1e-9
    }
}

/// Spatial-hash cell of `point` (same grid as
/// `geometry::mesh_ops::merge_close_vertices`, which this weld mirrors).
fn weld_hash_cell(point: [f32; 3], cell_size: f32) -> [i64; 3] {
    [
        (point[0] / cell_size).floor() as i64,
        (point[1] / cell_size).floor() as i64,
        (point[2] / cell_size).floor() as i64,
    ]
}

fn weld_dist_sq(a: [f32; 3], b: [f32; 3]) -> f32 {
    let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    dx * dx + dy * dy + dz * dz
}

/// Weld duplicate positions for charting.
///
/// Clustering is identical to `geometry::mesh_ops::merge_close_vertices`
/// (input-index order, first representative within `epsilon` wins, spatial
/// hash with cell size `epsilon`), but *every face is kept*: the face buffer
/// is rewritten through the weld map without dropping corners that collapsed
/// onto the same representative. Those faces become degenerate and xatlas
/// ignores them while keeping their face slot aligned with the input, which
/// is what [`UvAtlas::face_ids`] relies on.
///
/// Returns `(welded_vertices, welded_faces, weld_origin)` where
/// `welded_faces.len() == faces.len()`,
/// `weld_origin[welded_index]` is the ORIGINAL input vertex index that
/// supplied the representative position (used to keep [`UvAtlas::vmapping`]
/// in the original index space).
fn weld_uv_input(vertices: &[[f32; 3]], faces: &[u32]) -> (Vec<[f32; 3]>, Vec<u32>, Vec<u32>) {
    let epsilon = weld_epsilon(vertices);
    let epsilon_sq = epsilon * epsilon;
    // Spatial hash: cell key -> kept representative indices (insertion order).
    let mut cells: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
    let mut welded: Vec<[f32; 3]> = Vec::with_capacity(vertices.len());
    let mut weld_origin: Vec<u32> = Vec::with_capacity(vertices.len());
    // Original input vertex index -> welded representative index.
    let mut weld_of: Vec<u32> = Vec::with_capacity(vertices.len());
    for (index, &point) in vertices.iter().enumerate() {
        let cell = weld_hash_cell(point, epsilon);
        // First kept representative within epsilon wins (index-order scan).
        let mut target: Option<u32> = None;
        if point.iter().all(|value| value.is_finite()) {
            'search: for dx in -1_i64..=1 {
                for dy in -1_i64..=1 {
                    for dz in -1_i64..=1 {
                        let key = [
                            cell[0].saturating_add(dx),
                            cell[1].saturating_add(dy),
                            cell[2].saturating_add(dz),
                        ];
                        let Some(reps) = cells.get(&key) else {
                            continue;
                        };
                        for &rep in reps {
                            if weld_dist_sq(point, welded[rep as usize]) <= epsilon_sq {
                                target = Some(rep);
                                break 'search;
                            }
                        }
                    }
                }
            }
        }
        let target = match target {
            Some(target) => target,
            None => {
                let target = welded.len() as u32;
                welded.push(point);
                weld_origin.push(index as u32);
                cells.entry(cell).or_default().push(target);
                target
            }
        };
        weld_of.push(target);
    }
    let welded_faces: Vec<u32> = faces.iter().map(|&index| weld_of[index as usize]).collect();
    (welded, welded_faces, weld_origin)
}

/// Unwrap `vertices`/`faces` into ONE atlas with stock xatlas packing
/// (reference `uv_bake.cpp:1138-1183`). Returns the atlas mesh with
/// per-atlas-vertex UVs, the mapping back to the input vertices, and the
/// per-triangle input-face provenance (`face_ids`).
///
/// `texture_size` is kept for API compatibility: the atlas is packed at the
/// packer's own density estimate (`resolution = 0`) and the UVs are
/// normalized to `[0, 1]`, so the bake resolution no longer influences the
/// unwrap. It is only used as the reported `atlas_size` when the mesh has no
/// parameterizable face at all (fully degenerate input).
///
/// Errors (descriptive, no panics): empty face list, out-of-range or
/// non-finite input, an invalid `texture_size`, or an xatlas indexing
/// failure. Vertices must be in the reference `[-0.5, 0.5]^3` frame (xatlas
/// is frame-agnostic; the value only matters for texel density, which is
/// estimated against the atlas itself).
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
    // Welding keeps every face (corners that weld together only make the face
    // degenerate, and xatlas ignores those in place), so welded face slots
    // stay 1:1 with the input faces.
    let (welded_vertices, welded_faces, weld_origin) = weld_uv_input(vertices, faces);
    let welded_vertex_count = welded_vertices.len();
    let input_face_count = faces.len() / 3;
    debug_assert_eq!(faces.len(), welded_faces.len());

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
    // refinement (xatlas' internal LSCM/piecewise parameterization plays the
    // role of cumesh's `smooth_strength`).
    let chart_options = ChartOptions {
        max_iterations: 1,
        ..ChartOptions::default()
    };
    atlas.compute_charts(&chart_options);

    // Reference packing (`uv_bake.cpp:1138-1183`): ONE stock `PackOptions`
    // pass. With `resolution = 0` and `texels_per_unit = 0` xatlas grows a
    // single atlas at its own density estimate (targeting ~1024^2, limited
    // only by the charts' tight used extents), so `atlas_count` is 1 and the
    // atlas is never split across the bake resolution. The texel-space UVs
    // are normalized below (`uv_bake.cpp:1192`), which is what makes the
    // charts span the full [0,1]^2 frame regardless of the packer size.
    atlas.pack_charts(&PackOptions {
        resolution: 0,
        texels_per_unit: 0.0,
        padding: 0,
        bilinear: true,
        block_align: false,
        ..PackOptions::default()
    });

    let chart_count = atlas.chart_count();
    let packed_width = atlas.width();
    let packed_height = atlas.height();
    // Fully degenerate input (every face ignored, zero charts): nothing was
    // packed; all faces are still emitted point-collapsed below and the
    // requested texture_size is the reported atlas size.
    let atlas_size = if packed_width > 0 && packed_height > 0 {
        packed_width.max(packed_height)
    } else {
        texture_size
    };
    // Fallback detector: a multi-atlas result is not expected with the stock
    // pack. If it ever happens, keep the pre-Change-A behavior (vertical
    // merge of the per-atlas UV frames) and say so.
    let mut atlas_count = 1;
    let mut clusters_fell_back = false;
    if atlas.atlas_count() > 1 {
        atlas_count = atlas.atlas_count();
        clusters_fell_back = true;
        eprintln!(
            "uv_atlas: xatlas packed {atlas_count} atlases with stock PackOptions; \
             fell back to multi-atlas (vertical UV merge)"
        );
    }

    let mut out = UvAtlas {
        uvs: Vec::new(),
        vmapping: Vec::new(),
        indices: Vec::new(),
        vertices: Vec::new(),
        chart_count,
        face_ids: Vec::with_capacity(input_face_count),
        uncharted_face_count: 0,
        cluster_count: 1,
        clusters_fell_back,
        atlas_count,
        atlas_size,
    };
    if chart_count == 0 {
        // Fully degenerate input: xatlas produced no chart at all, so every
        // face is emitted point-collapsed onto the texture corner. Handling
        // this before `meshes()` also avoids the xatlas wrapper's null
        // chart-array slice on the zero-chart path.
        for (face_index, face) in welded_faces.chunks_exact(3).enumerate() {
            for &welded in face {
                let welded = welded as usize;
                out.indices.push(out.uvs.len() as u32);
                out.uvs.push([0.0, 0.0]);
                out.vmapping.push(weld_origin[welded]);
                out.vertices.push(welded_vertices[welded]);
            }
            out.uncharted_face_count += 1;
            out.face_ids.push(face_index as u32);
        }
        validate_uv_atlas(&out, input_face_count)?;
        return Ok(out);
    }
    // xatlas emits vertex UVs in ATLAS TEXEL units (xatlas.h: "Not
    // normalized - values are in Atlas width and height range"), so they are
    // divided by the packed atlas dimensions here.
    let width_f = packed_width.max(1) as f32;
    let height_f = packed_height.max(1) as f32;
    let atlas_count_f = atlas_count.max(1) as f32;
    // Normalize one atlas vertex UV into the `[0, 1]^2` frame. On the
    // multi-atlas fallback the per-atlas frames are stacked vertically
    // (pre-Change-A behavior).
    let normalize = |uv: [f32; 2], atlas_index: u32| -> [f32; 2] {
        let u = (uv[0] / width_f).clamp(0.0, 1.0);
        let v = (uv[1] / height_f).clamp(0.0, 1.0);
        if clusters_fell_back {
            [
                u,
                ((v + atlas_index as f32) / atlas_count_f).clamp(0.0, 1.0),
            ]
        } else {
            [u, v]
        }
    };
    let meshes = atlas.meshes();
    if meshes.len() != 1 {
        return Err(Lux3dError::InvalidInput(
            "UV unwrap expected exactly one output mesh (xatlas join)",
        ));
    }
    for mesh in &meshes {
        let base = out.vmapping.len() as u32;
        // Twin UV per welded position: the first charted copy of a position.
        // Uncharted faces collapse onto it so they sample a neighboring
        // chart's texel instead of the texture corner.
        let mut twin: HashMap<u32, [f32; 2]> = HashMap::with_capacity(mesh.vertex_array.len());
        for vertex in &mesh.vertex_array {
            if vertex.atlas_index < 0 {
                continue;
            }
            let atlas_index = vertex.atlas_index as u32;
            if atlas_index >= atlas_count {
                return Err(Lux3dError::InvalidInput(
                    "UV unwrap produced a vertex outside the packed atlas set",
                ));
            }
            twin.entry(vertex.xref)
                .or_insert_with(|| normalize(vertex.uv, atlas_index));
        }
        for vertex in &mesh.vertex_array {
            let xref = vertex.xref as usize;
            if xref >= welded_vertex_count {
                return Err(Lux3dError::InvalidInput(
                    "UV unwrap produced a vertex map entry outside the welded vertex buffer",
                ));
            }
            // atlasIndex = -1 marks vertices of non-parameterizable (zero
            // area in 3D) faces and failed charts; xatlas emits them with uv
            // (0, 0). They stay in the vertex array so the index buffer stays
            // aligned; their faces are re-emitted point-collapsed below.
            let uv = if vertex.atlas_index >= 0 {
                normalize(vertex.uv, vertex.atlas_index as u32)
            } else {
                twin.get(&vertex.xref).copied().unwrap_or([0.0, 0.0])
            };
            out.uvs.push(uv);
            out.vmapping.push(weld_origin[xref]);
            out.vertices.push(welded_vertices[xref]);
        }

        let face_count = mesh.index_array.len() / 3;
        if mesh.index_array.len() % 3 != 0 || face_count != input_face_count {
            // xatlas keeps one output face slot per input face (ignored faces
            // live in its invalid-geometry buffer, written back at the same
            // slot); a mismatch would silently desync `face_ids` from the
            // input faces.
            return Err(Lux3dError::InvalidInput(
                "UV unwrap changed the input triangle count (xatlas face slots are not aligned)",
            ));
        }
        // Faces owned by a real (non-Invalid) chart are parameterized; every
        // other slot is a face xatlas ignored (zero area, degenerate or
        // zero-length edges) or put in a failed chart.
        let mut charted = vec![false; face_count];
        for chart in &mesh.chart_array {
            if matches!(chart.type_, ChartType::Invalid) {
                continue;
            }
            for &face in chart.face_array {
                if face as usize >= face_count {
                    return Err(Lux3dError::InvalidInput(
                        "UV unwrap produced a chart face outside the face buffer",
                    ));
                }
                charted[face as usize] = true;
            }
        }
        for face_index in 0..face_count {
            let face = &mesh.index_array[face_index * 3..face_index * 3 + 3];
            if charted[face_index] {
                for &index in face {
                    out.indices.push(base + index);
                }
            } else {
                // Keep the face, POINT-COLLAPSED onto one UV: a charted twin
                // at the same position when one exists, (0, 0) otherwise
                // (reference `uv_bake.cpp`: "uncharted faces kept with
                // point-collapsed UVs"). Fresh atlas vertices are emitted so
                // the collapse cannot stretch any chart's UVs.
                let mut collapsed = [0.0_f32, 0.0];
                for &index in face {
                    let vertex =
                        mesh.vertex_array
                            .get(index as usize)
                            .ok_or(Lux3dError::InvalidInput(
                                "UV unwrap produced an atlas index outside the atlas vertex buffer",
                            ))?;
                    if let Some(uv) = twin.get(&vertex.xref) {
                        collapsed = *uv;
                        break;
                    }
                }
                for &index in face {
                    let vertex = &mesh.vertex_array[index as usize];
                    let xref = vertex.xref as usize;
                    if xref >= welded_vertex_count {
                        return Err(Lux3dError::InvalidInput(
                            "UV unwrap produced a vertex map entry outside the welded vertex buffer",
                        ));
                    }
                    out.indices.push(out.vmapping.len() as u32);
                    out.uvs.push(collapsed);
                    out.vmapping.push(weld_origin[xref]);
                    out.vertices.push(welded_vertices[xref]);
                }
                out.uncharted_face_count += 1;
            }
            // Face slots are aligned with the input faces (welding keeps every
            // face and xatlas writes ignored faces back at their slot), so the
            // slot index IS the input face index.
            out.face_ids.push(face_index as u32);
        }
    }

    validate_uv_atlas(&out, input_face_count)?;
    Ok(out)
}

/// Shared output invariants of [`uv_unwrap`]: exactly one output triangle per
/// input face, indices in range, and finite UVs.
fn validate_uv_atlas(atlas: &UvAtlas, input_face_count: usize) -> Result<()> {
    if atlas.indices.len() != input_face_count * 3 || atlas.face_ids.len() != input_face_count {
        return Err(Lux3dError::InvalidInput(
            "UV unwrap did not preserve the input face count",
        ));
    }
    for &index in &atlas.indices {
        if index as usize >= atlas.vmapping.len() {
            return Err(Lux3dError::InvalidInput(
                "UV unwrap produced an atlas index outside the atlas vertex buffer",
            ));
        }
    }
    for uv in &atlas.uvs {
        if uv.iter().any(|c| !c.is_finite()) {
            return Err(Lux3dError::InvalidInput(
                "UV unwrap produced a non-finite UV",
            ));
        }
    }
    Ok(())
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

        // Charts were produced; the stock single pack reports one atlas whose
        // size is the packer's density estimate (not `texture_size`).
        assert!(atlas.chart_count > 0, "cube must produce charts");
        assert_eq!(1, atlas.atlas_count);
        assert!(atlas.atlas_size > 0, "packed atlas must have an extent");

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

        // Indices are in range and every input triangle is represented
        // exactly once (face provenance, Change A).
        let atlas_vertices = atlas.vmapping.len();
        assert_eq!(0, atlas.indices.len() % 3);
        assert_eq!(faces.len(), atlas.indices.len(), "atlas keeps all faces");
        assert_eq!(faces.len() / 3, atlas.face_ids.len());
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

    /// Change A: one stock pack pass -> exactly one atlas, no multi-atlas
    /// vertical merge, and the cluster stat is the single global cluster.
    #[test]
    fn test_single_atlas_count_is_one() {
        let (vertices, faces) = cube();
        let atlas = uv_unwrap(&vertices, &faces, 2048).expect("cube unwrap");
        assert_eq!(1, atlas.atlas_count, "stock pack must yield one atlas");
        assert!(!atlas.clusters_fell_back, "single atlas must not fall back");
        assert_eq!(1, atlas.cluster_count);
        assert!(atlas.chart_count > 0);

        // `texture_size` no longer limits the pack: both requests produce the
        // same atlas at the packer's own density estimate.
        let small = uv_unwrap(&vertices, &faces, 256).expect("cube unwrap 256");
        assert_eq!(1, small.atlas_count);
        assert_eq!(atlas.atlas_size, small.atlas_size);
        assert_eq!(atlas.uvs, small.uvs);
    }

    /// Face provenance is a permutation of the input faces: every face kept
    /// exactly once, in input order on the single-mesh path.
    #[test]
    fn test_face_ids_permutation() {
        let (vertices, faces) = cube();
        let face_count = faces.len() / 3;
        let atlas = uv_unwrap(&vertices, &faces, 2048).expect("cube unwrap");
        assert_eq!(face_count, atlas.face_ids.len());
        let mut sorted = atlas.face_ids.clone();
        sorted.sort_unstable();
        assert_eq!((0..face_count as u32).collect::<Vec<_>>(), sorted);
        assert_eq!(Some(&0), atlas.face_ids.first());
        assert_eq!(Some(&(face_count as u32 - 1)), atlas.face_ids.last());
        assert_eq!(face_count * 3, atlas.indices.len());
        assert_eq!(0, atlas.uncharted_face_count, "cube has no degenerate face");
    }

    /// No dead band: the stock pack + texel->[0,1] normalization makes the
    /// charts span both axes (the old multi-atlas vertical merge left unused
    /// v strips).
    #[test]
    fn test_no_dead_v_band() {
        let (vertices, faces) = cube();
        let atlas = uv_unwrap(&vertices, &faces, 2048).expect("cube unwrap");
        let (mut max_u, mut max_v) = (0.0_f32, 0.0_f32);
        for uv in &atlas.uvs {
            max_u = max_u.max(uv[0]);
            max_v = max_v.max(uv[1]);
        }
        assert!(max_u > 0.9, "U must reach the atlas edge, max_u={max_u}");
        assert!(max_v > 0.9, "V must reach the atlas edge, max_v={max_v}");
    }

    /// A fully degenerate (zero-area) input is still a valid atlas product:
    /// the face is kept, point-collapsed to (0, 0) instead of being dropped.
    #[test]
    fn test_degenerate_face_kept_point_collapsed() {
        // Collinear triangle: zero 3D area, xatlas cannot chart it.
        let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let faces = vec![0, 1, 2];
        let atlas = uv_unwrap(&vertices, &faces, 512).expect("degenerate unwrap");
        assert_eq!(0, atlas.chart_count);
        assert_eq!(1, atlas.atlas_count);
        assert_eq!(512, atlas.atlas_size, "no pack: texture_size is reported");
        assert_eq!(1, atlas.uncharted_face_count);
        assert_eq!(vec![0], atlas.face_ids);
        assert_eq!(3, atlas.indices.len(), "the face is kept, not dropped");
        for uv in &atlas.uvs {
            assert_eq!([0.0, 0.0], *uv, "collapsed onto the texture corner");
        }
        for (position, &original) in atlas.vertices.iter().zip(&atlas.vmapping) {
            assert_eq!(&vertices[original as usize], position);
        }
    }

    /// An uncharted face that shares welded positions with a charted face is
    /// point-collapsed onto that face's UV (twin), not the texture corner.
    #[test]
    fn test_degenerate_face_collapses_onto_twin_uv() {
        // Unit square (two charted triangles) plus a zero-area triangle made
        // of two of the square's own vertices.
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let faces = vec![0, 1, 2, 0, 2, 3, 0, 1, 0];
        let atlas = uv_unwrap(&vertices, &faces, 256).expect("square unwrap");
        assert_eq!(3, atlas.face_ids.len());
        assert_eq!(9, atlas.indices.len());
        assert_eq!(1, atlas.uncharted_face_count);
        // The degenerate face is output triangle 2; its three vertices must
        // share ONE UV, and that UV must be an already-charted corner UV.
        let collapsed = &atlas.indices[6..9];
        let uvs: Vec<[f32; 2]> = collapsed.iter().map(|&i| atlas.uvs[i as usize]).collect();
        assert_eq!(uvs[0], uvs[1]);
        assert_eq!(uvs[1], uvs[2]);
        let mut charted_corner_uvs = Vec::new();
        for (&original, &uv) in atlas.vmapping.iter().zip(&atlas.uvs) {
            if original == 0 || original == 1 {
                charted_corner_uvs.push(uv);
            }
        }
        assert!(
            charted_corner_uvs.contains(&uvs[0]),
            "collapsed UV {:?} must be a charted twin, got {charted_corner_uvs:?}",
            uvs[0]
        );
    }

    /// Change A determinism: two identical unwraps produce identical atlases,
    /// including the new provenance/stat fields.
    #[test]
    fn test_determinism() {
        let (vertices, faces) = cube();
        let first = uv_unwrap(&vertices, &faces, 512).expect("first unwrap");
        let second = uv_unwrap(&vertices, &faces, 512).expect("second unwrap");
        assert_eq!(first, second);
        assert_eq!(first.face_ids, second.face_ids);
        assert_eq!(first.uncharted_face_count, second.uncharted_face_count);
        assert_eq!(first.clusters_fell_back, second.clusters_fell_back);
    }

    /// The charting weld must keep the documented `merge_close_vertices`
    /// vertex clustering while keeping every face in the buffer.
    #[test]
    fn weld_matches_merge_close_vertices_vertex_partition() {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            // Exact duplicates of 1 and 2 (repair_non_manifold_edges shape).
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            // Near-duplicate of vertex 1, inside the relative weld epsilon.
            [1.0 + 5.0e-7, 0.0, 0.0],
        ];
        // The last face collapses onto two welded indices: it is dropped by
        // `merge_close_vertices` but must stay in the welding output here.
        let faces = vec![0, 1, 2, 3, 4, 5, 6, 5, 2, 0, 1, 3];
        let (welded, welded_faces, weld_origin) = weld_uv_input(&vertices, &faces);
        assert_eq!(faces.len(), welded_faces.len(), "every face stays");

        let flat: Vec<f32> = vertices.iter().flatten().copied().collect();
        let merged =
            super::super::mesh_ops::merge_close_vertices(&flat, &faces, weld_epsilon(&vertices));
        let merged_positions: Vec<[f32; 3]> = merged
            .vertices
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect();
        assert_eq!(merged_positions, welded, "same representatives");
        assert_eq!(
            merged.vertex_remap, weld_origin,
            "same representative origins"
        );
        // Filtering the degenerate welded faces must reproduce the merge
        // utility's face output exactly.
        let kept: Vec<u32> = welded_faces
            .chunks_exact(3)
            .filter(|face| face[0] != face[1] && face[1] != face[2] && face[0] != face[2])
            .flatten()
            .copied()
            .collect();
        assert_eq!(merged.faces, kept, "same face partition");
    }
}
