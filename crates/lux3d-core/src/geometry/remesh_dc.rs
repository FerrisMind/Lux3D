//! Narrow-band dual-contouring remesher (TRELLIS.2 `to_glb` remesh branch).
//!
//! Algorithmic port of the official CuMesh implementation, which is the
//! authority for this stage:
//!
//! * `CuMesh/cumesh/remeshing.py:60-256` — sparse octree grid build, UDF
//!   sampling, topology construction, adaptive quad split and the optional
//!   project-back pass.
//! * `CuMesh/src/remesh/simple_dual_contour.cu:28-156` — the per-voxel dual
//!   vertex (mean of the 12 edge intersections) and the per-axis
//!   `intersected` signs.
//! * `CuMesh/src/remesh/api.h:19-67` — the C++ entry-point contract.
//!
//! The official call site is `o-voxel/o_voxel/postprocess.py:171-180`:
//!
//! ```text
//! center = aabb.mean(dim=0)
//! scale  = (aabb[1] - aabb[0]).max()
//! resolution = grid_size.max()
//! remesh_narrow_band_dc(vertices, faces,
//!     center = center,
//!     scale = (resolution + 3 * remesh_band) / resolution * scale,
//!     resolution = resolution,
//!     band = remesh_band,
//!     project_back = remesh_project,
//!     bvh = bvh)
//! ```
//!
//! # Algorithm
//!
//! 1. **Sparse grid.** `eps = band * scale / resolution` thickens the UDF so
//!    the extracted isosurface is `unsigned_distance - eps = 0`. The grid
//!    starts at `base_resolution = resolution`, repeatedly halved while it is
//!    greater than 32 (`remeshing.py:102-107`). All `base_resolution^3` cell
//!    centers `((coord + 0.5) / base - 0.5) * scale + center` are sampled
//!    against the BVH; a cell is kept when `|UDF - eps| < 0.87 * cell_size`
//!    (`0.87` is the official constant, approximately `sqrt(3)/2`, covering
//!    the half cell diagonal). While `base_resolution < resolution` the
//!    survivors are doubled and expanded into their eight children, in the
//!    official child order, and the filter runs again at the doubled
//!    resolution (`remeshing.py:118-141`).
//! 2. **Corner values.** The unique integer corners of the active voxels are
//!    sampled once: `unsigned_distance(corner) - eps` (`remeshing.py:154-165`)
//!    with `corner_position = (corner / resolution - 0.5) * scale + center`.
//!    There is no ray parity, no inside/outside sign and no QEF: the sign is
//!    purely the side of the thickened isosurface.
//! 3. **Dual vertices.** For every active voxel the 12 edges are traversed
//!    exactly like `simple_dual_contour.cu:52-143`. A crossing is
//!    `(v1 < 0 && v2 >= 0) || (v1 >= 0 && v2 < 0)`, its position is
//!    `grid_low + t` with `t = -v1 / (v2 - v1)`, and the dual vertex is the
//!    mean of all crossing positions; a voxel without crossings falls back to
//!    its center. `intersected[axis]` is `+1` when the `(u = 1, v = 1)` edge
//!    of that axis goes from negative to non-negative, `-1` for the reverse
//!    and `0` otherwise.
//! 4. **Topology.** Every nonzero `intersected[axis]` of every voxel becomes
//!    one quad joining the dual vertices of the four voxels around that grid
//!    edge, with the official edge-neighbor offsets
//!    (`remeshing.py:70-76,186-204`). Quads whose four voxels are not all in
//!    the active set are dropped (the official `0xffffffff` lookup miss), then
//!    unreferenced dual vertices are removed in ascending voxel-index order
//!    (`remeshing.py:205-210`).
//! 5. **Triangles.** The remesher emits triangles directly: each quad is
//!    split with the official `quad_split_1_{n,p}` / `quad_split_2_{n,p}`
//!    tables, the normal-alignment metric picks split 1 only when
//!    `align0 > align1` (strict, ties choose split 2), and the `p` variants
//!    are used for `intersected_dir == 1` (`remeshing.py:212-233`). The
//!    winding points from the negative (inside the thickened band) to the
//!    positive side, i.e. outward for a closed input; it never depends on the
//!    input face orientation.
//! 6. **Project back.** After triangulation each output vertex moves by
//!    `v -= project_back * (v - closest_point_on_input)` when
//!    `project_back > 0` (`remeshing.py:238-250`); `0` disables the pass.
//!
//! # Degenerate cases and assumptions
//!
//! * **Band.** `band = 0` removes the thickening: every corner value is an
//!   unsigned distance (`>= 0`), so the strict crossing tests never fire and
//!   the official algorithm returns an empty mesh. Callers that need a
//!   surface must use `band > 0` (the shipped value is `1`).
//! * **Closedness and shells.** The UDF has no inside/outside classification,
//!   so open or inconsistently wound inputs are handled geometrically; the
//!   result can still be open where the thickened shell leaves the grid. For a
//!   closed input and `band >= 1` the extracted surface is closed: every voxel
//!   touching a sign-changing edge is within half a cell diagonal of the
//!   isosurface and therefore inside the `0.87 * cell_size` narrow band.
//!   Because the zero set is `unsigned_distance - eps`, a closed solid yields
//!   TWO disjoint closed components — the outer offset surface at distance
//!   `eps` (outward winding) and the inner offset surface at depth `eps`
//!   (inward winding). Their signed volumes cancel, leaving the shell volume;
//!   the official `to_glb` remesh has the same output. Geometry thinner than
//!   `2 * eps` fills the interior with negative values and produces a single
//!   shell.
//! * **Sub-cell features.** Grid sampling cannot capture surface detail that
//!   does not flip a corner value across the band; the official algorithm has
//!   the same limitation.
//! * **Input contract.** `bvh` must have been built over the same
//!   `(vertices, faces)` buffers; the buffers are validated here, but the
//!   tree is not re-derived. `resolution` must be in
//!   `2..=MAX_GRID_SIZE_PER_AXIS` and must halve evenly down to 32 (the
//!   official `assert base_resolution % 2 == 0`), `center` / `band` /
//!   `project_back` must be finite, `band >= 0`, `project_back >= 0`, and
//!   `scale > 0` with a finite grid cube. `project_back` is a raw lerp
//!   factor, so values above 1 extrapolate past the closest point, matching
//!   the official call.

use rustc_hash::FxHashMap;

use crate::error::{Lux3dError, Result};

use super::dual_grid::MAX_GRID_SIZE_PER_AXIS;
use super::mesh_bvh::MeshBvh;

/// Official narrow-band refinement factor (`remeshing.py:129`): a cell is
/// kept when `|UDF - eps| < 0.87 * cell_size`. `0.87` is the reference
/// constant (approximately `sqrt(3) / 2`, the half cell diagonal).
const NARROW_BAND_FACTOR: f64 = 0.87;

/// Official child offsets of the octree expansion (`remeshing.py:85-88`),
/// in order: x fastest, then y, then z.
const OCTREE_CHILD_OFFSETS: [[i32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];

/// Voxel offsets of the four voxels around a grid edge, per edge axis
/// (`remeshing.py:71-76`).
const EDGE_NEIGHBOR_VOXEL_OFFSETS: [[[i32; 3]; 4]; 3] = [
    // x-axis edge: neighbors at (0,0,0), (0,0,1), (0,1,1), (0,1,0).
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
    // y-axis edge: neighbors at (0,0,0), (1,0,0), (1,0,1), (0,0,1).
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
    // z-axis edge: neighbors at (0,0,0), (0,1,0), (1,1,0), (1,0,0).
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
];

/// Official quad split patterns (`remeshing.py:80-83`), two triangles each.
/// `_n` is used for `intersected == -1`, `_p` for `intersected == +1`.
const QUAD_SPLIT_1_N: [usize; 6] = [0, 1, 2, 0, 2, 3];
const QUAD_SPLIT_1_P: [usize; 6] = [0, 2, 1, 0, 3, 2];
const QUAD_SPLIT_2_N: [usize; 6] = [0, 1, 3, 3, 1, 2];
const QUAD_SPLIT_2_P: [usize; 6] = [0, 3, 1, 3, 2, 1];

/// BVH queries are batched in chunks this large so the temporary closest-point
/// result vector stays bounded on high-resolution grids.
const QUERY_CHUNK: usize = 1 << 16;

/// Options for [`remesh_narrow_band_dc`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RemeshDcOptions {
    /// Grid-cube center (official `aabb.mean(dim=0)`).
    pub center: [f32; 3],
    /// Grid-cube side length. This is the call site's already-expanded value
    /// `(resolution + 3 * band) / resolution * mesh_extent`
    /// (`postprocess.py:174`); [`narrow_band_grid_scale`] computes it.
    pub scale: f32,
    /// Cells per axis (official `grid_size.max()`).
    pub resolution: u32,
    /// Narrow-band half-width in cells (official `band`; shipped value `1`).
    pub band: f32,
    /// Project-back lerp factor (official `project_back`; `0` disables).
    pub project_back: f32,
}

impl RemeshDcOptions {
    /// Builds options from the mesh AABB exactly like the official call site:
    /// the grid center is the AABB center and the grid side expands the
    /// largest AABB extent by `(resolution + 3 * band) / resolution`
    /// (`postprocess.py:166-180`).
    pub fn from_aabb(aabb: [[f32; 3]; 2], resolution: u32, band: f32, project_back: f32) -> Self {
        let center = [
            (aabb[0][0] + aabb[1][0]) * 0.5,
            (aabb[0][1] + aabb[1][1]) * 0.5,
            (aabb[0][2] + aabb[1][2]) * 0.5,
        ];
        let extent = (aabb[1][0] - aabb[0][0])
            .max(aabb[1][1] - aabb[0][1])
            .max(aabb[1][2] - aabb[0][2]);
        Self {
            center,
            scale: narrow_band_grid_scale(extent, resolution, band),
            resolution,
            band,
            project_back,
        }
    }
}

/// Official grid-cube side expansion (`postprocess.py:174`):
/// `(resolution + 3 * band) / resolution * scale`.
///
/// `resolution` must be non-zero; [`remesh_narrow_band_dc`] validates the
/// final options.
pub fn narrow_band_grid_scale(scale: f32, resolution: u32, band: f32) -> f32 {
    (resolution as f32 + 3.0 * band) / resolution as f32 * scale
}

/// Remeshed mesh: triangle topology, world-space vertex positions.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RemeshDcMesh {
    /// Flat `[x, y, z, ...]` vertex positions (mean edge intersections,
    /// converted to world space and projected back when requested). Only
    /// dual vertices referenced by at least one emitted quad are kept.
    pub vertices: Vec<f32>,
    /// Flat triangle indices, three per face, wound outward (from the inside
    /// of the thickened narrow band toward the outside).
    pub faces: Vec<u32>,
}

/// Remeshes `(vertices, faces)` with the official narrow-band dual contouring
/// algorithm.
///
/// * `vertices`: flat `[x, y, z, ...]` f32 positions.
/// * `faces`: flat `[F, 3]` zero-based triangle indices.
/// * `bvh`: triangle BVH built over the same `(vertices, faces)` buffers
///   (`MeshBvh::build`); the official call passes the pre-built
///   `cumesh.cuBVH` the same way (`postprocess.py:122, 179`).
/// * `options`: grid center/scale/resolution, band and project-back factor.
///
/// Returns a triangle mesh. See the module docs for the algorithm and the
/// documented degenerate-case assumptions.
pub fn remesh_narrow_band_dc<'mesh>(
    vertices: &'mesh [f32],
    faces: &'mesh [u32],
    bvh: &MeshBvh<'mesh>,
    options: &RemeshDcOptions,
) -> Result<RemeshDcMesh> {
    validate_options(options)?;
    validate_mesh(vertices, faces)?;
    validate_grid_bounds(options)?;

    let resolution = options.resolution;
    let resolution_f32 = resolution as f32;
    let scale = options.scale;
    let center = options.center;

    // Official `eps = band * scale / resolution` (`remeshing.py:100`). Python
    // evaluates it in double precision and PyTorch casts the scalar to the
    // f32 tensor dtype at the subtraction; mirror that.
    let eps = (options.band as f64 * scale as f64 / resolution as f64) as f32;
    if !eps.is_finite() {
        return Err(Lux3dError::InvalidInput(
            "remesh narrow-band dual contouring band epsilon is not finite",
        ));
    }

    let base_resolution = base_resolution(resolution)?;

    // ---- step 1: narrow-band sparse grid (official octree expansion) -------
    let mut coords: Vec<[i32; 3]> = Vec::with_capacity((base_resolution as usize).pow(3));
    for x in 0..base_resolution as i32 {
        for y in 0..base_resolution as i32 {
            for z in 0..base_resolution as i32 {
                coords.push([x, y, z]);
            }
        }
    }

    let mut current_resolution = base_resolution;
    loop {
        let divisor = current_resolution as f32;
        // Python computes `0.87 * (scale / base_resolution)` in double
        // precision and casts the comparison scalar to f32.
        let threshold = (NARROW_BAND_FACTOR * (scale as f64 / current_resolution as f64)) as f32;
        if !threshold.is_finite() {
            return Err(Lux3dError::InvalidInput(
                "remesh narrow-band dual contouring band threshold is not finite",
            ));
        }
        let mut kept: Vec<[i32; 3]> = Vec::with_capacity(coords.len());
        for chunk in coords.chunks(QUERY_CHUNK) {
            let points: Vec<[f32; 3]> = chunk
                .iter()
                .map(|&coord| cell_center_point(coord, divisor, scale, center))
                .collect();
            let hits = bvh.closest_points(&points);
            for (&coord, hit) in chunk.iter().zip(hits.iter()) {
                // `abs(unsigned_distance - eps) < 0.87 * cell_size`.
                if (hit.unsigned_distance - eps).abs() < threshold {
                    kept.push(coord);
                }
            }
        }
        coords = kept;

        if current_resolution >= resolution {
            break;
        }
        // Official `base_resolution *= 2` then 8-child expansion
        // (`remeshing.py:135-140`).
        current_resolution *= 2;
        let mut expanded: Vec<[i32; 3]> = Vec::with_capacity(coords.len() * 8);
        for coord in &coords {
            for offset in OCTREE_CHILD_OFFSETS {
                expanded.push([
                    coord[0] * 2 + offset[0],
                    coord[1] * 2 + offset[1],
                    coord[2] * 2 + offset[2],
                ]);
            }
        }
        coords = expanded;
    }

    // ---- step 2: unique corner values = unsigned distance - eps -----------
    // The official `get_sparse_voxel_grid_active_vertices` dedups the corners
    // of the active voxels; the values are sampled once per unique corner.
    let mut corner_indices: FxHashMap<u64, u32> =
        FxHashMap::with_capacity_and_hasher(coords.len() * 2, Default::default());
    let mut corners: Vec<[u32; 3]> = Vec::with_capacity(coords.len() * 2);
    for voxel in &coords {
        for local in 0..8i32 {
            let corner = [
                (voxel[0] + (local & 1)) as u32,
                (voxel[1] + ((local >> 1) & 1)) as u32,
                (voxel[2] + ((local >> 2) & 1)) as u32,
            ];
            let key = corner_key(corner);
            if let std::collections::hash_map::Entry::Vacant(entry) = corner_indices.entry(key) {
                entry.insert(corners.len() as u32);
                corners.push(corner);
            }
        }
    }

    let mut corner_values: Vec<f32> = Vec::with_capacity(corners.len());
    for chunk in corners.chunks(QUERY_CHUNK) {
        let points: Vec<[f32; 3]> = chunk
            .iter()
            .map(|&corner| corner_point(corner, resolution_f32, scale, center))
            .collect();
        let hits = bvh.closest_points(&points);
        for hit in hits {
            corner_values.push(hit.unsigned_distance - eps);
        }
    }

    // ---- step 3: one dual vertex per active voxel --------------------------
    // Mirrors `simple_dual_contour.cu:28-156`: mean of the 12 edge
    // intersections (grid coordinates) or the voxel center when no edge
    // crosses, plus the per-axis `intersected` sign of the (u=1, v=1) edge.
    let mut dual_vertices: Vec<[f32; 3]> = Vec::with_capacity(coords.len());
    let mut intersected: Vec<[i32; 3]> = Vec::with_capacity(coords.len());
    let mut values = [0.0f32; 8];

    for voxel in &coords {
        for (local, value) in values.iter_mut().enumerate() {
            let corner = [
                (voxel[0] + (local & 1) as i32) as u32,
                (voxel[1] + ((local >> 1) & 1) as i32) as u32,
                (voxel[2] + ((local >> 2) & 1) as i32) as u32,
            ];
            let index = corner_indices[&corner_key(corner)] as usize;
            *value = corner_values[index];
        }

        let mut sum = [0.0f32; 3];
        let mut count = 0u32;
        let mut signs = [0i32; 3];

        // X axis: val1 = (vx, vy+u, vz+v), val2 = (vx+1, vy+u, vz+v).
        for u in 0..=1i32 {
            for v in 0..=1i32 {
                let low = (u << 1 | v << 2) as usize;
                let high = low | 1;
                let val1 = values[low];
                let val2 = values[high];
                if edges_cross(val1, val2) {
                    let t = -val1 / (val2 - val1);
                    sum[0] += voxel[0] as f32 + t;
                    sum[1] += (voxel[1] + u) as f32;
                    sum[2] += (voxel[2] + v) as f32;
                    count += 1;
                }
                if u == 1 && v == 1 {
                    signs[0] = edge_sign(val1, val2);
                }
            }
        }

        // Y axis: val1 = (vx+u, vy, vz+v), val2 = (vx+u, vy+1, vz+v).
        for u in 0..=1i32 {
            for v in 0..=1i32 {
                let low = (u | v << 2) as usize;
                let high = low | 2;
                let val1 = values[low];
                let val2 = values[high];
                if edges_cross(val1, val2) {
                    let t = -val1 / (val2 - val1);
                    sum[0] += (voxel[0] + u) as f32;
                    sum[1] += voxel[1] as f32 + t;
                    sum[2] += (voxel[2] + v) as f32;
                    count += 1;
                }
                if u == 1 && v == 1 {
                    signs[1] = edge_sign(val1, val2);
                }
            }
        }

        // Z axis: val1 = (vx+u, vy+v, vz), val2 = (vx+u, vy+v, vz+1).
        for u in 0..=1i32 {
            for v in 0..=1i32 {
                let low = (u | v << 1) as usize;
                let high = low | 4;
                let val1 = values[low];
                let val2 = values[high];
                if edges_cross(val1, val2) {
                    let t = -val1 / (val2 - val1);
                    sum[0] += (voxel[0] + u) as f32;
                    sum[1] += (voxel[1] + v) as f32;
                    sum[2] += voxel[2] as f32 + t;
                    count += 1;
                }
                if u == 1 && v == 1 {
                    signs[2] = edge_sign(val1, val2);
                }
            }
        }

        let vertex = if count > 0 {
            let divisor = count as f32;
            [sum[0] / divisor, sum[1] / divisor, sum[2] / divisor]
        } else {
            [
                voxel[0] as f32 + 0.5,
                voxel[1] as f32 + 0.5,
                voxel[2] as f32 + 0.5,
            ]
        };
        dual_vertices.push(vertex);
        intersected.push(signs);
    }

    // ---- step 4: topology, exact official quad construction ----------------
    let mut voxel_indices: FxHashMap<u64, u32> =
        FxHashMap::with_capacity_and_hasher(coords.len() * 2, Default::default());
    for (index, voxel) in coords.iter().enumerate() {
        voxel_indices.insert(voxel_key(*voxel), index as u32);
    }

    let mut quads: Vec<[u32; 4]> = Vec::new();
    let mut quad_directions: Vec<i32> = Vec::new();
    for (voxel_index, voxel) in coords.iter().enumerate() {
        for axis in 0..3usize {
            let direction = intersected[voxel_index][axis];
            if direction == 0 {
                continue;
            }
            let mut quad = [0u32; 4];
            let mut complete = true;
            for (slot, offset) in EDGE_NEIGHBOR_VOXEL_OFFSETS[axis].iter().enumerate() {
                let neighbor = [
                    voxel[0] + offset[0],
                    voxel[1] + offset[1],
                    voxel[2] + offset[2],
                ];
                match voxel_indices.get(&voxel_key(neighbor)) {
                    Some(&index) => quad[slot] = index,
                    None => {
                        complete = false;
                        break;
                    }
                }
            }
            if complete {
                quads.push(quad);
                quad_directions.push(direction);
            }
        }
    }

    // Official `torch.unique` keeps referenced dual vertices in ascending
    // voxel-index order (`remeshing.py:205-210`).
    let mut used: Vec<u32> = quads.iter().flatten().copied().collect();
    used.sort_unstable();
    used.dedup();

    let mut vertex_map = vec![u32::MAX; dual_vertices.len()];
    let mut output_vertices: Vec<f32> = Vec::with_capacity(used.len() * 3);
    for (new_index, &old_index) in used.iter().enumerate() {
        vertex_map[old_index as usize] = new_index as u32;
        let dual = dual_vertices[old_index as usize];
        output_vertices.push((dual[0] / resolution_f32 - 0.5) * scale + center[0]);
        output_vertices.push((dual[1] / resolution_f32 - 0.5) * scale + center[1]);
        output_vertices.push((dual[2] / resolution_f32 - 0.5) * scale + center[2]);
    }
    for quad in &mut quads {
        for index in quad.iter_mut() {
            *index = vertex_map[*index as usize];
        }
    }

    // ---- step 5: official adaptive split into triangles -------------------
    let mut output_faces: Vec<u32> = Vec::with_capacity(quads.len() * 6);
    for (quad, &direction) in quads.iter().zip(quad_directions.iter()) {
        let chosen = triangulate_quad(&output_vertices, *quad, direction);
        output_faces.extend_from_slice(&chosen);
    }

    // ---- step 6: optional project-back to the input surface ---------------
    // Official guard: `if project_back > 0` (`remeshing.py:238`); `0`
    // disables and negative factors are rejected by validation.
    if options.project_back > 0.0 {
        for vertex in output_vertices.chunks_exact_mut(3) {
            let point = [vertex[0], vertex[1], vertex[2]];
            let hit = bvh.closest_point(point);
            for axis in 0..3 {
                vertex[axis] -= options.project_back * (point[axis] - hit.position[axis]);
            }
        }
    }

    Ok(RemeshDcMesh {
        vertices: output_vertices,
        faces: output_faces,
    })
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_options(options: &RemeshDcOptions) -> Result<()> {
    if options.center.iter().any(|value| !value.is_finite()) {
        return Err(Lux3dError::InvalidInput(
            "remesh narrow-band dual contouring requires a finite grid center",
        ));
    }
    if !options.scale.is_finite() || options.scale <= 0.0 {
        return Err(Lux3dError::InvalidInput(
            "remesh narrow-band dual contouring requires a finite positive grid scale",
        ));
    }
    if options.resolution < 2 || options.resolution > MAX_GRID_SIZE_PER_AXIS {
        return Err(Lux3dError::InvalidInput(
            "remesh narrow-band dual contouring requires 2 <= resolution <= 4096",
        ));
    }
    if !options.band.is_finite() || options.band < 0.0 {
        return Err(Lux3dError::InvalidInput(
            "remesh narrow-band dual contouring requires a finite non-negative band",
        ));
    }
    if !options.project_back.is_finite() || options.project_back < 0.0 {
        return Err(Lux3dError::InvalidInput(
            "remesh narrow-band dual contouring requires a finite non-negative project_back \
             factor (0 disables project-back)",
        ));
    }
    Ok(())
}

/// The grid cube must stay inside the finite f32 range: every sampled point
/// (cell center or corner) is `center ± scale / 2` at most, and
/// [`MeshBvh::closest_point`] requires finite query points.
fn validate_grid_bounds(options: &RemeshDcOptions) -> Result<()> {
    let half = options.scale as f64 * 0.5;
    let limit = f32::MAX as f64;
    for axis in 0..3 {
        let low = options.center[axis] as f64 - half;
        let high = options.center[axis] as f64 + half;
        if !low.is_finite() || !high.is_finite() || low.abs() > limit || high.abs() > limit {
            return Err(Lux3dError::InvalidInput(
                "remesh narrow-band dual contouring grid cube is not finite",
            ));
        }
    }
    Ok(())
}

/// Official base-resolution search (`remeshing.py:102-107`): halve while
/// greater than 32, asserting evenness at every step.
fn base_resolution(resolution: u32) -> Result<u32> {
    let mut base = resolution;
    while base > 32 {
        if base % 2 != 0 {
            return Err(Lux3dError::InvalidInput(
                "remesh narrow-band dual contouring requires a resolution that halves evenly \
                 down to 32 (official base-resolution loop)",
            ));
        }
        base /= 2;
    }
    Ok(base)
}

fn validate_mesh(vertices: &[f32], faces: &[u32]) -> Result<()> {
    if vertices.len() % 3 != 0 {
        return Err(Lux3dError::InvalidInput(
            "remesh narrow-band dual contouring vertex buffer length must be a multiple of 3",
        ));
    }
    if faces.len() % 3 != 0 {
        return Err(Lux3dError::InvalidInput(
            "remesh narrow-band dual contouring face buffer length must be a multiple of 3",
        ));
    }
    if vertices.is_empty() || faces.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "remesh narrow-band dual contouring requires a non-empty mesh",
        ));
    }
    let vertex_count = vertices.len() / 3;
    if faces.iter().any(|&index| index as usize >= vertex_count) {
        return Err(Lux3dError::InvalidInput(
            "remesh narrow-band dual contouring face index outside the vertex buffer",
        ));
    }
    if vertices.iter().any(|value| !value.is_finite()) {
        return Err(Lux3dError::InvalidInput(
            "remesh narrow-band dual contouring requires finite vertex positions",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Grid sampling and dual-contour edges
// ---------------------------------------------------------------------------

/// Official cell-center sample point (`remeshing.py:121`):
/// `((coord + 0.5) / base - 0.5) * scale + center`.
#[inline]
fn cell_center_point(coord: [i32; 3], base: f32, scale: f32, center: [f32; 3]) -> [f32; 3] {
    [
        ((coord[0] as f32 + 0.5) / base - 0.5) * scale + center[0],
        ((coord[1] as f32 + 0.5) / base - 0.5) * scale + center[1],
        ((coord[2] as f32 + 0.5) / base - 0.5) * scale + center[2],
    ]
}

/// Official corner sample point (`remeshing.py:162`):
/// `(corner / resolution - 0.5) * scale + center`.
#[inline]
fn corner_point(corner: [u32; 3], resolution: f32, scale: f32, center: [f32; 3]) -> [f32; 3] {
    [
        (corner[0] as f32 / resolution - 0.5) * scale + center[0],
        (corner[1] as f32 / resolution - 0.5) * scale + center[1],
        (corner[2] as f32 / resolution - 0.5) * scale + center[2],
    ]
}

/// Official sign-change test (`simple_dual_contour.cu:62`): the second branch
/// is `val2 < 0` with `val1 >= 0`, so a zero corner belongs to the
/// non-negative side.
#[inline]
fn edges_cross(val1: f32, val2: f32) -> bool {
    (val1 < 0.0 && val2 >= 0.0) || (val1 >= 0.0 && val2 < 0.0)
}

/// Official `intersected[axis]` sign (`simple_dual_contour.cu:71-81`).
#[inline]
fn edge_sign(val1: f32, val2: f32) -> i32 {
    if val1 < 0.0 && val2 >= 0.0 {
        1
    } else if val1 >= 0.0 && val2 < 0.0 {
        -1
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Official adaptive quad split
// ---------------------------------------------------------------------------

/// Picks the official split of one quad and returns its two triangles
/// (`remeshing.py:212-233`).
///
/// `attempt_0` uses the split-1 pattern, `attempt_1` the split-2 pattern; the
/// `_p` variants are used for `direction == 1` and the `_n` variants
/// otherwise. `align0 > align1` is strict, so ties choose split 2 exactly like
/// the reference `torch.where`.
fn triangulate_quad(vertices: &[f32], quad: [u32; 4], direction: i32) -> [u32; 6] {
    let (attempt_0, attempt_1) = if direction == 1 {
        (QUAD_SPLIT_1_P, QUAD_SPLIT_2_P)
    } else {
        (QUAD_SPLIT_1_N, QUAD_SPLIT_2_N)
    };
    let triangles_0 = apply_pattern(quad, attempt_0);
    let triangles_1 = apply_pattern(quad, attempt_1);
    let align_0 = split_alignment(vertices, &triangles_0);
    let align_1 = split_alignment(vertices, &triangles_1);
    if align_0 > align_1 {
        triangles_0
    } else {
        triangles_1
    }
}

#[inline]
fn apply_pattern(quad: [u32; 4], pattern: [usize; 6]) -> [u32; 6] {
    [
        quad[pattern[0]],
        quad[pattern[1]],
        quad[pattern[2]],
        quad[pattern[3]],
        quad[pattern[4]],
        quad[pattern[5]],
    ]
}

/// Official normal-alignment metric, verbatim from `remeshing.py:220-231`:
/// `normals0 = cross(v[t1] - v[t0], v[t2] - v[t0])`,
/// `normals1 = cross(v[t2] - v[t1], v[t3] - v[t1])`,
/// `align = abs(dot(normals0, normals1))`.
fn split_alignment(vertices: &[f32], triangles: &[u32; 6]) -> f32 {
    let position = |index: usize| -> [f32; 3] {
        let base = triangles[index] as usize * 3;
        [vertices[base], vertices[base + 1], vertices[base + 2]]
    };
    let normals_0 = cross_f32(
        sub_f32(position(1), position(0)),
        sub_f32(position(2), position(0)),
    );
    let normals_1 = cross_f32(
        sub_f32(position(2), position(1)),
        sub_f32(position(3), position(1)),
    );
    dot_f32(normals_0, normals_1).abs()
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

#[inline]
fn voxel_key(voxel: [i32; 3]) -> u64 {
    debug_assert!(voxel.iter().all(|&value| value >= 0));
    voxel[0] as u64 | (voxel[1] as u64) << 21 | (voxel[2] as u64) << 42
}

#[inline]
fn corner_key(corner: [u32; 3]) -> u64 {
    corner[0] as u64 | (corner[1] as u64) << 21 | (corner[2] as u64) << 42
}

#[inline]
fn sub_f32(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross_f32(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot_f32(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    /// UV sphere of `radius` with an exact single point at each pole
    /// (collapsed pole triangles are skipped so the input has no zero-area
    /// triangles).
    fn uv_sphere(stacks: usize, sectors: usize, radius: f32) -> (Vec<f32>, Vec<u32>) {
        let mut vertices = Vec::with_capacity((stacks + 1) * (sectors + 1) * 3);
        for i in 0..=stacks {
            let theta = std::f32::consts::PI * i as f32 / stacks as f32;
            for j in 0..=sectors {
                let phi = std::f32::consts::TAU * j as f32 / sectors as f32;
                if i == 0 || i == stacks {
                    vertices.push(0.0);
                    vertices.push(0.0);
                    vertices.push(radius * theta.cos());
                } else {
                    vertices.push(radius * theta.sin() * phi.cos());
                    vertices.push(radius * theta.sin() * phi.sin());
                    vertices.push(radius * theta.cos());
                }
            }
        }
        let at = |i: usize, j: usize| i * (sectors + 1) + j;
        let same = |a: usize, b: usize| {
            (0..3).all(|k| (vertices[a * 3 + k] - vertices[b * 3 + k]).abs() <= 1e-9)
        };
        let mut faces = Vec::new();
        for i in 0..stacks {
            for j in 0..sectors {
                let a = at(i, j);
                let b = at(i + 1, j);
                let c = at(i + 1, j + 1);
                let d = at(i, j + 1);
                if !same(a, b) && !same(b, c) && !same(c, a) {
                    faces.extend([a as u32, b as u32, c as u32]);
                }
                if !same(a, c) && !same(c, d) && !same(d, a) {
                    faces.extend([a as u32, c as u32, d as u32]);
                }
            }
        }
        (vertices, faces)
    }

    /// Unit cube `[0, 1]^3` with outward winding.
    fn cube() -> (Vec<f32>, Vec<u32>) {
        let vertices: Vec<f32> = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
        .iter()
        .flatten()
        .copied()
        .collect();
        let faces: Vec<u32> = [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ]
        .iter()
        .flatten()
        .copied()
        .collect();
        (vertices, faces)
    }

    fn remesh(vertices: &[f32], faces: &[u32], options: &RemeshDcOptions) -> RemeshDcMesh {
        let bvh = MeshBvh::build(vertices, faces).expect("build bvh");
        remesh_narrow_band_dc(vertices, faces, &bvh, options).expect("remesh")
    }

    fn triangle_point(mesh: &RemeshDcMesh, index: u32) -> [f32; 3] {
        let base = index as usize * 3;
        [
            mesh.vertices[base],
            mesh.vertices[base + 1],
            mesh.vertices[base + 2],
        ]
    }

    /// Divergence-theorem signed volume of a closed triangle mesh. Positive
    /// for outward winding.
    fn signed_volume(mesh: &RemeshDcMesh) -> f64 {
        signed_volume_parts(&mesh.vertices, &mesh.faces)
    }

    fn signed_volume_parts(vertices: &[f32], faces: &[u32]) -> f64 {
        let mut volume = 0.0f64;
        for triangle in faces.chunks_exact(3) {
            let mut points = [[0.0f64; 3]; 3];
            for (slot, &index) in points.iter_mut().zip(triangle.iter()) {
                let base = index as usize * 3;
                *slot = [
                    vertices[base] as f64,
                    vertices[base + 1] as f64,
                    vertices[base + 2] as f64,
                ];
            }
            let [a, b, c] = points;
            volume += (a[0] * (b[1] * c[2] - b[2] * c[1])
                + a[1] * (b[2] * c[0] - b[0] * c[2])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                / 6.0;
        }
        volume
    }

    fn find_root(parents: &mut [u32], mut index: u32) -> u32 {
        while parents[index as usize] != index {
            let parent = parents[index as usize];
            let grandparent = parents[parent as usize];
            parents[index as usize] = grandparent;
            index = grandparent;
        }
        index
    }

    fn union(parents: &mut [u32], a: u32, b: u32) {
        let root_a = find_root(parents, a);
        let root_b = find_root(parents, b);
        if root_a != root_b {
            parents[root_b as usize] = root_a;
        }
    }

    /// Watertightness (every undirected edge shared by exactly two triangles),
    /// the number of connected closed components, and the Euler characteristic
    /// of each genus-0 component. Returns the per-component signed volumes.
    fn assert_closed_triangle_mesh(mesh: &RemeshDcMesh, expected_components: usize) -> Vec<f64> {
        assert!(!mesh.faces.is_empty(), "remesh must produce faces");
        assert_eq!(mesh.faces.len() % 3, 0, "output faces are triangles");
        assert_eq!(
            0,
            mesh.faces.len() % 6,
            "the official splitter emits two triangles per quad"
        );
        assert_eq!(mesh.vertices.len() % 3, 0);
        assert!(
            mesh.vertices.iter().all(|value| value.is_finite()),
            "vertices must be finite"
        );
        let vertex_count = mesh.vertices.len() / 3;
        for &index in &mesh.faces {
            assert!(
                (index as usize) < vertex_count,
                "face index outside the vertex buffer"
            );
        }

        let mut edge_counts: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        let mut parents: Vec<u32> = (0..vertex_count as u32).collect();
        for triangle in mesh.faces.chunks_exact(3) {
            union(&mut parents, triangle[0], triangle[1]);
            union(&mut parents, triangle[1], triangle[2]);
            for edge in 0..3 {
                let a = triangle[edge];
                let b = triangle[(edge + 1) % 3];
                assert_ne!(a, b, "degenerate triangle edge");
                let key = if a < b { (a, b) } else { (b, a) };
                *edge_counts.entry(key).or_insert(0) += 1;
            }
        }
        for (edge, count) in &edge_counts {
            assert_eq!(*count, 2, "edge {edge:?} used {count} times");
        }

        let mut component_of_vertex = vec![u32::MAX; vertex_count];
        let mut component_count = 0u32;
        for &index in &mesh.faces {
            if component_of_vertex[index as usize] == u32::MAX {
                let root = find_root(&mut parents, index);
                component_of_vertex[index as usize] = component_count;
                // Point every vertex of this component at the component id.
                for (vertex, slot) in component_of_vertex.iter_mut().enumerate() {
                    if *slot == u32::MAX && find_root(&mut parents, vertex as u32) == root {
                        *slot = component_count;
                    }
                }
                component_count += 1;
            }
        }
        assert_eq!(
            component_count as usize, expected_components,
            "connected component count"
        );
        assert!(
            component_of_vertex
                .iter()
                .all(|&component| component != u32::MAX),
            "the remesher must remove unreferenced vertices"
        );

        let mut volumes = vec![0.0f64; component_count as usize];
        for triangle in mesh.faces.chunks_exact(3) {
            let a = triangle_point(mesh, triangle[0]);
            let b = triangle_point(mesh, triangle[1]);
            let c = triangle_point(mesh, triangle[2]);
            let a = [a[0] as f64, a[1] as f64, a[2] as f64];
            let b = [b[0] as f64, b[1] as f64, b[2] as f64];
            let c = [c[0] as f64, c[1] as f64, c[2] as f64];
            let determinant = a[0] * (b[1] * c[2] - b[2] * c[1])
                + a[1] * (b[2] * c[0] - b[0] * c[2])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
            volumes[component_of_vertex[triangle[0] as usize] as usize] += determinant / 6.0;
        }

        let euler = vertex_count as i64 - edge_counts.len() as i64 + (mesh.faces.len() / 3) as i64;
        assert_eq!(
            euler,
            2 * component_count as i64,
            "each component must be a closed genus-0 surface"
        );
        volumes
    }

    /// Every emitted vertex must lie within one band cell plus the half cell
    /// diagonal of the input surface.
    fn assert_vertices_near_surface(
        mesh: &RemeshDcMesh,
        vertices: &[f32],
        faces: &[u32],
        options: &RemeshDcOptions,
        maximum_factor: f32,
    ) {
        let bvh = MeshBvh::build(vertices, faces).expect("build bvh");
        let spacing = options.scale / options.resolution as f32;
        let maximum_distance = maximum_factor * spacing + 1e-4;
        for chunk in mesh.vertices.chunks_exact(3) {
            let point = [chunk[0], chunk[1], chunk[2]];
            let distance = bvh.closest_point(point).unsigned_distance;
            assert!(
                distance <= maximum_distance,
                "vertex {point:?} is {distance} from the surface (limit {maximum_distance})"
            );
        }
    }

    #[test]
    fn grid_scale_matches_official_formula() {
        // Exact power-of-two division: bit-identical f32 arithmetic.
        assert_eq!(narrow_band_grid_scale(1.0, 512, 1.0), 515.0 / 512.0);
        assert_eq!(narrow_band_grid_scale(2.5, 100, 2.0), 106.0 / 100.0 * 2.5);

        // The official call site: center = aabb.mean, scale = max extent.
        let options =
            RemeshDcOptions::from_aabb([[-0.5, -1.0, 0.0], [0.5, 1.0, 0.25]], 256, 1.0, 0.9);
        assert_eq!(options.center, [0.0, 0.0, 0.125]);
        assert_eq!(options.scale, narrow_band_grid_scale(2.0, 256, 1.0));
        assert_eq!(options.resolution, 256);
        assert_eq!(options.band, 1.0);
        assert_eq!(options.project_back, 0.9);
    }

    #[test]
    fn base_resolution_halving_matches_official_loop() {
        // Official: while base > 32 { assert even; /= 2 }.
        assert_eq!(base_resolution(2).unwrap(), 2);
        assert_eq!(base_resolution(16).unwrap(), 16);
        assert_eq!(base_resolution(32).unwrap(), 32);
        assert_eq!(base_resolution(64).unwrap(), 32);
        assert_eq!(base_resolution(512).unwrap(), 32);
        assert_eq!(base_resolution(4096).unwrap(), 32);
        assert_eq!(base_resolution(100).unwrap(), 25);
        assert_eq!(base_resolution(36).unwrap(), 18);
        assert_eq!(base_resolution(34).unwrap(), 17);
        // Odd resolutions above 32 fail the official evenness assertion.
        assert!(base_resolution(33).is_err());
        assert!(base_resolution(1000).is_err());
    }

    #[test]
    fn sphere_remesh_is_closed_and_outward_oriented() {
        let (vertices, faces) = uv_sphere(20, 28, 0.5);
        let options = RemeshDcOptions::from_aabb([[-0.5; 3], [0.5; 3]], 24, 1.0, 0.0);
        let mesh = remesh(&vertices, &faces, &options);
        let volumes = assert_closed_triangle_mesh(&mesh, 2);
        assert_vertices_near_surface(&mesh, &vertices, &faces, &options, 2.0);

        // `UDF - eps` has two closed components: the outer offset sphere at
        // +eps (positive/outward) and the inner offset sphere at -eps
        // (negative/inward).
        assert_eq!(2, volumes.len());
        let outer = volumes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let inner = volumes.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(
            outer > 0.0,
            "the outer shell must wind outward: {volumes:?}"
        );
        assert!(inner < 0.0, "the inner shell must wind inward: {volumes:?}");
        assert!(
            outer > -inner,
            "the outer shell must enclose the inner one: {volumes:?}"
        );

        let eps = (options.band * options.scale / options.resolution as f32) as f64;
        let expected_outer = 4.0 / 3.0 * std::f64::consts::PI * (0.5 + eps).powi(3);
        let expected_inner = 4.0 / 3.0 * std::f64::consts::PI * (0.5 - eps).powi(3);
        assert!(
            (outer - expected_outer).abs() < 0.1 * expected_outer,
            "outer shell volume {outer} far from {expected_outer}"
        );
        assert!(
            (inner + expected_inner).abs() < 0.1 * expected_inner,
            "inner shell volume {inner} far from {}",
            -expected_inner
        );

        // The signed volume of the two shells cancels to the shell volume and
        // stays positive.
        let volume = signed_volume(&mesh);
        assert!(
            volume > 0.0,
            "outward winding must give a positive signed volume, got {volume}"
        );
        let expected_shell = expected_outer - expected_inner;
        assert!(
            (volume - expected_shell).abs() < 0.1 * expected_shell,
            "remeshed shell volume {volume} far from {expected_shell}"
        );
    }

    #[test]
    fn box_remesh_is_closed() {
        let (vertices, faces) = cube();
        let options = RemeshDcOptions::from_aabb([[0.0; 3], [1.0; 3]], 24, 1.0, 0.0);
        let mesh = remesh(&vertices, &faces, &options);
        let volumes = assert_closed_triangle_mesh(&mesh, 2);
        assert_vertices_near_surface(&mesh, &vertices, &faces, &options, 2.0);

        // Two closed components: the outer Minkowski sum of the unit cube with
        // a ball of radius eps (positive) and the sharp eroded cube
        // `[eps, 1 - eps]^3` (negative).
        assert_eq!(2, volumes.len());
        let outer = volumes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let inner = volumes.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(
            outer > 0.0,
            "the outer shell must wind outward: {volumes:?}"
        );
        assert!(inner < 0.0, "the inner shell must wind inward: {volumes:?}");

        let eps = (options.band * options.scale / options.resolution as f32) as f64;
        let expected_outer = 1.0
            + 6.0 * eps
            + 3.0 * std::f64::consts::PI * eps * eps
            + 4.0 / 3.0 * std::f64::consts::PI * eps.powi(3);
        let expected_inner = (1.0 - 2.0 * eps).powi(3);
        assert!(
            (outer - expected_outer).abs() < 0.1 * expected_outer,
            "outer shell volume {outer} far from {expected_outer}"
        );
        assert!(
            (inner + expected_inner).abs() < 0.1 * expected_inner,
            "inner shell volume {inner} far from {}",
            -expected_inner
        );

        let volume = signed_volume(&mesh);
        assert!(volume > 0.0, "outward winding must give a positive volume");
        let expected_shell = expected_outer - expected_inner;
        assert!(
            (volume - expected_shell).abs() < 0.1 * expected_shell,
            "remeshed shell volume {volume} far from {expected_shell}"
        );
    }

    #[test]
    fn triangle_count_scales_with_resolution() {
        let (vertices, faces) = uv_sphere(16, 20, 0.5);
        let coarse = remesh(
            &vertices,
            &faces,
            &RemeshDcOptions::from_aabb([[-0.5; 3], [0.5; 3]], 16, 1.0, 0.0),
        );
        let fine = remesh(
            &vertices,
            &faces,
            &RemeshDcOptions::from_aabb([[-0.5; 3], [0.5; 3]], 32, 1.0, 0.0),
        );
        assert_eq!(0, coarse.faces.len() % 6);
        assert_eq!(0, fine.faces.len() % 6);
        let coarse_triangles = coarse.faces.len() / 3;
        let fine_triangles = fine.faces.len() / 3;
        // Surface area scales with resolution^2; allow a wide band around the
        // expected 4x ratio (the offset thickness shifts it slightly).
        assert!(
            fine_triangles > coarse_triangles * 2,
            "expected roughly 4x more triangles, got {coarse_triangles} -> {fine_triangles}"
        );
        assert!(
            fine_triangles < coarse_triangles * 8,
            "expected roughly 4x more triangles, got {coarse_triangles} -> {fine_triangles}"
        );
    }

    #[test]
    fn non_power_of_two_resolution_uses_official_halving() {
        let (vertices, faces) = uv_sphere(14, 18, 0.5);
        // 36 halves to 18, then doubles back to 36 once.
        let options = RemeshDcOptions::from_aabb([[-0.5; 3], [0.5; 3]], 36, 1.0, 0.0);
        let mesh = remesh(&vertices, &faces, &options);
        assert_closed_triangle_mesh(&mesh, 2);
    }

    #[test]
    fn project_back_monotonically_reduces_surface_distance() {
        let (vertices, faces) = uv_sphere(14, 18, 0.5);
        let bvh = MeshBvh::build(&vertices, &faces).expect("build bvh");
        let base = RemeshDcOptions::from_aabb([[-0.5; 3], [0.5; 3]], 16, 1.0, 0.0);

        let mean_distance = |factor: f32| -> f32 {
            let options = RemeshDcOptions {
                project_back: factor,
                ..base
            };
            let mesh = remesh_narrow_band_dc(&vertices, &faces, &bvh, &options).expect("remesh");
            assert!(!mesh.vertices.is_empty());
            let total: f32 = mesh
                .vertices
                .chunks_exact(3)
                .map(|chunk| {
                    bvh.closest_point([chunk[0], chunk[1], chunk[2]])
                        .unsigned_distance
                })
                .sum();
            total / (mesh.vertices.len() / 3) as f32
        };

        let none = mean_distance(0.0);
        let half = mean_distance(0.5);
        let full = mean_distance(1.0);
        assert!(none > 0.0, "the raw dual mesh must sit off the surface");
        assert!(
            half < none,
            "half projection must reduce the mean distance: {half} vs {none}"
        );
        assert!(
            full < half,
            "full projection must reduce it further: {full} vs {half}"
        );
        assert!(
            full < 1e-4,
            "full projection must land on the surface, got {full}"
        );
    }

    #[test]
    fn band_controls_the_offset_surface() {
        let (vertices, faces) = uv_sphere(14, 18, 0.5);
        let bvh = MeshBvh::build(&vertices, &faces).expect("build bvh");
        let base = RemeshDcOptions::from_aabb([[-0.5; 3], [0.5; 3]], 16, 1.0, 0.0);

        let mean_distance = |band: f32| -> (f32, usize) {
            let options = RemeshDcOptions { band, ..base };
            let mesh = remesh_narrow_band_dc(&vertices, &faces, &bvh, &options).expect("remesh");
            let total: f32 = mesh
                .vertices
                .chunks_exact(3)
                .map(|chunk| {
                    bvh.closest_point([chunk[0], chunk[1], chunk[2]])
                        .unsigned_distance
                })
                .sum();
            (total / (mesh.vertices.len() / 3) as f32, mesh.faces.len())
        };

        // band = 0: every corner value is an unsigned distance (>= 0), so the
        // official strict sign-change test never fires and the output is
        // empty.
        let (_, band_zero_faces) = mean_distance(0.0);
        assert_eq!(0, band_zero_faces, "band = 0 yields the empty mesh");

        // The extracted surface sits at distance eps = band * spacing.
        let spacing = base.scale / base.resolution as f32;
        let (one, one_faces) = mean_distance(1.0);
        let (two, two_faces) = mean_distance(2.0);
        assert!(one_faces > 0 && two_faces > 0);
        let eps_one = spacing;
        assert!(
            one > 0.5 * eps_one && one < 1.5 * eps_one,
            "band = 1 offset {one} not near eps {eps_one}"
        );
        assert!(
            two > one * 1.4,
            "band = 2 must sit farther out: {two} vs {one}"
        );
    }

    #[test]
    fn remesh_is_deterministic() {
        let (vertices, faces) = uv_sphere(16, 20, 0.5);
        let options = RemeshDcOptions::from_aabb([[-0.5; 3], [0.5; 3]], 20, 1.0, 0.25);
        let first = remesh(&vertices, &faces, &options);
        let second = remesh(&vertices, &faces, &options);
        assert_eq!(first, second);
    }

    #[test]
    fn quad_split_patterns_match_official_tables() {
        assert_eq!(QUAD_SPLIT_1_N, [0, 1, 2, 0, 2, 3]);
        assert_eq!(QUAD_SPLIT_1_P, [0, 2, 1, 0, 3, 2]);
        assert_eq!(QUAD_SPLIT_2_N, [0, 1, 3, 3, 1, 2]);
        assert_eq!(QUAD_SPLIT_2_P, [0, 3, 1, 3, 2, 1]);
    }

    #[test]
    fn adaptive_split_matches_official_alignment_rule() {
        let flat = |points: [[f32; 3]; 4]| -> Vec<f32> {
            points.iter().flatten().copied().collect::<Vec<f32>>()
        };
        let quad = [0, 1, 2, 3];

        // Planar unit square: align0 = 1 > align1 = 0, so split 1 wins.
        // direction -1 uses the `_n` pattern, direction 1 the `_p` pattern.
        let planar = flat([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]);
        assert_eq!(
            [0, 1, 2, 0, 2, 3],
            triangulate_quad(&planar, quad, -1),
            "split 1 must win for a non-degenerate planar quad"
        );
        assert_eq!(
            [0, 2, 1, 0, 3, 2],
            triangulate_quad(&planar, quad, 1),
            "direction 1 selects the official `_p` pattern"
        );

        // Degenerate quad (all four corners coincide): align0 == align1 == 0,
        // and the strict `align0 > align1` tie chooses split 2, exactly like
        // the reference `torch.where`.
        let degenerate = flat([[0.0, 0.0, 0.0]; 4]);
        assert_eq!(
            [0, 1, 3, 3, 1, 2],
            triangulate_quad(&degenerate, quad, -1),
            "a tie chooses split 2 (`_n`)"
        );
        assert_eq!(
            [0, 3, 1, 3, 2, 1],
            triangulate_quad(&degenerate, quad, 1),
            "a tie chooses split 2 (`_p`)"
        );
    }

    #[test]
    fn invalid_options_and_meshes_are_rejected() {
        let (vertices, faces) = uv_sphere(8, 10, 0.5);
        let valid = RemeshDcOptions::from_aabb([[-0.5; 3], [0.5; 3]], 16, 1.0, 0.0);
        let bvh = MeshBvh::build(&vertices, &faces).expect("build bvh");
        let run =
            |options: &RemeshDcOptions| remesh_narrow_band_dc(&vertices, &faces, &bvh, options);
        assert!(run(&valid).is_ok());

        assert!(
            run(&RemeshDcOptions {
                resolution: 1,
                ..valid
            })
            .is_err()
        );
        assert!(
            run(&RemeshDcOptions {
                resolution: MAX_GRID_SIZE_PER_AXIS + 1,
                ..valid
            })
            .is_err()
        );
        // Odd resolution above 32 fails the official base-resolution halving.
        assert!(
            run(&RemeshDcOptions {
                resolution: 33,
                ..valid
            })
            .is_err()
        );
        assert!(
            run(&RemeshDcOptions {
                scale: 0.0,
                ..valid
            })
            .is_err()
        );
        assert!(
            run(&RemeshDcOptions {
                scale: f32::NAN,
                ..valid
            })
            .is_err()
        );
        assert!(
            run(&RemeshDcOptions {
                center: [f32::NAN, 0.0, 0.0],
                ..valid
            })
            .is_err()
        );
        // The grid cube must stay inside the finite f32 range.
        assert!(
            run(&RemeshDcOptions {
                center: [f32::MAX, 0.0, 0.0],
                scale: f32::MAX,
                ..valid
            })
            .is_err()
        );
        assert!(
            run(&RemeshDcOptions {
                band: -0.5,
                ..valid
            })
            .is_err()
        );
        assert!(
            run(&RemeshDcOptions {
                band: f32::INFINITY,
                ..valid
            })
            .is_err()
        );
        assert!(
            run(&RemeshDcOptions {
                project_back: f32::NAN,
                ..valid
            })
            .is_err()
        );
        assert!(
            run(&RemeshDcOptions {
                project_back: -0.5,
                ..valid
            })
            .is_err()
        );
        assert!(
            run(&RemeshDcOptions {
                project_back: f32::NEG_INFINITY,
                ..valid
            })
            .is_err()
        );

        // Malformed buffers are rejected before any grid work.
        assert!(remesh_narrow_band_dc(&vertices[..8], &faces, &bvh, &valid).is_err());
        assert!(remesh_narrow_band_dc(&vertices, &faces[..2], &bvh, &valid).is_err());
        assert!(remesh_narrow_band_dc(&[], &faces, &bvh, &valid).is_err());
        assert!(remesh_narrow_band_dc(&vertices, &[], &bvh, &valid).is_err());
        let mut bad_faces = faces.clone();
        bad_faces[0] = u32::MAX;
        assert!(remesh_narrow_band_dc(&vertices, &bad_faces, &bvh, &valid).is_err());
    }

    // ---- gated official CuMesh goldens (OR1) ------------------------------

    fn read_npy_gz_f32(path: &std::path::Path) -> Vec<f32> {
        use flate2::read::GzDecoder;
        use ndarray::Array2;
        use ndarray_npy::ReadNpyExt;

        let file = std::fs::File::open(path).expect("open fixture");
        let decoder = GzDecoder::new(std::io::BufReader::new(file));
        let array: Array2<f32> = Array2::read_npy(decoder).expect("read npy");
        array.iter().copied().collect()
    }

    fn read_npy_gz_u32(path: &std::path::Path) -> Vec<u32> {
        use flate2::read::GzDecoder;
        use ndarray::Array2;
        use ndarray_npy::ReadNpyExt;

        let file = std::fs::File::open(path).expect("open fixture");
        let decoder = GzDecoder::new(std::io::BufReader::new(file));
        let array: Array2<i32> = Array2::read_npy(decoder).expect("read npy");
        array.iter().map(|&value| value as u32).collect()
    }

    /// Order-insensitive vertex-set comparison: every golden vertex must have
    /// its own computed vertex within `tolerance`. The official vertex order
    /// is a hash/index artifact, so it is not compared.
    fn assert_vertex_sets_match(ours: &[f32], golden: &[f32], tolerance: f32, label: &str) {
        use std::collections::HashMap;

        assert_eq!(ours.len(), golden.len(), "{label}: vertex buffer length");
        let cell = (tolerance * 2.0).max(1e-6);
        let cell_key = |point: [f32; 3]| -> (i32, i32, i32) {
            (
                (point[0] / cell).round() as i32,
                (point[1] / cell).round() as i32,
                (point[2] / cell).round() as i32,
            )
        };
        let mut buckets: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (index, chunk) in ours.chunks_exact(3).enumerate() {
            buckets
                .entry(cell_key([chunk[0], chunk[1], chunk[2]]))
                .or_default()
                .push(index);
        }

        let mut used = vec![false; ours.len() / 3];
        let mut worst = 0.0f32;
        for chunk in golden.chunks_exact(3) {
            let point = [chunk[0], chunk[1], chunk[2]];
            let (kx, ky, kz) = cell_key(point);
            let mut best = f32::INFINITY;
            let mut best_index = usize::MAX;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let Some(indices) = buckets.get(&(kx + dx, ky + dy, kz + dz)) else {
                            continue;
                        };
                        for &index in indices {
                            if used[index] {
                                continue;
                            }
                            let candidate =
                                [ours[index * 3], ours[index * 3 + 1], ours[index * 3 + 2]];
                            let distance = ((point[0] - candidate[0]).powi(2)
                                + (point[1] - candidate[1]).powi(2)
                                + (point[2] - candidate[2]).powi(2))
                            .sqrt();
                            if distance < best {
                                best = distance;
                                best_index = index;
                            }
                        }
                    }
                }
            }
            assert!(
                best_index != usize::MAX && best <= tolerance,
                "{label}: no computed vertex within {tolerance} of golden {point:?} \
                 (nearest unused distance {best})"
            );
            used[best_index] = true;
            worst = worst.max(best);
        }
        eprintln!("{label}: max golden/computed vertex distance {worst}");
    }

    /// Comparison against the official `cumesh.remeshing.remesh_narrow_band_dc`
    /// goldens produced by OR1 (`tests/fixtures/remesh/`, see
    /// `scripts/generate_cumesh_remesh_goldens.py`). Skips with a message when
    /// the fixture directory is absent; does not fail CI without the fixtures.
    #[test]
    fn official_cumesh_goldens_match() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/remesh");
        if !root.is_dir() {
            eprintln!(
                "skipping official CuMesh remesh goldens: {} is absent",
                root.display()
            );
            return;
        }

        let mut compared = 0usize;
        for name in ["cube_res32", "cube_res64", "sphere_res32", "sphere_res64"] {
            let dir = root.join(name);
            if !dir.join("params.json").is_file() {
                eprintln!("skipping {name}: fixture directory is incomplete");
                continue;
            }
            let params: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(dir.join("params.json"))
                    .expect("read remesh fixture params"),
            )
            .expect("parse remesh fixture params");
            let param_f32 = |key: &str| -> f32 {
                params[key]
                    .as_f64()
                    .unwrap_or_else(|| panic!("{name}: missing numeric param `{key}`"))
                    as f32
            };
            let center: Vec<f32> = params["center"]
                .as_array()
                .unwrap_or_else(|| panic!("{name}: missing center array"))
                .iter()
                .map(|value| value.as_f64().expect("center entry") as f32)
                .collect();
            assert_eq!(3, center.len(), "{name}: center length");
            let options = RemeshDcOptions {
                center: [center[0], center[1], center[2]],
                scale: param_f32("scale"),
                resolution: params["resolution"]
                    .as_u64()
                    .unwrap_or_else(|| panic!("{name}: missing integer param `resolution`"))
                    as u32,
                band: param_f32("band"),
                project_back: param_f32("project_back"),
            };

            let input_vertices = read_npy_gz_f32(&dir.join("input_vertices.npy.gz"));
            let input_faces = read_npy_gz_u32(&dir.join("input_faces.npy.gz"));
            let golden_vertices = read_npy_gz_f32(&dir.join("vertices.npy.gz"));
            let golden_faces = read_npy_gz_u32(&dir.join("faces.npy.gz"));

            let bvh = MeshBvh::build(&input_vertices, &input_faces).expect("build fixture bvh");
            let mesh = remesh_narrow_band_dc(&input_vertices, &input_faces, &bvh, &options)
                .unwrap_or_else(|error| panic!("{name}: remesh failed: {error}"));

            // Exact topology parity: the official quad count is `faces / 6`.
            assert_eq!(
                golden_vertices.len(),
                mesh.vertices.len(),
                "{name}: vertex count"
            );
            assert_eq!(golden_faces.len(), mesh.faces.len(), "{name}: face count");
            assert_eq!(
                0,
                mesh.faces.len() % 6,
                "{name}: two triangles per official quad"
            );

            assert_vertex_sets_match(&mesh.vertices, &golden_vertices, 1e-4, name);

            let golden_volume = signed_volume_parts(&golden_vertices, &golden_faces);
            let volume = signed_volume(&mesh);
            assert!(golden_volume > 0.0, "{name}: golden winding is outward");
            assert!(
                (volume - golden_volume).abs() <= 1e-4 * golden_volume.abs().max(1.0),
                "{name}: volume {volume} vs golden {golden_volume}"
            );
            compared += 1;
        }
        assert!(compared > 0, "no official remesh goldens were compared");
    }
}
