//! Narrow-band dual-contouring remesher (TRELLIS.2 `to_glb` remesh branch).
//!
//! Rust port of the observable algorithm of
//! `cumesh.remeshing.remesh_narrow_band_dc` as called from the official
//! postprocessor (`o-voxel/o_voxel/postprocess.py:171-180`):
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
//! `cumesh` is a pip dependency and its sources are not part of the reference
//! tree. This module therefore implements exactly the algorithm fixed by the
//! TR1 brief — uniform grid, narrow band via [`MeshBvh`] closest-point
//! distance, ray-parity SDF sign, dual contouring with the shared QEF solve
//! from the sibling `dual_grid` module, and BVH project-back — and adds no
//! other cumesh parameters.
//!
//! # Algorithm
//!
//! 1. **Grid.** A uniform cubic grid with `resolution` cells per axis and
//!    cell size `h = scale / resolution`, centered on `center` (the grid cube
//!    spans `center ± scale / 2`). `scale` is the already-expanded value the
//!    call site passes; [`narrow_band_grid_scale`] applies the official
//!    `(resolution + 3 * band) / resolution` factor to a mesh extent
//!    (`postprocess.py:166-180`).
//! 2. **Narrow band.** Candidate cells come from every triangle's AABB
//!    expanded by `band * h`. A candidate is kept when the BVH unsigned
//!    distance from its center to the mesh is at most `band * h`. With the
//!    shipped `band = 1` (`example.py:44`, `app.py:504`) this includes every
//!    cell the surface crosses, because a cell's center is at most
//!    `sqrt(3) / 2 * h` from any point on the cell boundary.
//! 3. **SDF samples.** The eight corners of every band cell receive a signed
//!    distance: magnitude from [`MeshBvh::closest_point`], sign from an
//!    even-odd ray-crossing count. Negative means inside.
//! 4. **Dual contouring.** Every sign-changing cell edge (SDF linearly
//!    interpolated along the edge) contributes a plane QEF term. The normal
//!    is the geometric normal of the BVH-closest triangle at the
//!    intersection point, oriented toward increasing SDF. Each cell with at
//!    least one crossing gets exactly one vertex: the clamped QEF minimizer
//!    (`dual_grid::solve_qef_cell`) or, when no usable normal exists
//!    (degenerate triangles), the mean of its crossing points.
//! 5. **Quads.** One quad per sign-changing grid edge joins the vertices of
//!    the four cells around that edge, wound so the face normal points along
//!    increasing SDF (outward). Each grid edge is emitted exactly once, by
//!    the cell that owns it at its max corner.
//! 6. **Project back.** Each output vertex moves toward its BVH closest point
//!    by the factor `project_back` when it is positive: `0` disables (the
//!    shipped demo value, `example.py:45`, `app.py:505`), `1` places every
//!    vertex on the input surface. Matching the official `if project_back >
//!    0` guard (`remeshing.py:238`), a negative factor is rejected by the
//!    option validation instead of extrapolating away from the surface.
//!
//! # Degenerate cases and assumptions
//!
//! * **Closed input meshes.** The ray-parity sign is only meaningful for a
//!   closed mesh with a well-defined inside. Open meshes can produce
//!   arbitrary signs; the official pipeline fills holes
//!   (`postprocess.py:108-113`) before remeshing, and callers should do the
//!   same (`geometry::mesh_ops::fill_holes`).
//! * **Ray direction.** The parity ray is a fixed generic direction with all
//!   components non-zero, so axis-aligned geometry (the box test) is never
//!   hit edge-on. Intersection ray parameters are de-duplicated within a
//!   small tolerance: a ray passing exactly through a shared triangle edge
//!   or vertex reports the same parameter from each incident triangle and
//!   counts once. Rays that start exactly on the surface or graze a
//!   silhouette tangentially are measure-zero inputs without special
//!   handling; a corner sampled exactly on the surface gets distance `0`,
//!   is classified as inside (`signed <= 0`), and interpolates to its own
//!   corner.
//! * **Sub-cell features.** As with every grid-sampled dual contourer,
//!   surface detail that does not flip the sign of any cell corner is not
//!   captured.
//! * **Band and connectivity.** A quad is emitted only when all four cells
//!   around its grid edge have a vertex. For `band >= 1` that always holds;
//!   with a smaller band a surface-crossing cell can fall outside the band
//!   and the affected quad is dropped (documented, not silently
//!   reclassified).
//! * **Input contract.** `bvh` must have been built over the same
//!   `(vertices, faces)` buffers; the buffers are validated here, but the
//!   tree is not re-derived. `resolution` must be in
//!   `2..=MAX_GRID_SIZE_PER_AXIS`, `center` / `band` / `project_back` must be
//!   finite, `band >= 0`, `project_back >= 0`, and `scale > 0`.
//!   `project_back` is a raw lerp factor, so values above 1 extrapolate
//!   toward (and past) the closest point, matching the official call.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::error::{Lux3dError, Result};

use super::dual_grid::{plane_q, q4_from_sym, solve_qef_cell, MAX_GRID_SIZE_PER_AXIS};
use super::mesh_bvh::MeshBvh;

/// Fixed generic ray direction for the even-odd inside/outside test. Every
/// component is non-zero, so rays never run parallel to axis-aligned triangle
/// edges (the closed synthetic meshes exercised by the tests).
const RAY_DIRECTION: [f64; 3] = [1.0, 0.371_390_676_354_103_7, 0.618_033_988_749_894_9];

/// The 12 cube edges as local-corner index pairs. Local corner index bits:
/// bit 0 = +x, bit 1 = +y, bit 2 = +z.
const CUBE_EDGES: [(usize, usize); 12] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7), // along x
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7), // along y
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7), // along z
];

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

/// Remeshed mesh: quad topology, world-space vertex positions.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RemeshDcMesh {
    /// Flat `[x, y, z, ...]` vertex positions (QEF-placed, clamped to their
    /// cells, then projected back when requested).
    pub vertices: Vec<f32>,
    /// Flat quad indices, four per face, wound outward (toward increasing
    /// SDF).
    pub faces: Vec<u32>,
}

/// Remeshes `(vertices, faces)` with narrow-band dual contouring.
///
/// * `vertices`: flat `[x, y, z, ...]` f32 positions.
/// * `faces`: flat `[F, 3]` zero-based triangle indices.
/// * `bvh`: triangle BVH built over the same `(vertices, faces)` buffers
///   (`MeshBvh::build`); the official call passes the pre-built
///   `cumesh.cuBVH` the same way (`postprocess.py:122, 179`).
/// * `options`: grid center/scale/resolution, band and project-back factor.
///
/// Returns quad faces (four indices each). See the module docs for the
/// algorithm and the documented degenerate-case assumptions.
pub fn remesh_narrow_band_dc<'mesh>(
    vertices: &'mesh [f32],
    faces: &'mesh [u32],
    bvh: &MeshBvh<'mesh>,
    options: &RemeshDcOptions,
) -> Result<RemeshDcMesh> {
    validate_options(options)?;
    validate_mesh(vertices, faces)?;

    let resolution = options.resolution as i32;
    let spacing = options.scale / options.resolution as f32;
    if !spacing.is_finite() || spacing <= 0.0 {
        return Err(Lux3dError::InvalidInput(
            "remesh narrow-band dual contouring produced a degenerate cell size",
        ));
    }
    let origin = [
        options.center[0] - options.scale * 0.5,
        options.center[1] - options.scale * 0.5,
        options.center[2] - options.scale * 0.5,
    ];
    if origin.iter().any(|value| !value.is_finite()) {
        return Err(Lux3dError::InvalidInput(
            "remesh narrow-band dual contouring grid origin is not finite",
        ));
    }
    let band_distance = options.band * spacing;
    if !band_distance.is_finite() {
        return Err(Lux3dError::InvalidInput(
            "remesh narrow-band dual contouring band distance is not finite",
        ));
    }

    // ---- step 1+2: narrow-band cells from triangle-expanded AABB checks ----
    let mut candidates: FxHashSet<[i32; 3]> = FxHashSet::default();
    for triangle in faces.chunks_exact(3) {
        let a = vertex_f32(vertices, triangle[0]);
        let b = vertex_f32(vertices, triangle[1]);
        let c = vertex_f32(vertices, triangle[2]);
        let mut low = [0i32; 3];
        let mut high = [0i32; 3];
        for axis in 0..3 {
            let minimum = a[axis].min(b[axis]).min(c[axis]) - band_distance;
            let maximum = a[axis].max(b[axis]).max(c[axis]) + band_distance;
            // Over-inclusive by at most one cell per axis: the distance
            // filter below removes candidates whose center is too far.
            low[axis] = clamp_cell_index((minimum - origin[axis]) / spacing, resolution);
            high[axis] = clamp_cell_index((maximum - origin[axis]) / spacing, resolution);
        }
        if low[0] > high[0] || low[1] > high[1] || low[2] > high[2] {
            continue;
        }
        for z in low[2]..=high[2] {
            for y in low[1]..=high[1] {
                for x in low[0]..=high[0] {
                    candidates.insert([x, y, z]);
                }
            }
        }
    }

    let mut active_cells: Vec<[i32; 3]> = Vec::new();
    for cell in candidates {
        let center = [
            origin[0] + (cell[0] as f32 + 0.5) * spacing,
            origin[1] + (cell[1] as f32 + 0.5) * spacing,
            origin[2] + (cell[2] as f32 + 0.5) * spacing,
        ];
        if bvh.closest_point(center).unsigned_distance <= band_distance {
            active_cells.push(cell);
        }
    }
    // Deterministic vertex/face order: the candidate set is hash-ordered, so
    // sort by grid key before anything allocates output indices.
    active_cells.sort_unstable_by_key(|cell| cell_key(*cell));

    // ---- step 3+4: SDF corners and one QEF vertex per crossing cell --------
    let parity = RayParity::build(vertices, faces);
    let mut sampler = CornerSampler {
        bvh,
        parity: &parity,
        origin,
        spacing,
        values: FxHashMap::default(),
        scratch: Vec::new(),
    };

    let mut output_vertices: Vec<f32> = Vec::new();
    let mut cell_vertices: FxHashMap<u64, u32> = FxHashMap::default();

    for cell in &active_cells {
        let mut corner_positions = [[0.0f32; 3]; 8];
        let mut corner_sdf = [0.0f32; 8];
        for (local, (position, value)) in corner_positions
            .iter_mut()
            .zip(corner_sdf.iter_mut())
            .enumerate()
        {
            let corner = [
                (cell[0] + (local & 1) as i32) as u32,
                (cell[1] + ((local >> 1) & 1) as i32) as u32,
                (cell[2] + ((local >> 2) & 1) as i32) as u32,
            ];
            *position = [
                origin[0] + corner[0] as f32 * spacing,
                origin[1] + corner[1] as f32 * spacing,
                origin[2] + corner[2] as f32 * spacing,
            ];
            *value = sampler.value(corner);
        }

        let mut q_sym = [0.0f32; 10];
        let mut crossing_sum = [0.0f64; 3];
        let mut crossing_count = 0u32;
        let mut plane_terms = 0u32;

        for &(corner_a, corner_b) in CUBE_EDGES.iter() {
            let sdf_a = corner_sdf[corner_a];
            let sdf_b = corner_sdf[corner_b];
            if (sdf_a <= 0.0) == (sdf_b <= 0.0) {
                continue;
            }
            let position_a = corner_positions[corner_a];
            let position_b = corner_positions[corner_b];
            let denominator = sdf_a - sdf_b;
            let t = if denominator != 0.0 {
                sdf_a / denominator
            } else {
                0.5
            };
            let intersection = [
                position_a[0] + (position_b[0] - position_a[0]) * t,
                position_a[1] + (position_b[1] - position_a[1]) * t,
                position_a[2] + (position_b[2] - position_a[2]) * t,
            ];
            crossing_sum[0] += intersection[0] as f64;
            crossing_sum[1] += intersection[1] as f64;
            crossing_sum[2] += intersection[2] as f64;
            crossing_count += 1;

            // Plane QEF term from the BVH-closest triangle at the crossing,
            // oriented along increasing SDF (outward for the negative-inside
            // convention).
            let hit = bvh.closest_point(intersection);
            let Some(normal) = triangle_normal(vertices, faces, hit.tri_id) else {
                continue;
            };
            let edge = [
                position_b[0] - position_a[0],
                position_b[1] - position_a[1],
                position_b[2] - position_a[2],
            ];
            let increasing = if sdf_b > sdf_a { 1.0f32 } else { -1.0f32 };
            let normal = if dot_f32(normal, edge) * increasing < 0.0 {
                [-normal[0], -normal[1], -normal[2]]
            } else {
                normal
            };
            let term = plane_q(normal, -dot_f32(normal, intersection));
            for (accumulated, value) in q_sym.iter_mut().zip(term.iter()) {
                *accumulated += *value;
            }
            plane_terms += 1;
        }

        if crossing_count == 0 {
            continue;
        }

        let min_corner = [
            origin[0] as f64 + cell[0] as f64 * spacing as f64,
            origin[1] as f64 + cell[1] as f64 * spacing as f64,
            origin[2] as f64 + cell[2] as f64 * spacing as f64,
        ];
        let max_corner = [
            min_corner[0] + spacing as f64,
            min_corner[1] + spacing as f64,
            min_corner[2] + spacing as f64,
        ];
        let mut position = if plane_terms > 0 {
            let q = q4_from_sym(&q_sym);
            solve_qef_cell(&q, min_corner, max_corner)
        } else {
            // No usable triangle normal (degenerate input triangles): fall
            // back to the mean of the edge intersection points, which lies in
            // the cell by construction.
            let inverse = 1.0 / crossing_count as f64;
            [
                crossing_sum[0] * inverse,
                crossing_sum[1] * inverse,
                crossing_sum[2] * inverse,
            ]
        };
        for axis in 0..3 {
            position[axis] = position[axis].clamp(min_corner[axis], max_corner[axis]);
        }

        let vertex_index = (output_vertices.len() / 3) as u32;
        output_vertices.push(position[0] as f32);
        output_vertices.push(position[1] as f32);
        output_vertices.push(position[2] as f32);
        cell_vertices.insert(cell_key(*cell), vertex_index);
    }

    // ---- step 5: one quad per sign-changing grid edge ---------------------
    // Each grid edge is owned by the cell for which the edge starts at the
    // cell's max corner in the two perpendicular axes and at its low corner
    // along the edge axis, so every edge is visited exactly once.
    let mut output_faces: Vec<u32> = Vec::new();
    for cell in &active_cells {
        if !cell_vertices.contains_key(&cell_key(*cell)) {
            continue;
        }
        for axis in 0..3usize {
            let axis_u = (axis + 1) % 3;
            let axis_v = (axis + 2) % 3;

            let mut base_corner = [cell[0] as u32, cell[1] as u32, cell[2] as u32];
            base_corner[axis_u] += 1;
            base_corner[axis_v] += 1;
            let mut high_corner = base_corner;
            high_corner[axis] += 1;

            let sdf_low = sampler.value(base_corner);
            let sdf_high = sampler.value(high_corner);
            if (sdf_low <= 0.0) == (sdf_high <= 0.0) {
                continue;
            }

            // Cells around the edge, counter-clockwise in the (u, v) plane so
            // that the quad normal is +axis when the SDF increases along the
            // edge.
            const OFFSETS: [(i32, i32); 4] = [(-1, -1), (0, -1), (0, 0), (-1, 0)];
            let mut quad = [0u32; 4];
            let mut complete = true;
            for (index, (du, dv)) in OFFSETS.iter().enumerate() {
                let mut neighbor = [0i32; 3];
                neighbor[axis] = base_corner[axis] as i32;
                neighbor[axis_u] = base_corner[axis_u] as i32 + du;
                neighbor[axis_v] = base_corner[axis_v] as i32 + dv;
                if neighbor
                    .iter()
                    .any(|&value| value < 0 || value >= resolution)
                {
                    complete = false;
                    break;
                }
                match cell_vertices.get(&cell_key(neighbor)) {
                    Some(&vertex) => quad[index] = vertex,
                    None => {
                        complete = false;
                        break;
                    }
                }
            }
            if !complete {
                continue;
            }
            debug_assert_eq!(quad[0], cell_vertices[&cell_key(*cell)]);
            if sdf_low >= sdf_high {
                quad.reverse();
            }
            output_faces.extend_from_slice(&quad);
        }
    }

    // ---- step 6: project back to the input surface ------------------------
    // Official guard: `if project_back > 0` (`remeshing.py:238`); `0`
    // disables and negative factors are rejected by validation.
    if options.project_back > 0.0 {
        for vertex in output_vertices.chunks_exact_mut(3) {
            let point = [vertex[0], vertex[1], vertex[2]];
            let hit = bvh.closest_point(point);
            for axis in 0..3 {
                vertex[axis] =
                    point[axis] + options.project_back * (hit.position[axis] - point[axis]);
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
// Corner sampling (SDF magnitude from the BVH, sign from ray parity)
// ---------------------------------------------------------------------------

struct CornerSampler<'a, 'mesh> {
    bvh: &'a MeshBvh<'mesh>,
    parity: &'a RayParity<'mesh>,
    origin: [f32; 3],
    spacing: f32,
    values: FxHashMap<u64, f32>,
    scratch: Vec<f64>,
}

impl CornerSampler<'_, '_> {
    fn value(&mut self, corner: [u32; 3]) -> f32 {
        let key = corner_key(corner);
        if let Some(&value) = self.values.get(&key) {
            return value;
        }
        let position = [
            self.origin[0] + corner[0] as f32 * self.spacing,
            self.origin[1] + corner[1] as f32 * self.spacing,
            self.origin[2] + corner[2] as f32 * self.spacing,
        ];
        let inside = self.parity.is_inside(position, &mut self.scratch);
        let distance = self.bvh.closest_point(position).unsigned_distance;
        let value = if inside { -distance } else { distance };
        self.values.insert(key, value);
        value
    }
}

/// Even-odd ray-crossing counter with a 2D bucket index over the ray-plane
/// invariant coordinates `u = y dx - dy x`, `w = z dx - dz x` (constant along
/// a ray with direction `(dx, dy, dz)`).
struct RayParity<'a> {
    vertices: &'a [f32],
    faces: &'a [u32],
    min: [f64; 2],
    max: [f64; 2],
    buckets_per_axis: usize,
    inv_extent: [f64; 2],
    buckets: Vec<Vec<u32>>,
    t_epsilon: f64,
}

impl<'a> RayParity<'a> {
    fn build(vertices: &'a [f32], faces: &'a [u32]) -> Self {
        let direction = RAY_DIRECTION;
        let triangle_count = faces.len() / 3;

        let mut triangle_boxes: Vec<([f64; 2], [f64; 2])> = Vec::with_capacity(triangle_count);
        let mut min = [f64::INFINITY; 2];
        let mut max = [f64::NEG_INFINITY; 2];
        let mut mesh_min = [f64::INFINITY; 3];
        let mut mesh_max = [f64::NEG_INFINITY; 3];

        for triangle in faces.chunks_exact(3) {
            let a = vertex_f64(vertices, triangle[0]);
            let b = vertex_f64(vertices, triangle[1]);
            let c = vertex_f64(vertices, triangle[2]);
            for axis in 0..3 {
                mesh_min[axis] = mesh_min[axis].min(a[axis]).min(b[axis]).min(c[axis]);
                mesh_max[axis] = mesh_max[axis].max(a[axis]).max(b[axis]).max(c[axis]);
            }
            let (au, aw) = invariant(a, direction);
            let (bu, bw) = invariant(b, direction);
            let (cu, cw) = invariant(c, direction);
            let low = [au.min(bu).min(cu), aw.min(bw).min(cw)];
            let high = [au.max(bu).max(cu), aw.max(bw).max(cw)];
            min[0] = min[0].min(low[0]);
            min[1] = min[1].min(low[1]);
            max[0] = max[0].max(high[0]);
            max[1] = max[1].max(high[1]);
            triangle_boxes.push((low, high));
        }
        debug_assert_eq!(triangle_boxes.len(), triangle_count);

        let diagonal = ((mesh_max[0] - mesh_min[0]).powi(2)
            + (mesh_max[1] - mesh_min[1]).powi(2)
            + (mesh_max[2] - mesh_min[2]).powi(2))
        .sqrt();
        let t_epsilon = (diagonal * 1e-6).max(f64::MIN_POSITIVE);
        let padding = (diagonal * 1e-5).max(1e-9);
        for (low, high) in &mut triangle_boxes {
            low[0] -= padding;
            low[1] -= padding;
            high[0] += padding;
            high[1] += padding;
        }
        min[0] -= padding;
        min[1] -= padding;
        max[0] += padding;
        max[1] += padding;

        let target_buckets = (triangle_count as f64).sqrt().ceil() as usize;
        let buckets_per_axis = target_buckets.clamp(4, 128);
        let mut inv_extent = [0.0f64; 2];
        for axis in 0..2 {
            let span = max[axis] - min[axis];
            if span > 0.0 {
                inv_extent[axis] = buckets_per_axis as f64 / span;
            }
        }

        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); buckets_per_axis * buckets_per_axis];
        for (triangle_id, (low, high)) in triangle_boxes.iter().enumerate() {
            let low_u = bucket_index(low[0], min[0], inv_extent[0], buckets_per_axis);
            let high_u = bucket_index(high[0], min[0], inv_extent[0], buckets_per_axis);
            let low_w = bucket_index(low[1], min[1], inv_extent[1], buckets_per_axis);
            let high_w = bucket_index(high[1], min[1], inv_extent[1], buckets_per_axis);
            for w in low_w..=high_w {
                for u in low_u..=high_u {
                    buckets[w * buckets_per_axis + u].push(triangle_id as u32);
                }
            }
        }

        Self {
            vertices,
            faces,
            min,
            max,
            buckets_per_axis,
            inv_extent,
            buckets,
            t_epsilon,
        }
    }

    /// Even-odd inside test: `true` when the ray from `point` along
    /// [`RAY_DIRECTION`] crosses the mesh an odd number of times.
    fn is_inside(&self, point: [f32; 3], scratch: &mut Vec<f64>) -> bool {
        let origin = [point[0] as f64, point[1] as f64, point[2] as f64];
        let (u, w) = invariant(origin, RAY_DIRECTION);
        if u < self.min[0] || u > self.max[0] || w < self.min[1] || w > self.max[1] {
            return false;
        }
        let bucket = self.bucket_index(w, 1) * self.buckets_per_axis + self.bucket_index(u, 0);

        scratch.clear();
        for &triangle_id in &self.buckets[bucket] {
            let base = triangle_id as usize * 3;
            let a = vertex_f64(self.vertices, self.faces[base]);
            let b = vertex_f64(self.vertices, self.faces[base + 1]);
            let c = vertex_f64(self.vertices, self.faces[base + 2]);
            if let Some(parameter) = ray_triangle_parameter(origin, RAY_DIRECTION, a, b, c) {
                scratch.push(parameter);
            }
        }
        if scratch.is_empty() {
            return false;
        }
        scratch.sort_by(f64::total_cmp);
        // Count distinct crossing parameters: a ray through a shared triangle
        // edge/vertex reports one parameter per incident triangle.
        let mut crossings = 1usize;
        for pair in scratch.windows(2) {
            if pair[1] - pair[0] > self.t_epsilon {
                crossings += 1;
            }
        }
        crossings % 2 == 1
    }

    fn bucket_index(&self, value: f64, axis: usize) -> usize {
        bucket_index(
            value,
            self.min[axis],
            self.inv_extent[axis],
            self.buckets_per_axis,
        )
    }
}

fn bucket_index(value: f64, min: f64, inv_extent: f64, buckets_per_axis: usize) -> usize {
    if inv_extent <= 0.0 {
        return 0;
    }
    let index = ((value - min) * inv_extent).floor();
    if !index.is_finite() {
        return 0;
    }
    index.clamp(0.0, (buckets_per_axis - 1) as f64) as usize
}

/// Ray-plane invariant coordinates for direction `(dx, dy, dz)`, scaled by
/// `dx` to avoid a division: `u = y dx - dy x`, `w = z dx - dz x`.
fn invariant(point: [f64; 3], direction: [f64; 3]) -> (f64, f64) {
    (
        point[1] * direction[0] - direction[1] * point[0],
        point[2] * direction[0] - direction[2] * point[0],
    )
}

/// Möller–Trumbore ray/triangle parameter for `t > 0` (the ray direction is
/// not necessarily normalized, so `t` is in direction-length units).
fn ray_triangle_parameter(
    origin: [f64; 3],
    direction: [f64; 3],
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
) -> Option<f64> {
    let edge1 = sub_f64(b, a);
    let edge2 = sub_f64(c, a);
    let pvec = cross_f64(direction, edge2);
    let determinant = dot_f64(edge1, pvec);
    // Parallel / degenerate triangles: no usable crossing.
    if determinant.abs() <= 1e-12 {
        return None;
    }
    let inverse_determinant = 1.0 / determinant;
    let tvec = sub_f64(origin, a);
    let barycentric_u = dot_f64(tvec, pvec) * inverse_determinant;
    if !(0.0..=1.0).contains(&barycentric_u) {
        return None;
    }
    let qvec = cross_f64(tvec, edge1);
    let barycentric_v = dot_f64(direction, qvec) * inverse_determinant;
    if barycentric_v < 0.0 || barycentric_u + barycentric_v > 1.0 {
        return None;
    }
    let parameter = dot_f64(edge2, qvec) * inverse_determinant;
    if parameter > 0.0 {
        Some(parameter)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

#[inline]
fn cell_key(cell: [i32; 3]) -> u64 {
    debug_assert!(cell.iter().all(|&value| value >= 0));
    cell[0] as u64 | (cell[1] as u64) << 21 | (cell[2] as u64) << 42
}

#[inline]
fn corner_key(corner: [u32; 3]) -> u64 {
    corner[0] as u64 | (corner[1] as u64) << 21 | (corner[2] as u64) << 42
}

/// Clamps a float cell index into `[0, resolution - 1]` (non-finite values
/// clamp to the corresponding grid edge, NaN to the low edge).
fn clamp_cell_index(value: f32, resolution: i32) -> i32 {
    if value.is_nan() {
        return 0;
    }
    if value == f32::NEG_INFINITY {
        return 0;
    }
    if value == f32::INFINITY {
        return resolution - 1;
    }
    (value as i64).clamp(0, resolution as i64 - 1) as i32
}

#[inline]
fn vertex_f32(vertices: &[f32], index: u32) -> [f32; 3] {
    let base = index as usize * 3;
    [vertices[base], vertices[base + 1], vertices[base + 2]]
}

#[inline]
fn vertex_f64(vertices: &[f32], index: u32) -> [f64; 3] {
    let base = index as usize * 3;
    [
        vertices[base] as f64,
        vertices[base + 1] as f64,
        vertices[base + 2] as f64,
    ]
}

/// Unit geometric normal of triangle `triangle_id`, or `None` for
/// degenerate (zero-area) triangles.
fn triangle_normal(vertices: &[f32], faces: &[u32], triangle_id: u32) -> Option<[f32; 3]> {
    let base = triangle_id as usize * 3;
    let a = vertex_f32(vertices, faces[base]);
    let b = vertex_f32(vertices, faces[base + 1]);
    let c = vertex_f32(vertices, faces[base + 2]);
    let normal = cross_f32(sub_f32(b, a), sub_f32(c, a));
    let length_squared = dot_f32(normal, normal);
    if !length_squared.is_finite() || length_squared <= 0.0 {
        return None;
    }
    let inverse_length = 1.0 / length_squared.sqrt();
    Some([
        normal[0] * inverse_length,
        normal[1] * inverse_length,
        normal[2] * inverse_length,
    ])
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

#[inline]
fn sub_f64(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross_f64(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot_f64(a: [f64; 3], b: [f64; 3]) -> f64 {
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

    fn quad_point(mesh: &RemeshDcMesh, index: u32) -> [f32; 3] {
        vertex_f32(&mesh.vertices, index)
    }

    /// Divergence-theorem signed volume of a closed quad mesh (each quad is
    /// split into two triangles). Positive for outward winding.
    fn signed_volume(mesh: &RemeshDcMesh) -> f64 {
        let mut volume = 0.0f64;
        for quad in mesh.faces.chunks_exact(4) {
            let a = vertex_f64(&mesh.vertices, quad[0]);
            let b = vertex_f64(&mesh.vertices, quad[1]);
            let c = vertex_f64(&mesh.vertices, quad[2]);
            let d = vertex_f64(&mesh.vertices, quad[3]);
            volume += dot_f64(a, cross_f64(b, c)) / 6.0;
            volume += dot_f64(a, cross_f64(c, d)) / 6.0;
        }
        volume
    }

    /// Watertightness (every undirected edge shared by exactly two quads)
    /// plus the Euler characteristic of the resulting closed quad surface.
    fn assert_closed_quad_mesh(mesh: &RemeshDcMesh, expected_euler: i64) {
        assert!(!mesh.faces.is_empty(), "remesh must produce faces");
        assert_eq!(mesh.faces.len() % 4, 0, "output faces are quads");
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
        for quad in mesh.faces.chunks_exact(4) {
            for edge in 0..4 {
                let a = quad[edge];
                let b = quad[(edge + 1) % 4];
                assert_ne!(a, b, "degenerate quad edge");
                let key = if a < b { (a, b) } else { (b, a) };
                *edge_counts.entry(key).or_insert(0) += 1;
            }
        }
        for (edge, count) in &edge_counts {
            assert_eq!(*count, 2, "edge {edge:?} used {count} times");
        }

        let euler = vertex_count as i64 - edge_counts.len() as i64 + (mesh.faces.len() / 4) as i64;
        assert_eq!(euler, expected_euler, "Euler characteristic");
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
    fn sphere_remesh_is_closed_and_outward_oriented() {
        let (vertices, faces) = uv_sphere(20, 28, 0.5);
        let options = RemeshDcOptions::from_aabb([[-0.5; 3], [0.5; 3]], 24, 1.0, 0.0);
        let mesh = remesh(&vertices, &faces, &options);
        assert_closed_quad_mesh(&mesh, 2);

        // Every vertex stays inside the grid cube and within one band cell
        // plus the half cell diagonal of the input surface.
        let spacing = options.scale / options.resolution as f32;
        let maximum_distance = (options.band + 0.5 * 3.0f32.sqrt()) * spacing + 1e-4;
        let bvh = MeshBvh::build(&vertices, &faces).expect("build bvh");
        let half_extent = options.scale * 0.5;
        for chunk in mesh.vertices.chunks_exact(3) {
            let point = [chunk[0], chunk[1], chunk[2]];
            let distance = bvh.closest_point(point).unsigned_distance;
            assert!(
                distance <= maximum_distance,
                "vertex {point:?} is {distance} from the surface (limit {maximum_distance})"
            );
            for axis in 0..3 {
                assert!(point[axis] >= options.center[axis] - half_extent - 1e-5);
                assert!(point[axis] <= options.center[axis] + half_extent + 1e-5);
            }
        }

        // Winding: for a convex sphere the quad normal must point away from
        // the center. The Newell normal is used so quads whose first three
        // corners are collinear still report a meaningful direction. Strongly
        // non-planar quads (adjacent QEF vertices clamped onto the same
        // point) can have a geometric normal that disagrees with the
        // topological orientation, so instead of asserting every quad the
        // test requires that inverted normals are a small minority; the
        // global signed volume below is the exact orientation check.
        let mut inverted = 0usize;
        let mut quad_count = 0usize;
        for quad in mesh.faces.chunks_exact(4) {
            quad_count += 1;
            let corners = [
                quad_point(&mesh, quad[0]),
                quad_point(&mesh, quad[1]),
                quad_point(&mesh, quad[2]),
                quad_point(&mesh, quad[3]),
            ];
            let mut normal = [0.0f32; 3];
            let mut centroid = [0.0f32; 3];
            for index in 0..4 {
                let current = corners[index];
                let next = corners[(index + 1) % 4];
                normal[0] += (current[1] - next[1]) * (current[2] + next[2]);
                normal[1] += (current[2] - next[2]) * (current[0] + next[0]);
                normal[2] += (current[0] - next[0]) * (current[1] + next[1]);
                centroid[0] += current[0] * 0.25;
                centroid[1] += current[1] * 0.25;
                centroid[2] += current[2] * 0.25;
            }
            if dot_f32(normal, centroid) <= 0.0 {
                inverted += 1;
            }
        }
        assert!(
            inverted * 10 < quad_count,
            "{inverted} of {quad_count} quads have a geometric inward normal"
        );

        // Global orientation: the divergence-theorem signed volume of a
        // closed outward-wound surface is positive and approximates the
        // sphere volume.
        let volume = signed_volume(&mesh);
        assert!(
            volume > 0.0,
            "outward winding must give a positive signed volume, got {volume}"
        );
        assert!(
            (volume - 4.0 / 3.0 * std::f64::consts::PI * 0.5f64.powi(3)).abs() < 0.15,
            "remeshed volume {volume} far from the sphere volume"
        );
    }

    #[test]
    fn box_remesh_is_closed() {
        let (vertices, faces) = cube();
        let options = RemeshDcOptions::from_aabb([[0.0; 3], [1.0; 3]], 24, 1.0, 0.0);
        let mesh = remesh(&vertices, &faces, &options);
        assert_closed_quad_mesh(&mesh, 2);
        let volume = signed_volume(&mesh);
        assert!(volume > 0.0, "outward winding must give a positive volume");
        assert!(
            (volume - 1.0).abs() < 0.2,
            "remeshed box volume {volume} far from 1.0"
        );
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
        assert!(none > 0.0, "the raw QEF mesh must sit off the surface");
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
    fn remesh_is_deterministic() {
        let (vertices, faces) = uv_sphere(16, 20, 0.5);
        let options = RemeshDcOptions::from_aabb([[-0.5; 3], [0.5; 3]], 20, 1.0, 0.25);
        let first = remesh(&vertices, &faces, &options);
        let second = remesh(&vertices, &faces, &options);
        assert_eq!(first, second);
    }

    #[test]
    fn invalid_options_and_meshes_are_rejected() {
        let (vertices, faces) = uv_sphere(8, 10, 0.5);
        let valid = RemeshDcOptions::from_aabb([[-0.5; 3], [0.5; 3]], 16, 1.0, 0.0);
        let bvh = MeshBvh::build(&vertices, &faces).expect("build bvh");
        let run =
            |options: &RemeshDcOptions| remesh_narrow_band_dc(&vertices, &faces, &bvh, options);
        assert!(run(&valid).is_ok());

        assert!(run(&RemeshDcOptions {
            resolution: 1,
            ..valid
        })
        .is_err());
        assert!(run(&RemeshDcOptions {
            resolution: MAX_GRID_SIZE_PER_AXIS + 1,
            ..valid
        })
        .is_err());
        assert!(run(&RemeshDcOptions {
            scale: 0.0,
            ..valid
        })
        .is_err());
        assert!(run(&RemeshDcOptions {
            scale: f32::NAN,
            ..valid
        })
        .is_err());
        assert!(run(&RemeshDcOptions {
            center: [f32::NAN, 0.0, 0.0],
            ..valid
        })
        .is_err());
        assert!(run(&RemeshDcOptions {
            band: -0.5,
            ..valid
        })
        .is_err());
        assert!(run(&RemeshDcOptions {
            band: f32::INFINITY,
            ..valid
        })
        .is_err());
        assert!(run(&RemeshDcOptions {
            project_back: f32::NAN,
            ..valid
        })
        .is_err());
        assert!(run(&RemeshDcOptions {
            project_back: -0.5,
            ..valid
        })
        .is_err());
        assert!(run(&RemeshDcOptions {
            project_back: f32::NEG_INFINITY,
            ..valid
        })
        .is_err());

        // Malformed buffers are rejected before any grid work.
        assert!(remesh_narrow_band_dc(&vertices[..8], &faces, &bvh, &valid).is_err());
        assert!(remesh_narrow_band_dc(&vertices, &faces[..2], &bvh, &valid).is_err());
        assert!(remesh_narrow_band_dc(&[], &faces, &bvh, &valid).is_err());
        assert!(remesh_narrow_band_dc(&vertices, &[], &bvh, &valid).is_err());
        let mut bad_faces = faces.clone();
        bad_faces[0] = u32::MAX;
        assert!(remesh_narrow_band_dc(&vertices, &bad_faces, &bvh, &valid).is_err());
    }
}
