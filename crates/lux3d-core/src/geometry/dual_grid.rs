//! Mesh → O-Voxel flexible dual-grid voxelization (CPU port).
//!
//! Rust port of the reference implementation used by the TRELLIS.2 texturing
//! pipeline:
//!
//! * wrapper semantics: `o-voxel/o_voxel/convert/flexible_dual_grid.py`
//!   (`mesh_to_flexible_dual_grid`)
//! * CPU kernel: `o-voxel/src/convert/flexible_dual_grid.cpp`
//!   (`mesh_to_flexible_dual_grid_cpu`: `intersect_qef` scan-line
//!   voxelization, `face_qef` plane tests, `boundry_qef` DDA boundary terms,
//!   per-voxel QEF solve with voxel-cell clamping)
//!
//! The TRELLIS.2 shape-SLAT encoder
//! (`trellis2/pipelines/trellis2_texturing.py::encode_shape_slat`) runs this
//! op on an existing mesh before the shape SLAT encoder network; this module
//! is the inverse of [`crate::runtime::flexible_dual_grid_to_mesh`].
//!
//! # Coordinate frame (upstream-exact)
//!
//! The Python wrapper shifts vertices by `-aabb[0]`, voxelizes, and returns
//! the kernel output unchanged (it does NOT add `aabb[0]` back). The
//! returned `dual_vertices` are therefore absolute positions in the
//! *shifted* grid frame: voxel `k` spans `[k * voxel_size, (k + 1) *
//! voxel_size]` per axis, and each dual vertex is clamped into its voxel
//! cell. With the encoder's `aabb = [[-0.5; 3], [0.5; 3]]` and
//! `grid_size = R` (`voxel_size = 1 / R`), the per-cell SLAT features are
//! `dual_vertices * R - voxel_indices` (per-cell fractions in `[0, 1]^3`) —
//! exactly what `encode_shape_slat` and `o-voxel/examples/mesh2ovox.py`
//! compute. To feed [`crate::runtime::flexible_dual_grid_to_mesh`], convert
//! to per-cell fractions with `dual / voxel_size - voxel_index`; that
//! function re-adds the `aabb[0]` offset itself.
//!
//! # Weights
//!
//! [`MeshToDualGridOptions::new`] uses the upstream wrapper defaults
//! (`face_weight = 1.0`, `boundary_weight = 1.0`,
//! `regularization_weight = 0.1`). The shape-SLAT encoder call uses
//! `face_weight = 1.0`, `boundary_weight = 0.2`,
//! `regularization_weight = 1e-2` — see
//! [`MeshToDualGridOptions::for_shape_slat_encoder`]. Note an upstream
//! quirk: `face_weight` and `regularization_weight` only act as `> 0` gates
//! in the C++ kernel (the face term is added unscaled); only
//! `boundary_weight` actually scales its term. The port replicates that
//! behavior exactly.
//!
//! # Algorithm
//!
//! 1. `intersect_qef`: per-triangle scan-line voxelization along all three
//!    axes with 4-connected 2x2 voxel stamps; accumulates the mean
//!    intersection point, the hit count, the `intersected[axis]` flag (set
//!    for the `(dx, dy) = (0, 0)` stamp) and the plane QEF
//!    `(n, d) (n, d)^T` per voxel.
//! 2. `face_qef`: exact triangle/voxel-overlap tests (Akenine-Möller style
//!    plane + three projected-edge tests) add the unscaled plane QEF to
//!    every overlapped voxel that already exists.
//! 3. `boundry_qef`: open mesh edges (edges counted exactly once) are
//!    DDA-traversed; every visited in-grid voxel that already exists gets
//!    `boundary_weight * [[I - dd^T, b], [b^T, c]]` added.
//! 4. Per-voxel solve: regularization toward the mean intersection
//!    (`regularization_weight * cnt * [[I, -p], [-p^T, p^T p]]`), then the
//!    unconstrained minimizer of the homogeneous quadratic `v^T Q v`,
//!    clamped into the voxel cell by exhaustive 1/2/3-axis-constrained
//!    enumeration in exactly the upstream candidate order (strict `<`
//!    comparison keeps the first best candidate, so tie behavior matches).
//!
//! Output rows are in hash-table *insertion order* — the same deterministic
//! order as the upstream C++ `std::vector` push sequence (no sorting), so a
//! golden reference comparison would be row-order exact.
//!
//! # Deviations from upstream
//!
//! * The final QEF solve runs in `f64` (upstream: `f32`
//!   `colPivHouseholderQr`). Positive-definite systems (always the case when
//!   `regularization_weight > 0`, i.e. every production call) take a
//!   closed-form Cramer fast path; singular / rank-deficient systems (only
//!   possible with `regularization_weight = 0`) fall back to a symmetric
//!   eigen pseudo-inverse with a `1e-12` relative eigenvalue cutoff. All
//!   routes agree to floating-point noise for the regularized systems; with
//!   `regularization_weight = 0` and a rank-deficient voxel, upstream's
//!   answer is solver-defined while this port returns the minimum-norm
//!   least-squares solution.
//! * A degenerate (zero-area) triangle is rejected as invalid input where
//!   upstream propagates `NaN` normals into the QEF. The check uses the
//!   exact f32 edge differences the kernel crosses (`v1 - v0`, `v2 - v1`),
//!   so triangles whose cross product rounds to zero in f32 (micro-slivers
//!   whose tiny edge vanishes when added to a long edge) are caught before
//!   they can poison a voxel.
//! * An auto-computed aabb that is degenerate along an axis (flat mesh +
//!   `grid_size` mode leaves `voxel_size = 0` upstream) is rejected;
//!   upstream produces `inf`/`NaN` voxel indices there. Pass an explicit
//!   `aabb` for flat meshes (the encoder always does).
//! * Weights must be finite and non-negative.
//! * The per-axis grid size is capped at [`MAX_GRID_SIZE_PER_AXIS`] to keep
//!   the encoded voxel keys and `i32` coordinate math safe.
//!
//! # Reference parity status
//!
//! The `o_voxel` Python extension is not built in this environment (verified
//! via `python -c "import o_voxel"`; only `torch` is available), so the
//! tests here are structural: invariants, coordinate-frame checks against
//! the already-ported `flexible_dual_grid_to_mesh`, and determinism. No
//! golden voxel-index comparison is possible without building the extension.

use std::collections::{BTreeMap, HashMap};
use std::hash::BuildHasherDefault;

use nalgebra::linalg::SymmetricEigen;
use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};

use crate::error::{Lux3dError, Result};

/// Maximum accepted per-axis grid size. `grid_size >= 2` per axis is
/// required by the auto-padding formula `(max - min) / (grid_size - 1)`; the
/// cap keeps the encoded voxel keys (21 bits per axis) and `i32` coordinate
/// arithmetic safe. The production pipeline uses 512 / 1024.
pub const MAX_GRID_SIZE_PER_AXIS: u32 = 4096;

/// Options for [`mesh_to_flexible_dual_grid`].
#[derive(Debug, Clone, PartialEq)]
pub struct MeshToDualGridOptions {
    /// Voxel grid resolution per axis (must be `>= 2` per axis; capped at
    /// [`MAX_GRID_SIZE_PER_AXIS`]).
    pub grid_size: [u32; 3],
    /// Explicit axis-aligned bounding box `[[min_x, min_y, min_z],
    /// [max_x, max_y, max_z]]`. When `None`, the box is computed from the
    /// vertex extremes and expanded by half a cell per axis (upstream
    /// `grid_size`-mode padding: `padding = (max - min) / (grid_size - 1)`,
    /// `min -= padding / 2`, `max += padding / 2`).
    pub aabb: Option<[[f32; 3]; 2]>,
    /// Face QEF term gate (`> 0` enables the term; the C++ kernel adds it
    /// unscaled). Upstream wrapper default `1.0`; encoder uses `1.0`.
    pub face_weight: f32,
    /// Boundary QEF term weight (open edges; actually scales the term).
    /// Upstream wrapper default `1.0`; encoder uses `0.2`.
    pub boundary_weight: f32,
    /// Regularization toward the mean intersection point. Upstream wrapper
    /// default `0.1`; encoder uses `1e-2`.
    pub regularization_weight: f32,
}

impl MeshToDualGridOptions {
    /// Upstream wrapper defaults: `face_weight = 1.0`, `boundary_weight =
    /// 1.0`, `regularization_weight = 0.1`, auto aabb.
    pub fn new(grid_size: [u32; 3]) -> Self {
        Self {
            grid_size,
            aabb: None,
            face_weight: 1.0,
            boundary_weight: 1.0,
            regularization_weight: 0.1,
        }
    }

    /// Uniform resolution shorthand for [`MeshToDualGridOptions::new`].
    pub fn uniform(resolution: u32) -> Self {
        Self::new([resolution, resolution, resolution])
    }

    /// The TRELLIS.2 shape-SLAT encoder call
    /// (`encode_shape_slat`): explicit `[-0.5, 0.5]^3` aabb, `face_weight =
    /// 1.0`, `boundary_weight = 0.2`, `regularization_weight = 1e-2`.
    pub fn for_shape_slat_encoder(resolution: u32) -> Self {
        Self {
            grid_size: [resolution, resolution, resolution],
            aabb: Some([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]),
            face_weight: 1.0,
            boundary_weight: 0.2,
            regularization_weight: 1e-2,
        }
    }
}

/// Sparse flexible dual-grid voxelization of a triangle mesh.
///
/// Row order across the three buffers is the hash-table insertion order of
/// the upstream kernel (deterministic, unsorted).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DualGridVoxelization {
    /// `[N, 3]` active voxel coordinates, all inside `[0, grid_size)^3`.
    pub voxel_indices: Vec<[i32; 3]>,
    /// `[N, 3]` dual vertex positions in the shifted grid frame
    /// (`vertices - aabb[0]`); each vertex lies inside its voxel cell.
    pub dual_vertices: Vec<[f32; 3]>,
    /// `[N, 3]` per-axis flags marking whether the voxel's low-corner
    /// 2x2 stamp sample hit the surface for that scan axis.
    pub intersected: Vec<[bool; 3]>,
}

/// Voxelize a triangle mesh into the O-Voxel flexible dual grid.
///
/// * `vertices`: flat `[V, 3]` f32 positions (all finite).
/// * `faces`: flat `[F, 3]` zero-based triangle indices (all in range,
///   no zero-area triangles).
/// * `options`: grid size, optional aabb and QEF weights.
///
/// See the module docs for the output coordinate frame and the exact
/// upstream-deviation list.
pub fn mesh_to_flexible_dual_grid(
    vertices: &[f32],
    faces: &[u32],
    options: &MeshToDualGridOptions,
) -> Result<DualGridVoxelization> {
    // ---- input validation --------------------------------------------------
    if vertices.len() % 3 != 0 {
        return Err(Lux3dError::InvalidInput(
            "dual grid voxelization vertex buffer length must be a multiple of 3",
        ));
    }
    if faces.len() % 3 != 0 {
        return Err(Lux3dError::InvalidInput(
            "dual grid voxelization face buffer length must be a multiple of 3",
        ));
    }
    if vertices.is_empty() || faces.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "dual grid voxelization requires a non-empty mesh",
        ));
    }
    let vertex_count = vertices.len() / 3;
    for &index in faces {
        if index as usize >= vertex_count {
            return Err(Lux3dError::InvalidInput(
                "dual grid voxelization face index outside the vertex buffer",
            ));
        }
    }
    if vertices.iter().any(|v| !v.is_finite()) {
        return Err(Lux3dError::InvalidInput(
            "dual grid voxelization requires finite vertex positions",
        ));
    }
    for tri in faces.chunks_exact(3) {
        let v0 = vertex(vertices, tri[0]);
        let v1 = vertex(vertices, tri[1]);
        let v2 = vertex(vertices, tri[2]);
        // Validate with the same f32 edge differences the kernel uses
        // (`e0 = v1 - v0`, `e1 = v2 - v1`): a triangle whose kernel cross
        // rounds to zero would otherwise poison the QEF with NaN normals.
        let normal = cross3(sub3(v1, v0), sub3(v2, v1));
        let norm2 = dot3(normal, normal);
        if !norm2.is_finite() || norm2 <= 0.0 {
            return Err(Lux3dError::InvalidInput(
                "dual grid voxelization rejected a degenerate (zero-area) triangle",
            ));
        }
    }
    for &axis_size in &options.grid_size {
        if axis_size < 2 {
            return Err(Lux3dError::InvalidInput(
                "dual grid voxelization requires grid_size >= 2 per axis",
            ));
        }
        if axis_size > MAX_GRID_SIZE_PER_AXIS {
            return Err(Lux3dError::InvalidInput(
                "dual grid voxelization grid_size exceeds the per-axis cap",
            ));
        }
    }
    for weight in [
        options.face_weight,
        options.boundary_weight,
        options.regularization_weight,
    ] {
        if !weight.is_finite() || weight < 0.0 {
            return Err(Lux3dError::InvalidInput(
                "dual grid voxelization weights must be finite and non-negative",
            ));
        }
    }

    // ---- aabb and voxel size (wrapper semantics) ---------------------------
    let aabb = resolve_aabb(vertices, options.grid_size, options.aabb)?;
    let grid_size = options.grid_size;
    let voxel_size = [
        (aabb[1][0] - aabb[0][0]) / grid_size[0] as f32,
        (aabb[1][1] - aabb[0][1]) / grid_size[1] as f32,
        (aabb[1][2] - aabb[0][2]) / grid_size[2] as f32,
    ];
    if voxel_size.iter().any(|s| !s.is_finite() || *s <= 0.0) {
        return Err(Lux3dError::InvalidInput(
            "dual grid voxelization produced a degenerate voxel size",
        ));
    }

    // Shift the mesh into grid space: `vertices - aabb[0]` (wrapper step).
    let mut shifted = Vec::with_capacity(vertices.len());
    for (index, &value) in vertices.iter().enumerate() {
        shifted.push(value - aabb[0][index % 3]);
    }

    let grid_min = [0i32; 3];
    let grid_max = [
        grid_size[0] as i32,
        grid_size[1] as i32,
        grid_size[2] as i32,
    ];

    // ---- voxelization kernel ------------------------------------------------
    let mut state = VoxelState::default();
    intersect_qef(voxel_size, grid_min, grid_max, &shifted, faces, &mut state);
    if state.voxels.len() > u32::MAX as usize {
        return Err(Lux3dError::InvalidInput(
            "dual grid voxelization produced more voxels than addressable",
        ));
    }
    if options.face_weight > 0.0 {
        face_qef(voxel_size, grid_min, grid_max, &shifted, faces, &mut state);
    }
    if options.boundary_weight > 0.0 {
        let segments = collect_boundary_segments(&shifted, faces);
        boundry_qef(
            voxel_size,
            grid_min,
            grid_max,
            &segments,
            options.boundary_weight,
            &mut state,
        );
    }
    let dual_vertices = solve_dual_vertices(&state, voxel_size, options.regularization_weight);

    Ok(DualGridVoxelization {
        voxel_indices: state.voxels,
        dual_vertices,
        intersected: state.intersected,
    })
}

/// Upstream auto-aabb: expand the vertex bounding box by half a cell per
/// axis (`padding = (max - min) / (grid_size - 1)`), or validate the
/// user-provided one.
fn resolve_aabb(
    vertices: &[f32],
    grid_size: [u32; 3],
    aabb: Option<[[f32; 3]; 2]>,
) -> Result<[[f32; 3]; 2]> {
    match aabb {
        Some(aabb) => {
            for corner in &aabb {
                if corner.iter().any(|v| !v.is_finite()) {
                    return Err(Lux3dError::InvalidInput(
                        "dual grid voxelization aabb must be finite",
                    ));
                }
            }
            for axis in 0..3 {
                if !(aabb[0][axis] < aabb[1][axis]) {
                    return Err(Lux3dError::InvalidInput(
                        "dual grid voxelization aabb min must be below max on every axis",
                    ));
                }
            }
            Ok(aabb)
        }
        None => {
            let mut min = [f32::INFINITY; 3];
            let mut max = [f32::NEG_INFINITY; 3];
            for chunk in vertices.chunks_exact(3) {
                for axis in 0..3 {
                    min[axis] = min[axis].min(chunk[axis]);
                    max[axis] = max[axis].max(chunk[axis]);
                }
            }
            for axis in 0..3 {
                let padding = (max[axis] - min[axis]) / (grid_size[axis] - 1) as f32;
                min[axis] -= padding * 0.5;
                max[axis] += padding * 0.5;
                if !min[axis].is_finite() || !max[axis].is_finite() || !(max[axis] > min[axis]) {
                    return Err(Lux3dError::InvalidInput(
                        "dual grid voxelization auto aabb is degenerate or too large along an \
                         axis; provide an explicit aabb for flat meshes",
                    ));
                }
            }
            Ok([min, max])
        }
    }
}

// ---------------------------------------------------------------------------
// Voxel state
// ---------------------------------------------------------------------------

/// Sparse per-voxel accumulators (one entry per active voxel, insertion
/// ordered).
#[derive(Default)]
struct VoxelState {
    /// Encoded voxel coordinate -> row index (`u32` addressing is checked
    /// after the scan; only `intersect_qef` inserts).
    map: HashMap<u64, u32, BuildHasherDefault<EncodedKeyHasher>>,
    voxels: Vec<[i32; 3]>,
    /// Running mean intersection point (sum + count like upstream).
    means: Vec<[f32; 3]>,
    cnt: Vec<f32>,
    intersected: Vec<[bool; 3]>,
    /// Upper triangle of the symmetric 4x4 QEF, row-major:
    /// `[q00, q01, q02, q03, q11, q12, q13, q22, q23, q33]`.
    qefs: Vec<[f32; 10]>,
}

#[inline]
fn sym_index(i: usize, j: usize) -> usize {
    const TABLE: [[usize; 4]; 4] = [[0, 1, 2, 3], [1, 4, 5, 6], [2, 5, 7, 8], [3, 6, 8, 9]];
    TABLE[i][j]
}

/// Packs non-negative voxel coordinates into a collision-free `u64` key
/// (21 bits per axis; the grid cap keeps coordinates far below `2^21`).
/// Negative coordinates (possible only on lookups from the boundary DDA)
/// set high bits and can therefore never collide with stored keys.
#[inline]
fn encode_key(coord: [i32; 3]) -> u64 {
    debug_assert!(coord[0] >= 0 && coord[1] >= 0 && coord[2] >= 0);
    coord[0] as u64 | (coord[1] as u64) << 21 | (coord[2] as u64) << 42
}

/// Trivial deterministic hasher for the pre-encoded keys.
#[derive(Default)]
struct EncodedKeyHasher(u64);

impl std::hash::Hasher for EncodedKeyHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 ^ byte as u64).wrapping_mul(0x100000001b3);
        }
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.0 = value.wrapping_mul(0x9E3779B97F4A7C15);
    }
}

/// Insert-or-accumulate one 2x2 stamp hit (upstream `intersect_qef` inner
/// block).
#[inline]
fn stamp(
    state: &mut VoxelState,
    coord: [i32; 3],
    intersect: [f32; 3],
    axis: usize,
    corner: bool,
    q: &[f32; 10],
) {
    let key = encode_key(coord);
    if let Some(&index) = state.map.get(&key) {
        let index = index as usize;
        for k in 0..3 {
            state.means[index][k] += intersect[k];
        }
        state.cnt[index] += 1.0;
        if corner {
            state.intersected[index][axis] = true;
        }
        for k in 0..10 {
            state.qefs[index][k] += q[k];
        }
    } else {
        let index = state.voxels.len() as u32;
        state.map.insert(key, index);
        state.voxels.push(coord);
        state.means.push(intersect);
        state.cnt.push(1.0);
        let mut flags = [false; 3];
        if corner {
            flags[axis] = true;
        }
        state.intersected.push(flags);
        state.qefs.push(*q);
    }
}

#[inline]
fn add_q(state: &mut VoxelState, index: usize, q: &[f32; 10]) {
    for k in 0..10 {
        state.qefs[index][k] += q[k];
    }
}

// ---------------------------------------------------------------------------
// Stage 1: intersect_qef (scan-line voxelization)
// ---------------------------------------------------------------------------

fn intersect_qef(
    voxel_size: [f32; 3],
    grid_min: [i32; 3],
    grid_max: [i32; 3],
    vertices: &[f32],
    faces: &[u32],
    state: &mut VoxelState,
) {
    for tri in faces.chunks_exact(3) {
        let v0 = vertex(vertices, tri[0]);
        let v1 = vertex(vertices, tri[1]);
        let v2 = vertex(vertices, tri[2]);
        let e0 = sub3(v1, v0);
        let e1 = sub3(v2, v1);
        let n = normalize3(cross3(e0, e1));
        let q = plane_q(n, -dot3(n, v0));
        for ax2 in 0..3 {
            scan_line_fill(ax2, v0, v1, v2, &q, voxel_size, grid_min, grid_max, state);
        }
    }
}

/// Per-axis scan-line fill (upstream `scan_line_fill` closure). Projects the
/// triangle onto the `(ax0, ax1)` plane, sorts by `ax1`, rasterizes the two
/// halves and stamps 4-connected 2x2 voxel neighborhoods per sample.
fn scan_line_fill(
    ax2: usize,
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    q: &[f32; 10],
    voxel_size: [f32; 3],
    grid_min: [i32; 3],
    grid_max: [i32; 3],
    state: &mut VoxelState,
) {
    let ax0 = (ax2 + 1) % 3;
    let ax1 = (ax2 + 2) % 3;
    // Canonical triangle in (ax0, ax1, ax2) coordinates, sorted by ax1.
    let mut t: [[f64; 3]; 3] = [
        [v0[ax0] as f64, v0[ax1] as f64, v0[ax2] as f64],
        [v1[ax0] as f64, v1[ax1] as f64, v1[ax2] as f64],
        [v2[ax0] as f64, v2[ax1] as f64, v2[ax2] as f64],
    ];
    t.sort_by(|a, b| a[1].total_cmp(&b[1]));

    let vs1 = voxel_size[ax1] as f64;
    let row_lo = grid_min[ax1];
    let row_hi = grid_max[ax1] - 1;
    let start = clamp_i(t[0][1] / vs1, row_lo, row_hi);
    let mid = clamp_i(t[1][1] / vs1, row_lo, row_hi);
    let end = clamp_i(t[2][1] / vs1, row_lo, row_hi);

    scan_line_half(
        start, mid, t[0], t[1], t[2], ax0, ax1, ax2, q, voxel_size, grid_min, grid_max, state,
    );
    scan_line_half(
        mid, end, t[2], t[1], t[0], ax0, ax1, ax2, q, voxel_size, grid_min, grid_max, state,
    );
}

/// One scan-line triangle half (upstream `scan_line_half`). The sample point
/// sits on the `(x_idx + 1, y_idx + 1)` grid corner; the 2x2 stamp covers
/// the 4-connected neighborhood and the `(0, 0)` stamp carries the
/// `intersected[axis2]` flag.
#[allow(clippy::too_many_arguments)]
fn scan_line_half(
    row_start: i32,
    row_end: i32,
    t0: [f64; 3],
    t1: [f64; 3],
    t2: [f64; 3],
    ax0: usize,
    ax1: usize,
    ax2: usize,
    q: &[f32; 10],
    voxel_size: [f32; 3],
    grid_min: [i32; 3],
    grid_max: [i32; 3],
    state: &mut VoxelState,
) {
    let vs0 = voxel_size[ax0] as f64;
    let vs2 = voxel_size[ax2] as f64;
    let col_lo = grid_min[ax0];
    let col_hi = grid_max[ax0] - 1;
    for y_idx in row_start..row_end {
        let y = ((y_idx + 1) as f32 * voxel_size[ax1]) as f64;
        let mut t3 = lerp2(t0[1], t1[1], y, [t0[0], t0[2]], [t1[0], t1[2]]);
        let mut t4 = lerp2(t0[1], t2[1], y, [t0[0], t0[2]], [t2[0], t2[2]]);
        if t3[0] > t4[0] {
            std::mem::swap(&mut t3, &mut t4);
        }
        let line_start = clamp_i(t3[0] / vs0, col_lo, col_hi);
        let line_end = clamp_i(t4[0] / vs0, col_lo, col_hi);
        for x_idx in line_start..line_end {
            let x = ((x_idx + 1) as f32 * voxel_size[ax0]) as f64;
            let z = lerp1(t3[0], t4[0], x, t3[1], t4[1]);
            let z_idx = (z / vs2) as i32;
            if z_idx < grid_min[ax2] || z_idx >= grid_max[ax2] {
                continue;
            }
            for dx in 0..2i32 {
                for dy in 0..2i32 {
                    let mut coord = [0i32; 3];
                    coord[ax0] = x_idx + dx;
                    coord[ax1] = y_idx + dy;
                    coord[ax2] = z_idx;
                    let mut intersect = [0.0f32; 3];
                    intersect[ax0] = x as f32;
                    intersect[ax1] = y as f32;
                    intersect[ax2] = z as f32;
                    stamp(state, coord, intersect, ax2, dx == 0 && dy == 0, q);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stage 2: face_qef (triangle/voxel overlap plane tests)
// ---------------------------------------------------------------------------

fn face_qef(
    voxel_size: [f32; 3],
    grid_min: [i32; 3],
    grid_max: [i32; 3],
    vertices: &[f32],
    faces: &[u32],
    state: &mut VoxelState,
) {
    for tri in faces.chunks_exact(3) {
        let v0 = vertex(vertices, tri[0]);
        let v1 = vertex(vertices, tri[1]);
        let v2 = vertex(vertices, tri[2]);

        let e0 = sub3(v1, v0);
        let e1 = sub3(v2, v1);
        let e2 = sub3(v0, v2);
        let n = normalize3(cross3(e0, e1));
        let q = plane_q(n, -dot3(n, v0));

        // Triangle bounding box in voxel coordinates.
        let mut bb_min = [0i32; 3];
        let mut bb_max = [0i32; 3];
        for axis in 0..3 {
            let lo = v0[axis].min(v1[axis]).min(v2[axis]) / voxel_size[axis];
            let hi = v0[axis].max(v1[axis]).max(v2[axis]) / voxel_size[axis];
            bb_min[axis] = (lo as i32).max(grid_min[axis]);
            bb_max[axis] = ((hi + 1.0) as i32).min(grid_max[axis]);
        }

        // Plane through box test setup.
        let c = [
            if n[0] > 0.0 { voxel_size[0] } else { 0.0 },
            if n[1] > 0.0 { voxel_size[1] } else { 0.0 },
            if n[2] > 0.0 { voxel_size[2] } else { 0.0 },
        ];
        let d1 = dot3(n, sub3(c, v0));
        let d2 = dot3(n, sub3(sub3(voxel_size, c), v0));

        // XY projection edge tests.
        let mul_xy: f32 = if n[2] < 0.0 { -1.0 } else { 1.0 };
        let n_xy_e0 = [-mul_xy * e0[1], mul_xy * e0[0]];
        let n_xy_e1 = [-mul_xy * e1[1], mul_xy * e1[0]];
        let n_xy_e2 = [-mul_xy * e2[1], mul_xy * e2[0]];
        let d_xy_e0 =
            -dot2(n_xy_e0, [v0[0], v0[1]]) + dot2(max0_2(n_xy_e0), [voxel_size[0], voxel_size[1]]);
        let d_xy_e1 =
            -dot2(n_xy_e1, [v1[0], v1[1]]) + dot2(max0_2(n_xy_e1), [voxel_size[0], voxel_size[1]]);
        let d_xy_e2 =
            -dot2(n_xy_e2, [v2[0], v2[1]]) + dot2(max0_2(n_xy_e2), [voxel_size[0], voxel_size[1]]);

        // YZ projection edge tests.
        let mul_yz: f32 = if n[0] < 0.0 { -1.0 } else { 1.0 };
        let n_yz_e0 = [-mul_yz * e0[2], mul_yz * e0[1]];
        let n_yz_e1 = [-mul_yz * e1[2], mul_yz * e1[1]];
        let n_yz_e2 = [-mul_yz * e2[2], mul_yz * e2[1]];
        let d_yz_e0 =
            -dot2(n_yz_e0, [v0[1], v0[2]]) + dot2(max0_2(n_yz_e0), [voxel_size[1], voxel_size[2]]);
        let d_yz_e1 =
            -dot2(n_yz_e1, [v1[1], v1[2]]) + dot2(max0_2(n_yz_e1), [voxel_size[1], voxel_size[2]]);
        let d_yz_e2 =
            -dot2(n_yz_e2, [v2[1], v2[2]]) + dot2(max0_2(n_yz_e2), [voxel_size[1], voxel_size[2]]);

        // ZX projection edge tests.
        let mul_zx: f32 = if n[1] < 0.0 { -1.0 } else { 1.0 };
        let n_zx_e0 = [-mul_zx * e0[0], mul_zx * e0[2]];
        let n_zx_e1 = [-mul_zx * e1[0], mul_zx * e1[2]];
        let n_zx_e2 = [-mul_zx * e2[0], mul_zx * e2[2]];
        let d_zx_e0 =
            -dot2(n_zx_e0, [v0[2], v0[0]]) + dot2(max0_2(n_zx_e0), [voxel_size[2], voxel_size[0]]);
        let d_zx_e1 =
            -dot2(n_zx_e1, [v1[2], v1[0]]) + dot2(max0_2(n_zx_e1), [voxel_size[2], voxel_size[0]]);
        let d_zx_e2 =
            -dot2(n_zx_e2, [v2[2], v2[0]]) + dot2(max0_2(n_zx_e2), [voxel_size[2], voxel_size[0]]);

        // Loop over candidate voxels inside the bounding box.
        for z in bb_min[2]..bb_max[2] {
            for y in bb_min[1]..bb_max[1] {
                for x in bb_min[0]..bb_max[0] {
                    let p = [
                        x as f32 * voxel_size[0],
                        y as f32 * voxel_size[1],
                        z as f32 * voxel_size[2],
                    ];

                    // Plane through box test.
                    let n_dot_p = dot3(n, p);
                    if (n_dot_p + d1) * (n_dot_p + d2) > 0.0 {
                        continue;
                    }

                    // XY projection test.
                    let p_xy = [p[0], p[1]];
                    if dot2(n_xy_e0, p_xy) + d_xy_e0 < 0.0 {
                        continue;
                    }
                    if dot2(n_xy_e1, p_xy) + d_xy_e1 < 0.0 {
                        continue;
                    }
                    if dot2(n_xy_e2, p_xy) + d_xy_e2 < 0.0 {
                        continue;
                    }

                    // YZ projection test.
                    let p_yz = [p[1], p[2]];
                    if dot2(n_yz_e0, p_yz) + d_yz_e0 < 0.0 {
                        continue;
                    }
                    if dot2(n_yz_e1, p_yz) + d_yz_e1 < 0.0 {
                        continue;
                    }
                    if dot2(n_yz_e2, p_yz) + d_yz_e2 < 0.0 {
                        continue;
                    }

                    // ZX projection test.
                    let p_zx = [p[2], p[0]];
                    if dot2(n_zx_e0, p_zx) + d_zx_e0 < 0.0 {
                        continue;
                    }
                    if dot2(n_zx_e1, p_zx) + d_zx_e1 < 0.0 {
                        continue;
                    }
                    if dot2(n_zx_e2, p_zx) + d_zx_e2 < 0.0 {
                        continue;
                    }

                    // Passed all tests — accumulate the unscaled plane QEF.
                    if let Some(&index) = state.map.get(&encode_key([x, y, z])) {
                        add_q(state, index as usize, &q);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stage 3: boundry_qef (DDA over open edges)
// ---------------------------------------------------------------------------

/// Open edges (undirected edges referenced by exactly one triangle) as
/// vertex-position segment pairs, in sorted edge order (upstream
/// `std::map<std::pair<int,int>,int>` iteration order).
fn collect_boundary_segments(vertices: &[f32], faces: &[u32]) -> Vec<[f32; 3]> {
    let mut edge_count: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    for tri in faces.chunks_exact(3) {
        for k in 0..3 {
            let a = tri[k];
            let b = tri[(k + 1) % 3];
            let key = if a <= b { (a, b) } else { (b, a) };
            *edge_count.entry(key).or_insert(0) += 1;
        }
    }
    let mut segments = Vec::new();
    for (&(a, b), &count) in edge_count.iter() {
        if count == 1 {
            segments.push(vertex(vertices, a));
            segments.push(vertex(vertices, b));
        }
    }
    segments
}

fn boundry_qef(
    voxel_size: [f32; 3],
    grid_min: [i32; 3],
    grid_max: [i32; 3],
    segments: &[[f32; 3]],
    boundary_weight: f32,
    state: &mut VoxelState,
) {
    for seg in segments.chunks_exact(2) {
        let v0 = seg[0];
        let v1 = seg[1];

        // Segment direction (f32 difference widened to f64, like upstream).
        let mut dir = [
            (v1[0] - v0[0]) as f64,
            (v1[1] - v0[1]) as f64,
            (v1[2] - v0[2]) as f64,
        ];
        let segment_length = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if segment_length < 1e-6 {
            continue;
        }
        for d in &mut dir {
            *d /= segment_length;
        }

        // A = I - d d^T (computed in f64, cast to f32 like upstream).
        let mut a = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] = (if i == j { 1.0f32 } else { 0.0f32 }) - (dir[i] * dir[j]) as f32;
            }
        }
        let a_v0 = mat3_vec3(a, v0);
        let b = [-a_v0[0], -a_v0[1], -a_v0[2]];
        let c = dot3(v0, a_v0);
        let q = boundary_q(a, b, c);

        // DDA traversal.
        let start_voxel = [
            (v0[0] / voxel_size[0]).floor() as i32,
            (v0[1] / voxel_size[1]).floor() as i32,
            (v0[2] / voxel_size[2]).floor() as i32,
        ];
        let step = [
            if dir[0] > 0.0 { 1i32 } else { -1 },
            if dir[1] > 0.0 { 1 } else { -1 },
            if dir[2] > 0.0 { 1 } else { -1 },
        ];
        let mut t_max = [0.0f64; 3];
        let mut t_delta = [0.0f64; 3];
        for axis in 0..3 {
            if dir[axis] == 0.0 {
                t_max[axis] = f64::INFINITY;
                t_delta[axis] = f64::INFINITY;
            } else {
                let border: f32 = voxel_size[axis]
                    * (start_voxel[axis] + if step[axis] > 0 { 1 } else { 0 }) as f32;
                t_max[axis] = ((border - v0[axis]) as f64) / dir[axis];
                t_delta[axis] = voxel_size[axis] as f64 / dir[axis].abs();
            }
        }

        let mut current = start_voxel;
        let mut visited = vec![current];
        loop {
            let axis = if t_max[0] < t_max[1] {
                if t_max[0] < t_max[2] { 0 } else { 2 }
            } else if t_max[1] < t_max[2] {
                1
            } else {
                2
            };
            if t_max[axis] > segment_length {
                break;
            }
            current[axis] += step[axis];
            t_max[axis] += t_delta[axis];
            visited.push(current);
        }

        // Accumulate the weighted boundary QEF into existing in-grid voxels.
        for coord in visited {
            if (coord[0] < grid_min[0] || coord[0] >= grid_max[0])
                || (coord[1] < grid_min[1] || coord[1] >= grid_max[1])
                || (coord[2] < grid_min[2] || coord[2] >= grid_max[2])
            {
                continue;
            }
            if let Some(&index) = state.map.get(&encode_key(coord)) {
                let index = index as usize;
                for k in 0..10 {
                    state.qefs[index][k] += boundary_weight * q[k];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stage 4: per-voxel QEF solve
// ---------------------------------------------------------------------------

fn solve_dual_vertices(
    state: &VoxelState,
    voxel_size: [f32; 3],
    regularization_weight: f32,
) -> Vec<[f32; 3]> {
    let mut dual_vertices = Vec::with_capacity(state.voxels.len());
    for index in 0..state.voxels.len() {
        let coord = state.voxels[index];
        let min_corner = [
            coord[0] as f32 * voxel_size[0],
            coord[1] as f32 * voxel_size[1],
            coord[2] as f32 * voxel_size[2],
        ];
        let max_corner = [
            (coord[0] + 1) as f32 * voxel_size[0],
            (coord[1] + 1) as f32 * voxel_size[1],
            (coord[2] + 1) as f32 * voxel_size[2],
        ];
        let min_c = [
            min_corner[0] as f64,
            min_corner[1] as f64,
            min_corner[2] as f64,
        ];
        let max_c = [
            max_corner[0] as f64,
            max_corner[1] as f64,
            max_corner[2] as f64,
        ];

        // Copy the symmetric QEF and add the regularization term (f32
        // accumulation like upstream, then widened for the solve).
        let mut q_sym = state.qefs[index];
        if regularization_weight > 0.0 {
            let cnt = state.cnt[index];
            let p = [
                state.means[index][0] / cnt,
                state.means[index][1] / cnt,
                state.means[index][2] / cnt,
            ];
            let p_dot_p = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
            let scale = regularization_weight * cnt;
            let qreg = [
                scale,           // (0,0)
                0.0,             // (0,1)
                0.0,             // (0,2)
                scale * -p[0],   // (0,3)
                scale,           // (1,1)
                0.0,             // (1,2)
                scale * -p[1],   // (1,3)
                scale,           // (2,2)
                scale * -p[2],   // (2,3)
                scale * p_dot_p, // (3,3)
            ];
            for k in 0..10 {
                q_sym[k] += qreg[k];
            }
        }
        let q = q4_from_sym(&q_sym);

        // Unconstrained minimizer of v^T Q v (upstream: colPivHouseholderQr
        // on the top-left 3x3 with b = -Q(0:3, 3); here: f64 symmetric
        // eigen pseudo-inverse).
        let a3 = [
            [q[0][0], q[0][1], q[0][2]],
            [q[1][0], q[1][1], q[1][2]],
            [q[2][0], q[2][1], q[2][2]],
        ];
        let b3 = [-q[0][3], -q[1][3], -q[2][3]];
        let mut v_new = solve_sym3_pinv(&a3, &b3);

        let inside = (0..3).all(|axis| v_new[axis] >= min_c[axis] && v_new[axis] <= max_c[axis]);
        if !inside {
            let mut best = f64::INFINITY;

            // Single-constraint candidates (fix one axis at a cell bound).
            for fixed in 0..3 {
                let ax1 = (fixed + 1) % 3;
                let ax2 = (fixed + 2) % 3;
                let a2 = [[q[ax1][ax1], q[ax1][ax2]], [q[ax2][ax1], q[ax2][ax2]]];
                let fixed_col = [q[ax1][fixed], q[ax2][fixed]];
                let const_col = [q[ax1][3], q[ax2][3]];
                for &bound in &[min_c[fixed], max_c[fixed]] {
                    let b2 = [
                        -(fixed_col[0] * bound + const_col[0]),
                        -(fixed_col[1] * bound + const_col[1]),
                    ];
                    let x = solve_sym2_pinv(&a2, &b2);
                    if x[0] >= min_c[ax1]
                        && x[0] <= max_c[ax1]
                        && x[1] >= min_c[ax2]
                        && x[1] <= max_c[ax2]
                    {
                        let mut p = [0.0f64; 4];
                        p[fixed] = bound;
                        p[ax1] = x[0];
                        p[ax2] = x[1];
                        p[3] = 1.0;
                        let err = qef_err(&q, p);
                        if err < best {
                            best = err;
                            v_new = [p[0], p[1], p[2]];
                        }
                    }
                }
            }

            // Two-constraint candidates (free one axis, both others fixed).
            for free in 0..3 {
                let ax1 = (free + 1) % 3;
                let ax2 = (free + 2) % 3;
                let a = q[free][free];
                if a == 0.0 {
                    // Upstream divides and relies on the range check to drop
                    // the inf/NaN candidates; skipping is equivalent.
                    continue;
                }
                let b3 = [q[free][ax1], q[free][ax2], q[free][3]];
                let bounds1 = [min_c[ax1], min_c[ax1], max_c[ax1], max_c[ax1]];
                let bounds2 = [min_c[ax2], max_c[ax2], min_c[ax2], max_c[ax2]];
                for combo in 0..4 {
                    let qv = [bounds1[combo], bounds2[combo], 1.0];
                    let x = -(b3[0] * qv[0] + b3[1] * qv[1] + b3[2] * qv[2]) / a;
                    if x >= min_c[free] && x <= max_c[free] {
                        let mut p = [0.0f64; 4];
                        p[free] = x;
                        p[ax1] = bounds1[combo];
                        p[ax2] = bounds2[combo];
                        p[3] = 1.0;
                        let err = qef_err(&q, p);
                        if err < best {
                            best = err;
                            v_new = [p[0], p[1], p[2]];
                        }
                    }
                }
            }

            // Three-constraint candidates (the 8 cell corners).
            for x_constraint in 0..2 {
                for y_constraint in 0..2 {
                    for z_constraint in 0..2 {
                        let p = [
                            if x_constraint == 0 {
                                max_c[0]
                            } else {
                                min_c[0]
                            },
                            if y_constraint == 0 {
                                max_c[1]
                            } else {
                                min_c[1]
                            },
                            if z_constraint == 0 {
                                max_c[2]
                            } else {
                                min_c[2]
                            },
                            1.0,
                        ];
                        let err = qef_err(&q, p);
                        if err < best {
                            best = err;
                            v_new = [p[0], p[1], p[2]];
                        }
                    }
                }
            }
        }

        dual_vertices.push([v_new[0] as f32, v_new[1] as f32, v_new[2] as f32]);
    }
    dual_vertices
}

/// Symmetric 3x3 solve. Positive-definite systems (always the case when
/// `regularization_weight > 0`, since the QEF is a sum of PSD matrices plus
/// `regularization * cnt * I`) take a closed-form Cramer fast path;
/// singular/rank-deficient systems fall back to a symmetric eigen
/// pseudo-inverse (minimum-norm least squares).
fn solve_sym3_pinv(a: &[[f64; 3]; 3], b: &[f64; 3]) -> [f64; 3] {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.is_finite() && det > 0.0 {
        // PSD with positive determinant => positive definite.
        let x0 = (b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
            + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]))
            / det;
        let x1 = (a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
            - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]))
            / det;
        let x2 = (a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
            - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
            + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
            / det;
        if x0.is_finite() && x1.is_finite() && x2.is_finite() {
            return [x0, x1, x2];
        }
    }
    solve_sym3_eigen_pinv(a, b)
}

/// Symmetric eigen pseudo-inverse for a 3x3 system.
fn solve_sym3_eigen_pinv(a: &[[f64; 3]; 3], b: &[f64; 3]) -> [f64; 3] {
    let m = Matrix3::new(
        a[0][0], a[0][1], a[0][2], //
        a[1][0], a[1][1], a[1][2], //
        a[2][0], a[2][1], a[2][2],
    );
    let eigen = SymmetricEigen::new(m);
    let max_abs = eigen
        .eigenvalues
        .iter()
        .map(|l| l.abs())
        .fold(0.0f64, f64::max);
    let tol = max_abs * 1e-12;
    let b_vec = Vector3::new(b[0], b[1], b[2]);
    let v_t_b = eigen.eigenvectors.transpose() * b_vec;
    let mut x = [0.0f64; 3];
    for k in 0..3 {
        let lambda = eigen.eigenvalues[k];
        if lambda > tol {
            let w = v_t_b[k] / lambda;
            for r in 0..3 {
                x[r] += eigen.eigenvectors[(r, k)] * w;
            }
        }
    }
    x
}

/// Symmetric 2x2 solve (Cramer fast path for PD systems, eigen
/// pseudo-inverse fallback).
fn solve_sym2_pinv(a: &[[f64; 2]; 2], b: &[f64; 2]) -> [f64; 2] {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.is_finite() && det > 0.0 && a[0][0] > 0.0 {
        let x0 = (b[0] * a[1][1] - a[0][1] * b[1]) / det;
        let x1 = (a[0][0] * b[1] - a[1][0] * b[0]) / det;
        if x0.is_finite() && x1.is_finite() {
            return [x0, x1];
        }
    }
    let m = Matrix2::new(a[0][0], a[0][1], a[1][0], a[1][1]);
    let eigen = SymmetricEigen::new(m);
    let max_abs = eigen
        .eigenvalues
        .iter()
        .map(|l| l.abs())
        .fold(0.0f64, f64::max);
    let tol = max_abs * 1e-12;
    let b_vec = Vector2::new(b[0], b[1]);
    let v_t_b = eigen.eigenvectors.transpose() * b_vec;
    let mut x = [0.0f64; 2];
    for k in 0..2 {
        let lambda = eigen.eigenvalues[k];
        if lambda > tol {
            let w = v_t_b[k] / lambda;
            for r in 0..2 {
                x[r] += eigen.eigenvectors[(r, k)] * w;
            }
        }
    }
    x
}

/// `p^T Q p` for the homogeneous 4-vector `p`.
#[inline]
fn qef_err(q: &[[f64; 4]; 4], p: [f64; 4]) -> f64 {
    let mut total = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            total += p[i] * q[i][j] * p[j];
        }
    }
    total
}

/// Widens the stored symmetric upper triangle into a full 4x4 f64 matrix.
#[inline]
fn q4_from_sym(sym: &[f32; 10]) -> [[f64; 4]; 4] {
    let mut q = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            q[i][j] = sym[sym_index(i, j)] as f64;
        }
    }
    q
}

// ---------------------------------------------------------------------------
// Small math helpers (f32 grid math mirrors the C++ kernel)
// ---------------------------------------------------------------------------

#[inline]
fn vertex(vertices: &[f32], index: u32) -> [f32; 3] {
    let base = index as usize * 3;
    [vertices[base], vertices[base + 1], vertices[base + 2]]
}

#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn dot2(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

#[inline]
fn max0_2(a: [f32; 2]) -> [f32; 2] {
    [a[0].max(0.0), a[1].max(0.0)]
}

#[inline]
fn mat3_vec3(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// `n / |n|` with the upstream `norm() = sqrt(squaredNorm())` order.
#[inline]
fn normalize3(n: [f32; 3]) -> [f32; 3] {
    let norm = (dot3(n, n)).sqrt();
    [n[0] / norm, n[1] / norm, n[2] / norm]
}

/// Homogeneous plane QEF: `(n, d) (n, d)^T` upper triangle.
#[inline]
fn plane_q(n: [f32; 3], d: f32) -> [f32; 10] {
    [
        n[0] * n[0],
        n[0] * n[1],
        n[0] * n[2],
        n[0] * d,
        n[1] * n[1],
        n[1] * n[2],
        n[1] * d,
        n[2] * n[2],
        n[2] * d,
        d * d,
    ]
}

/// Boundary edge QEF: `[[A, b], [b^T, c]]` with `A = I - d d^T`.
#[inline]
fn boundary_q(a: [[f32; 3]; 3], b: [f32; 3], c: f32) -> [f32; 10] {
    [
        a[0][0], a[0][1], a[0][2], b[0], //
        a[1][1], a[1][2], b[1], //
        a[2][2], b[2], //
        c,
    ]
}

/// Upstream scalar `lerp`: exact-equality short circuit then linear blend.
#[inline]
fn lerp1(a: f64, b: f64, t: f64, val_a: f64, val_b: f64) -> f64 {
    if a == b {
        return val_a;
    }
    let alpha = (t - a) / (b - a);
    (1.0 - alpha) * val_a + alpha * val_b
}

/// Upstream 2-vector `lerp`.
#[inline]
fn lerp2(a: f64, b: f64, t: f64, val_a: [f64; 2], val_b: [f64; 2]) -> [f64; 2] {
    if a == b {
        return val_a;
    }
    let alpha = (t - a) / (b - a);
    [
        (1.0 - alpha) * val_a[0] + alpha * val_b[0],
        (1.0 - alpha) * val_a[1] + alpha * val_b[1],
    ]
}

/// C-style `int(double)` cast (truncation toward zero) followed by a clamp
/// into `[lo, hi]`; mirrors `std::clamp(int(x / voxel_size), lo, hi)`.
#[inline]
fn clamp_i(x: f64, lo: i32, hi: i32) -> i32 {
    (x as i32).clamp(lo, hi)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::flexible_dual_grid_to_mesh;

    /// Unit cube [0,1]^3, outward winding (same layout as the `trellis2`
    /// stage tests).
    fn cube_mesh() -> (Vec<f32>, Vec<u32>) {
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

    /// One flat triangle in the z = 0.5 plane (needs an explicit aabb: the
    /// auto-aabb is degenerate along z, matching upstream grid-size-mode
    /// behavior which this port rejects).
    fn flat_triangle() -> (Vec<f32>, Vec<u32>) {
        (
            vec![0.1, 0.1, 0.5, 0.9, 0.1, 0.5, 0.5, 0.9, 0.5],
            vec![0, 1, 2],
        )
    }

    /// UV sphere of radius 0.5 centered at the origin; collapsed pole quads
    /// are skipped so every triangle is non-degenerate. Roughly
    /// `2 * stacks * sectors` triangles.
    fn uv_sphere(stacks: usize, sectors: usize) -> (Vec<f32>, Vec<u32>) {
        let mut vertices = Vec::with_capacity((stacks + 1) * (sectors + 1) * 3);
        for i in 0..=stacks {
            let theta = std::f32::consts::PI * i as f32 / stacks as f32;
            for j in 0..=sectors {
                let phi = std::f32::consts::TAU * j as f32 / sectors as f32;
                if i == 0 || i == stacks {
                    // Exact single pole point (sin(pi) in f32 is ~-8.7e-8,
                    // which would leave a micro-ring of 1e-9-scale sliver
                    // triangles around the pole).
                    vertices.push(0.0);
                    vertices.push(0.0);
                    vertices.push(0.5 * theta.cos());
                } else {
                    vertices.push(0.5 * theta.sin() * phi.cos());
                    vertices.push(0.5 * theta.sin() * phi.sin());
                    vertices.push(0.5 * theta.cos());
                }
            }
        }
        let at = |i: usize, j: usize| {
            let j = j % (sectors + 1);
            i * (sectors + 1) + j
        };
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
                // Skip triangles collapsed by the pole rows.
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

    fn voxelization(
        vertices: &[f32],
        faces: &[u32],
        options: &MeshToDualGridOptions,
    ) -> DualGridVoxelization {
        mesh_to_flexible_dual_grid(vertices, faces, options).expect("voxelization")
    }

    fn assert_duals_inside_cells(result: &DualGridVoxelization, voxel_size: [f32; 3]) {
        let eps = 1e-4;
        for (coord, dual) in result.voxel_indices.iter().zip(&result.dual_vertices) {
            for axis in 0..3 {
                assert!(dual[axis].is_finite(), "dual vertex must be finite");
                let lo = coord[axis] as f32 * voxel_size[axis];
                let hi = (coord[axis] + 1) as f32 * voxel_size[axis];
                assert!(
                    dual[axis] >= lo - eps && dual[axis] <= hi + eps,
                    "dual {dual:?} outside cell {coord:?} on axis {axis} (cell [{lo}, {hi}])"
                );
            }
        }
    }

    fn voxel_size_for(aabb: [[f32; 3]; 2], grid_size: u32) -> [f32; 3] {
        [
            (aabb[1][0] - aabb[0][0]) / grid_size as f32,
            (aabb[1][1] - aabb[0][1]) / grid_size as f32,
            (aabb[1][2] - aabb[0][2]) / grid_size as f32,
        ]
    }

    /// Cube at grid 16 with the auto aabb: non-empty, coordinates inside the
    /// grid and spanning the full shell, dual vertices finite and inside
    /// their cells, intersected flags present, and the output byte-identical
    /// across runs.
    #[test]
    fn cube_voxelization_structural_invariants_and_determinism() {
        let (vertices, faces) = cube_mesh();
        let options = MeshToDualGridOptions::uniform(16);
        let result = voxelization(&vertices, &faces, &options);

        assert!(!result.voxel_indices.is_empty(), "cube must produce voxels");
        assert_eq!(result.voxel_indices.len(), result.dual_vertices.len());
        assert_eq!(result.voxel_indices.len(), result.intersected.len());

        let grid = 16u32;
        for coord in &result.voxel_indices {
            for axis in 0..3 {
                assert!(coord[axis] >= 0 && (coord[axis] as u32) < grid);
            }
        }
        // The cube shell spans the full padded grid extent on every axis.
        for axis in 0..3 {
            let min = result.voxel_indices.iter().map(|c| c[axis]).min().unwrap();
            let max = result.voxel_indices.iter().map(|c| c[axis]).max().unwrap();
            assert_eq!(min, 0, "axis {axis} must reach the low shell");
            assert_eq!(
                max,
                grid as i32 - 1,
                "axis {axis} must reach the high shell"
            );
        }

        // Auto aabb for the unit cube: padding = 1 / (16 - 1).
        let padding = 1.0f32 / 15.0;
        let aabb = [[0.0 - padding * 0.5; 3], [1.0 + padding * 0.5; 3]];
        assert_duals_inside_cells(&result, voxel_size_for(aabb, 16));

        let flagged = result
            .intersected
            .iter()
            .filter(|flags| flags.iter().any(|f| *f))
            .count();
        assert!(flagged > 0, "the cube surface must set intersected flags");
        assert!(flagged < result.voxel_indices.len() + 1);

        // Deterministic output: identical across runs.
        let again = voxelization(&vertices, &faces, &options);
        assert_eq!(result, again);
    }

    /// Auto-aabb must equal the upstream padding formula: voxelizing with
    /// the explicitly padded box is byte-identical to the auto run.
    #[test]
    fn auto_aabb_matches_upstream_padding_formula() {
        let (vertices, faces) = cube_mesh();
        let auto = voxelization(&vertices, &faces, &MeshToDualGridOptions::uniform(16));

        let padding = 1.0f32 / 15.0;
        let explicit = MeshToDualGridOptions {
            grid_size: [16; 3],
            aabb: Some([[0.0 - padding * 0.5; 3], [1.0 + padding * 0.5; 3]]),
            ..MeshToDualGridOptions::new([16; 3])
        };
        let manual = voxelization(&vertices, &faces, &explicit);

        assert_eq!(auto.voxel_indices, manual.voxel_indices);
        assert_eq!(auto.dual_vertices, manual.dual_vertices);
        assert_eq!(auto.intersected, manual.intersected);
    }

    /// The shape-SLAT encoder frame: `feats = dual * resolution - voxel`
    /// lands in the unit cell (upstream clamps to [0, 1] when packing).
    #[test]
    fn shape_slat_encoder_frame_feats_in_unit_cell() {
        let (vertices, faces) = cube_mesh();
        let options = MeshToDualGridOptions::for_shape_slat_encoder(32);
        let result = voxelization(&vertices, &faces, &options);

        for (coord, dual) in result.voxel_indices.iter().zip(&result.dual_vertices) {
            for axis in 0..3 {
                let feat = dual[axis] * 32.0 - coord[axis] as f32;
                assert!(
                    feat > -1e-3 && feat < 1.0 + 1e-3,
                    "encoder feat {feat} outside the unit cell"
                );
            }
        }
    }

    /// A single flat triangle: voxels appear only along its surface slab,
    /// everything stays finite, and the scan sets z-axis flags.
    #[test]
    fn single_triangle_voxelizes_along_surface() {
        let (vertices, faces) = flat_triangle();
        let options = MeshToDualGridOptions {
            grid_size: [16; 3],
            aabb: Some([[0.0; 3], [1.0; 3]]),
            ..MeshToDualGridOptions::new([16; 3])
        };
        let result = voxelization(&vertices, &faces, &options);

        assert!(!result.voxel_indices.is_empty());
        // The triangle lies in z = 0.5 = 8 * (1/16): only the ax2 = 2 scan
        // produces stamps (the other two projections have empty row/col
        // spans), so every voxel sits in z-cell 8.
        for coord in &result.voxel_indices {
            assert_eq!(coord[2], 8, "voxel {coord:?} off the surface slab");
            assert!(coord[0] >= 0 && coord[0] < 16);
            assert!(coord[1] >= 0 && coord[1] < 16);
        }
        assert_duals_inside_cells(&result, [1.0 / 16.0; 3]);
        assert!(
            result.intersected.iter().any(|flags| flags[2]),
            "z scan must set flags"
        );
    }

    /// Roundtrip: voxelize a cube, convert the dual vertices to per-cell
    /// fractions and extract with the existing
    /// `runtime::flexible_dual_grid_to_mesh`; the resulting mesh must stay
    /// inside the cube bounds and be non-empty.
    #[test]
    fn roundtrip_extraction_stays_within_cube_bounds() {
        let (vertices, faces) = cube_mesh();
        let aabb = [[0.0f32; 3], [1.0; 3]];
        let options = MeshToDualGridOptions {
            grid_size: [32; 3],
            aabb: Some(aabb),
            ..MeshToDualGridOptions::new([32; 3])
        };
        let result = voxelization(&vertices, &faces, &options);
        let voxel_size = voxel_size_for(aabb, 32);

        // Dual (absolute, shifted frame) -> per-cell fraction.
        let fractions: Vec<[f32; 3]> = result
            .dual_vertices
            .iter()
            .zip(&result.voxel_indices)
            .map(|(dual, coord)| {
                [
                    dual[0] / voxel_size[0] - coord[0] as f32,
                    dual[1] / voxel_size[1] - coord[1] as f32,
                    dual[2] / voxel_size[2] - coord[2] as f32,
                ]
            })
            .collect();
        let split_weight = vec![1.0f32; result.voxel_indices.len()];
        let mesh = flexible_dual_grid_to_mesh(
            &result.voxel_indices,
            &fractions,
            &result.intersected,
            &split_weight,
            [32usize; 3],
            aabb,
        );

        assert!(!mesh.faces.is_empty(), "extraction must produce faces");
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for chunk in mesh.vertices.chunks_exact(3) {
            for axis in 0..3 {
                min[axis] = min[axis].min(chunk[axis]);
                max[axis] = max[axis].max(chunk[axis]);
            }
        }
        for axis in 0..3 {
            assert!(min[axis] >= -1e-3, "min {min:?} outside the cube");
            assert!(max[axis] <= 1.0 + 1e-3, "max {max:?} outside the cube");
        }
    }

    /// Boundary weight scales the open-edge QEF: it must not change the
    /// voxel set (boundary terms only update existing voxels) but must move
    /// at least one dual vertex for an open mesh.
    #[test]
    fn boundary_weight_affects_dual_vertices_but_not_voxel_set() {
        let (vertices, faces) = flat_triangle();
        let make = |boundary_weight: f32| MeshToDualGridOptions {
            grid_size: [16; 3],
            aabb: Some([[0.0; 3], [1.0; 3]]),
            boundary_weight,
            ..MeshToDualGridOptions::new([16; 3])
        };
        let without = voxelization(&vertices, &faces, &make(0.0));
        let with = voxelization(&vertices, &faces, &make(0.2));

        assert_eq!(without.voxel_indices, with.voxel_indices);
        assert_ne!(without.dual_vertices, with.dual_vertices);
    }

    /// `face_weight = 0` disables the face term (upstream gate) while the
    /// scan-line term still produces a complete finite voxelization.
    #[test]
    fn zero_face_weight_skips_face_term() {
        let (vertices, faces) = cube_mesh();
        let options = MeshToDualGridOptions {
            grid_size: [16; 3],
            aabb: Some([[0.0; 3], [1.0; 3]]),
            face_weight: 0.0,
            ..MeshToDualGridOptions::new([16; 3])
        };
        let result = voxelization(&vertices, &faces, &options);
        assert!(!result.voxel_indices.is_empty());
        assert_duals_inside_cells(&result, [1.0 / 16.0; 3]);
    }

    /// Malformed inputs are rejected instead of producing NaN output.
    #[test]
    fn malformed_inputs_are_rejected() {
        let (vertices, faces) = cube_mesh();
        let base = MeshToDualGridOptions::uniform(16);
        let voxelization_err =
            |v: &[f32], f: &[u32], o: &MeshToDualGridOptions| mesh_to_flexible_dual_grid(v, f, o);

        // Empty mesh pieces and ragged buffers.
        assert!(voxelization_err(&[], &faces, &base).is_err());
        assert!(voxelization_err(&vertices, &[], &base).is_err());
        assert!(voxelization_err(&vertices[..8], &faces, &base).is_err());
        assert!(voxelization_err(&vertices, &faces[..2], &base).is_err());

        // Out-of-range face index.
        let bad_faces = vec![0u32, 1, 999];
        assert!(voxelization_err(&vertices, &bad_faces, &base).is_err());

        // Non-finite vertex.
        let mut nan_vertices = vertices.clone();
        nan_vertices[0] = f32::NAN;
        assert!(voxelization_err(&nan_vertices, &faces, &base).is_err());

        // Degenerate triangles: repeated index and collinear points.
        let repeated = vec![0u32, 0, 1];
        assert!(voxelization_err(&vertices, &repeated, &base).is_err());
        let collinear_vertices = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        assert!(voxelization_err(&collinear_vertices, &vec![0, 1, 2], &base).is_err());

        // Grid size bounds.
        let small = MeshToDualGridOptions {
            grid_size: [1, 16, 16],
            ..base.clone()
        };
        assert!(voxelization_err(&vertices, &faces, &small).is_err());
        let large = MeshToDualGridOptions {
            grid_size: [MAX_GRID_SIZE_PER_AXIS + 1, 2, 2],
            ..base.clone()
        };
        assert!(voxelization_err(&vertices, &faces, &large).is_err());
        // The cap boundary itself is accepted. A tiny mesh keeps the
        // stamping cost proportional to the surface, not the grid volume.
        let tiny = vec![0.0f32, 0.0, 0.4, 0.02, 0.0, 0.4, 0.0, 0.02, 0.4];
        let at_cap = MeshToDualGridOptions {
            grid_size: [MAX_GRID_SIZE_PER_AXIS; 3],
            aabb: Some([[-0.5; 3], [0.5; 3]]),
            ..base.clone()
        };
        assert!(voxelization_err(&tiny, &vec![0u32, 1, 2], &at_cap).is_ok());

        // Bad explicit aabb: inverted, zero-extent, non-finite.
        let inverted = MeshToDualGridOptions {
            grid_size: [16; 3],
            aabb: Some([[1.0; 3], [0.0; 3]]),
            ..base.clone()
        };
        assert!(voxelization_err(&vertices, &faces, &inverted).is_err());
        let zero_extent = MeshToDualGridOptions {
            grid_size: [16; 3],
            aabb: Some([[0.0; 3], [1.0, 1.0, 0.0]]),
            ..base.clone()
        };
        assert!(voxelization_err(&vertices, &faces, &zero_extent).is_err());
        let non_finite = MeshToDualGridOptions {
            grid_size: [16; 3],
            aabb: Some([[0.0; 3], [1.0, 1.0, f32::NAN]]),
            ..base.clone()
        };
        assert!(voxelization_err(&vertices, &faces, &non_finite).is_err());

        // Negative / non-finite weights.
        let negative = MeshToDualGridOptions {
            boundary_weight: -0.1,
            ..base.clone()
        };
        assert!(voxelization_err(&vertices, &faces, &negative).is_err());
        let nan_weight = MeshToDualGridOptions {
            regularization_weight: f32::NAN,
            ..base.clone()
        };
        assert!(voxelization_err(&vertices, &faces, &nan_weight).is_err());

        // Flat mesh with auto aabb (degenerate axis) is rejected; the same
        // mesh with an explicit aabb works.
        let (tri_vertices, tri_faces) = flat_triangle();
        assert!(voxelization_err(&tri_vertices, &tri_faces, &base).is_err());
    }

    /// Performance probe matching the production envelope: ~50k triangles
    /// voxelized into a 1024^3 grid with the encoder settings.
    #[test]
    #[ignore = "performance probe; run with: cargo test -p lux3d-core --release --lib dual_grid -- --ignored --nocapture"]
    fn perf_50k_tri_mesh_1024_grid() {
        let (vertices, faces) = uv_sphere(158, 158);
        let tri_count = faces.len() / 3;
        let options = MeshToDualGridOptions::for_shape_slat_encoder(1024);
        let start = std::time::Instant::now();
        let result = voxelization(&vertices, &faces, &options);
        let elapsed = start.elapsed();
        println!(
            "mesh_to_flexible_dual_grid: {tri_count} tris, grid 1024^3 -> {} voxels in {elapsed:?}",
            result.voxel_indices.len()
        );
        assert!(!result.voxel_indices.is_empty());
        assert_duals_inside_cells(&result, [1.0 / 1024.0; 3]);
    }
}
