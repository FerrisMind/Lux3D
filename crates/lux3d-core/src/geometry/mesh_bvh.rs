//! CPU triangle BVH for closest-point queries (TRELLIS.2 remesh/bake
//! foundation).
//!
//! Upstream evidence (`G:/jobs/candle-based-projects/candle-3d-refs/TRELLIS.2/o-voxel`):
//!
//! * `o_voxel/postprocess.py:122` builds a BVH over the cleaned
//!   `(vertices, faces)` mesh (`bvh = cumesh.cuBVH(vertices, faces)`) right
//!   after `fill_holes` and before either post-process branch.
//! * `o_voxel/postprocess.py:254-256` queries it with
//!   `bvh.unsigned_distance(valid_pos, return_uvw=True)`. The returned
//!   `(distance, face_id, uvw)` triple is consumed as
//!   `orig_tri_verts = vertices[faces[face_id.long()]]` followed by
//!   `valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)`, i.e.
//!   `uvw` holds the barycentric weights of the hit triangle's vertices
//!   `(v0, v1, v2)` and sums to 1.
//!
//! This module provides that primitive on CPU for the TR0 foundation (used by
//! TR1 narrow-band remeshing and TR3 bake snap-back):
//!
//! * [`MeshBvh::build`] from the same flat `(vertices, faces)` buffers the
//!   rest of `geometry::mesh_ops` uses (zero-copy: the tree borrows them).
//! * [`MeshBvh::closest_point`] -> [`ClosestPoint`]
//!   (`tri_id`, `uvw`, `position`, `unsigned_distance`), plus the batched
//!   [`MeshBvh::closest_points`].
//!
//! Scope and locked design notes:
//!
//! * The distance is UNSIGNED, exactly like the official
//!   `unsigned_distance` call. No inside/outside sign is computed here: for
//!   TR1 the sign is locked to ray-parity even-odd crossing counts, not to
//!   the side of the closest surface point / normal.
//! * Build is a median split on the largest centroid extent (leaf size 8).
//!   SAH is not needed: the official tree is opaque and only the
//!   nearest-triangle semantics are observable, so this is the smallest
//!   faithful BVH.
//! * Queries are deterministic: split ties order by triangle id and equal
//!   distances return the lowest triangle id.
//!
//! Conventions:
//!
//! * Vertices are a flat `[x0, y0, z0, x1, ...]` `f32` buffer; faces are
//!   zero-based `u32` triangles, 3 per face (same contract as
//!   `geometry::mesh_ops`).
//! * Build rejects malformed input with [`Lux3dError::InvalidInput`] instead
//!   of panicking: buffers whose lengths are not multiples of 3, empty
//!   meshes, out-of-range face indices, non-finite vertex coordinates, and
//!   more triangles than `u32` can address.
//! * Zero-area (degenerate) triangles are accepted; their closest point is
//!   well defined on the triangle edges.
//! * Query points MUST be finite: [`MeshBvh::closest_point`] /
//!   [`MeshBvh::closest_points`] assert the contract in every build profile
//!   instead of returning the internal sentinel triangle id.

use std::cmp::Ordering;

use crate::error::Lux3dError;
use crate::Result;

/// Triangles per BVH leaf: median splits stop once a node holds this many
/// (or fewer) triangles.
const MAX_LEAF_TRIANGLES: usize = 8;

/// Relative slack on the node-pruning bound. The AABB lower bound is
/// conservative in exact arithmetic; the slack keeps f32 rounding from
/// pruning a node that holds the true minimum.
const PRUNE_RELATIVE_SLACK: f32 = 1e-5;

/// Absolute companion of [`PRUNE_RELATIVE_SLACK`] (handles a zero best
/// distance, e.g. a query point lying exactly on the surface).
const PRUNE_ABSOLUTE_SLACK: f32 = 1e-9;

/// Result of a closest-point query against a [`MeshBvh`].
///
/// Mirrors the official `bvh.unsigned_distance(point, return_uvw=True)`
/// triple plus the reconstructed surface position
/// (`postprocess.py:254-256`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClosestPoint {
    /// Zero-based triangle (face) index of the closest triangle.
    pub tri_id: u32,
    /// Barycentric weights for the hit triangle's vertices `(v0, v1, v2)`
    /// (sum to 1). `position == uvw[0] * v0 + uvw[1] * v1 + uvw[2] * v2`.
    pub uvw: [f32; 3],
    /// Closest point on the mesh surface.
    pub position: [f32; 3],
    /// Unsigned distance from the query point to `position`.
    pub unsigned_distance: f32,
}

/// One BVH node. `count > 0` marks a leaf whose triangles live at
/// `tri_order[left..left + count]`; otherwise `left` / `right` are child node
/// indices.
#[derive(Debug, Clone, Copy)]
struct BvhNode {
    min: [f32; 3],
    max: [f32; 3],
    left: u32,
    right: u32,
    count: u32,
}

/// Median-split triangle BVH over borrowed mesh buffers.
///
/// The tree stores only the node hierarchy, the triangle permutation and its
/// bounds; vertex/face data stays in the caller's buffers (zero-copy).
#[derive(Debug, Clone)]
pub struct MeshBvh<'a> {
    vertices: &'a [f32],
    faces: &'a [u32],
    nodes: Vec<BvhNode>,
    /// Triangle ids grouped by leaf; leaves index contiguous ranges.
    tri_order: Vec<u32>,
}

impl<'a> MeshBvh<'a> {
    /// Build the BVH over a flat `[x, y, z, ...]` vertex buffer and `u32`
    /// triangle indices.
    ///
    /// Matches the official `cumesh.cuBVH(vertices, faces)` construction
    /// (`postprocess.py:122`) in what it observes: nearest-triangle ids and
    /// barycentric coordinates.
    ///
    /// Errors (never panics) on malformed buffers; see the module docs.
    pub fn build(vertices: &'a [f32], faces: &'a [u32]) -> Result<Self> {
        if vertices.len() % 3 != 0 {
            return Err(Lux3dError::InvalidInput(
                "mesh bvh vertex buffer length must be a multiple of 3",
            ));
        }
        if faces.len() % 3 != 0 {
            return Err(Lux3dError::InvalidInput(
                "mesh bvh face buffer length must be a multiple of 3",
            ));
        }
        if vertices.is_empty() || faces.is_empty() {
            return Err(Lux3dError::InvalidInput(
                "mesh bvh requires a non-empty mesh",
            ));
        }
        let vertex_count = vertices.len() / 3;
        for &index in faces {
            if index as usize >= vertex_count {
                return Err(Lux3dError::InvalidInput(
                    "mesh bvh face index outside the vertex buffer",
                ));
            }
        }
        if vertices.iter().any(|value| !value.is_finite()) {
            return Err(Lux3dError::InvalidInput(
                "mesh bvh requires finite vertex positions",
            ));
        }
        let triangle_count = faces.len() / 3;
        if triangle_count > u32::MAX as usize {
            return Err(Lux3dError::InvalidInput(
                "mesh bvh triangle count exceeds u32 addressing",
            ));
        }

        let mut centroids = Vec::with_capacity(triangle_count);
        for triangle in faces.chunks_exact(3) {
            let a = vertex(vertices, triangle[0] as usize);
            let b = vertex(vertices, triangle[1] as usize);
            let c = vertex(vertices, triangle[2] as usize);
            centroids.push([
                (a[0] + b[0] + c[0]) / 3.0,
                (a[1] + b[1] + c[1]) / 3.0,
                (a[2] + b[2] + c[2]) / 3.0,
            ]);
        }

        let mut order: Vec<u32> = (0..triangle_count as u32).collect();
        let mut nodes = Vec::with_capacity(2 * triangle_count.div_ceil(MAX_LEAF_TRIANGLES));
        build_node(&mut nodes, &mut order, 0, &centroids, vertices, faces);
        Ok(Self {
            vertices,
            faces,
            nodes,
            tri_order: order,
        })
    }

    /// Closest point on the mesh to `point`.
    ///
    /// Returns the hit triangle id, its barycentric `uvw`, the surface
    /// position and the unsigned distance — the data the official
    /// `unsigned_distance(..., return_uvw=True)` call exposes
    /// (`postprocess.py:254-256`).
    ///
    /// # Contract
    ///
    /// `point` must be finite. The contract is enforced with an assertion in
    /// every build profile: a non-finite query has no meaningful nearest
    /// triangle and would otherwise return the internal sentinel triangle
    /// id. Every in-tree caller validates its positions (remesh checks the
    /// input vertices; the bake path interpolates validated atlas vertices).
    pub fn closest_point(&self, point: [f32; 3]) -> ClosestPoint {
        let mut stack = Vec::with_capacity(64);
        self.query(point, &mut stack)
    }

    /// Batched [`Self::closest_point`]: one result per input point, in input
    /// order. Every point must satisfy the same finite-input contract.
    pub fn closest_points(&self, points: &[[f32; 3]]) -> Vec<ClosestPoint> {
        let mut stack = Vec::with_capacity(64);
        points
            .iter()
            .map(|&point| self.query(point, &mut stack))
            .collect()
    }

    fn query(&self, point: [f32; 3], stack: &mut Vec<u32>) -> ClosestPoint {
        assert!(
            point.iter().all(|value| value.is_finite()),
            "MeshBvh::closest_point requires a finite query point, got {point:?}"
        );
        stack.clear();
        stack.push(0);

        let mut best_distance_sq = f32::INFINITY;
        let mut best_tri_id = u32::MAX;
        let mut best_position = [0.0; 3];
        let mut best_uvw = [0.0; 3];

        while let Some(node_index) = stack.pop() {
            let node = self.nodes[node_index as usize];
            let node_distance_sq = aabb_distance_sq(point, node.min, node.max);
            let prune_limit =
                best_distance_sq * (1.0 + PRUNE_RELATIVE_SLACK) + PRUNE_ABSOLUTE_SLACK;
            if node_distance_sq > prune_limit {
                continue;
            }
            if node.count > 0 {
                let start = node.left as usize;
                let end = start + node.count as usize;
                for &tri_id in &self.tri_order[start..end] {
                    let base = tri_id as usize * 3;
                    let a = vertex(self.vertices, self.faces[base] as usize);
                    let b = vertex(self.vertices, self.faces[base + 1] as usize);
                    let c = vertex(self.vertices, self.faces[base + 2] as usize);
                    let (position, uvw) = closest_point_on_triangle(point, a, b, c);
                    let candidate_distance_sq = dist_sq(point, position);
                    if candidate_distance_sq < best_distance_sq
                        || (candidate_distance_sq == best_distance_sq && tri_id < best_tri_id)
                    {
                        best_distance_sq = candidate_distance_sq;
                        best_tri_id = tri_id;
                        best_position = position;
                        best_uvw = uvw;
                    }
                }
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }

        debug_assert_ne!(best_tri_id, u32::MAX, "a built mesh bvh has a hit");
        ClosestPoint {
            tri_id: best_tri_id,
            uvw: best_uvw,
            position: best_position,
            unsigned_distance: best_distance_sq.sqrt(),
        }
    }
}

/// Recursively split `order[offset..offset + order.len()]` into a median-split
/// subtree and return the node index. `offset` is the slice's position in the
/// full triangle order, stored by leaves.
fn build_node(
    nodes: &mut Vec<BvhNode>,
    order: &mut [u32],
    offset: usize,
    centroids: &[[f32; 3]],
    vertices: &[f32],
    faces: &[u32],
) -> u32 {
    let count = order.len();

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for &tri_id in order.iter() {
        let base = tri_id as usize * 3;
        for corner in 0..3 {
            let position = vertex(vertices, faces[base + corner] as usize);
            for axis in 0..3 {
                min[axis] = min[axis].min(position[axis]);
                max[axis] = max[axis].max(position[axis]);
            }
        }
    }

    let node_index = nodes.len() as u32;
    nodes.push(BvhNode {
        min,
        max,
        left: 0,
        right: 0,
        count: 0,
    });

    if count <= MAX_LEAF_TRIANGLES {
        let node = &mut nodes[node_index as usize];
        node.left = offset as u32;
        node.count = count as u32;
        return node_index;
    }

    let mut centroid_min = [f32::INFINITY; 3];
    let mut centroid_max = [f32::NEG_INFINITY; 3];
    for &tri_id in order.iter() {
        let centroid = centroids[tri_id as usize];
        for axis in 0..3 {
            centroid_min[axis] = centroid_min[axis].min(centroid[axis]);
            centroid_max[axis] = centroid_max[axis].max(centroid[axis]);
        }
    }
    let extent = [
        centroid_max[0] - centroid_min[0],
        centroid_max[1] - centroid_min[1],
        centroid_max[2] - centroid_min[2],
    ];
    // Largest centroid extent; stable x -> y -> z tie order.
    let axis = if extent[0] >= extent[1] && extent[0] >= extent[2] {
        0
    } else if extent[1] >= extent[2] {
        1
    } else {
        2
    };
    if extent[axis] <= 0.0 {
        // All centroids coincide: splitting cannot separate them, so stop.
        let node = &mut nodes[node_index as usize];
        node.left = offset as u32;
        node.count = count as u32;
        return node_index;
    }

    let mid = count / 2;
    order.select_nth_unstable_by(mid, |&a, &b| centroids_cmp(a, b, axis, centroids));
    let (left_order, right_order) = order.split_at_mut(mid);
    let left = build_node(nodes, left_order, offset, centroids, vertices, faces);
    let right = build_node(nodes, right_order, offset + mid, centroids, vertices, faces);

    let node = &mut nodes[node_index as usize];
    node.left = left;
    node.right = right;
    node_index
}

/// Total order on triangle centroids along `axis`: coordinate first, then
/// triangle id so the split is deterministic when centroids tie.
fn centroids_cmp(a: u32, b: u32, axis: usize, centroids: &[[f32; 3]]) -> Ordering {
    let left = centroids[a as usize][axis];
    let right = centroids[b as usize][axis];
    left.partial_cmp(&right)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.cmp(&b))
}

/// Squared distance from `point` to an axis-aligned box (0 inside).
fn aabb_distance_sq(point: [f32; 3], min: [f32; 3], max: [f32; 3]) -> f32 {
    let mut distance_sq = 0.0;
    for axis in 0..3 {
        let offset = if point[axis] < min[axis] {
            min[axis] - point[axis]
        } else if point[axis] > max[axis] {
            point[axis] - max[axis]
        } else {
            0.0
        };
        distance_sq += offset * offset;
    }
    distance_sq
}

/// Closest point on a triangle plus its barycentric weights for
/// `(a, b, c)`; the weights sum to 1 and reconstruct the returned position.
fn closest_point_on_triangle(
    point: [f32; 3],
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let uvw = triangle_uvw(point, a, b, c);
    let position = [
        uvw[0] * a[0] + uvw[1] * b[0] + uvw[2] * c[0],
        uvw[0] * a[1] + uvw[1] * b[1] + uvw[2] * c[1],
        uvw[0] * a[2] + uvw[1] * b[2] + uvw[2] * c[2],
    ];
    (position, uvw)
}

/// Barycentric weights of the closest point on triangle `(a, b, c)`.
///
/// Region-based closest-point-on-triangle test (vertex, edge, then interior
/// Voronoi regions), with the usual edge projection for zero-area triangles.
fn triangle_uvw(point: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(point, a);

    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return [1.0, 0.0, 0.0];
    }

    let bp = sub(point, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return [0.0, 1.0, 0.0];
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let denominator = d1 - d3;
        let v = if denominator != 0.0 {
            d1 / denominator
        } else {
            0.0
        };
        return [1.0 - v, v, 0.0];
    }

    let cp = sub(point, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return [0.0, 0.0, 1.0];
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let denominator = d2 - d6;
        let w = if denominator != 0.0 {
            d2 / denominator
        } else {
            0.0
        };
        return [1.0 - w, 0.0, w];
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let denominator = (d4 - d3) + (d5 - d6);
        let w = if denominator != 0.0 {
            (d4 - d3) / denominator
        } else {
            0.0
        };
        return [0.0, 1.0 - w, w];
    }

    let denominator = va + vb + vc;
    if denominator > 0.0 && denominator.is_finite() {
        let inverse = 1.0 / denominator;
        let v = vb * inverse;
        let w = vc * inverse;
        return [1.0 - v - w, v, w];
    }

    // Degenerate (zero-area) triangle or overflow: fall back to its edges.
    degenerate_triangle_uvw(point, a, b, c)
}

/// Closest point over the three edges of a degenerate triangle.
fn degenerate_triangle_uvw(point: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let candidates = [
        edge_candidate(point, a, b, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        edge_candidate(point, b, c, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        edge_candidate(point, c, a, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
    ];
    let mut best = candidates[0];
    let mut best_distance_sq = dist_sq(point, best.0);
    for &candidate in &candidates[1..] {
        let candidate_distance_sq = dist_sq(point, candidate.0);
        if candidate_distance_sq < best_distance_sq {
            best = candidate;
            best_distance_sq = candidate_distance_sq;
        }
    }
    best.1
}

/// Closest point on segment `start..end` and the barycentric weights
/// interpolated between the segment's endpoint weights.
fn edge_candidate(
    point: [f32; 3],
    start: [f32; 3],
    end: [f32; 3],
    start_uvw: [f32; 3],
    end_uvw: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let direction = sub(end, start);
    let length_sq = dot(direction, direction);
    let t = if length_sq > 0.0 {
        (dot(sub(point, start), direction) / length_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let position = add(start, scale(direction, t));
    let uvw = [
        start_uvw[0] + (end_uvw[0] - start_uvw[0]) * t,
        start_uvw[1] + (end_uvw[1] - start_uvw[1]) * t,
        start_uvw[2] + (end_uvw[2] - start_uvw[2]) * t,
    ];
    (position, uvw)
}

/// Flat-buffer vertex accessor (`index` is a vertex index, not a coordinate
/// offset).
fn vertex(vertices: &[f32], index: usize) -> [f32; 3] {
    let base = index * 3;
    [vertices[base], vertices[base + 1], vertices[base + 2]]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(vector: [f32; 3], factor: f32) -> [f32; 3] {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn dist_sq(a: [f32; 3], b: [f32; 3]) -> f32 {
    let difference = sub(a, b);
    dot(difference, difference)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG for fuzz points (no external RNG dependency).
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }

        fn next_f32(&mut self) -> f32 {
            self.next_u32() as f32 / (u32::MAX as f32 + 1.0)
        }

        fn range(&mut self, low: f32, high: f32) -> f32 {
            low + (high - low) * self.next_f32()
        }

        fn point(&mut self, low: f32, high: f32) -> [f32; 3] {
            [
                self.range(low, high),
                self.range(low, high),
                self.range(low, high),
            ]
        }
    }

    /// Unit cube `[0, 1]^3` with the `geometry::mesh_ops` layout.
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

    /// UV sphere of `radius`; pole rings contain degenerate triangles by
    /// construction (good extra coverage for the edge fallback).
    fn sphere(rings: usize, segments: usize, radius: f32) -> (Vec<f32>, Vec<u32>) {
        let mut vertices = Vec::with_capacity((rings + 1) * segments * 3);
        for ring in 0..=rings {
            let theta = std::f32::consts::PI * ring as f32 / rings as f32;
            let (sin_theta, cos_theta) = theta.sin_cos();
            for segment in 0..segments {
                let phi = 2.0 * std::f32::consts::PI * segment as f32 / segments as f32;
                let (sin_phi, cos_phi) = phi.sin_cos();
                vertices.push(radius * sin_theta * cos_phi);
                vertices.push(radius * cos_theta);
                vertices.push(radius * sin_theta * sin_phi);
            }
        }
        let mut faces = Vec::with_capacity(rings * segments * 6);
        for ring in 0..rings {
            for segment in 0..segments {
                let next = (segment + 1) % segments;
                let a = (ring * segments + segment) as u32;
                let b = (ring * segments + next) as u32;
                let c = ((ring + 1) * segments + segment) as u32;
                let d = ((ring + 1) * segments + next) as u32;
                faces.extend_from_slice(&[a, c, b]);
                faces.extend_from_slice(&[b, c, d]);
            }
        }
        (vertices, faces)
    }

    fn torus(
        major_radius: f32,
        minor_radius: f32,
        major_segments: usize,
        minor_segments: usize,
    ) -> (Vec<f32>, Vec<u32>) {
        let mut vertices = Vec::with_capacity(major_segments * minor_segments * 3);
        for major in 0..major_segments {
            let u = 2.0 * std::f32::consts::PI * major as f32 / major_segments as f32;
            let (sin_u, cos_u) = u.sin_cos();
            for minor in 0..minor_segments {
                let v = 2.0 * std::f32::consts::PI * minor as f32 / minor_segments as f32;
                let (sin_v, cos_v) = v.sin_cos();
                let ring = major_radius + minor_radius * cos_v;
                vertices.push(ring * cos_u);
                vertices.push(minor_radius * sin_v);
                vertices.push(ring * sin_u);
            }
        }
        let mut faces = Vec::with_capacity(major_segments * minor_segments * 6);
        for major in 0..major_segments {
            let next_major = (major + 1) % major_segments;
            for minor in 0..minor_segments {
                let next_minor = (minor + 1) % minor_segments;
                let a = (major * minor_segments + minor) as u32;
                let b = (next_major * minor_segments + minor) as u32;
                let c = (major * minor_segments + next_minor) as u32;
                let d = (next_major * minor_segments + next_minor) as u32;
                faces.extend_from_slice(&[a, b, c]);
                faces.extend_from_slice(&[b, d, c]);
            }
        }
        (vertices, faces)
    }

    /// Exhaustive reference query reusing the same per-triangle routine (so
    /// only the tree traversal is under test).
    fn brute_force(vertices: &[f32], faces: &[u32], point: [f32; 3]) -> ClosestPoint {
        let mut best_distance_sq = f32::INFINITY;
        let mut best_tri_id = u32::MAX;
        let mut best_position = [0.0; 3];
        let mut best_uvw = [0.0; 3];
        for (tri_id, triangle) in faces.chunks_exact(3).enumerate() {
            let a = vertex(vertices, triangle[0] as usize);
            let b = vertex(vertices, triangle[1] as usize);
            let c = vertex(vertices, triangle[2] as usize);
            let (position, uvw) = closest_point_on_triangle(point, a, b, c);
            let candidate_distance_sq = dist_sq(point, position);
            let tri_id = tri_id as u32;
            if candidate_distance_sq < best_distance_sq
                || (candidate_distance_sq == best_distance_sq && tri_id < best_tri_id)
            {
                best_distance_sq = candidate_distance_sq;
                best_tri_id = tri_id;
                best_position = position;
                best_uvw = uvw;
            }
        }
        ClosestPoint {
            tri_id: best_tri_id,
            uvw: best_uvw,
            position: best_position,
            unsigned_distance: best_distance_sq.sqrt(),
        }
    }

    fn assert_matches_brute_force(
        bvh: &MeshBvh<'_>,
        vertices: &[f32],
        faces: &[u32],
        point: [f32; 3],
        context: &str,
    ) {
        let hit = bvh.closest_point(point);
        let reference = brute_force(vertices, faces, point);
        assert_eq!(
            hit.tri_id, reference.tri_id,
            "{context}: triangle id for point {point:?}"
        );
        let tolerance = 1e-5 * (1.0 + point.iter().fold(0.0f32, |acc, v| acc.max(v.abs())));
        assert!(
            (hit.unsigned_distance - reference.unsigned_distance).abs() <= tolerance,
            "{context}: distance {} vs brute force {} for point {point:?}",
            hit.unsigned_distance,
            reference.unsigned_distance
        );
        for axis in 0..3 {
            assert!(
                (hit.position[axis] - reference.position[axis]).abs() <= tolerance,
                "{context}: position axis {axis} {} vs brute force {}",
                hit.position[axis],
                reference.position[axis]
            );
            assert!(
                (hit.uvw[axis] - reference.uvw[axis]).abs() <= tolerance,
                "{context}: uvw axis {axis} {} vs brute force {}",
                hit.uvw[axis],
                reference.uvw[axis]
            );
        }

        // uvw must be barycentric for the hit triangle and reconstruct the
        // reported position (official reconstruction, `postprocess.py:255-256`).
        assert!(
            (hit.uvw.iter().sum::<f32>() - 1.0).abs() <= 1e-4,
            "{context}: uvw does not sum to 1: {:?}",
            hit.uvw
        );
        let base = hit.tri_id as usize * 3;
        for axis in 0..3 {
            let reconstructed = hit.uvw[0] * vertices[faces[base] as usize * 3 + axis]
                + hit.uvw[1] * vertices[faces[base + 1] as usize * 3 + axis]
                + hit.uvw[2] * vertices[faces[base + 2] as usize * 3 + axis];
            assert!(
                (reconstructed - hit.position[axis]).abs() <= 1e-4,
                "{context}: uvw reconstruction axis {axis} {reconstructed} vs {}",
                hit.position[axis]
            );
        }
    }

    #[test]
    fn cube_closest_points_match_expected_geometry() {
        let (vertices, faces) = cube();
        let bvh = MeshBvh::build(&vertices, &faces).expect("build cube bvh");

        let center = bvh.closest_point([0.5, 0.5, 0.5]);
        assert!((center.unsigned_distance - 0.5).abs() <= 1e-6);

        let on_face = bvh.closest_point([0.5, 0.5, 0.0]);
        assert!(on_face.unsigned_distance <= 1e-6);

        let above_face = bvh.closest_point([0.5, 0.5, 2.0]);
        assert!((above_face.unsigned_distance - 1.0).abs() <= 1e-6);
        assert!((above_face.position[2] - 1.0).abs() <= 1e-6);

        let corner_outside = bvh.closest_point([2.0, 2.0, 2.0]);
        assert!((corner_outside.unsigned_distance - 3.0f32.sqrt()).abs() <= 1e-6);
        for axis in 0..3 {
            assert!((corner_outside.position[axis] - 1.0).abs() <= 1e-6);
        }
    }

    #[test]
    fn cube_queries_match_brute_force() {
        let (vertices, faces) = cube();
        let bvh = MeshBvh::build(&vertices, &faces).expect("build cube bvh");
        let mut rng = Lcg::new(0x5eed);
        for _ in 0..500 {
            let point = rng.point(-1.5, 2.5);
            assert_matches_brute_force(&bvh, &vertices, &faces, point, "cube");
        }
    }

    #[test]
    fn sphere_queries_match_brute_force() {
        let (vertices, faces) = sphere(16, 24, 1.0);
        let bvh = MeshBvh::build(&vertices, &faces).expect("build sphere bvh");
        let mut rng = Lcg::new(0x5c4e3e);
        for _ in 0..400 {
            let point = rng.point(-1.6, 1.6);
            assert_matches_brute_force(&bvh, &vertices, &faces, point, "sphere");
        }
    }

    #[test]
    fn torus_queries_match_brute_force() {
        let (vertices, faces) = torus(1.0, 0.35, 20, 14);
        let bvh = MeshBvh::build(&vertices, &faces).expect("build torus bvh");
        let mut rng = Lcg::new(0x70e05);
        for _ in 0..400 {
            let point = rng.point(-1.8, 1.8);
            assert_matches_brute_force(&bvh, &vertices, &faces, point, "torus");
        }
    }

    #[test]
    fn uvw_is_barycentric_for_the_hit_triangle() {
        let vertices = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let faces = [0u32, 1, 2];
        let bvh = MeshBvh::build(&vertices, &faces).expect("build single triangle");

        let hit = bvh.closest_point([0.25, 0.25, 2.0]);
        assert_eq!(hit.tri_id, 0);
        assert!((hit.position[0] - 0.25).abs() <= 1e-6);
        assert!((hit.position[1] - 0.25).abs() <= 1e-6);
        assert!(hit.position[2].abs() <= 1e-6);
        assert!((hit.uvw[0] - 0.5).abs() <= 1e-6);
        assert!((hit.uvw[1] - 0.25).abs() <= 1e-6);
        assert!((hit.uvw[2] - 0.25).abs() <= 1e-6);
        assert!((hit.unsigned_distance - 2.0).abs() <= 1e-6);

        // Vertex region: the query projects past vertex `b`.
        let vertex_hit = bvh.closest_point([3.0, -1.0, 0.0]);
        assert_eq!(vertex_hit.uvw, [0.0, 1.0, 0.0]);
        assert!((vertex_hit.unsigned_distance - 5.0f32.sqrt()).abs() <= 1e-6);
    }

    #[test]
    fn batched_queries_match_scalar_queries() {
        let (vertices, faces) = torus(1.0, 0.35, 20, 14);
        let bvh = MeshBvh::build(&vertices, &faces).expect("build torus bvh");
        let mut rng = Lcg::new(0xba7c4);
        let points: Vec<[f32; 3]> = (0..128).map(|_| rng.point(-1.8, 1.8)).collect();
        let batched = bvh.closest_points(&points);
        assert_eq!(batched.len(), points.len());
        for (point, batched_hit) in points.iter().zip(batched.iter()) {
            let scalar_hit = bvh.closest_point(*point);
            assert_eq!(*batched_hit, scalar_hit, "batched query for {point:?}");
        }
    }

    #[test]
    fn build_and_queries_are_deterministic() {
        let (vertices, faces) = sphere(16, 24, 1.0);
        let first = MeshBvh::build(&vertices, &faces).expect("build first bvh");
        let second = MeshBvh::build(&vertices, &faces).expect("build second bvh");
        let mut rng = Lcg::new(0xd37e4);
        for _ in 0..100 {
            let point = rng.point(-1.6, 1.6);
            let a = first.closest_point(point);
            let b = first.closest_point(point);
            let c = second.closest_point(point);
            assert_eq!(a, b, "repeated query for {point:?}");
            assert_eq!(a, c, "rebuild query for {point:?}");
        }
    }

    #[test]
    fn build_rejects_malformed_meshes() {
        assert!(MeshBvh::build(&[], &[]).is_err());
        assert!(MeshBvh::build(&[0.0, 0.0], &[]).is_err());

        let triangle = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        // Two indices are not a triangle.
        assert!(MeshBvh::build(&triangle, &[0, 1]).is_err());
        // Index outside the vertex buffer.
        assert!(MeshBvh::build(&triangle, &[0, 1, 3]).is_err());
        // Non-finite vertex coordinate.
        let non_finite = [0.0f32, 0.0, 0.0, 1.0, f32::NAN, 0.0, 0.0, 1.0, 0.0];
        assert!(MeshBvh::build(&non_finite, &[0, 1, 2]).is_err());
        // The well-formed case builds.
        assert!(MeshBvh::build(&triangle, &[0, 1, 2]).is_ok());
    }

    #[test]
    fn degenerate_triangles_have_defined_closest_points() {
        // Colinear (zero-area) triangle along the x axis.
        let vertices = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let faces = [0u32, 1, 2];
        let bvh = MeshBvh::build(&vertices, &faces).expect("build degenerate bvh");

        let hit = bvh.closest_point([0.5, 1.0, 0.0]);
        assert!((hit.position[0] - 0.5).abs() <= 1e-6);
        assert!(hit.position[1].abs() <= 1e-6);
        assert!((hit.unsigned_distance - 1.0).abs() <= 1e-6);
        assert!((hit.uvw.iter().sum::<f32>() - 1.0).abs() <= 1e-6);
    }

    /// The finite-input contract is enforced in every build profile: a
    /// non-finite query panics instead of returning the internal sentinel
    /// triangle id.
    #[test]
    #[should_panic(expected = "requires a finite query point")]
    fn non_finite_queries_are_rejected() {
        let vertices = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let bvh = MeshBvh::build(&vertices, &[0, 1, 2]).expect("build bvh");
        let _ = bvh.closest_point([f32::NAN, 0.0, 0.0]);
    }
}
