//! CPU port of the cumesh normal-cone chart pre-clustering that TRELLIS.2's
//! UV bake runs before xatlas (trellis.cpp `src/uv_bake.cpp:850-1000`, the
//! "coarse pre-clustering of the big faces" block; upstream
//! `cumesh.compute_charts`, atlas.cu:1071-1210, spec 27 section 6.2).
//!
//! Bottom-up chart merging. Every non-degenerate face starts as its own
//! chart; each round, every manifold chart-pair is scored with
//!
//! ```text
//! cost = merged_cone_half_angle
//!      + area_weight * (areaA + areaB)
//!      + perimeter_area_weight * (perimA + perimB - 2 * shared)^2 / (areaA + areaB)
//! ```
//!
//! where `merged_cone_half_angle` is half the angular span of the cone that
//! contains both charts (`(hi - lo) * 0.5` in the reference, with
//! `lo = min(-halfA, angleAB - halfB)`, `hi = max(halfA, angleAB + halfB)`).
//! A pair collapses only if it is the minimum-cost pair of BOTH charts and
//! `cost <= cone_half_angle_rad`. Chart cones (axis = normalized sum of the
//! face normals, half-angle = maximum angular deviation) are recomputed from
//! scratch every round, exactly like the reference.
//!
//! Determinism: the manifold edge list is built in `BTreeMap` order, the
//! chart-pair cost map is a `BTreeMap` (iterated in `(c0, c1)` order), the
//! argmin tie-break keeps the lowest pair index, faces inside a cluster are
//! sorted and clusters are ordered by their minimum face id. Two calls with
//! identical input therefore return an identical partition.
//!
//! Deviations from the reference, all intentional and documented:
//!
//! * Adjacency is by vertex INDEX, as in the reference (edge key = sorted
//!   vertex-index pair). Coincident-but-distinct vertices are not welded
//!   here; welding happens upstream (`geometry::uv_atlas` welds by position
//!   before charting).
//! * Zero-area (or non-finite-area) faces are split out as singleton
//!   clusters and reported in [`NormalConePartition::degenerate_faces`]. The
//!   reference has no degeneracy handling and would feed zero-length normals
//!   into its cone math; cumesh's caller is assumed to have cleaned the mesh.
//! * `max_rounds` defaults to [`NORMAL_CONE_MAX_ROUNDS`] (128); the C++
//!   block caps its merge loop at 256 rounds.
//! * A non-finite cost (only reachable when f32 coordinate magnitudes
//!   overflow the area/perimeter arithmetic) is saturated to `f32::INFINITY`
//!   so it can never be selected for a collapse.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use crate::Lux3dError;

/// Default cap on the merged cone half-angle, and the default collapse
/// threshold: `pi/2` (the upstream
/// `threshold_cone_half_angle_rad`, cumesh's most permissive cone).
pub const NORMAL_CONE_HALF_ANGLE_RAD: f32 = std::f32::consts::FRAC_PI_2;

/// Default weight of the merged-pair area term.
pub const NORMAL_CONE_AREA_WEIGHT: f32 = 0.1;

/// Default weight of the merged-pair perimeter-vs-area term.
pub const NORMAL_CONE_PERIMETER_AREA_WEIGHT: f32 = 1e-4;

/// Default maximum number of merge rounds.
pub const NORMAL_CONE_MAX_ROUNDS: u32 = 128;

/// Tuning for [`normal_cone_clusters`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalConeOptions {
    /// Collapse threshold for the pair cost (radians). Default
    /// [`NORMAL_CONE_HALF_ANGLE_RAD`].
    pub cone_half_angle_rad: f32,
    /// Weight of `areaA + areaB`. Default [`NORMAL_CONE_AREA_WEIGHT`].
    pub area_weight: f32,
    /// Weight of the squared non-shared perimeter over merged area. Default
    /// [`NORMAL_CONE_PERIMETER_AREA_WEIGHT`].
    pub perimeter_area_weight: f32,
    /// Maximum number of merge rounds. Default [`NORMAL_CONE_MAX_ROUNDS`].
    pub max_rounds: u32,
}

impl Default for NormalConeOptions {
    fn default() -> Self {
        Self {
            cone_half_angle_rad: NORMAL_CONE_HALF_ANGLE_RAD,
            area_weight: NORMAL_CONE_AREA_WEIGHT,
            perimeter_area_weight: NORMAL_CONE_PERIMETER_AREA_WEIGHT,
            max_rounds: NORMAL_CONE_MAX_ROUNDS,
        }
    }
}

/// Result of [`normal_cone_clusters`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NormalConePartition {
    /// Faces per cluster, ascending. Clusters are ordered by their minimum
    /// face id, so cluster 0 contains face 0 unless face 0 is degenerate (in
    /// which case it is still the first singleton).
    pub clusters: Vec<Vec<u32>>,
    /// Cluster index per face; `face_cluster.len()` equals the face count
    /// (`faces.len() / 3`). Every face, degenerate ones included, is covered
    /// exactly once.
    pub face_cluster: Vec<u32>,
    /// Merge rounds executed: each round costs every chart pair and then
    /// collapses the mutual argmin pairs, so a round that merges nothing is
    /// counted, while a round that starts with a single chart (or a mesh with
    /// no non-degenerate faces) is not.
    pub rounds: u32,
    /// Zero-area faces (ascending). Each is its own singleton cluster and
    /// takes part in no adjacency, perimeter or cone math.
    pub degenerate_faces: Vec<u32>,
}

/// Precomputed geometry for one non-degenerate face.
#[derive(Debug, Clone, Copy)]
struct FaceGeom {
    /// Unit face normal; zero only when the (non-zero) cross length is below
    /// `1e-20` (reference behaviour).
    normal: [f32; 3],
    area: f32,
    /// Lengths of the edges `v0v1`, `v1v2`, `v2v0`.
    edge_len: [f32; 3],
}

/// Second-face registration state for one undirected vertex-index edge.
#[derive(Debug, Clone, Copy)]
enum EdgeSlot {
    /// One incident face so far.
    First(u32),
    /// Exactly two incident faces.
    Second(u32, u32),
    /// Three or more incident faces: non-manifold, never adjacent.
    NonManifold,
}

#[inline]
fn edge_key(a: u32, b: u32) -> u64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    ((lo as u64) << 32) | (hi as u64)
}

#[inline]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn length(a: [f32; 3]) -> f32 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Merged float that is safe to compare against a threshold: `-0.0` and
/// `+0.0` collapse to the same ordering, and NaN becomes `+inf`.
#[inline]
fn sanitized_cost(cost: f32) -> f32 {
    if cost.is_finite() {
        cost.max(-f32::INFINITY)
    } else {
        f32::INFINITY
    }
}

/// Cluster `faces` into normal-cone charts, a CPU port of cumesh's
/// `compute_charts` merge phase.
///
/// `faces` is a flat triangle list (`len` must be a multiple of 3) indexing
/// `vertices`. All face indices must be in range and every vertex finite.
///
/// Degenerate faces (zero area) become singleton clusters and are listed in
/// [`NormalConePartition::degenerate_faces`]; edges shared by more than two
/// faces are non-manifold and never produce a chart adjacency.
///
/// # Errors
///
/// [`Lux3dError::InvalidInput`] when the index count is not a multiple of 3,
/// a face index is out of range, or a vertex component is not finite.
pub fn normal_cone_clusters(
    vertices: &[[f32; 3]],
    faces: &[u32],
    options: &NormalConeOptions,
) -> crate::Result<NormalConePartition> {
    if faces.len() % 3 != 0 {
        return Err(Lux3dError::InvalidInput(
            "normal_cone_clusters: index count must be a multiple of 3",
        ));
    }
    for vertex in vertices {
        if !(vertex[0].is_finite() && vertex[1].is_finite() && vertex[2].is_finite()) {
            return Err(Lux3dError::InvalidInput(
                "normal_cone_clusters: vertices must be finite",
            ));
        }
    }
    let face_count = faces.len() / 3;
    for &index in faces {
        if index as usize >= vertices.len() {
            return Err(Lux3dError::InvalidInput(
                "normal_cone_clusters: face index out of range",
            ));
        }
    }

    // Per-face geometry, and the degenerate split. `face_geom` is indexed by
    // original face id; degenerate faces keep a placeholder entry that no
    // consumer reads (`nd_faces` never points at them).
    let mut face_geom: Vec<Option<FaceGeom>> = vec![None; face_count];
    let mut degenerate_faces: Vec<u32> = Vec::new();
    for face in 0..face_count {
        let p0 = vertices[faces[3 * face] as usize];
        let p1 = vertices[faces[3 * face + 1] as usize];
        let p2 = vertices[faces[3 * face + 2] as usize];
        let e1 = sub(p1, p0);
        let e2 = sub(p2, p0);
        let normal_raw = cross(e1, e2);
        let cross_len = length(normal_raw);
        // Zero area, and the overflow/NaN spill of finite-but-extreme
        // coordinates, are both treated as degenerate so no NaN can reach the
        // cone math.
        if !cross_len.is_finite() || cross_len <= 0.0 {
            degenerate_faces.push(face as u32);
            continue;
        }
        let inv = 1.0 / cross_len;
        face_geom[face] = Some(FaceGeom {
            normal: [
                normal_raw[0] * inv,
                normal_raw[1] * inv,
                normal_raw[2] * inv,
            ],
            area: 0.5 * cross_len,
            edge_len: [length(e1), length(sub(p2, p1)), length(sub(p0, p2))],
        });
    }

    // Local (0..nb) index per non-degenerate face; u32::MAX elsewhere.
    let mut nd_index: Vec<u32> = vec![u32::MAX; face_count];
    let mut nd_faces: Vec<u32> = Vec::with_capacity(face_count - degenerate_faces.len());
    for face in 0..face_count {
        if face_geom[face].is_some() {
            nd_index[face] = nd_faces.len() as u32;
            nd_faces.push(face as u32);
        }
    }
    let nb = nd_faces.len();

    // Manifold edge list. Every face counts towards manifoldness, so an edge
    // shared with a degenerate face is still a non-manifold edge when three
    // faces meet on it; only non-degenerate pairs can produce an adjacency.
    let mut edge2f: BTreeMap<u64, EdgeSlot> = BTreeMap::new();
    for face in 0..face_count {
        for corner in 0..3 {
            let a = faces[3 * face + corner];
            let b = faces[3 * face + (corner + 1) % 3];
            match edge2f.entry(edge_key(a, b)) {
                Entry::Vacant(slot) => {
                    slot.insert(EdgeSlot::First(face as u32));
                }
                Entry::Occupied(mut slot) => match *slot.get() {
                    EdgeSlot::First(first) => {
                        slot.insert(EdgeSlot::Second(first, face as u32));
                    }
                    EdgeSlot::Second(..) | EdgeSlot::NonManifold => {
                        slot.insert(EdgeSlot::NonManifold);
                    }
                },
            }
        }
    }

    // (local a, local b, shared edge length) in sorted edge-key order.
    let mut adjacency: Vec<(u32, u32, f32)> = Vec::new();
    for (&key, slot) in &edge2f {
        let EdgeSlot::Second(a, b) = *slot else {
            continue;
        };
        let la = nd_index[a as usize];
        let lb = nd_index[b as usize];
        if la == u32::MAX || lb == u32::MAX {
            continue;
        }
        let v0 = vertices[(key >> 32) as usize];
        let v1 = vertices[(key & 0xFFFF_FFFF) as usize];
        adjacency.push((la, lb, length(sub(v0, v1))));
    }

    // Bottom-up merge. `chart` maps local face -> chart id; ids are
    // compressed to 0..C at the start of every round, exactly like the
    // reference.
    let mut chart: Vec<u32> = (0..nb as u32).collect();
    let mut remap: Vec<u32> = vec![u32::MAX; nb];
    let mut rounds: u32 = 0;
    let max_rounds = options.max_rounds;

    for _ in 0..max_rounds {
        remap.fill(u32::MAX);
        let mut chart_count: u32 = 0;
        for &chart_id in &chart {
            if remap[chart_id as usize] == u32::MAX {
                remap[chart_id as usize] = chart_count;
                chart_count += 1;
            }
        }
        for chart_id in &mut chart {
            *chart_id = remap[*chart_id as usize];
        }
        if chart_count <= 1 {
            break;
        }
        let c = chart_count as usize;
        rounds += 1;

        // Per-chart cone: normalized sum-of-normals axis, max deviation.
        let mut axis = vec![[0.0f32; 3]; c];
        let mut half = vec![0.0f32; c];
        let mut carea = vec![0.0f32; c];
        for (face, &chart_id) in chart.iter().enumerate() {
            let geom = face_geom[nd_faces[face] as usize].expect("non-degenerate face");
            let ci = chart_id as usize;
            axis[ci][0] += geom.normal[0];
            axis[ci][1] += geom.normal[1];
            axis[ci][2] += geom.normal[2];
            carea[ci] += geom.area;
        }
        for ax in &mut axis {
            let len = length(*ax);
            if len.is_finite() && len > 1e-20 {
                ax[0] /= len;
                ax[1] /= len;
                ax[2] /= len;
            } else {
                *ax = [0.0, 0.0, 1.0];
            }
        }
        for (face, &chart_id) in chart.iter().enumerate() {
            let geom = face_geom[nd_faces[face] as usize].expect("non-degenerate face");
            let ci = chart_id as usize;
            let deviation = dot(axis[ci], geom.normal).clamp(-1.0, 1.0).acos();
            half[ci] = half[ci].max(deviation);
        }

        // Chart-pair shared boundary length; same-chart manifold edges are
        // interior and get subtracted from the perimeter.
        let mut pair_len: BTreeMap<u64, f32> = BTreeMap::new();
        let mut shared_in = vec![0.0f32; c];
        for &(a, b, len) in &adjacency {
            let ca = chart[a as usize] as usize;
            let cb = chart[b as usize] as usize;
            if ca == cb {
                shared_in[ca] += 2.0 * len;
                continue;
            }
            let (lo, hi) = if ca < cb { (ca, cb) } else { (cb, ca) };
            *pair_len
                .entry(((lo as u64) << 32) | hi as u64)
                .or_insert(0.0) += len;
        }

        // perimeter = total face-edge length - interior manifold length.
        let mut perim = vec![0.0f32; c];
        for (face, &chart_id) in chart.iter().enumerate() {
            let geom = face_geom[nd_faces[face] as usize].expect("non-degenerate face");
            perim[chart_id as usize] += geom.edge_len[0] + geom.edge_len[1] + geom.edge_len[2];
        }
        for ci in 0..c {
            perim[ci] -= shared_in[ci];
        }

        // Pair costs. The BTreeMap iterates in (c0, c1) order, so `pairs` is
        // sorted and the argmin tie-break is deterministic.
        let mut pairs: Vec<(usize, usize, f32)> = Vec::with_capacity(pair_len.len());
        for (&key, &len) in &pair_len {
            let c0 = (key >> 32) as usize;
            let c1 = (key & 0xFFFF_FFFF) as usize;
            debug_assert!(c0 < c1);
            let angle = dot(axis[c0], axis[c1]).clamp(-1.0, 1.0).acos();
            let lo = (-half[c0]).min(angle - half[c1]);
            let hi = half[c0].max(angle + half[c1]);
            let merged_half_angle = (hi - lo) * 0.5;
            let merged_area = carea[c0] + carea[c1];
            let non_shared = perim[c0] + perim[c1] - 2.0 * len;
            let cost = merged_half_angle
                + options.area_weight * merged_area
                + options.perimeter_area_weight * (non_shared * non_shared / merged_area);
            pairs.push((c0, c1, sanitized_cost(cost)));
        }

        // Per-chart argmin pair (lowest index wins ties).
        let mut cmin: Vec<usize> = vec![usize::MAX; c];
        for (index, &(c0, c1, cost)) in pairs.iter().enumerate() {
            for chart_id in [c0, c1] {
                if cmin[chart_id] == usize::MAX || cost < pairs[cmin[chart_id]].2 {
                    cmin[chart_id] = index;
                }
            }
        }

        // Collapse mutual-argmin pairs under the threshold. A chart can be
        // the second endpoint of at most one such pair, so one application of
        // `cmap` is enough (no merge chains).
        let mut cmap: Vec<u32> = (0..chart_count).collect();
        let mut merged_any = false;
        for (index, &(c0, c1, cost)) in pairs.iter().enumerate() {
            if !(cost <= options.cone_half_angle_rad) {
                continue;
            }
            if cmin[c0] != index || cmin[c1] != index {
                continue;
            }
            cmap[c1] = c0 as u32;
            merged_any = true;
        }
        if !merged_any {
            break;
        }
        for chart_id in &mut chart {
            *chart_id = cmap[*chart_id as usize];
        }
    }

    // Final chart-id compression and gather. First appearance follows local
    // face order, which follows original face order, so clusters come out in
    // ascending-min-face order before the degenerate singletons are added.
    remap.fill(u32::MAX);
    let mut chart_count: u32 = 0;
    for &chart_id in &chart {
        if remap[chart_id as usize] == u32::MAX {
            remap[chart_id as usize] = chart_count;
            chart_count += 1;
        }
    }
    let mut clusters: Vec<Vec<u32>> = vec![Vec::new(); chart_count as usize];
    for (face, &chart_id) in chart.iter().enumerate() {
        clusters[remap[chart_id as usize] as usize].push(nd_faces[face]);
    }
    for &face in &degenerate_faces {
        clusters.push(vec![face]);
    }
    for cluster in &mut clusters {
        cluster.sort_unstable();
    }
    clusters.sort_by_key(|cluster| cluster[0]);

    let mut face_cluster: Vec<u32> = vec![u32::MAX; face_count];
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        for &face in cluster {
            face_cluster[face as usize] = cluster_index as u32;
        }
    }
    debug_assert!(face_cluster.iter().all(|&cluster| cluster != u32::MAX));

    Ok(NormalConePartition {
        clusters,
        face_cluster,
        rounds,
        degenerate_faces,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_partition(vertices: &[[f32; 3]], faces: &[u32]) -> NormalConePartition {
        normal_cone_clusters(vertices, faces, &NormalConeOptions::default())
            .expect("valid mesh must partition")
    }

    /// Two triangles per quad, each quad on its own four vertices.
    fn quad_faces(base: u32) -> [u32; 6] {
        [base, base + 1, base + 2, base, base + 2, base + 3]
    }

    fn assert_partition_consistent(partition: &NormalConePartition, face_count: usize) {
        assert_eq!(partition.face_cluster.len(), face_count);
        assert_eq!(
            partition.clusters.iter().map(Vec::len).sum::<usize>(),
            face_count
        );
        for (cluster_index, cluster) in partition.clusters.iter().enumerate() {
            assert!(!cluster.is_empty(), "clusters must not be empty");
            assert!(
                cluster.windows(2).all(|pair| pair[0] < pair[1]),
                "cluster faces must be strictly ascending"
            );
            assert_eq!(cluster[0], *cluster.iter().min().unwrap());
            for &face in cluster {
                assert_eq!(partition.face_cluster[face as usize], cluster_index as u32);
            }
        }
        assert!(
            partition
                .clusters
                .windows(2)
                .all(|pair| pair[0][0] < pair[1][0]),
            "clusters must be ordered by minimum face id"
        );
        // Every face covered exactly once.
        let mut seen = vec![false; face_count];
        for cluster in &partition.clusters {
            for &face in cluster {
                assert!(!seen[face as usize], "face {face} appears twice");
                seen[face as usize] = true;
            }
        }
        assert!(seen.iter().all(|&hit| hit), "some face is unclustered");
    }

    /// Test 1: a cube whose six quads carry their own (welded-within-face)
    /// vertices. The quads share no indices, so each face is an isolated
    /// component of two triangles and collapses to exactly one cluster.
    #[test]
    fn welded_cube_six_faces_yields_six_clusters() {
        let mut vertices: Vec<[f32; 3]> = Vec::new();
        let mut faces: Vec<u32> = Vec::new();
        let quads: [[[f32; 3]; 4]; 6] = [
            // +z
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            // -z
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            // +y
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            // -y
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            // +x
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
            ],
            // -x
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
        ];
        for quad in quads {
            let base = vertices.len() as u32;
            vertices.extend_from_slice(&quad);
            faces.extend_from_slice(&quad_faces(base));
        }
        assert_eq!(vertices.len(), 24);
        assert_eq!(faces.len() / 3, 12);

        let partition = default_partition(&vertices, &faces);
        assert_eq!(partition.clusters.len(), 6);
        assert_eq!(partition.rounds, 2);
        assert!(partition.degenerate_faces.is_empty());
        for (cluster_index, cluster) in partition.clusters.iter().enumerate() {
            assert_eq!(
                cluster,
                &vec![2 * cluster_index as u32, 2 * cluster_index as u32 + 1]
            );
        }
        assert_partition_consistent(&partition, 12);
    }

    /// Test 2: two 8-unit triangles sharing one edge folded by 90 degrees.
    /// The merged cone half angle is pi/4 and the area term adds 1.6, so the
    /// pair cost (about 2.39) exceeds the pi/2 threshold and the fold is kept.
    #[test]
    fn folded_pair_keeps_two_clusters() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, -4.0],
        ];
        let faces = [0, 1, 2, 0, 1, 3];
        let partition = default_partition(&vertices, &faces);
        assert_eq!(partition.clusters, vec![vec![0], vec![1]]);
        assert_eq!(partition.rounds, 1);
        assert!(
            partition.clusters.iter().all(|cluster| cluster.len() == 1),
            "90-degree fold must not merge"
        );
        assert_partition_consistent(&partition, 2);
    }

    /// Test 3: two quads with disjoint vertex sets share no manifold edge, so
    /// the only merge is the diagonal inside each quad.
    #[test]
    fn disjoint_quads_never_merge() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
            [4.0, 1.0, 0.0],
        ];
        let faces = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];
        let partition = default_partition(&vertices, &faces);
        assert_eq!(partition.clusters, vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(partition.rounds, 2);
        assert_partition_consistent(&partition, 4);
    }

    /// Welded 4x4 planar grid: shared vertices, so quads merge along shared
    /// edges but the area term stops the merge well short of one cluster.
    fn welded_grid(n: u32) -> (Vec<[f32; 3]>, Vec<u32>) {
        let side = n + 1;
        let mut vertices = Vec::with_capacity((side * side) as usize);
        for y in 0..side {
            for x in 0..side {
                vertices.push([x as f32, y as f32, 0.0]);
            }
        }
        let mut faces = Vec::with_capacity((n * n * 6) as usize);
        for y in 0..n {
            for x in 0..n {
                let v00 = y * side + x;
                let v10 = v00 + 1;
                let v01 = v00 + side;
                let v11 = v01 + 1;
                faces.extend_from_slice(&[v00, v10, v11, v00, v11, v01]);
            }
        }
        (vertices, faces)
    }

    /// Test 4: welded grid merges into a handful of charts.
    #[test]
    fn welded_grid_merges_within_four_clusters() {
        let (vertices, faces) = welded_grid(4);
        assert_eq!(faces.len() / 3, 32);
        let partition = default_partition(&vertices, &faces);
        assert!(
            partition.clusters.len() <= 4,
            "expected at most 4 clusters, got {}",
            partition.clusters.len()
        );
        assert!(
            partition.clusters.len() > 1,
            "area weight must keep the whole grid from collapsing into one chart"
        );
        assert!(partition.degenerate_faces.is_empty());
        assert_partition_consistent(&partition, 32);
    }

    /// Test 5: identical input yields an identical partition, including the
    /// round count.
    #[test]
    fn partition_is_deterministic() {
        let (vertices, faces) = welded_grid(4);
        let first = default_partition(&vertices, &faces);
        let second = default_partition(&vertices, &faces);
        assert_eq!(first, second);
    }

    /// Test 6: three triangles on one edge form a non-manifold edge, which is
    /// dropped instead of producing adjacency, so nothing merges.
    #[test]
    fn non_manifold_edge_is_ignored() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let faces = [0, 1, 2, 0, 1, 3, 0, 1, 4];
        let partition = default_partition(&vertices, &faces);
        assert_eq!(partition.clusters, vec![vec![0], vec![1], vec![2]]);
        assert_eq!(partition.rounds, 1);
        assert!(partition.degenerate_faces.is_empty());
        assert_partition_consistent(&partition, 3);
    }

    /// Test 7: a collinear triangle is a zero-area face: it becomes its own
    /// singleton cluster and is reported as degenerate, while the valid quad
    /// next to it still merges.
    #[test]
    fn zero_area_face_is_singleton() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ];
        let faces = [0, 1, 2, 0, 2, 3, 4, 5, 6];
        let partition = default_partition(&vertices, &faces);
        assert_eq!(partition.degenerate_faces, vec![2]);
        assert_eq!(partition.clusters, vec![vec![0, 1], vec![2]]);
        assert_eq!(partition.face_cluster, vec![0, 0, 1]);
        assert_eq!(partition.rounds, 1);
        assert_partition_consistent(&partition, 3);
    }

    /// Test 8: `face_cluster` and `clusters` agree on a mesh that exercises
    /// several merges plus a degenerate face.
    #[test]
    fn face_cluster_matches_clusters() {
        let (mut vertices, mut faces) = welded_grid(3);
        let base = vertices.len() as u32;
        let degenerate_face = (faces.len() / 3) as u32;
        vertices.extend_from_slice(&[[-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        faces.extend_from_slice(&[base, base + 1, base + 2]);
        let partition = default_partition(&vertices, &faces);
        assert_eq!(
            partition.degenerate_faces,
            vec![degenerate_face],
            "collinear face must be reported degenerate"
        );
        assert_partition_consistent(&partition, 19);

        let mut recovered = vec![vec![0u32; 0]; partition.clusters.len()];
        for (face, &cluster_index) in partition.face_cluster.iter().enumerate() {
            recovered[cluster_index as usize].push(face as u32);
        }
        // `face_cluster` iterates faces in ascending order already.
        assert_eq!(recovered, partition.clusters);
    }

    #[test]
    fn rejects_bad_index_count() {
        let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let error =
            normal_cone_clusters(&vertices, &[0, 1], &NormalConeOptions::default()).unwrap_err();
        assert!(matches!(error, Lux3dError::InvalidInput(_)));
    }

    #[test]
    fn rejects_out_of_range_index() {
        let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let error =
            normal_cone_clusters(&vertices, &[0, 1, 3], &NormalConeOptions::default()).unwrap_err();
        assert!(matches!(error, Lux3dError::InvalidInput(_)));
    }

    #[test]
    fn rejects_non_finite_vertex() {
        let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, f32::NAN, 0.0]];
        let error =
            normal_cone_clusters(&vertices, &[0, 1, 2], &NormalConeOptions::default()).unwrap_err();
        assert!(matches!(error, Lux3dError::InvalidInput(_)));
    }

    #[test]
    fn empty_mesh_is_empty_partition() {
        let partition = default_partition(&[], &[]);
        assert_eq!(partition, NormalConePartition::default());
    }

    #[test]
    fn zero_rounds_cap_keeps_faces_separate() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let faces = [0, 1, 2, 0, 2, 3];
        let options = NormalConeOptions {
            max_rounds: 0,
            ..NormalConeOptions::default()
        };
        let partition = normal_cone_clusters(&vertices, &faces, &options).expect("valid mesh");
        assert_eq!(partition.rounds, 0);
        assert_eq!(partition.clusters, vec![vec![0], vec![1]]);
        assert_partition_consistent(&partition, 2);
    }
}
