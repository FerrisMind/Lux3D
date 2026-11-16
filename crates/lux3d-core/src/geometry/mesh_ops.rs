//! Reusable CPU mesh post-processing ops (hole filling, decimation, cleanup).
//!
//! Pure-`std` triangle-mesh utilities shared by every model family's export
//! path. They mirror the upstream post-processing semantics without pulling in
//! any framework-specific types:
//!
//! - TRELLIS.2 runs `cumesh` `fill_holes` / `simplify` /
//!   `remove_duplicate_faces` / `repair_non_manifold_edges` /
//!   `remove_small_connected_components` / `unify_face_orientations` on the
//!   decoded dual-grid mesh before export.
//! - TripoSG runs pymeshlab merge-close-vertices + quadric decimation for
//!   `--faces`.
//!
//! Conventions:
//! - Vertices are a flat `[x0, y0, z0, x1, ...]` `f32` buffer; a "vertex
//!   count" is always `vertices.len() / 3` (a trailing partial vertex is
//!   ignored).
//! - Faces are triangles of zero-based `u32` vertex indices, 3 per face.
//! - Every function validates index bounds and never panics on malformed
//!   input: out-of-range faces are skipped (or yield an empty result), and
//!   faces with repeated indices are treated as degenerate.
//! - All operations are deterministic: results depend only on the input
//!   buffers (no HashMap iteration order leaks into any output, and the
//!   simplifier is the deterministic CuMesh QEM port in `geometry::qem`).

use std::collections::{HashMap, HashSet, VecDeque};

/// Result of an edit that may add, drop, or reorder vertices.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MeshEditResult {
    /// Vertex positions of the edited mesh, flat `[x, y, z, ...]`.
    pub vertices: Vec<f32>,
    /// Triangle indices into `vertices` (3 per face).
    pub faces: Vec<u32>,
    /// Maps *new* vertex index -> *old* (input) vertex index, so callers can
    /// transfer per-vertex attributes (e.g. colors) with
    /// `new_color[i] = old_color[vertex_remap[i]]`. `vertex_remap.len() ==
    /// vertices.len() / 3` always holds.
    pub vertex_remap: Vec<u32>,
}

/// Drop unreferenced vertices, keeping the surviving vertices in first-use
/// (face-scan) order.
///
/// Faces with any out-of-range index are dropped; all other faces are kept
/// verbatim (deduplication of degenerate faces is
/// [`remove_duplicate_faces`]'s job). `vertex_remap[new] == old`.
pub fn compact_mesh(vertices: &[f32], faces: &[u32]) -> MeshEditResult {
    let vertex_count = vertices.len() / 3;
    let mut old_to_new = vec![u32::MAX; vertex_count];
    let mut vertex_remap = Vec::new();
    let mut new_vertices = Vec::new();
    let mut new_faces = Vec::with_capacity(faces.len());

    for face in faces.chunks_exact(3) {
        let (a, b, c) = (face[0] as usize, face[1] as usize, face[2] as usize);
        if a >= vertex_count || b >= vertex_count || c >= vertex_count {
            continue;
        }
        for &old in &[a, b, c] {
            if old_to_new[old] == u32::MAX {
                old_to_new[old] = vertex_remap.len() as u32;
                vertex_remap.push(old as u32);
                new_vertices.extend_from_slice(&vertices[old * 3..old * 3 + 3]);
            }
        }
        new_faces.extend_from_slice(&[old_to_new[a], old_to_new[b], old_to_new[c]]);
    }

    MeshEditResult {
        vertices: new_vertices,
        faces: new_faces,
        vertex_remap,
    }
}

/// Drop faces with repeated indices and faces whose unordered index triple
/// already appeared earlier (first occurrence wins, winding is ignored).
///
/// `vertices_len` is the vertex count of the mesh the faces refer to;
/// out-of-range faces are dropped as well.
pub fn remove_duplicate_faces(vertices_len: usize, faces: &[u32]) -> Vec<u32> {
    let vertices_len = vertices_len as u32;
    let mut seen = HashSet::with_capacity(faces.len() / 3);
    let mut out = Vec::with_capacity(faces.len());
    for face in faces.chunks_exact(3) {
        let (a, b, c) = (face[0], face[1], face[2]);
        if a >= vertices_len || b >= vertices_len || c >= vertices_len {
            continue;
        }
        if a == b || b == c || a == c {
            continue;
        }
        let mut key = [a, b, c];
        key.sort_unstable();
        if seen.insert(key) {
            out.extend_from_slice(face);
        }
    }
    out
}

/// Fill boundary loops whose perimeter is `< max_hole_perimeter` (CuMesh
/// masks loops with `LessThanOp`).
///
/// A boundary edge is an undirected vertex pair used by exactly one face;
/// boundary edges are chained into loops by following their winding. Each
/// closed loop with at least 3 edges and an unambiguous walk (exactly one
/// unused outgoing boundary edge at every step and no repeated vertices, i.e.
/// the boundary is not pinched through a non-manifold vertex) whose total
/// edge length fits the threshold is closed: 3-edge loops with a single
/// triangle, larger loops by adding one centroid vertex per loop and fanning
/// the loop edges onto it (which also handles non-planar loops).
///
/// **Virtual vertices:** this function's return type carries faces only. Any
/// newly added centroid vertex is referred to by an index `>=` the input
/// vertex count: centroid `k` (loops are filled in boundary-discovery order)
/// sits at index `vertex_count + k`, and its position is the mean of that
/// loop's vertices. Use [`fill_holes_with_centroids`] when you need the
/// extended vertex buffer, or run the result through [`compact_mesh`] /
/// [`postprocess_mesh`], which materialize the centroids.
pub fn fill_holes(vertices: &[f32], faces: &[u32], max_hole_perimeter: f32) -> Vec<u32> {
    fill_holes_with_centroids(vertices, faces, max_hole_perimeter).1
}

/// [`fill_holes`] plus the materialized centroid vertices.
///
/// Returns `(vertices, faces)` where `vertices` is the input vertex buffer
/// with one centroid appended per filled loop (in fill order) and `faces`
/// indexes into that extended buffer.
pub fn fill_holes_with_centroids(
    vertices: &[f32],
    faces: &[u32],
    max_hole_perimeter: f32,
) -> (Vec<f32>, Vec<u32>) {
    let vertex_count = vertices.len() / 3;
    let vertex_count_u32 = vertex_count as u32;
    let mut valid_faces: Vec<[u32; 3]> = Vec::with_capacity(faces.len() / 3);
    let mut edge_count: HashMap<[u32; 2], u32> = HashMap::new();
    for face in faces.chunks_exact(3) {
        let (a, b, c) = (face[0], face[1], face[2]);
        let in_range = a < vertex_count_u32 && b < vertex_count_u32 && c < vertex_count_u32;
        if !in_range || a == b || b == c || a == c {
            continue;
        }
        valid_faces.push([a, b, c]);
        for (u, v) in [(a, b), (b, c), (c, a)] {
            let key = if u <= v { [u, v] } else { [v, u] };
            *edge_count.entry(key).or_insert(0) += 1;
        }
    }

    // Directed boundary edges (their undirected edge has exactly one incident
    // face), grouped by source vertex in face-scan order.
    let mut outgoing: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut boundary_dirs: Vec<(u32, u32)> = Vec::new();
    for face in &valid_faces {
        for (a, b) in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
            let key = if a <= b { [a, b] } else { [b, a] };
            if edge_count[&key] == 1 {
                boundary_dirs.push((a, b));
                outgoing.entry(a).or_default().push(b);
            }
        }
    }

    let mut new_vertices = vertices.to_vec();
    let mut new_faces: Vec<u32> = Vec::with_capacity(valid_faces.len() * 3);
    for face in &valid_faces {
        new_faces.extend_from_slice(face);
    }

    // Walk candidate loops from every unvisited boundary edge. A failed walk
    // (pinched boundary, winding break) does not consume its edges: a later
    // attempt starting elsewhere on the loop may still succeed. Successful
    // walks commit their edges so each clean loop is filled exactly once.
    let mut committed: HashSet<(u32, u32)> = HashSet::new();
    for &(start, first_next) in &boundary_dirs {
        if committed.contains(&(start, first_next)) {
            continue;
        }
        // Walk the loop from the directed edge `start -> first_next`.
        let mut walk_edges: Vec<(u32, u32)> = vec![(start, first_next)];
        let mut walk_edge_set: HashSet<(u32, u32)> = [(start, first_next)].into_iter().collect();
        let mut loop_vertices = vec![start, first_next];
        let mut loop_vertex_set: HashSet<u32> = [start, first_next].into_iter().collect();
        let mut valid_loop = true;
        let mut cur = first_next;
        while cur != start {
            // Exactly one unused outgoing boundary edge must exist at `cur`;
            // more means a pinched (non-manifold) boundary, none means a
            // winding break. Either way the loop is not safely fillable.
            let candidates: Vec<u32> = outgoing
                .get(&cur)
                .map(|nexts| {
                    nexts
                        .iter()
                        .copied()
                        .filter(|&next| !walk_edge_set.contains(&(cur, next)))
                        .collect()
                })
                .unwrap_or_default();
            if candidates.len() != 1 {
                valid_loop = false;
                break;
            }
            let next = candidates[0];
            walk_edges.push((cur, next));
            walk_edge_set.insert((cur, next));
            if next == start {
                break;
            }
            if !loop_vertex_set.insert(next) {
                // The loop revisits a vertex (figure-eight pinch): skip.
                valid_loop = false;
                break;
            }
            loop_vertices.push(next);
            cur = next;
        }
        if !valid_loop || loop_vertices.len() < 3 {
            continue;
        }
        for edge in walk_edges {
            committed.insert(edge);
        }

        // Perimeter (including the closing edge) in loop order.
        let n = loop_vertices.len();
        let mut perimeter = 0.0f32;
        for i in 0..n {
            let a = vertex(vertices, loop_vertices[i]);
            let b = vertex(vertices, loop_vertices[(i + 1) % n]);
            perimeter += dist(a, b);
        }
        if perimeter >= max_hole_perimeter || perimeter.is_nan() {
            continue;
        }

        // Triangular holes close with a single triangle (reversed winding,
        // no new vertex); larger loops get one centroid vertex per loop and
        // a fan of the loop edges onto it. `new_vertices` already holds all
        // input vertices, so the next free index is its length / 3.
        if n == 3 {
            new_faces.extend_from_slice(&[loop_vertices[0], loop_vertices[2], loop_vertices[1]]);
            continue;
        }
        let centroid_index = (new_vertices.len() / 3) as u32;
        let mut centroid = [0.0f32; 3];
        for &v in &loop_vertices {
            let p = vertex(vertices, v);
            centroid[0] += p[0];
            centroid[1] += p[1];
            centroid[2] += p[2];
        }
        let centroid = [
            centroid[0] / n as f32,
            centroid[1] / n as f32,
            centroid[2] / n as f32,
        ];
        new_vertices.extend_from_slice(&centroid);

        // Boundary directed edges run `loop_vertices[i] -> loop_vertices[i+1]`
        // with the existing surface on their right; the fill triangles must
        // traverse each shared edge the opposite way, hence the reversed fan
        // `(next, cur, centroid)`.
        for i in 0..n {
            let cur = loop_vertices[i];
            let next = loop_vertices[(i + 1) % n];
            new_faces.extend_from_slice(&[next, cur, centroid_index]);
        }
    }

    (new_vertices, new_faces)
}

/// Drop connected components whose total triangle area is `<= min_area`.
///
/// Faces are connected when they share a vertex (vertex adjacency, not edge
/// adjacency). Areas are computed per valid triangle; out-of-range and
/// degenerate faces are dropped. A component survives only if its total area
/// is strictly greater than `min_area`.
pub fn remove_small_connected_components(
    vertices: &[f32],
    faces: &[u32],
    min_area: f32,
) -> Vec<u32> {
    let vertex_count = vertices.len() / 3;

    // CuMesh defines connected components through MANIFOLD face adjacency:
    // two faces are neighbours when they share an edge used by exactly two
    // faces (non-manifold sheets do not join components). Faces are unioned,
    // components are summed by face area, and components with
    // `area >= min_area` are kept.
    let mut kept: Vec<[u32; 3]> = Vec::with_capacity(faces.len() / 3);
    for face in faces.chunks_exact(3) {
        let (a, b, c) = (face[0], face[1], face[2]);
        let in_range =
            a < vertex_count as u32 && b < vertex_count as u32 && c < vertex_count as u32;
        if !in_range || a == b || b == c || a == c {
            continue;
        }
        kept.push([a, b, c]);
    }
    if kept.is_empty() {
        return Vec::new();
    }

    let mut edge_faces: HashMap<[u32; 2], Vec<usize>> = HashMap::new();
    for (fi, face) in kept.iter().enumerate() {
        for local in 0..3 {
            let u = face[local];
            let v = face[(local + 1) % 3];
            let key = if u <= v { [u, v] } else { [v, u] };
            edge_faces.entry(key).or_default().push(fi);
        }
    }

    let mut parent: Vec<usize> = (0..kept.len()).collect();
    fn find(parent: &mut [usize], mut node: usize) -> usize {
        while parent[node] != node {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        node
    }
    for incident in edge_faces.values() {
        if incident.len() != 2 {
            continue;
        }
        let root_a = find(&mut parent, incident[0]);
        let root_b = find(&mut parent, incident[1]);
        if root_a != root_b {
            if root_a < root_b {
                parent[root_b] = root_a;
            } else {
                parent[root_a] = root_b;
            }
        }
    }

    let mut component_area: HashMap<usize, f32> = HashMap::new();
    for (fi, face) in kept.iter().enumerate() {
        let root = find(&mut parent, fi);
        let area = triangle_area(
            vertex(vertices, face[0]),
            vertex(vertices, face[1]),
            vertex(vertices, face[2]),
        );
        *component_area.entry(root).or_insert(0.0) += area;
    }

    let mut out = Vec::with_capacity(faces.len());
    for (fi, face) in kept.iter().enumerate() {
        let root = find(&mut parent, fi);
        if component_area[&root] >= min_area {
            out.extend_from_slice(face);
        }
    }
    out
}

/// Split vertices to repair non-manifold edges (CuMesh
/// `repair_non_manifold_edges`).
///
/// CuMesh does **not** drop faces here. It unions the face-corner occurrences
/// of the same input vertex across every MANIFOLD edge (an undirected edge
/// used by exactly two faces), then renumbers those corner groups: a vertex
/// touched by several non-manifold sheets becomes one output vertex per
/// sheet. All faces are preserved; only vertex sharing changes.
///
/// Concretely, for each edge used by exactly two faces `f0`, `f1`, the two
/// shared endpoint vertices are found in both faces and the corresponding
/// corner slots (`3 * face + local_index`) are unioned. Corner groups are
/// emitted in ascending root order (the CuMesh `compress_ids` order), which is
/// first-use order because unions attach the larger root under the smaller
/// one.
///
/// Out-of-range and degenerate faces are dropped before the repair (they
/// cannot participate in manifold adjacency); every other face is kept.
pub fn repair_non_manifold_edges(vertices: &[f32], faces: &[u32]) -> MeshEditResult {
    let vertex_count = vertices.len() / 3;
    let mut kept: Vec<[u32; 3]> = Vec::with_capacity(faces.len() / 3);
    for face in faces.chunks_exact(3) {
        let (a, b, c) = (face[0], face[1], face[2]);
        let in_range =
            a < vertex_count as u32 && b < vertex_count as u32 && c < vertex_count as u32;
        if !in_range || a == b || b == c || a == c {
            continue;
        }
        kept.push([a, b, c]);
    }
    if kept.is_empty() {
        return compact_mesh(vertices, &[]);
    }

    // edge (lo, hi) -> [(face index, local corner of lo, local corner of hi)]
    let mut edge_faces: HashMap<[u32; 2], Vec<(usize, usize, usize)>> = HashMap::new();
    for (fi, face) in kept.iter().enumerate() {
        for local in 0..3 {
            let u = face[local];
            let v = face[(local + 1) % 3];
            let (lo, hi, lo_local, hi_local) = if u <= v {
                (u, v, local, (local + 1) % 3)
            } else {
                (v, u, (local + 1) % 3, local)
            };
            edge_faces
                .entry([lo, hi])
                .or_default()
                .push((fi, lo_local, hi_local));
        }
    }

    // Union-find over the 3 * F face corners.
    let corner_count = kept.len() * 3;
    let mut parent: Vec<usize> = (0..corner_count).collect();
    fn find(parent: &mut [usize], mut node: usize) -> usize {
        while parent[node] != node {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        node
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let root_a = find(parent, a);
        let root_b = find(parent, b);
        if root_a == root_b {
            return;
        }
        // Attach the larger root under the smaller one so the root of a
        // corner group is its lowest corner index (CuMesh id order).
        if root_a < root_b {
            parent[root_b] = root_a;
        } else {
            parent[root_a] = root_b;
        }
    }
    for incident in edge_faces.values() {
        if incident.len() != 2 {
            continue;
        }
        let (face_a, lo_a, hi_a) = incident[0];
        let (face_b, lo_b, hi_b) = incident[1];
        union(&mut parent, 3 * face_a + lo_a, 3 * face_b + lo_b);
        union(&mut parent, 3 * face_a + hi_a, 3 * face_b + hi_b);
    }

    // Compress roots in ascending order (root == lowest corner index).
    let mut root_to_new: HashMap<usize, u32> = HashMap::with_capacity(corner_count);
    let mut new_vertices: Vec<f32> = Vec::new();
    let mut vertex_remap: Vec<u32> = Vec::new();
    let mut new_faces: Vec<u32> = Vec::with_capacity(corner_count);
    for (fi, face) in kept.iter().enumerate() {
        for local in 0..3 {
            let root = find(&mut parent, 3 * fi + local);
            let new_index = *root_to_new.entry(root).or_insert_with(|| {
                let original = face[local] as usize;
                vertex_remap.push(original as u32);
                new_vertices.extend_from_slice(&vertices[original * 3..original * 3 + 3]);
                vertex_remap.len() as u32 - 1
            });
            new_faces.push(new_index);
        }
    }

    MeshEditResult {
        vertices: new_vertices,
        faces: new_faces,
        vertex_remap,
    }
}

/// Make face winding consistent across manifold adjacency and orient every
/// connected component with positive signed volume.
///
/// Faces are connected when they share an edge used by exactly two faces
/// (manifold adjacency); a neighbour whose shared-edge direction agrees with
/// its own is flipped so the two faces traverse the edge in opposite
/// directions. Non-manifold edges do not propagate orientation. After the
/// consistency pass, each connected component's signed volume (sum of
/// `dot(v0, cross(v1, v2)) / 6`) is made positive; components with zero
/// volume (open surfaces) keep their propagated orientation.
///
/// Out-of-range and degenerate faces are dropped.
pub fn unify_face_orientations(vertices: &[f32], faces: &[u32]) -> Vec<u32> {
    let vertex_count_u32 = (vertices.len() / 3) as u32;
    let mut valid: Vec<[u32; 3]> = Vec::with_capacity(faces.len() / 3);
    for face in faces.chunks_exact(3) {
        let (a, b, c) = (face[0], face[1], face[2]);
        if a >= vertex_count_u32 || b >= vertex_count_u32 || c >= vertex_count_u32 {
            continue;
        }
        if a == b || b == c || a == c {
            continue;
        }
        valid.push([a, b, c]);
    }

    // Manifold adjacency: edge -> the two faces using it, with the direction
    // each face traverses the (lo, hi) edge. Edge keys are visited in sorted
    // order so adjacency lists (and therefore BFS order) are deterministic.
    let mut edge_faces: HashMap<[u32; 2], Vec<(usize, bool)>> = HashMap::new();
    for (fi, face) in valid.iter().enumerate() {
        for (u, v) in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
            let (lo, hi, winds_lo_to_hi) = if u <= v { (u, v, true) } else { (v, u, false) };
            edge_faces
                .entry([lo, hi])
                .or_default()
                .push((fi, winds_lo_to_hi));
        }
    }
    let mut edge_keys: Vec<[u32; 2]> = edge_faces.keys().copied().collect();
    edge_keys.sort_unstable();

    let mut adjacency: Vec<Vec<(usize, bool)>> = vec![Vec::new(); valid.len()];
    for key in &edge_keys {
        let incident = &edge_faces[key];
        if incident.len() != 2 {
            continue;
        }
        let (f, df) = incident[0];
        let (g, dg) = incident[1];
        adjacency[f].push((g, df == dg));
        adjacency[g].push((f, df == dg));
    }

    // BFS consistency: a neighbour reached across an edge whose two faces
    // traverse it in the same direction must be flipped. Constraint for the
    // edge (f, g): `flipped[f] ^ flipped[g] == same_dir`.
    let mut flipped = vec![false; valid.len()];
    let mut visited = vec![false; valid.len()];
    let mut component_of = vec![usize::MAX; valid.len()];
    let mut component_count = 0usize;
    for root in 0..valid.len() {
        if visited[root] {
            continue;
        }
        let component = component_count;
        component_count += 1;
        visited[root] = true;
        component_of[root] = component;
        let mut queue = VecDeque::new();
        queue.push_back(root);
        while let Some(f) = queue.pop_front() {
            for &(g, same_dir) in &adjacency[f] {
                if visited[g] {
                    // Conflicting constraints (non-orientable patches) are
                    // tolerated: the first assignment wins.
                    continue;
                }
                if flipped[f] ^ flipped[g] ^ same_dir {
                    flipped[g] = !flipped[g];
                }
                visited[g] = true;
                component_of[g] = component;
                queue.push_back(g);
            }
        }
    }

    // Per-component signed volume with the (possibly flipped) windings.
    let mut component_volume = vec![0.0f32; component_count];
    for (fi, face) in valid.iter().enumerate() {
        let (a, b, c) = if flipped[fi] {
            (face[0], face[2], face[1])
        } else {
            (face[0], face[1], face[2])
        };
        component_volume[component_of[fi]] += signed_volume_contribution(
            vertex(vertices, a),
            vertex(vertices, b),
            vertex(vertices, c),
        );
    }

    let mut out = Vec::with_capacity(valid.len() * 3);
    for (fi, face) in valid.iter().enumerate() {
        let component_negative = component_volume[component_of[fi]] < 0.0;
        if flipped[fi] ^ component_negative {
            out.extend_from_slice(&[face[0], face[2], face[1]]);
        } else {
            out.extend_from_slice(face);
        }
    }
    out
}

/// Merge vertices closer than `epsilon` (pymeshlab
/// `meshing_merge_close_vertices` equivalent).
///
/// Upstream `inference_triposg.py` runs `ms.meshing_merge_close_vertices()`
/// (no arguments) before `meshing_decimation_quadric_edge_collapse` for the
/// `--faces` target. MeshLab's filter collapses every cluster of vertices
/// that fit inside the threshold distance onto a single representative
/// vertex; pymeshlab documents the default threshold as `1/10000 of bounding
/// box diagonal` (the rendered `PercentageValue = 1%` row in the filter list
/// conflicts with its own description text; we follow the description, see
/// `crates/lux3d-core/src/runtime/triposg.rs` for the pipeline constant).
///
/// Semantics (documented, deterministic):
/// - Vertices are visited in input-index order. A vertex joins the first
///   already-kept representative within `epsilon` Euclidean distance
///   (candidates found through a spatial hash grid with cell size
///   `epsilon`; a point within `epsilon` always lies in the 3x3x3 cell
///   neighbourhood, so the search is exhaustive). Otherwise it becomes a new
///   representative. The representative position is the FIRST vertex of the
///   cluster (lowest input index) — positions are not averaged.
/// - `vertex_remap[new] == old` points at the representative's input index.
/// - Faces are rewritten through the merge remap; faces with out-of-range
///   indices and faces that collapse to a repeated vertex (pre-existing or
///   produced by the merge) are dropped, matching what the following
///   decimation pass would discard anyway.
/// - `epsilon` is an absolute distance in world units. `epsilon <= 0`, NaN,
///   or an empty mesh returns the compacted input unchanged (pymeshlab with
///   a zero threshold merges nothing).
/// - Non-finite vertices are treated as unmergeable (they stay their own
///   representatives, and only their out-of-range faces are dropped).
pub fn merge_close_vertices(vertices: &[f32], faces: &[u32], epsilon: f32) -> MeshEditResult {
    let vertex_count = vertices.len() / 3;
    if vertex_count == 0 {
        return MeshEditResult::default();
    }
    if !(epsilon > 0.0) || !epsilon.is_finite() {
        return compact_mesh(vertices, faces);
    }

    let epsilon_sq = if epsilon.is_infinite() {
        f32::INFINITY
    } else {
        epsilon * epsilon
    };
    let cell_size = epsilon;

    // Spatial hash: cell key -> kept representative indices (insertion order).
    let mut cells: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
    // Kept representative positions, in first-kept (input-index) order.
    let mut kept: Vec<[f32; 3]> = Vec::with_capacity(vertex_count);
    // Input index of the vertex that created each representative.
    let mut kept_origin: Vec<u32> = Vec::with_capacity(vertex_count);
    let mut remap = vec![u32::MAX; vertex_count];
    for index in 0..vertex_count {
        let point = vertex(vertices, index as u32);
        let cell = hash_cell(point, cell_size);
        // First kept representative within epsilon wins (index-order scan).
        let mut target: Option<u32> = None;
        if point.iter().all(|value| value.is_finite()) {
            'search: for dx in -1_i64..=1 {
                for dy in -1_i64..=1 {
                    for dz in -1_i64..=1 {
                        let key = [cell[0] + dx, cell[1] + dy, cell[2] + dz];
                        let Some(reps) = cells.get(&key) else {
                            continue;
                        };
                        for &rep in reps {
                            let rep_point = kept[rep as usize];
                            if dist_sq(point, rep_point) <= epsilon_sq {
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
                let target = kept.len() as u32;
                kept.push(point);
                kept_origin.push(index as u32);
                cells.entry(cell).or_default().push(target);
                target
            }
        };
        remap[index] = target;
    }

    let mut new_vertices = Vec::with_capacity(kept.len() * 3);
    for point in &kept {
        new_vertices.extend_from_slice(point);
    }
    let mut new_faces = Vec::with_capacity(faces.len());
    for face in faces.chunks_exact(3) {
        let (a, b, c) = (face[0] as usize, face[1] as usize, face[2] as usize);
        if a >= vertex_count || b >= vertex_count || c >= vertex_count {
            continue;
        }
        let mapped = [remap[a], remap[b], remap[c]];
        if mapped[0] == mapped[1] || mapped[1] == mapped[2] || mapped[0] == mapped[2] {
            continue;
        }
        new_faces.extend_from_slice(&mapped);
    }

    MeshEditResult {
        vertices: new_vertices,
        faces: new_faces,
        vertex_remap: kept_origin,
    }
}

/// Spatial-hash cell key for `point` with cell size `cell_size` (floor of the
/// scaled coordinates; float -> int casts saturate, so coordinates beyond the
/// i64 range collapse into the boundary cells rather than panicking).
fn hash_cell(point: [f32; 3], cell_size: f32) -> [i64; 3] {
    [
        (point[0] / cell_size).floor() as i64,
        (point[1] / cell_size).floor() as i64,
        (point[2] / cell_size).floor() as i64,
    ]
}

fn dist_sq(a: [f32; 3], b: [f32; 3]) -> f32 {
    let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    dx * dx + dy * dy + dz * dz
}

/// Quadric-error mesh decimation.
///
/// Reduces the mesh toward `target_faces` triangles using meshopt's
/// quadric-error simplifier. The O-Voxel dual-grid mesh fed to this stage is
/// heavily non-manifold, so a single meshopt pass can stall above the target;
/// the simplifier is therefore re-run (up to [`MAX_SIMPLIFY_PASSES`] times)
/// until it reaches the target or stops making meaningful progress. The
/// upstream CuMesh QEM collapser is GPU-oriented and needs hundreds of
/// full passes on this topology on CPU, so meshopt is used for the bulk
/// reduction instead.
///
/// The result references a compacted vertex buffer; `vertex_remap` maps new
/// vertex indices back to the input vertex buffer (see [`MeshEditResult`]).
///
/// - Faces with out-of-range or repeated indices are dropped before
///   simplification.
/// - `target_faces == 0` (or a mesh without valid faces) yields an empty
///   mesh; a target above the current face count leaves the mesh unchanged
///   (compacted).
/// - The simplifier never grows the mesh and is deterministic.
pub fn simplify_mesh(vertices: &[f32], faces: &[u32], target_faces: usize) -> MeshEditResult {
    const MAX_SIMPLIFY_PASSES: usize = 8;
    const MIN_RELATIVE_PROGRESS: f32 = 0.005;

    let vertex_count = vertices.len() / 3;
    let mut indices: Vec<u32> = Vec::with_capacity(faces.len());
    for face in faces.chunks_exact(3) {
        if face.iter().all(|&index| (index as usize) < vertex_count)
            && face[0] != face[1]
            && face[1] != face[2]
            && face[0] != face[2]
        {
            indices.extend_from_slice(face);
        }
    }
    if target_faces == 0 || indices.is_empty() {
        return MeshEditResult::default();
    }
    if indices.len() / 3 <= target_faces {
        return compact_mesh(vertices, &indices);
    }
    let positions: Vec<[f32; 3]> = vertices
        .chunks_exact(3)
        .map(|vertex| [vertex[0], vertex[1], vertex[2]])
        .collect();
    for _ in 0..MAX_SIMPLIFY_PASSES {
        let before = indices.len() / 3;
        let target_indices = target_faces.saturating_mul(3).min(indices.len());
        let simplified = meshopt::simplify::simplify_decoder(
            &indices,
            &positions,
            target_indices,
            1.0,
            meshopt::simplify::SimplifyOptions::Permissive,
            None,
        );
        let len = simplified.len() - simplified.len() % 3;
        indices = simplified[..len].to_vec();
        let after = indices.len() / 3;
        if after <= target_faces || after >= before {
            break;
        }
        if (before - after) as f32 / (before as f32) < MIN_RELATIVE_PROGRESS {
            break;
        }
    }
    compact_mesh(vertices, &indices)
}

/// Full upstream-style post-processing pipeline.
///
/// Mirrors the TRELLIS.2 export order (cumesh calls). The first simplify pass
/// is the reference aggressive pass at `decimation_target * 3` triangles; the
/// second targets `decimation_target`:
///
/// 1. `fill_holes(fill_hole_perimeter)`
/// 2. `simplify(decimation_target * 3)` — skipped when `decimation_target == 0`
/// 3. `remove_duplicate_faces`
/// 4. `repair_non_manifold_edges`
/// 5. `remove_small_connected_components(min_component_area)`
/// 6. `fill_holes`
/// 7. `simplify(decimation_target)` — skipped when `decimation_target == 0`
/// 8. `remove_duplicate_faces`
/// 9. `repair_non_manifold_edges`
/// 10. `remove_small_connected_components(min_component_area)`
/// 11. `fill_holes`
/// 12. `unify_face_orientations`
/// 13. `compact_mesh`
///
/// Free of any TRELLIS.2/TripoSG-specific types; usable by any family.
pub fn postprocess_mesh(
    vertices: &[f32],
    faces: &[u32],
    fill_hole_perimeter: f32,
    decimation_target: usize,
    min_component_area: f32,
) -> MeshEditResult {
    let (vertices, faces) = fill_holes_with_centroids(vertices, faces, fill_hole_perimeter);
    let result = if decimation_target > 0 {
        // Upstream step 1: aggressive simplification to 3x the final target.
        simplify_mesh(&vertices, &faces, decimation_target.saturating_mul(3))
    } else {
        compact_mesh(&vertices, &faces)
    };
    let (vertices, faces) = (result.vertices, result.faces);

    let faces = remove_duplicate_faces(vertices.len() / 3, &faces);
    let repaired = repair_non_manifold_edges(&vertices, &faces);
    let (vertices, faces) = (repaired.vertices, repaired.faces);
    let faces = remove_small_connected_components(&vertices, &faces, min_component_area);

    let (vertices, faces) = fill_holes_with_centroids(&vertices, &faces, fill_hole_perimeter);
    let result = if decimation_target > 0 {
        simplify_mesh(&vertices, &faces, decimation_target)
    } else {
        compact_mesh(&vertices, &faces)
    };
    let (vertices, faces) = (result.vertices, result.faces);

    let faces = remove_duplicate_faces(vertices.len() / 3, &faces);
    let repaired = repair_non_manifold_edges(&vertices, &faces);
    let (vertices, faces) = (repaired.vertices, repaired.faces);
    let faces = remove_small_connected_components(&vertices, &faces, min_component_area);

    let (vertices, faces) = fill_holes_with_centroids(&vertices, &faces, fill_hole_perimeter);
    let faces = unify_face_orientations(&vertices, &faces);
    compact_mesh(&vertices, &faces)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Position of vertex `i` (callers guarantee `i < vertices.len() / 3`).
fn vertex(vertices: &[f32], i: u32) -> [f32; 3] {
    let base = i as usize * 3;
    [vertices[base], vertices[base + 1], vertices[base + 2]]
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Triangle area `|cross(b - a, c - a)| / 2`.
fn triangle_area(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let (ab, ac) = (sub(b, a), sub(c, a));
    let n = cross(ab, ac);
    0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
}

/// Signed tetrahedron contribution `dot(a, cross(b, c)) / 6`.
fn signed_volume_contribution(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let bc = cross(b, c);
    (a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2]) / 6.0
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Unit cube: 8 vertices, 12 triangles, outward (positive-volume) winding.
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

    fn signed_volume(vertices: &[f32], faces: &[u32]) -> f32 {
        faces
            .chunks_exact(3)
            .map(|f| {
                signed_volume_contribution(
                    vertex(vertices, f[0]),
                    vertex(vertices, f[1]),
                    vertex(vertices, f[2]),
                )
            })
            .sum()
    }

    /// Map undirected edge -> number of incident faces.
    fn edge_incidence(faces: &[u32]) -> HashMap<[u32; 2], usize> {
        let mut counts = HashMap::new();
        for face in faces.chunks_exact(3) {
            for (u, v) in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
                let key = if u <= v { [u, v] } else { [v, u] };
                *counts.entry(key).or_insert(0usize) += 1;
            }
        }
        counts
    }

    /// Every undirected edge is traversed once in each direction.
    fn is_consistently_oriented(faces: &[u32]) -> bool {
        let mut dirs: HashMap<[u32; 2], i32> = HashMap::new();
        for face in faces.chunks_exact(3) {
            for (u, v) in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
                let key = if u <= v { [u, v] } else { [v, u] };
                let sign = if u <= v { 1 } else { -1 };
                *dirs.entry(key).or_insert(0) += sign;
            }
        }
        dirs.values().all(|&balance| balance == 0)
    }

    /// Rippled `n x n` grid plane (open mesh), 2*n*n triangles.
    fn grid(n: usize) -> (Vec<f32>, Vec<u32>) {
        let mut vertices = Vec::with_capacity((n + 1) * (n + 1) * 3);
        for j in 0..=n {
            for i in 0..=n {
                let (fi, fj) = (i as f32, j as f32);
                vertices.push(fi / n as f32);
                vertices.push(0.05 * (fi * 0.9).sin() * (fj * 1.1).cos());
                vertices.push(fj / n as f32);
            }
        }
        let mut faces = Vec::with_capacity(2 * n * n * 3);
        for j in 0..n {
            for i in 0..n {
                let v00 = (j * (n + 1) + i) as u32;
                let v10 = v00 + 1;
                let v01 = v00 + n as u32 + 1;
                let v11 = v01 + 1;
                faces.extend_from_slice(&[v00, v01, v11]);
                faces.extend_from_slice(&[v00, v11, v10]);
            }
        }
        (vertices, faces)
    }

    #[test]
    fn compact_mesh_keeps_first_use_order_and_remap() {
        let vertices: Vec<f32> = (0..5)
            .flat_map(|v| [v as f32, v as f32 * 10.0, v as f32 * 100.0])
            .collect();
        let faces = [3u32, 1, 2, 1, 4, 2];
        let result = compact_mesh(&vertices, &faces);
        // First-use order: 3, 1, 2, 4 (vertex 0 unreferenced -> dropped).
        assert_eq!(result.vertex_remap, vec![3, 1, 2, 4]);
        assert_eq!(result.vertices.len(), 12);
        for (new, &old) in result.vertex_remap.iter().enumerate() {
            assert_eq!(
                &result.vertices[new * 3..new * 3 + 3],
                &vertices[old as usize * 3..old as usize * 3 + 3]
            );
        }
        assert_eq!(result.faces, vec![0, 1, 2, 1, 3, 2]);
    }

    #[test]
    fn compact_mesh_drops_out_of_range_faces_and_handles_empty() {
        let (vertices, faces) = cube();
        let mut broken = faces.to_vec();
        broken.extend_from_slice(&[0, 1, 999]);
        let result = compact_mesh(&vertices, &broken);
        assert_eq!(result.faces.len(), 36);
        // All 8 vertices are referenced; compact relabels them into
        // first-use order, so positions must be compared through the remap.
        let mut sorted_remap = result.vertex_remap.clone();
        sorted_remap.sort_unstable();
        assert_eq!(sorted_remap, (0..8).collect::<Vec<u32>>());
        for (new, &old) in result.vertex_remap.iter().enumerate() {
            assert_eq!(
                &result.vertices[new * 3..new * 3 + 3],
                &vertices[old as usize * 3..old as usize * 3 + 3]
            );
        }
        assert!((signed_volume(&result.vertices, &result.faces) - 1.0).abs() < 1e-6);

        let empty = compact_mesh(&[], &[]);
        assert!(
            empty.vertices.is_empty() && empty.faces.is_empty() && empty.vertex_remap.is_empty()
        );
    }

    #[test]
    fn remove_duplicate_faces_drops_duplicates_degenerate_and_out_of_range() {
        let (vertices, faces) = cube();
        let mut faces = faces.to_vec();
        faces.extend_from_slice(&[0, 2, 1]); // exact duplicate of face 0
        faces.extend_from_slice(&[1, 2, 0]); // same unordered triple, rotated
        faces.extend_from_slice(&[5, 5, 6]); // repeated index
        faces.extend_from_slice(&[0, 1, 77]); // out of range
        let out = remove_duplicate_faces(vertices.len() / 3, &faces);
        assert_eq!(out, faces[..36].to_vec());
    }

    #[test]
    fn fill_holes_restores_cube_with_one_face_removed() {
        let (vertices, faces) = cube();
        // Drop one triangle of the top face (flat indices 6..9) -> 11 faces;
        // the square's diagonal becomes boundary and the hole is a closed
        // triangular loop with perimeter 2 + sqrt(2).
        let holey: Vec<u32> = faces[..6]
            .iter()
            .chain(faces[9..].iter())
            .copied()
            .collect();
        assert_eq!(holey.len(), 33);

        let filled = fill_holes(&vertices, &holey, 4.0);
        assert_eq!(filled.len() / 3, 12);
        assert!(is_consistently_oriented(&filled));
        let incidence = edge_incidence(&filled);
        assert!(incidence.values().all(|&count| count == 2));

        // Below the loop perimeter nothing is filled.
        let not_filled = fill_holes(&vertices, &holey, 3.4);
        assert_eq!(not_filled, holey);
    }

    #[test]
    fn fill_holes_closes_quad_hole_with_centroid_fan() {
        let (vertices, faces) = cube();
        // Drop the whole top quad (flat indices 6..12) -> 10 faces, one
        // square hole with perimeter exactly 4.0: the CuMesh threshold is
        // strict (`perimeter < max`), so 4.001 fills it and 4.0 does not.
        let holey: Vec<u32> = faces[..6]
            .iter()
            .chain(faces[12..].iter())
            .copied()
            .collect();
        assert_eq!(holey.len(), 30);

        let filled = fill_holes(&vertices, &holey, 4.001);
        assert_eq!(filled.len() / 3, 14);
        assert!(is_consistently_oriented(&filled));
        assert!(edge_incidence(&filled).values().all(|&count| count == 2));

        let not_filled = fill_holes(&vertices, &holey, 3.999);
        assert_eq!(not_filled, holey);

        // Centroid variant materializes exactly one new vertex at the loop
        // centroid (0.5, 0.5, 1).
        let (new_vertices, new_faces) = fill_holes_with_centroids(&vertices, &holey, 100.0);
        assert_eq!(new_vertices.len() / 3, 9);
        assert_eq!(&new_vertices[24..27], &[0.5, 0.5, 1.0]);
        assert_eq!(new_faces.len() / 3, 14);
        assert!((signed_volume(&new_vertices, &new_faces) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn fill_holes_skips_open_chains_and_closed_meshes() {
        let (vertices, faces) = cube();

        // Closed mesh: no boundary, nothing to do.
        let (v2, f2) = fill_holes_with_centroids(&vertices, &faces, 100.0);
        assert_eq!(f2, faces);
        assert_eq!(v2, vertices);

        // Two front-face triangles wound inconsistently: edge (0, 5) is used
        // twice in the same direction, so the boundary chain 0->1->5 and
        // 0->4->5 never closes. The walk must bail out instead of filling.
        let open_chain: Vec<u32> = [0, 1, 5, 0, 4, 5].to_vec();
        let (v3, f3) = fill_holes_with_centroids(&vertices, &open_chain, 100.0);
        assert_eq!(f3, open_chain);
        assert_eq!(v3, vertices);
    }

    #[test]
    fn fill_holes_ignores_malformed_faces() {
        let (vertices, faces) = cube();
        let mut broken = faces.clone();
        broken.extend_from_slice(&[0, 1, 999, 7, 7, 7, 4, 5, 6]);
        // The in-range top-face pair is duplicated, so dedup-wise the loop
        // walk simply sees a closed cube plus junk; nothing may panic and the
        // valid faces survive.
        let filled = fill_holes(&vertices, &broken, 100.0);
        assert!(filled.chunks_exact(3).all(|f| f.iter().all(|&i| i < 9)));
        assert!(filled.starts_with(&faces));
    }

    #[test]
    fn remove_small_connected_components_drops_tiny_cube() {
        let (vertices, faces) = cube();
        // Tiny 0.05 cube far away, reusing fresh vertex slots 8..16.
        let mut tiny_vertices = vertices.clone();
        for v in 0..8 {
            let base = [10.0, 10.0, 10.0];
            for axis in 0..3 {
                tiny_vertices[v * 3 + axis] = base[axis] + vertices[v * 3 + axis] * 0.05;
            }
        }
        let tiny_faces: Vec<u32> = faces.iter().map(|&i| i + 8).collect();
        let all_faces: Vec<u32> = faces.iter().chain(tiny_faces.iter()).copied().collect();
        let all_vertices: Vec<f32> = vertices
            .iter()
            .chain(tiny_vertices.iter())
            .copied()
            .collect();

        // Tiny cube total area = 6 * 0.05^2 = 0.015.
        let dropped = remove_small_connected_components(&all_vertices, &all_faces, 1.0);
        assert_eq!(dropped, faces);
        let kept = remove_small_connected_components(&all_vertices, &all_faces, 0.01);
        assert_eq!(kept.len() / 3, 24);
        let dropped_again = remove_small_connected_components(&all_vertices, &all_faces, 0.02);
        assert_eq!(dropped_again, faces);
    }

    #[test]
    fn remove_small_components_drops_malformed_faces() {
        let (vertices, faces) = cube();
        let mut broken = faces.clone();
        broken.extend_from_slice(&[0, 1, 999, 2, 2, 3]);
        let out = remove_small_connected_components(&vertices, &broken, 0.0);
        assert_eq!(out, faces);
    }

    #[test]
    fn repair_non_manifold_edges_splits_sheets_and_keeps_faces() {
        // Three faces share edge (0, 1): the edge is non-manifold, so no
        // manifold union happens and every face corner becomes its own
        // output vertex. All three faces must survive.
        let vertices: Vec<f32> = (0..5).flat_map(|i| [i as f32, 0.0, 0.0]).collect();
        let faces: Vec<u32> = [0, 1, 2, 0, 1, 3, 0, 1, 4].to_vec();
        let repaired = repair_non_manifold_edges(&vertices, &faces);
        assert_eq!(repaired.faces.len(), faces.len());
        assert_eq!(repaired.vertices.len() / 3, faces.len());
        // Face corner positions match the input face positions.
        for (out_face, in_face) in repaired.faces.chunks_exact(3).zip(faces.chunks_exact(3)) {
            for corner in 0..3 {
                assert_eq!(
                    vertex(&repaired.vertices, out_face[corner]),
                    vertex(&vertices, in_face[corner])
                );
            }
        }
        // No edge is shared by more than one face after the split.
        let mut edge_counts: HashMap<[u32; 2], usize> = HashMap::new();
        for face in repaired.faces.chunks_exact(3) {
            for local in 0..3 {
                let u = face[local];
                let v = face[(local + 1) % 3];
                *edge_counts
                    .entry(if u <= v { [u, v] } else { [v, u] })
                    .or_default() += 1;
            }
        }
        assert!(edge_counts.values().all(|&count| count <= 2));
    }

    #[test]
    fn repair_non_manifold_edges_splits_bowtie_vertex() {
        // Two triangles meeting at vertex 0 only: no manifold edge connects
        // the sheets, so vertex 0 is duplicated (one copy per sheet).
        let vertices: Vec<f32> = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, -1.0, 0.0],
        ]
        .iter()
        .flatten()
        .copied()
        .collect();
        let faces: Vec<u32> = [0, 1, 2, 0, 3, 4].to_vec();
        let repaired = repair_non_manifold_edges(&vertices, &faces);
        assert_eq!(repaired.faces.len(), faces.len());
        assert_eq!(repaired.vertices.len() / 3, 6);
        // The two copies of the origin are distinct output vertices.
        let origins: Vec<u32> = repaired.faces.chunks_exact(3).map(|face| face[0]).collect();
        assert_ne!(origins[0], origins[1]);
        for &origin in &origins {
            assert_eq!(vertex(&repaired.vertices, origin), [0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn repair_non_manifold_edges_leaves_manifold_mesh_untouched() {
        // A closed manifold cube: every vertex corner is connected through
        // manifold edges, so the repair only renumbers (8 vertices, same
        // triangles up to the remapped indices).
        let (vertices, faces) = cube();
        let repaired = repair_non_manifold_edges(&vertices, &faces);
        assert_eq!(repaired.vertices.len(), vertices.len());
        assert_eq!(repaired.faces.len(), faces.len());
        for (out_face, in_face) in repaired.faces.chunks_exact(3).zip(faces.chunks_exact(3)) {
            for corner in 0..3 {
                assert_eq!(
                    vertex(&repaired.vertices, out_face[corner]),
                    vertex(&vertices, in_face[corner])
                );
            }
        }
        // Structurally degenerate faces are dropped, valid ones untouched.
        let with_junk: Vec<u32> = faces.iter().copied().chain([9, 9, 9]).collect();
        let repaired = repair_non_manifold_edges(&vertices, &with_junk);
        assert_eq!(repaired.faces.len(), faces.len());
    }

    #[test]
    fn unify_orientations_flips_inverted_cube_back() {
        let (vertices, faces) = cube();
        let inverted: Vec<u32> = faces
            .chunks_exact(3)
            .flat_map(|f| [f[0], f[2], f[1]])
            .collect();
        assert!(signed_volume(&vertices, &inverted) < 0.0);

        let unified = unify_face_orientations(&vertices, &inverted);
        assert_eq!(unified, faces);
        assert!((signed_volume(&vertices, &unified) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn unify_orientations_repairs_single_inverted_face() {
        let (vertices, mut faces) = cube();
        // Invert only the first triangle.
        faces.swap(1, 2);
        assert!(!is_consistently_oriented(&faces));

        let unified = unify_face_orientations(&vertices, &faces);
        assert_eq!(unified.len() / 3, 12);
        assert!(is_consistently_oriented(&unified));
        assert!((signed_volume(&vertices, &unified) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn unify_orientations_keeps_consistent_cube_and_open_strip() {
        let (vertices, faces) = cube();
        assert_eq!(unify_face_orientations(&vertices, &faces), faces);

        // Open two-triangle strip, already consistent: zero volume keeps the
        // propagated orientation.
        let strip: Vec<u32> = [0, 1, 5, 0, 5, 4].to_vec();
        assert_eq!(unify_face_orientations(&vertices, &strip), strip);
    }

    #[test]
    fn unify_orientations_drops_malformed_faces() {
        let (vertices, faces) = cube();
        let mut broken = faces.clone();
        broken.extend_from_slice(&[0, 1, 999, 3, 3, 3]);
        let out = unify_face_orientations(&vertices, &broken);
        assert_eq!(out, faces);
    }

    #[test]
    fn simplify_reduces_subdivided_grid_and_stays_valid() {
        let (vertices, faces) = grid(16);
        assert_eq!(faces.len() / 3, 512);

        let result = simplify_mesh(&vertices, &faces, 100);
        let face_count = result.faces.len() / 3;
        assert!(face_count > 0 && face_count <= 100, "got {face_count}");
        assert_eq!(result.vertices.len() / 3, result.vertex_remap.len());
        let vertex_count = result.vertices.len() / 3;
        assert!(result.faces.iter().all(|&i| (i as usize) < vertex_count));
        assert!(result.vertices.iter().all(|v| v.is_finite()));
        // Remap is unique and points back into the original vertex buffer.
        let mut seen = result.vertex_remap.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), vertex_count);
        assert!(result.vertex_remap.iter().all(|&o| (o as usize) < 289));

        // Deterministic.
        let again = simplify_mesh(&vertices, &faces, 100);
        assert_eq!(again, result);
    }

    #[test]
    fn simplify_above_target_returns_compacted_input() {
        let (vertices, faces) = cube();
        let result = simplify_mesh(&vertices, &faces, 600);
        assert_eq!(result.faces.len() / 3, 12);
        assert_eq!(result.vertex_remap.len(), 8);
        // Compact relabels vertices into first-use order; the mesh itself
        // must be unchanged (positions transfer via remap, volume intact).
        let mut sorted_remap = result.vertex_remap.clone();
        sorted_remap.sort_unstable();
        assert_eq!(sorted_remap, (0..8).collect::<Vec<u32>>());
        for (new, &old) in result.vertex_remap.iter().enumerate() {
            assert_eq!(
                &result.vertices[new * 3..new * 3 + 3],
                &vertices[old as usize * 3..old as usize * 3 + 3]
            );
        }
        assert!((signed_volume(&result.vertices, &result.faces) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn simplify_zero_target_and_malformed_faces() {
        let (vertices, faces) = cube();
        let result = simplify_mesh(&vertices, &faces, 0);
        assert!(result.faces.is_empty());
        assert!(result.vertices.is_empty() && result.vertex_remap.is_empty());

        let mut broken = faces.clone();
        broken.extend_from_slice(&[0, 1, 999, 4, 4, 4]);
        let result = simplify_mesh(&vertices, &broken, 600);
        assert_eq!(result.faces.len() / 3, 12);
        assert!((signed_volume(&result.vertices, &result.faces) - 1.0).abs() < 1e-6);

        let empty = simplify_mesh(&[], &[], 10);
        assert!(empty.faces.is_empty() && empty.vertices.is_empty());
    }

    #[test]
    fn postprocess_restores_holey_cube_without_decimation() {
        let (vertices, faces) = cube();
        // One top triangle removed -> postprocess restores a closed cube
        // (the triangular hole closes with a single triangle, no new vertex).
        let holey: Vec<u32> = faces[..6]
            .iter()
            .chain(faces[9..].iter())
            .copied()
            .collect();
        let result = postprocess_mesh(&vertices, &holey, 10.0, 0, 0.0);
        assert_eq!(result.faces.len() / 3, 12);
        assert_eq!(result.vertices.len() / 3, 8);
        let incidence = edge_incidence(&result.faces);
        assert!(incidence.values().all(|&count| count == 2));
        assert!((signed_volume(&result.vertices, &result.faces) - 1.0).abs() < 1e-5);
        let mut sorted_remap = result.vertex_remap.clone();
        sorted_remap.sort_unstable();
        assert_eq!(sorted_remap, (0..8).collect::<Vec<u32>>());
    }

    #[test]
    fn postprocess_decimates_grid_and_is_deterministic() {
        let (vertices, faces) = grid(16);
        let result = postprocess_mesh(&vertices, &faces, 0.0, 48, 0.0);
        let face_count = result.faces.len() / 3;
        assert!(face_count > 0 && face_count <= 48, "got {face_count}");
        let vertex_count = result.vertices.len() / 3;
        assert_eq!(result.vertex_remap.len(), vertex_count);
        assert!(result.faces.iter().all(|&i| (i as usize) < vertex_count));
        assert!(result.vertices.iter().all(|v| v.is_finite()));
        assert_eq!(result, postprocess_mesh(&vertices, &faces, 0.0, 48, 0.0));

        // Aggressive decimation of the holey cube still yields a sane,
        // positively-oriented mesh.
        let (cube_vertices, cube_faces) = cube();
        let holey: Vec<u32> = cube_faces[..6]
            .iter()
            .chain(cube_faces[12..].iter())
            .copied()
            .collect();
        let result = postprocess_mesh(&cube_vertices, &holey, 10.0, 4, 0.0);
        assert!(!result.faces.is_empty());
        let vertex_count = result.vertices.len() / 3;
        assert!(result.faces.iter().all(|&i| (i as usize) < vertex_count));
        // Aggressive decimation may open the surface (zero volume), but
        // unify_face_orientations never leaves a component negative.
        assert!(signed_volume(&result.vertices, &result.faces) >= 0.0);
    }

    #[test]
    fn postprocess_all_malformed_faces_yields_empty() {
        let out = postprocess_mesh(&[0.0; 6], &[0, 1, 42, 5, 5, 5], 10.0, 8, 0.0);
        assert!(out.faces.is_empty() && out.vertices.is_empty() && out.vertex_remap.is_empty());
    }

    /// Two shared triangles whose right-edge vertices are duplicated at a
    /// tiny offset (the DMC/TripoSG welding scenario): [0,1,2] + [3,4,5] with
    /// 1 ~ 4 and 2 ~ 5 merged.
    #[test]
    fn merge_close_vertices_welds_duplicates_and_keeps_first_position() {
        let vertices: Vec<f32> = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], // triangle 0
            [0.0, 0.0, 0.0],
            [1.0 + 1e-6, 0.0, 0.0],
            [0.0, 1.0 + 1e-6, 0.0], // duplicate triangle (offset 1e-6)
        ]
        .iter()
        .flatten()
        .copied()
        .collect();
        let faces: Vec<u32> = [0, 1, 2, 3, 4, 5].to_vec();

        let result = merge_close_vertices(&vertices, &faces, 1e-4);
        // All six input vertices collapse onto three representatives.
        assert_eq!(result.vertices.len() / 3, 3);
        assert_eq!(result.faces, vec![0u32, 1, 2, 0, 1, 2]);
        // Representatives are the first (lowest-index) vertices of their
        // cluster; the merged duplicates point at them.
        assert_eq!(result.vertex_remap, vec![0, 1, 2]);
        // The kept positions are the FIRST cluster vertices (not averaged).
        assert_eq!(&result.vertices[..3], &[0.0, 0.0, 0.0]);
        assert_eq!(&result.vertices[3..6], &[1.0, 0.0, 0.0]);
        assert_eq!(&result.vertices[6..9], &[0.0, 1.0, 0.0]);
    }

    #[test]
    fn merge_close_vertices_respects_epsilon_and_determinism() {
        let vertices: Vec<f32> = [
            [0.0, 0.0, 0.0],
            [0.009, 0.0, 0.0], // within 0.01 of vertex 0
            [0.02, 0.0, 0.0],  // 0.011 away from 0.009 -> separate cluster
        ]
        .iter()
        .flatten()
        .copied()
        .collect();
        let faces: Vec<u32> = [0, 1, 2].to_vec();

        let merged = merge_close_vertices(&vertices, &faces, 0.01);
        assert_eq!(merged.vertices.len() / 3, 2);
        assert_eq!(merged.vertex_remap, vec![0, 2]);
        // Face collapses to a line -> dropped.
        assert!(merged.faces.is_empty());

        // Same input -> same output (no HashMap-order leaks).
        let again = merge_close_vertices(&vertices, &faces, 0.01);
        assert_eq!(again, merged);

        // A smaller epsilon keeps all three vertices apart.
        let apart = merge_close_vertices(&vertices, &faces, 0.005);
        assert_eq!(apart.vertices.len() / 3, 3);
        assert_eq!(apart.faces, vec![0u32, 1, 2]);
    }

    #[test]
    fn merge_close_vertices_degenerate_and_zero_epsilon_no_ops() {
        let (vertices, faces) = cube();

        // Zero / negative / NaN epsilon: compacted input unchanged.
        for epsilon in [0.0_f32, -1.0, f32::NAN] {
            let result = merge_close_vertices(&vertices, &faces, epsilon);
            assert_eq!(result.faces.len() / 3, 12);
            let mut sorted_remap = result.vertex_remap.clone();
            sorted_remap.sort_unstable();
            assert_eq!(sorted_remap, (0..8).collect::<Vec<u32>>());
            assert!((signed_volume(&result.vertices, &result.faces) - 1.0).abs() < 1e-6);
        }

        // Out-of-range faces are dropped, empty meshes stay empty. The
        // cube's shortest edge is 1.0, so a 1e-6 epsilon welds nothing.
        let mut broken = faces.clone();
        broken.extend_from_slice(&[0, 1, 999]);
        let result = merge_close_vertices(&vertices, &broken, 1e-6);
        assert_eq!(result.faces.len() / 3, 12);

        let empty = merge_close_vertices(&[], &[], 1.0);
        assert!(empty.vertices.is_empty() && empty.faces.is_empty());

        // Pre-existing degenerate faces are dropped by the merge rewrite.
        let with_degenerate: Vec<u32> = faces.iter().copied().chain([2, 2, 3]).collect();
        let result = merge_close_vertices(&vertices, &with_degenerate, 1e-6);
        assert_eq!(result.faces.len() / 3, 12);
    }

    /// Merge must survive the 512-face grid: a huge epsilon collapses the
    /// whole plane onto one vertex (every face degenerate), a moderate one
    /// only welds the near-duplicates we inject.
    #[test]
    fn merge_close_vertices_on_grid_scale() {
        let (vertices, mut faces) = grid(16);
        // Duplicate vertex 0 (grid corner) as a fresh vertex slot and add a
        // triangle referencing the duplicates.
        let duplicate: Vec<f32> =
            [vertices[0] + 1e-7, vertices[1] + 1e-7, vertices[2] + 1e-7].to_vec();
        let mut welded_vertices = vertices.clone();
        welded_vertices.extend_from_slice(&duplicate);
        faces.extend_from_slice(&[0, 1, (vertices.len() / 3) as u32]);

        let merged = merge_close_vertices(&welded_vertices, &faces, 1e-4);
        // The duplicate joined vertex 0; the added triangle collapsed.
        assert_eq!(merged.vertices.len() / 3, vertices.len() / 3);
        assert_eq!(merged.faces.len() / 3, 512);
        assert_eq!(merged.vertex_remap[0], 0);
        assert_eq!(merged.vertex_remap[1], 1);

        // Fully degenerate case: epsilon = mesh diagonal merges everything.
        let everything = merge_close_vertices(&vertices, &faces, 10.0);
        assert_eq!(everything.vertices.len() / 3, 1);
        assert!(everything.faces.is_empty());
    }
}
