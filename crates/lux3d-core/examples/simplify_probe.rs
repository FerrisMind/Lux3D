//! Offline probe for `geometry::simplify_mesh`: loads raw `<f32 positions>` and
//! `<u32 faces>` binaries, simplifies toward a face target, prints topology
//! stats and writes the result back as raw binaries.
//!
//! Usage:
//!   cargo run --release --example simplify_probe -- \
//!     verts.bin faces.bin TARGET_FACES out_prefix [simplify|postprocess]

use std::collections::HashSet;
use std::env;
use std::fs;

use lux3d_core::geometry::{
    fill_holes_with_centroids, postprocess_mesh, remove_duplicate_faces,
    remove_small_connected_components, repair_non_manifold_edges, simplify_mesh,
    unify_face_orientations, MeshEditResult,
};

fn read_f32(path: &str) -> Vec<f32> {
    let bytes = fs::read(path).expect("read f32 file");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_u32(path: &str) -> Vec<u32> {
    let bytes = fs::read(path).expect("read u32 file");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 || args.len() > 6 {
        eprintln!(
            "usage: simplify_probe VERTS.bin FACES.bin TARGET_FACES OUT_PREFIX [simplify|postprocess]"
        );
        std::process::exit(2);
    }
    let vertices = read_f32(&args[1]);
    let faces = read_u32(&args[2]);
    let target: usize = args[3].parse().expect("target faces");
    let prefix = &args[4];
    let stage = args.get(5).map(String::as_str).unwrap_or("simplify");

    let unique_before: HashSet<[u32; 3]> = vertices
        .chunks_exact(3)
        .map(|v| [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()])
        .collect();
    println!(
        "input: {} verts ({} unique positions), {} faces",
        vertices.len() / 3,
        unique_before.len(),
        faces.len() / 3
    );

    let result: MeshEditResult = match stage {
        "simplify" => simplify_mesh(&vertices, &faces, target),
        "postprocess" => postprocess_mesh(&vertices, &faces, 3e-2, target, 1e-5),
        "fill" => {
            let (vertices, faces) = fill_holes_with_centroids(&vertices, &faces, 3e-2);
            MeshEditResult {
                vertices,
                faces,
                vertex_remap: Vec::new(),
            }
        }
        "preclean" => {
            // Repair the raw non-manifold topology BEFORE the aggressive
            // decimation (upstream postprocess simplifies first).
            let (vertices, faces) = fill_holes_with_centroids(&vertices, &faces, 3e-2);
            let faces = remove_duplicate_faces(vertices.len() / 3, &faces);
            let repaired = repair_non_manifold_edges(&vertices, &faces);
            let faces =
                remove_small_connected_components(&repaired.vertices, &repaired.faces, 1e-5);
            let (vertices, faces) = fill_holes_with_centroids(&repaired.vertices, &faces, 3e-2);
            let simplified = simplify_mesh(&vertices, &faces, target);
            let faces = remove_duplicate_faces(simplified.vertices.len() / 3, &simplified.faces);
            let repaired = repair_non_manifold_edges(&simplified.vertices, &faces);
            let faces =
                remove_small_connected_components(&repaired.vertices, &repaired.faces, 1e-5);
            let (vertices, faces) = fill_holes_with_centroids(&repaired.vertices, &faces, 3e-2);
            let faces = unify_face_orientations(&vertices, &faces);
            MeshEditResult {
                vertices,
                faces,
                vertex_remap: Vec::new(),
            }
        }
        "halving" => {
            // Progressive ~2x simplification passes (meshopt preserves
            // quality far better at moderate ratios).
            let (mut vertices, mut faces) = fill_holes_with_centroids(&vertices, &faces, 3e-2);
            let mut current = faces.len() / 3;
            while current > target {
                let next = (current / 2).max(target);
                let simplified = simplify_mesh(&vertices, &faces, next);
                vertices = simplified.vertices;
                faces = simplified.faces;
                let before = current;
                current = faces.len() / 3;
                eprintln!("halving: {before} -> {current}");
                if current >= before {
                    break;
                }
            }
            let faces = unify_face_orientations(&vertices, &faces);
            MeshEditResult {
                vertices,
                faces,
                vertex_remap: Vec::new(),
            }
        }
        other => {
            eprintln!("unknown stage {other:?} (simplify|postprocess|fill|preclean|halving)");
            std::process::exit(2);
        }
    };
    let unique_after: HashSet<[u32; 3]> = result
        .vertices
        .chunks_exact(3)
        .map(|v| [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()])
        .collect();
    println!(
        "output: {} verts ({} unique positions), {} faces",
        result.vertices.len() / 3,
        unique_after.len(),
        result.faces.len() / 3
    );

    let verts_bytes: Vec<u8> = result
        .vertices
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let faces_bytes: Vec<u8> = result.faces.iter().flat_map(|f| f.to_le_bytes()).collect();
    fs::write(format!("{prefix}_verts.bin"), verts_bytes).expect("write verts");
    fs::write(format!("{prefix}_faces.bin"), faces_bytes).expect("write faces");
}
