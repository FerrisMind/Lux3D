//! TEMPORARY cluster-parity replay harness (test/replay tooling only).
//!
//! Purpose: re-run the TRELLIS.2 UV-unwrap + PBR-bake export blocks on the
//! EXACT pre-UV mesh and FROZEN PBR volume of
//! `trellis2_f16_512_12_tex50k_T_fix.glb` + `.pbrvox.bin`, for two source
//! roots (baseline pack-only vs normal-cone clustered). No inference, no
//! model weights, CPU only.
//!
//! Input reconstruction (lossless w.r.t. the bake input):
//! * the GLB positions are the baked atlas vertices in the glTF frame
//!   (`to_glb_axes`: reference `(x, y, z) -> (x, z, -y)`); applying the
//!   inverse `(x, y, z) -> (x, -z, y)` recovers the reference-frame bake
//!   positions bit-exactly (the map is a sign/permutation, no rounding);
//! * the GLB index buffer is the slot-aligned atlas index buffer; the old
//!   artifact kept all 48,378 faces (log `[bake] ... 48378 atlas faces`,
//!   GLB index count 145,134), so the soup is the exact bake input up to
//!   duplicate-position vertex identity, which `uv_unwrap` welds by
//!   position anyway;
//! * the `.pbrvox.bin` sidecar is the frozen 512^3 sparse 6-channel PBR
//!   volume (`L3DPBR1\0` v1: 24-byte header + 1,852,145 * (3 i32 + 6 f32)).
//!
//! The test writes one textured GLB plus atlas dumps and a stats JSON under
//! `PARITY_OUT`; it never edits source or overwrites existing artifacts.
//!
//! Env:
//!   PARITY_GLB        frozen artifact (default: the 50k fix GLB)
//!   PARITY_PBRVOX     frozen PBR sidecar (default: matching .pbrvox.bin)
//!   PARITY_OUT        output directory (required; created if absent)
//!   PARITY_LABEL      label written into stats (default: "unlabeled")
//!   PARITY_TEXSIZE    bake size (default 2048)
//!   PARITY_SNAP       "1" = BVH snap bake (default 0 = plain bake)
//!   PARITY_RAW_VERTS  raw pre-postprocess positions bin (snap source)
//!   PARITY_RAW_FACES  raw pre-postprocess faces bin (snap source)
//!   PARITY_RAW_ROT    raw->reference axis map, e.g. "x,-z,y" (default)
//!   PARITY_RAW_SCALE  uniform raw scale before rotation (default 1.0)
//!   PARITY_RAW_OFFSET "x,y,z" translation after rotation (default 0,0,0)
//!
//! The snap source frame/default is documented in the QA report; when the
//! historical BVH source is not reproducible the plain bake is the primary
//! like-for-like pair and the snap run is a sensitivity check.

use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use lux3d_core::export::{load_mesh, Trellis2ExportStage};
use lux3d_core::geometry::{
    bake_pbr_texture, bake_pbr_texture_with_snap, uv_unwrap, welded_vertex_normals, MeshBvh,
};
use lux3d_core::runtime::Trellis2TexturedMesh;

const PBRVOX_MAGIC: &[u8; 8] = b"L3DPBR1\0";
const PBRVOX_VERSION: u32 = 1;
const PBR_CHANNELS: usize = 6;

struct PbrVolume {
    resolution: u32,
    coords: Vec<[i32; 3]>,
    attrs: Vec<f32>,
}

fn env_or(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn read_f32_bin(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "{} not an f32 bin", path.display());
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_u32_bin(path: &Path) -> Vec<u32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "{} not a u32 bin", path.display());
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn write_f32_bin(path: &Path, values: &[f32]) {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn write_u32_bin(path: &Path, values: &[u32]) {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn read_pbrvox(path: &Path) -> PbrVolume {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(bytes.len() >= 24, "pbrvox header truncated");
    assert_eq!(&bytes[0..8], PBRVOX_MAGIC, "pbrvox magic");
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    assert_eq!(version, PBRVOX_VERSION, "pbrvox version");
    let resolution = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    let count = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
    let record = 3 * 4 + PBR_CHANNELS * 4;
    assert_eq!(
        bytes.len(),
        24 + count * record,
        "pbrvox payload length does not match header count"
    );
    let mut coords = Vec::with_capacity(count);
    let mut attrs = Vec::with_capacity(count * PBR_CHANNELS);
    for index in 0..count {
        let base = 24 + index * record;
        let x = i32::from_le_bytes(bytes[base..base + 4].try_into().unwrap());
        let y = i32::from_le_bytes(bytes[base + 4..base + 8].try_into().unwrap());
        let z = i32::from_le_bytes(bytes[base + 8..base + 12].try_into().unwrap());
        coords.push([x, y, z]);
        for channel in 0..PBR_CHANNELS {
            let at = base + 12 + channel * 4;
            attrs.push(f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()));
        }
    }
    PbrVolume {
        resolution,
        coords,
        attrs,
    }
}

/// Raw->reference axis map: three tokens, each one of `x`, `-x`, `y`, `-y`,
/// `z`, `-z`; output row k is the signed source axis.
fn parse_axis_map(text: &str) -> [[f32; 3]; 3] {
    let tokens: Vec<&str> = text.split(',').map(str::trim).collect();
    assert_eq!(tokens.len(), 3, "PARITY_RAW_ROT needs three tokens");
    let mut map = [[0.0_f32; 3]; 3];
    for (row, token) in tokens.iter().enumerate() {
        let (axis, sign) = match token.chars().next().unwrap() {
            '-' => (token[1..].chars().next().unwrap(), -1.0),
            c => (c, 1.0),
        };
        let column = match axis {
            'x' | 'X' => 0,
            'y' | 'Y' => 1,
            'z' | 'Z' => 2,
            other => panic!("unknown axis {other:?} in PARITY_RAW_ROT"),
        };
        map[row][column] = sign;
    }
    map
}

fn surface_volume_hit_fraction(
    vertices: &[[f32; 3]],
    faces: &[u32],
    volume: &PbrVolume,
) -> f64 {
    let mut present: HashSet<[i32; 3]> = HashSet::with_capacity(volume.coords.len());
    for &coord in &volume.coords {
        present.insert(coord);
    }
    let resolution = volume.resolution as f32;
    let mut hits = 0usize;
    let mut total = 0usize;
    for face in faces.chunks_exact(3) {
        let a = vertices[face[0] as usize];
        let b = vertices[face[1] as usize];
        let c = vertices[face[2] as usize];
        let centroid = [
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0,
        ];
        let mut key = [0_i32; 3];
        for axis in 0..3 {
            let cell = ((centroid[axis] + 0.5) * resolution).floor();
            key[axis] = (cell as i32).clamp(0, volume.resolution as i32 - 1);
        }
        total += 1;
        if present.contains(&key) {
            hits += 1;
        }
    }
    hits as f64 / total as f64
}

struct AtlasDump<'a> {
    out: &'a Path,
}

impl<'a> AtlasDump<'a> {
    fn write(&self, atlas: &lux3d_core::geometry::UvAtlas) {
        let uvs: Vec<f32> = atlas.uvs.iter().flatten().copied().collect();
        let positions: Vec<f32> = atlas.vertices.iter().flatten().copied().collect();
        write_f32_bin(&self.out.join("atlas_uvs.f32.bin"), &uvs);
        write_f32_bin(&self.out.join("atlas_positions.f32.bin"), &positions);
        write_u32_bin(&self.out.join("atlas_indices.u32.bin"), &atlas.indices);
        write_u32_bin(&self.out.join("atlas_vmapping.u32.bin"), &atlas.vmapping);
        write_u32_bin(&self.out.join("atlas_face_ids.u32.bin"), &atlas.face_ids);
    }
}

#[test]
fn cluster_parity_replay() {
    let glb_path = PathBuf::from(env_or(
        "PARITY_GLB",
        "G:/jobs/candle-based-projects/3d/refs-out/ours/trellis2_f16_512_12_tex50k_T_fix.glb",
    ));
    let pbrvox_path = PathBuf::from(env_or(
        "PARITY_PBRVOX",
        "G:/jobs/candle-based-projects/3d/refs-out/ours/trellis2_f16_512_12_tex50k_T_fix.pbrvox.bin",
    ));
    let out = PathBuf::from(env::var("PARITY_OUT").expect("PARITY_OUT is required"));
    let label = env_or("PARITY_LABEL", "unlabeled");
    let texture_size: usize = env_or("PARITY_TEXSIZE", "2048").parse().unwrap();
    let snap = env_or("PARITY_SNAP", "0") == "1";
    fs::create_dir_all(&out).expect("create PARITY_OUT");
    let out = out
        .canonicalize()
        .expect("canonicalize PARITY_OUT (must not be a fresh path)");

    println!("[replay] label {label}");
    println!("[replay] glb {}", glb_path.display());
    println!("[replay] pbrvox {}", pbrvox_path.display());
    println!("[replay] out {}", out.display());
    println!("[replay] texture_size {texture_size} snap {snap}");

    // 1. Frozen mesh: GLB positions are glTF-frame; invert the writer's
    //    axis conversion to recover the reference-frame bake positions.
    let imported = load_mesh(&glb_path).expect("load frozen GLB");
    let glb_vertex_count = imported.vertices.len() / 3;
    let glb_face_count = imported.faces.len() / 3;
    let mut vertices: Vec<[f32; 3]> = Vec::with_capacity(glb_vertex_count);
    for chunk in imported.vertices.chunks_exact(3) {
        vertices.push([chunk[0], -chunk[2], chunk[1]]);
    }
    let faces: Vec<u32> = imported.faces.clone();
    println!(
        "[replay] frozen mesh: {} vertices, {} faces, uvs {} (ignored: replay re-unwraps)",
        glb_vertex_count,
        glb_face_count,
        imported.uvs.is_some()
    );
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for vertex in &vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    let mut unique_bits: HashSet<[u32; 3]> = HashSet::new();
    for vertex in &vertices {
        unique_bits.insert([
            vertex[0].to_bits(),
            vertex[1].to_bits(),
            vertex[2].to_bits(),
        ]);
    }
    println!(
        "[replay] reference frame bbox min {:?} max {:?}; {} unique positions",
        min,
        max,
        unique_bits.len()
    );

    // 2. Frozen PBR volume.
    let volume = read_pbrvox(&pbrvox_path);
    println!(
        "[replay] pbrvox: res {} count {} channels {}",
        volume.resolution,
        volume.coords.len(),
        PBR_CHANNELS
    );
    let hit = surface_volume_hit_fraction(&vertices, &faces, &volume);
    println!("[replay] surface volume nearest-cell hit fraction {hit:.4}");
    assert!(
        hit > 0.5,
        "reference-frame mesh does not align with the frozen volume (hit {hit:.4}); \
         check the GLB->reference axis conversion"
    );

    // 3. Actual UV unwrap (source-root selected by the binary built here).
    let t_uv = Instant::now();
    let atlas = uv_unwrap(&vertices, &faces, texture_size as u32).expect("uv unwrap");
    let uv_ms = t_uv.elapsed().as_secs_f64() * 1000.0;

    // 4. Face-preservation bookkeeping (positions must be the exact input
    //    corners; clustering may reorder slots through face_ids).
    assert_eq!(
        atlas.indices.len(),
        3 * faces.len() / 3,
        "atlas must keep one output triangle per input face"
    );
    assert_eq!(atlas.face_ids.len(), faces.len() / 3, "face_ids length");
    let mut slot_mismatch = 0usize;
    let mut max_position_delta = 0.0_f32;
    for (slot, &input_face) in atlas.face_ids.iter().enumerate() {
        let input_base = input_face as usize * 3;
        for corner in 0..3 {
            let atlas_vertex = atlas.indices[slot * 3 + corner] as usize;
            let expected = vertices[faces[input_base + corner] as usize];
            let got = atlas.vertices[atlas_vertex];
            let delta = (0..3)
                .map(|axis| (got[axis] - expected[axis]).abs())
                .fold(0.0_f32, f32::max);
            if delta > 0.0 {
                slot_mismatch += 1;
                max_position_delta = max_position_delta.max(delta);
            }
        }
    }
    println!(
        "[replay] atlas: charts {} clusters {} atlases {} atlas_size {} uncharted {} \
         fell_back {}; atlas vertices {}; face_ids {}; slot-position mismatches {} max_delta {:.3e}",
        atlas.chart_count,
        atlas.cluster_count,
        atlas.atlas_count,
        atlas.atlas_size,
        atlas.uncharted_face_count,
        atlas.clusters_fell_back,
        atlas.vertices.len(),
        atlas.face_ids.len(),
        slot_mismatch,
        max_position_delta
    );
    assert_eq!(slot_mismatch, 0, "atlas triangle corners moved off the input surface");

    // 5. Normals on the pre-unwrap topology, gathered through vmapping
    //    (same order as `build_trellis2_textured_mesh`).
    let base_normals = welded_vertex_normals(&vertices, &faces);
    let normals: Vec<[f32; 3]> = atlas
        .vmapping
        .iter()
        .map(|&original| base_normals[original as usize])
        .collect();

    // 6. Actual PBR bake against the frozen volume.
    let t_bake = Instant::now();
    let texture = if snap {
        let raw_verts_path = PathBuf::from(
            env::var("PARITY_RAW_VERTS").expect("PARITY_RAW_VERTS required with PARITY_SNAP=1"),
        );
        let raw_faces_path = PathBuf::from(
            env::var("PARITY_RAW_FACES").expect("PARITY_RAW_FACES required with PARITY_SNAP=1"),
        );
        let map = parse_axis_map(&env_or("PARITY_RAW_ROT", "x,-z,y"));
        let scale: f32 = env_or("PARITY_RAW_SCALE", "1.0").parse().unwrap();
        let offset: Vec<f32> = env_or("PARITY_RAW_OFFSET", "0,0,0")
            .split(',')
            .map(|v| v.trim().parse::<f32>().unwrap())
            .collect();
        assert_eq!(offset.len(), 3);
        let raw_positions = read_f32_bin(&raw_verts_path);
        let raw_faces = read_u32_bin(&raw_faces_path);
        assert_eq!(raw_positions.len() % 3, 0);
        let mut original_vertices = Vec::with_capacity(raw_positions.len());
        for chunk in raw_positions.chunks_exact(3) {
            let p = [chunk[0] * scale, chunk[1] * scale, chunk[2] * scale];
            for row in 0..3 {
                original_vertices.push(
                    map[row][0] * p[0] + map[row][1] * p[1] + map[row][2] * p[2]
                        + offset[row],
                );
            }
        }
        println!(
            "[replay] snap BVH source: {} vertices, {} faces, rot {:?} scale {scale} offset {:?}",
            original_vertices.len() / 3,
            raw_faces.len() / 3,
            map,
            offset
        );
        let original_surface =
            MeshBvh::build(&original_vertices, &raw_faces).expect("build original-surface BVH");
        bake_pbr_texture_with_snap(
            &atlas.vertices,
            &atlas.uvs,
            &atlas.indices,
            &original_surface,
            &volume.coords,
            &volume.attrs,
            volume.resolution,
            texture_size,
        )
    } else {
        bake_pbr_texture(
            &atlas.vertices,
            &atlas.uvs,
            &atlas.indices,
            &volume.coords,
            &volume.attrs,
            volume.resolution,
            texture_size,
        )
    }
    .expect("pbr bake");
    let bake_ms = t_bake.elapsed().as_secs_f64() * 1000.0;
    println!(
        "[replay] bake: {} faces, {} valid texels, {} inpainted, {} ms",
        texture.stats.triangle_count, texture.stats.valid_texels, texture.stats.inpainted_texels, bake_ms
    );

    // 7. Compose the same mesh `build_trellis2_textured_mesh` produces and
    //    write it through the real textured-GLB writer.
    let textured = Trellis2TexturedMesh {
        vertices: atlas.vertices.iter().flatten().copied().collect(),
        faces: atlas.indices.clone(),
        normals: normals.iter().flatten().copied().collect(),
        uvs: atlas.uvs.iter().flatten().copied().collect(),
        texture,
        chart_count: atlas.chart_count,
        atlas_count: atlas.atlas_count,
        atlas_size: atlas.atlas_size,
    };
    let glb_out = out.join("export.glb");
    Trellis2ExportStage
        .write_textured_glb(&textured, &glb_out, true)
        .expect("write textured GLB");
    println!("[replay] wrote {}", glb_out.display());

    // 8. Atlas dumps for seam/chart analysis.
    AtlasDump { out: &out }.write(&atlas);

    // 9. Stats JSON (manual; no serde dependency in the test).
    let stats = format!(
        concat!(
            "{{\n",
            "  \"label\": \"{}\",\n",
            "  \"glb_input\": \"{}\",\n",
            "  \"pbrvox_input\": \"{}\",\n",
            "  \"texture_size\": {},\n",
            "  \"snap\": {},\n",
            "  \"input_vertices\": {},\n",
            "  \"input_faces\": {},\n",
            "  \"unique_positions\": {},\n",
            "  \"volume_resolution\": {},\n",
            "  \"volume_voxels\": {},\n",
            "  \"surface_volume_hit_fraction\": {:.6},\n",
            "  \"chart_count\": {},\n",
            "  \"cluster_count\": {},\n",
            "  \"atlas_count\": {},\n",
            "  \"atlas_size\": {},\n",
            "  \"atlas_vertices\": {},\n",
            "  \"uncharted_face_count\": {},\n",
            "  \"clusters_fell_back\": {},\n",
            "  \"faces_preserved\": {},\n",
            "  \"slot_position_mismatches\": {},\n",
            "  \"max_slot_position_delta\": {:.9e},\n",
            "  \"bake_triangle_count\": {},\n",
            "  \"valid_texels\": {},\n",
            "  \"inpainted_texels\": {},\n",
            "  \"texel_count\": {},\n",
            "  \"uv_unwrap_ms\": {:.3},\n",
            "  \"pbr_bake_ms\": {:.3}\n",
            "}}\n"
        ),
        label,
        glb_path.display(),
        pbrvox_path.display(),
        texture_size,
        snap,
        glb_vertex_count,
        glb_face_count,
        unique_bits.len(),
        volume.resolution,
        volume.coords.len(),
        hit,
        atlas.chart_count,
        atlas.cluster_count,
        atlas.atlas_count,
        atlas.atlas_size,
        atlas.vertices.len(),
        atlas.uncharted_face_count,
        atlas.clusters_fell_back,
        atlas.face_ids.len(),
        slot_mismatch,
        max_position_delta,
        textured.texture.stats.triangle_count,
        textured.texture.stats.valid_texels,
        textured.texture.stats.inpainted_texels,
        textured.texture.stats.texel_count,
        uv_ms,
        bake_ms
    );
    fs::write(out.join("atlas_stats.json"), stats).expect("write atlas_stats.json");
    println!("[replay] uv_unwrap {uv_ms:.1} ms, pbr_bake {bake_ms:.1} ms");
}
