//! Integration tests for the o_voxel command layer (`mesh2ovox`,
//! `ovox2mesh`, `ovox2glb`).
//!
//! Fixtures are generated as textured GLBs in `CARGO_TARGET_TMPDIR`; nothing
//! is written into the repository's tracked `out/` directory.

use std::path::{Path, PathBuf};

use lux3d_core::export::ovox::vxz::VxzCompression;
use lux3d_core::export::ovox::{read_voxels, write_voxels};
use lux3d_core::export::{
    mesh2ovox, ovox2glb, ovox2mesh, write_textured_glb, MeshToOvoxOptions, OvoxDecodeOptions,
    OvoxToGlbOptions, TexturedAlphaMode, TexturedGlbMesh,
};

const BASE_COLOR: [u8; 4] = [64, 128, 192, 255];
const METALLIC_ROUGHNESS: [u8; 4] = [0, 191, 64, 255]; // [_, roughness, metallic, _]

fn temp_dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create test temp dir");
    dir
}

/// Axis-aligned cube with four vertices and two triangles per face (outward
/// winding), constant-color PBR textures.
fn textured_cube(half_extent: f32) -> TexturedGlbMesh {
    let frames: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ];
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut faces = Vec::new();
    for (normal, u, v) in frames {
        let base = (vertices.len() / 3) as u32;
        let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
        let corner_uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        for (corner, uv) in corners.iter().zip(corner_uvs) {
            for axis in 0..3 {
                vertices.push(
                    normal[axis] * half_extent
                        + u[axis] * half_extent * corner.0
                        + v[axis] * half_extent * corner.1,
                );
            }
            normals.extend_from_slice(&normal);
            uvs.push(uv.0);
            uvs.push(uv.1);
        }
        faces.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    let texture_size = 16;
    let base_color_rgba = BASE_COLOR
        .iter()
        .copied()
        .cycle()
        .take(texture_size * texture_size * 4)
        .collect();
    let metallic_roughness = METALLIC_ROUGHNESS
        .iter()
        .copied()
        .cycle()
        .take(texture_size * texture_size * 4)
        .collect();
    TexturedGlbMesh {
        vertices,
        normals,
        uvs,
        faces,
        base_color_rgba,
        metallic_roughness,
        texture_size,
        alpha_mode: TexturedAlphaMode::Opaque,
        double_sided: true,
        already_gltf_axes: false,
    }
}

fn write_fixture(path: &Path, half_extent: f32) {
    write_textured_glb(&textured_cube(half_extent), path).expect("write textured GLB fixture");
}

/// Unsigned distance from a point to the boundary of the axis-aligned cube
/// `[-h, h]^3` (the normalized fixture surface).
fn cube_surface_distance(point: [f32; 3], half: f32) -> f32 {
    let outside: Vec<f32> = point
        .iter()
        .map(|coordinate| (coordinate.abs() - half).max(0.0))
        .collect();
    if outside.iter().any(|distance| *distance > 0.0) {
        outside.iter().map(|distance| distance * distance).sum::<f32>().sqrt()
    } else {
        point
            .iter()
            .map(|coordinate| half - coordinate.abs())
            .fold(f32::INFINITY, f32::min)
    }
}

#[test]
fn mesh_to_vxz_attribute_layout_matches_official_channels() {
    let dir = temp_dir("ovox_attr_layout");
    let input = dir.join("cube.glb");
    write_fixture(&input, 0.5);

    for compression in [VxzCompression::None, VxzCompression::Deflate] {
        let output = dir.join(format!("cube_{}.vxz", compression.as_str()));
        let options = MeshToOvoxOptions {
            grid_size: Some([16, 16, 16]),
            compression,
            ..MeshToOvoxOptions::default()
        };
        let stats = mesh2ovox(&input, &output, &options).expect("mesh2ovox");
        assert_eq!(stats.grid_size, [16, 16, 16]);
        assert!(stats.voxel_count > 0);
        assert_eq!(stats.attribute_channels.get("base_color"), Some(&3));
        assert_eq!(stats.attribute_channels.get("metallic"), Some(&1));
        assert_eq!(stats.attribute_channels.get("roughness"), Some(&1));
        assert_eq!(stats.attribute_channels.get("emissive"), Some(&3));
        assert_eq!(stats.attribute_channels.get("alpha"), Some(&1));
        assert_eq!(stats.attribute_channels.get("normal"), Some(&3));
        assert_eq!(stats.attribute_channels.get("dual_vertices"), Some(&3));
        assert_eq!(stats.attribute_channels.get("intersected"), Some(&1));
        assert!(stats.output_bytes > 0);

        // Round-trip the container: every attribute must survive byte-exact
        // (the writer reorders rows by z-order, so compare row-keyed data).
        let grid = read_voxels(&output).expect("read .vxz");
        grid.validate().expect("valid grid");
        let copied = dir.join("copied.vxz");
        write_voxels(&copied, &grid).expect("write .vxz");
        let copied_grid = read_voxels(&copied).expect("read copied .vxz");
        assert_eq!(grid.coords, copied_grid.coords);
        assert_eq!(grid.attrs, copied_grid.attrs);

        // Official PBR layout: material channels are quantized constants.
        let count = grid.len();
        for index in 0..count {
            let base = grid.attr("base_color").unwrap().row(index).unwrap();
            assert!((base[0] as i32 - 64).abs() <= 1, "base_color red {base:?}");
            assert!((base[1] as i32 - 128).abs() <= 1, "base_color green {base:?}");
            assert!((base[2] as i32 - 192).abs() <= 1, "base_color blue {base:?}");
            let metallic = grid.attr("metallic").unwrap().row(index).unwrap()[0];
            assert!((metallic as i32 - 64).abs() <= 2, "metallic {metallic}");
            let roughness = grid.attr("roughness").unwrap().row(index).unwrap()[0];
            assert!((roughness as i32 - 191).abs() <= 2, "roughness {roughness}");
            assert_eq!(grid.attr("alpha").unwrap().row(index).unwrap()[0], 255);
            assert_eq!(grid.attr("emissive").unwrap().row(index).unwrap(), &[0, 0, 0]);
            let dual = grid.attr("dual_vertices").unwrap().row(index).unwrap();
            assert_eq!(dual.len(), 3);
            let intersected = grid.attr("intersected").unwrap().row(index).unwrap()[0];
            assert!(intersected < 8, "intersected bits {intersected}");
        }
    }
}

#[test]
fn mesh_vxz_mesh_roundtrip_stays_within_one_voxel() {
    let dir = temp_dir("ovox_roundtrip");
    let input = dir.join("cube.glb");
    write_fixture(&input, 0.5);
    let volume = dir.join("cube.vxz");
    let resolution = 64u32;
    let options = MeshToOvoxOptions {
        grid_size: Some([resolution; 3]),
        ..MeshToOvoxOptions::default()
    };
    let voxelized = mesh2ovox(&input, &volume, &options).expect("mesh2ovox");

    let ply = dir.join("cube.ply");
    let decode_options = OvoxDecodeOptions {
        grid_size: [resolution; 3],
        aabb: [[-0.5; 3], [0.5; 3]],
    };
    let decoded = ovox2mesh(&volume, &ply, &decode_options).expect("ovox2mesh");
    assert_eq!(decoded.voxel_count, voxelized.voxel_count);
    assert_eq!(decoded.vertices, decoded.voxel_count);
    assert!(decoded.faces > 0);
    assert!(decoded.vertex_colors);
    assert!(decoded.output_bytes > 0);

    // The fixture normalizes to a cube of half extent 0.99999 / 2.
    let half = 0.99999 / 2.0;
    let voxel_size = 1.0 / resolution as f32;
    let mut max_distance = 0.0f32;
    let mut sum_distance = 0.0f64;
    let mut max_bounds_error = 0.0f32;
    for axis in 0..3 {
        max_bounds_error = max_bounds_error
            .max((decoded.bbox_min[axis] + 0.5).abs())
            .max((decoded.bbox_max[axis] - 0.5).abs());
    }
    let vertices = read_ply_vertices(&ply);
    for vertex in &vertices {
        let distance = cube_surface_distance(*vertex, half);
        max_distance = max_distance.max(distance);
        sum_distance += distance as f64;
    }
    let mean_distance = sum_distance / vertices.len() as f64;
    println!(
        "roundtrip res {resolution}: voxels={} verts={} faces={} max_dist={max_distance:.6} \
         mean_dist={mean_distance:.6} max_bbox_err={max_bounds_error:.6} voxel={voxel_size:.6}",
        voxelized.voxel_count, decoded.vertices, decoded.faces
    );
    assert!(
        max_bounds_error <= 0.5 * voxel_size + 1e-4,
        "reconstructed bbox {decoded:?} left the voxel box"
    );
    // Dual vertices are packed to 1/255 of a cell and the QEF vertex may sit
    // in a neighbouring surface voxel; two voxel diagonals is a safe bound.
    let tolerance = 2.0 * voxel_size * 3.0f32.sqrt();
    assert!(
        max_distance <= tolerance,
        "max distance {max_distance} exceeds tolerance {tolerance}"
    );
    assert!(
        mean_distance <= 0.5 * voxel_size as f64,
        "mean distance {mean_distance} exceeds half a voxel"
    );
}

#[test]
fn ovox2glb_end_to_end_at_official_resolution() {
    let dir = temp_dir("ovox_e2e_512");
    let input = dir.join("small_cube.glb");
    // A small object inside the default [-0.5, 0.5] box keeps the 512^3 run
    // fast while still exercising the official resolution, decimation target
    // and texture size.
    write_fixture(&input, 0.05);
    let volume = dir.join("small_cube.vxz");
    let mesh_options = MeshToOvoxOptions {
        grid_size: Some([512, 512, 512]),
        aabb: Some([[-0.5; 3], [0.5; 3]]),
        normalize_to_unit_cube: false,
        ..MeshToOvoxOptions::default()
    };
    let voxelized = mesh2ovox(&input, &volume, &mesh_options).expect("mesh2ovox at 512");
    assert_eq!(voxelized.grid_size, [512, 512, 512]);

    let ply = dir.join("small_cube.ply");
    let decoded = ovox2mesh(&volume, &ply, &OvoxDecodeOptions::default()).expect("ovox2mesh");
    assert!(decoded.faces > 0);

    let glb = dir.join("small_cube.glb.out.glb");
    let glb_options = OvoxToGlbOptions::default();
    let baked = ovox2glb(&volume, &glb, &glb_options).expect("ovox2glb");
    assert_eq!(baked.texture_size, 2048);
    assert!(baked.atlas_vertices > 0);
    assert!(baked.atlas_faces > 0);
    assert!(baked.chart_count > 0);
    assert!(baked.valid_texels > 0);
    assert!(baked.double_sided);
    assert!(baked.output_bytes > 0);
    assert!(glb.exists());

    println!(
        "e2e res 512: voxels={} extracted_verts={} extracted_faces={} post_verts={} \
         post_faces={} atlas_verts={} atlas_faces={} charts={} texture={} valid_texels={} \
         glb_bytes={} ply_verts={} ply_faces={}",
        voxelized.voxel_count,
        baked.extracted_vertices,
        baked.extracted_faces,
        baked.postprocessed_vertices,
        baked.postprocessed_faces,
        baked.atlas_vertices,
        baked.atlas_faces,
        baked.chart_count,
        baked.texture_size,
        baked.valid_texels,
        baked.output_bytes,
        decoded.vertices,
        decoded.faces
    );

    // Read the baked GLB back to confirm the textured mesh contract.
    let reloaded =
        lux3d_core::export::load_textured_mesh(&glb).expect("reload baked GLB");
    assert_eq!(reloaded.vertex_count(), baked.atlas_vertices);
    assert_eq!(reloaded.face_count(), baked.atlas_faces);
    assert!(!reloaded.materials.is_empty());
    assert_eq!(reloaded.materials[0].base_color_texture.as_ref().unwrap().width, 2048);
    assert_eq!(reloaded.materials[0].metallic_texture.as_ref().unwrap().width, 2048);
    assert_eq!(reloaded.materials[0].roughness_texture.as_ref().unwrap().width, 2048);
}

/// Minimal ASCII PLY vertex reader (the writer emits `format ascii 1.0` with
/// float x/y/z properties).
fn read_ply_vertices(path: &Path) -> Vec<[f32; 3]> {
    let text = std::fs::read_to_string(path).expect("read PLY");
    let mut lines = text.lines();
    let mut vertex_count = 0usize;
    let mut in_vertices = false;
    let mut vertices = Vec::new();
    for line in lines.by_ref() {
        if let Some(count) = line.strip_prefix("element vertex ") {
            vertex_count = count.trim().parse().expect("vertex count");
        }
        if line.trim() == "end_header" {
            in_vertices = true;
            break;
        }
    }
    assert!(in_vertices);
    for line in lines.take(vertex_count) {
        let values: Vec<f32> = line
            .split_whitespace()
            .take(3)
            .map(|value| value.parse().expect("PLY coordinate"))
            .collect();
        assert_eq!(values.len(), 3);
        vertices.push([values[0], values[1], values[2]]);
    }
    assert_eq!(vertices.len(), vertex_count);
    vertices
}
