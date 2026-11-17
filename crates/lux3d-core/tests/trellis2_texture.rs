//! TRELLIS.2 texturing pipeline core tests:
//!
//! 1. Non-gated: `preprocess_mesh_for_texturing` golden (center/scale/axis
//!    swap, degenerate rejection), `Trellis2TexturingOptions` validation,
//!    and the guide-synthesis builder on synthetic occupancy chains (the
//!    synthesized logits replayed through the REAL `channel_to_spatial`
//!    must reconstruct the voxelization coords exactly — the coordinate
//!    tree the guided texture decode will walk).
//! 2. Gated (`models/TRELLIS.2-4B` raw dir or `LUX3D_TRELLIS2_RAW_DIR`):
//!    shape-encoder weight loading (raw stage file through
//!    `ModelAssetOptions`, raw-dir pipeline `load_shape_encoder`, clear
//!    error when absent), and the real encode -> guide -> guided tex decode
//!    coordinate chain at a small resolution (decode coords must equal the
//!    voxelization coords as a set — the upstream SparseTensor-cache
//!    invariant).
//! 3. Env-gated (`LUX3D_TRELLIS2_RUN_TEXTURE_E2E=1`): the full
//!    `texture_mesh` run (voxelize -> encode -> tex DiT -> guided decode ->
//!    UV unwrap + PBR bake). CPU cost is heavy at the production 1024
//!    resolution (the encoder runs the full-resolution voxel grid); run it
//!    in release with `LUX3D_TRELLIS2_TEX_E2E_RESOLUTION` (default 512),
//!    `LUX3D_TRELLIS2_TEX_E2E_STEPS` (sampler override, default 12),
//!    `LUX3D_TRELLIS2_TEX_E2E_DEVICE` (cpu|cuda|vulkan|wgpu, default cpu)
//!    and `LUX3D_TRELLIS2_TEX_E2E_UVS=1` to exercise the input-UV reuse
//!    branch (no xatlas unwrap; the UVs must round-trip through the
//!    textured GLB):
//!    `LUX3D_TRELLIS2_RUN_TEXTURE_E2E=1 cargo test --release -p lux3d-core
//!    --test trellis2_texture texture_mesh_run_e2e -- --nocapture
//!    --test-threads 1`.

use std::collections::BTreeSet;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use lux3d_core::contracts::Trellis2RawInput;
use lux3d_core::geometry::mesh_to_flexible_dual_grid;
use lux3d_core::runtime::{
    channel_to_spatial, preprocess_mesh_for_texturing, synthesize_tex_guide_subs, Trellis2Pipeline,
    Trellis2SamplerOverrides, Trellis2TexturingOptions,
};
use lux3d_core::test_support::workspace_root;
use lux3d_core::weights::ModelAssetOptions;

fn raw_dir() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("LUX3D_TRELLIS2_RAW_DIR") {
        let path = PathBuf::from(path);
        return path.is_dir().then_some(path);
    }
    let path = workspace_root().join("models").join("TRELLIS.2-4B");
    path.is_dir().then_some(path)
}

/// The 8-cube unit cube fixture (outward winding; same layout as the
/// `mesh_ops` tests).
fn cube_vertices() -> Vec<f32> {
    [
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
    .collect()
}

fn cube_faces() -> Vec<u32> {
    [
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
    .collect()
}

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

/// Golden: the unit cube maps to the centered/scaled frame and the y/z swap
/// is the reference `tmp = y; y = -z; z = tmp`.
#[test]
fn preprocess_mesh_golden_cube() {
    let out =
        preprocess_mesh_for_texturing(&cube_vertices(), &cube_faces()).expect("cube preprocess");
    assert_eq!(out.len(), 24);
    // Extents are all 1: scale = 0.99999, every corner lands on
    // +/-0.499995; the swap maps (x, y, z) -> (x, -z, y).
    let expected_corner = 0.99999_f32 * 0.5;
    let expected: Vec<[f32; 3]> = [
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
    .map(|v| {
        let x = (v[0] - 0.5) * 0.99999;
        let y = (v[1] - 0.5) * 0.99999;
        let z = (v[2] - 0.5) * 0.99999;
        [x, -z, y]
    })
    .collect();
    for (row, exp) in out.chunks_exact(3).zip(expected.iter()) {
        for (v, e) in row.iter().zip(exp.iter()) {
            assert!(
                approx_eq(*v, *e, 1e-6),
                "vertex {row:?} vs expected {exp:?}"
            );
        }
    }
    // Range contract (the reference assert).
    assert!(
        out.iter().all(|&v| v >= -0.5 && v <= 0.5),
        "preprocessed cube out of range"
    );
    assert!(approx_eq(expected_corner, 0.499995, 1e-7));
}

/// Golden: an asymmetric box pins down the center, the max-extent scale and
/// the axis swap independently.
#[test]
fn preprocess_mesh_golden_asymmetric_box() {
    // Box [1, 2] x [0, 4] x [10, 11]: extents (1, 4, 1), center
    // (1.5, 2.0, 10.5), scale = 0.99999 / 4.
    let vertices: Vec<f32> = [[1.0, 0.0, 10.0], [2.0, 4.0, 11.0], [1.5, 2.0, 10.5]]
        .iter()
        .flatten()
        .copied()
        .collect();
    let faces = vec![0, 1, 2];
    let out = preprocess_mesh_for_texturing(&vertices, &faces).expect("box preprocess");
    let scale = 0.99999_f32 / 4.0;
    let expected: Vec<[f32; 3]> = [[1.0, 0.0, 10.0], [2.0, 4.0, 11.0], [1.5, 2.0, 10.5]]
        .iter()
        .map(|v| {
            let x = (v[0] - 1.5) * scale;
            let y = (v[1] - 2.0) * scale;
            let z = (v[2] - 10.5) * scale;
            [x, -z, y]
        })
        .collect();
    for (row, exp) in out.chunks_exact(3).zip(expected.iter()) {
        for (v, e) in row.iter().zip(exp.iter()) {
            assert!(
                approx_eq(*v, *e, 1e-6),
                "vertex {row:?} vs expected {exp:?}"
            );
        }
    }
    // The largest extent (y here) maps to +/-0.499995 after the swap moves
    // it onto z'.
    assert!(
        approx_eq(out[5], 0.499995, 1e-6),
        "max extent onto z: {}",
        out[5]
    );
    assert!(out.iter().all(|&v| v >= -0.5 && v <= 0.5));
}

/// The scale preserves face winding (the swap is a proper rotation, det +1),
/// so faces pass through unchanged.
#[test]
fn preprocess_mesh_preserves_winding() {
    // Signed tetrahedron volume must keep its sign through the transform.
    let vertices: Vec<f32> = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 5.0],
    ]
    .iter()
    .flatten()
    .copied()
    .collect();
    let faces = vec![0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3];
    let out = preprocess_mesh_for_texturing(&vertices, &faces).expect("tetra preprocess");
    let volume = |verts: &[f32]| -> f32 {
        let p = |i: usize| [verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2]];
        let (a, b, c, d) = (p(0), p(1), p(2), p(3));
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let ad = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
        let cross = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        cross[0] * ad[0] + cross[1] * ad[1] + cross[2] * ad[2]
    };
    let before = volume(&vertices);
    let after = volume(&out);
    assert!(before > 0.0 && after > 0.0, "{before} vs {after}");
}

#[test]
fn preprocess_mesh_rejects_degenerate_and_garbage() {
    let faces = vec![0, 1, 2];
    // Non-finite vertex.
    let mut vertices = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    vertices[4] = f32::NAN;
    let error =
        preprocess_mesh_for_texturing(&vertices, &faces).expect_err("non-finite vertex rejected");
    assert!(error.to_string().contains("non-finite"));
    // Zero extent (all vertices identical).
    let degenerate = vec![0.5; 9];
    let error =
        preprocess_mesh_for_texturing(&degenerate, &faces).expect_err("degenerate mesh rejected");
    assert!(error.to_string().contains("degenerate"));
    // Buffer length not a multiple of 3.
    let error = preprocess_mesh_for_texturing(&[0.0, 0.0, 0.0, 1.0], &faces)
        .expect_err("ragged vertex buffer rejected");
    assert!(error.to_string().contains("multiple of 3"));
}

/// Options validation: resolution set and texture-size contract.
#[test]
fn texturing_options_validation() {
    let options = Trellis2TexturingOptions::default();
    assert_eq!(options.seed, 42);
    assert_eq!(options.resolution, 1024);
    assert_eq!(options.texture_size, 2048);
    assert!(options.preprocess_image);
    assert!(options.preprocess_mesh);
    options.validate().expect("defaults valid");
    assert_eq!(options.cond_resolution(), 1024);

    let options = Trellis2TexturingOptions {
        resolution: 512,
        ..Trellis2TexturingOptions::default()
    };
    options.validate().expect("512 valid");
    assert_eq!(options.cond_resolution(), 512);

    for bad_resolution in [0, 256, 768, 2048] {
        let options = Trellis2TexturingOptions {
            resolution: bad_resolution,
            ..Trellis2TexturingOptions::default()
        };
        assert!(
            options.validate().is_err(),
            "resolution {bad_resolution} must be rejected"
        );
    }
    for bad_texture in [0, 12, 100, 8200] {
        let options = Trellis2TexturingOptions {
            texture_size: bad_texture,
            ..Trellis2TexturingOptions::default()
        };
        assert!(
            options.validate().is_err(),
            "texture_size {bad_texture} must be rejected"
        );
    }
    // Sampler overrides ride along untouched.
    let options = Trellis2TexturingOptions {
        tex_sampler: Trellis2SamplerOverrides {
            steps: Some(4),
            ..Trellis2SamplerOverrides::default()
        },
        ..Trellis2TexturingOptions::default()
    };
    options.validate().expect("overrides valid");
}

/// Replay the synthesized guides through the REAL `channel_to_spatial` (the
/// op the guided decoder runs): the coordinate chain must reconstruct the
/// voxelization exactly, level set by level set.
#[test]
fn guide_subs_replay_reconstructs_the_voxelization() {
    let device = Device::Cpu;
    // Deliberate multi-level occupancy: a full 2x2x2 child block under one
    // latent parent, a partial sibling pair, and isolated voxels at other
    // corners. Coordinates unsorted, duplicates absent.
    let mut voxels: Vec<[i32; 3]> = Vec::new();
    for dz in 0..2_i32 {
        for dy in 0..2 {
            for dx in 0..2 {
                voxels.push([16 + dx, 16 + dy, 16 + dz]);
            }
        }
    }
    // Two children of one level-1 parent (32, 16, 16).
    voxels.push([32, 16, 16]);
    voxels.push([33, 17, 16]);
    // Isolated voxels near other corners.
    voxels.push([0, 0, 0]);
    voxels.push([47, 3, 31]);
    voxels.push([63, 62, 63]);
    voxels.push([5, 40, 22]);
    let latent: Vec<[i32; 3]> = {
        let mut parents: Vec<[i32; 3]> = voxels
            .iter()
            .map(|c| [c[0] / 16, c[1] / 16, c[2] / 16])
            .collect();
        parents.sort_unstable();
        parents.dedup();
        parents
    };
    let guides = synthesize_tex_guide_subs(&voxels, &latent, &device).expect("guide synthesis");
    assert_eq!(guides.len(), 4, "one logits tensor per C2S stage");
    assert_eq!(guides[0].dims(), &[latent.len(), 8]);

    // Walk the chain exactly like the guided decoder: C2S over an all-ones
    // [Np, 8] feature block with the synthesized logits.
    let mut parents = latent.clone();
    let divs = [8_i32, 4, 2, 1];
    for (stage, guide) in guides.iter().enumerate() {
        let ones = Tensor::ones((parents.len(), 8), DType::F32, &device).expect("ones");
        let expanded = channel_to_spatial(&parents, &ones, guide).expect("c2s");
        parents = expanded.coords;
        // The stage output set must equal unique(V / div).
        let expected: BTreeSet<[i32; 3]> = voxels
            .iter()
            .map(|c| [c[0] / divs[stage], c[1] / divs[stage], c[2] / divs[stage]])
            .collect();
        let got: BTreeSet<[i32; 3]> = parents.iter().copied().collect();
        assert_eq!(got, expected, "stage {stage} coord set");
    }
    // The final chain is exactly the voxelization (each voxel once).
    let mut final_sorted = parents;
    final_sorted.sort_unstable();
    let mut voxel_sorted = voxels;
    voxel_sorted.sort_unstable();
    assert_eq!(final_sorted, voxel_sorted, "decode coords == voxelization");
}

/// Hand case: a two-voxel occupancy produces the exact expected bit blocks.
#[test]
fn guide_subs_hand_case_two_voxels() {
    let device = Device::Cpu;
    let voxels = vec![[0, 0, 0], [1, 0, 0]];
    let latent = vec![[0, 0, 0]];
    let guides = synthesize_tex_guide_subs(&voxels, &latent, &device).expect("guide synthesis");
    let values = |index: usize| -> Vec<Vec<f32>> {
        guides[index]
            .to_device(&Device::Cpu)
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.to_vec2::<f32>())
            .expect("guide values")
    };
    // Stages 0-2: the single parent's only occupied child is (0, 0, 0)
    // (subidx 0) at every level; stages 1-2 rows follow the C2S expansion
    // order (one row of one parent each).
    for stage in 0..3 {
        let rows = values(stage);
        assert_eq!(rows.len(), 1, "stage {stage} rows");
        assert_eq!(rows[0][0], 1.0, "stage {stage} subidx 0 selected");
        assert!(
            rows[0][1..].iter().all(|&v| v == -1.0),
            "stage {stage} other children dropped: {:?}",
            rows[0]
        );
    }
    // Stage 3: children of (0, 0, 0) on the full grid: (0, 0, 0) and
    // (1, 0, 0) — subidx 0 and 1.
    let rows = values(3);
    assert_eq!(rows[0][0], 1.0);
    assert_eq!(rows[0][1], 1.0);
    assert!(rows[0][2..].iter().all(|&v| v == -1.0));
}

#[test]
fn guide_subs_rejects_bad_inputs() {
    let device = Device::Cpu;
    let voxels = vec![[16, 16, 16], [17, 16, 16]];
    // Latent not equal to unique(V / 16) (which is [[1, 1, 1]] here).
    let error = synthesize_tex_guide_subs(&voxels, &[[2, 2, 2]], &device)
        .expect_err("latent mismatch rejected");
    assert!(error.to_string().contains("unique(voxels / 16)"));
    // Duplicate voxelization rows.
    let error = synthesize_tex_guide_subs(&[[16, 16, 16]; 2], &[[1, 1, 1]], &device)
        .expect_err("duplicate voxels rejected");
    assert!(error.to_string().contains("duplicates"));
    // Negative coords.
    let error = synthesize_tex_guide_subs(&[[-16, 16, 16]], &[[-1, 1, 1]], &device)
        .expect_err("negative coords rejected");
    assert!(error.to_string().contains("non-negative"));
    // Empty inputs.
    assert!(synthesize_tex_guide_subs(&[], &[[0, 0, 0]], &device).is_err());
    assert!(synthesize_tex_guide_subs(&[[0, 0, 0]], &[], &device).is_err());
}

/// Gated: the shape encoder loads from the raw model dir through
/// `ModelAssetOptions` (284 tensors) and through the raw-dir pipeline.
#[test]
fn shape_encoder_loads_from_model_assets() {
    let Some(dir) = raw_dir() else {
        eprintln!("SKIP: TRELLIS.2 raw model dir not found (set LUX3D_TRELLIS2_RAW_DIR)");
        return;
    };
    let device = Device::Cpu;
    let options = ModelAssetOptions {
        canonical_dir: Some(dir.clone()),
        cache_dir: None,
    };
    let encoder =
        lux3d_core::runtime::load_trellis2_shape_encoder_from_model_assets(&options, &device)
            .expect("shape encoder from model assets");
    assert_eq!(encoder.config().in_channels, 6);
    assert_eq!(encoder.config().latent_channels, 32);
    assert_eq!(encoder.config().num_blocks, vec![0, 4, 8, 16, 4]);
    drop(encoder);

    // The raw-dir pipeline loads the same encoder through its own weights.
    let pipeline = Trellis2Pipeline::load_from_raw_dir(&dir, &device).expect("pipeline");
    let _ = pipeline
        .load_shape_encoder(&device)
        .expect("pipeline shape encoder");
}

/// Gated: the loader errors with a clear message when the checkpoint is
/// missing (a directory without the stage file, and the canonical package
/// which has no `shape_enc` stage today).
#[test]
fn shape_encoder_missing_checkpoint_errors_clearly() {
    let device = Device::Cpu;
    let empty = std::env::temp_dir().join(format!(
        "lux3d-trellis2-texture-empty-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&empty);
    std::fs::create_dir_all(&empty).expect("temp dir");
    let options = ModelAssetOptions {
        canonical_dir: Some(empty.clone()),
        cache_dir: None,
    };
    let error =
        lux3d_core::runtime::load_trellis2_shape_encoder_from_model_assets(&options, &device)
            .expect_err("missing checkpoint must fail");
    let message = error.to_string();
    assert!(
        message.contains("shape_enc"),
        "error must name the shape encoder: {message}"
    );
    let _ = std::fs::remove_dir_all(&empty);
}

/// Gated structural chain: voxelize a small cube, encode it with the REAL
/// shape encoder, synthesize the guides and run the REAL guided tex decode
/// with a constant latent. The decoded coords must equal the voxelization
/// coords as a set (the upstream SparseTensor-cache invariant) and the PBR
/// attribute rows must line up.
#[test]
fn encode_guide_decode_coord_chain_gated() {
    let Some(dir) = raw_dir() else {
        eprintln!("SKIP: TRELLIS.2 raw model dir not found (set LUX3D_TRELLIS2_RAW_DIR)");
        return;
    };
    let device = Device::Cpu;
    let pipeline = Trellis2Pipeline::load_from_raw_dir(&dir, &device).expect("pipeline");

    // A small cube filling the [-0.5, 0.5] frame; resolution 32 keeps the
    // encoder/decoder cost tiny (the encoder is resolution-agnostic).
    let resolution = 32_u32;
    let scale = 0.99999_f32;
    let vertices: Vec<f32> = cube_vertices()
        .chunks_exact(3)
        .flat_map(|v| {
            [
                (v[0] - 0.5) * scale,
                (v[1] - 0.5) * scale,
                (v[2] - 0.5) * scale,
            ]
        })
        .collect();
    let faces = cube_faces();
    let vox = mesh_to_flexible_dual_grid(
        &vertices,
        &faces,
        &lux3d_core::geometry::MeshToDualGridOptions::for_shape_slat_encoder(resolution),
    )
    .expect("voxelization");
    assert!(!vox.voxel_indices.is_empty());

    let rows = vox.voxel_indices.len();
    let dual = Tensor::from_vec(
        vox.dual_vertices
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
        (rows, 3),
        &device,
    )
    .expect("dual tensor");
    let flags = Tensor::from_vec(
        vox.intersected
            .iter()
            .flat_map(|f| f.iter().map(|&b| b as u8 as f32))
            .collect::<Vec<_>>(),
        (rows, 3),
        &device,
    )
    .expect("flags tensor");
    let encoder = pipeline.load_shape_encoder(&device).expect("encoder");
    let slat = encoder
        .encode_shape_slat(
            &vox.voxel_indices,
            &dual,
            &flags,
            resolution as usize,
            &device,
        )
        .expect("encode");
    drop(encoder);
    // The encoder's latent coords are the guide builder's expectation
    // (unique(V / 16), sorted) — the cross-module contract.
    let guides = synthesize_tex_guide_subs(&vox.voxel_indices, &slat.coords, &device)
        .expect("guide synthesis from real encoder coords");
    assert_eq!(slat.feats.dims(), &[slat.coords.len(), 32]);

    // Constant texture latent (decode-time values are irrelevant to the
    // coordinate chain).
    let tex_latent =
        Tensor::zeros((slat.coords.len(), 32), DType::F32, &device).expect("tex latent");
    let (tex_coords, attrs) = pipeline
        .decode_tex_attrs(&slat.coords, &tex_latent, &guides, &device)
        .expect("guided tex decode");
    let mut decoded_sorted = tex_coords.clone();
    decoded_sorted.sort_unstable();
    let mut voxel_sorted = vox.voxel_indices.clone();
    voxel_sorted.sort_unstable();
    assert_eq!(
        decoded_sorted, voxel_sorted,
        "guided tex decode coords must equal the voxelization coords as a set"
    );
    assert_eq!(attrs.len(), tex_coords.len() * 6);
    // NOTE: the tex decoder is nonlinear (SiLU/LayerNorm/biases), so a
    // constant latent does NOT decode to a constant attribute value — the
    // meaningful invariant here is the coord chain above, plus the output
    // contract: 6 finite PBR channels clamped to [0, 1] per row.
    assert!(
        attrs
            .iter()
            .all(|&v| v.is_finite() && (0.0..=1.0).contains(&v)),
        "PBR attrs must be finite and in [0, 1]"
    );
}

fn e2e_device() -> Device {
    let requested = std::env::var("LUX3D_TRELLIS2_TEX_E2E_DEVICE")
        .unwrap_or_else(|_| "cpu".to_string())
        .to_lowercase();
    match requested.as_str() {
        "cpu" => Device::Cpu,
        "cuda" => {
            Device::new_cuda(0).expect("LUX3D_TRELLIS2_TEX_E2E_DEVICE=cuda requires a CUDA device")
        }
        "vulkan" => Device::new_vulkan(0)
            .expect("LUX3D_TRELLIS2_TEX_E2E_DEVICE=vulkan requires a Vulkan device"),
        "wgpu" => {
            Device::new_wgpu(0).expect("LUX3D_TRELLIS2_TEX_E2E_DEVICE=wgpu requires a wgpu device")
        }
        other => panic!("LUX3D_TRELLIS2_TEX_E2E_DEVICE must be cpu|cuda|vulkan|wgpu, got {other}"),
    }
}

/// Env-gated full run (`LUX3D_TRELLIS2_RUN_TEXTURE_E2E=1`): cube mesh +
/// synthetic RGBA condition image -> textured atlas mesh + GLB smoke. Set
/// `LUX3D_TRELLIS2_TEX_E2E_UVS=1` to pass a complete per-corner UV layout
/// (the UV-reuse branch: no unwrap, UVs round-trip through the GLB). See the
/// module doc comment for the exact command and env knobs.
#[test]
fn texture_mesh_run_e2e() {
    if std::env::var_os("LUX3D_TRELLIS2_RUN_TEXTURE_E2E").is_none() {
        eprintln!(
            "SKIP: set LUX3D_TRELLIS2_RUN_TEXTURE_E2E=1 to run the full texturing e2e \
             (see tests/trellis2_texture.rs docs; release build recommended)"
        );
        return;
    }
    let Some(dir) = raw_dir() else {
        eprintln!("SKIP: TRELLIS.2 raw model dir not found (set LUX3D_TRELLIS2_RAW_DIR)");
        return;
    };
    let device = e2e_device();
    let resolution: u32 = std::env::var("LUX3D_TRELLIS2_TEX_E2E_RESOLUTION")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let steps: usize = std::env::var("LUX3D_TRELLIS2_TEX_E2E_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);

    let pipeline = Trellis2Pipeline::load_from_raw_dir(&dir, &device).expect("pipeline");
    let options = Trellis2TexturingOptions {
        seed: 42,
        resolution,
        tex_sampler: Trellis2SamplerOverrides {
            steps: Some(steps),
            ..Trellis2SamplerOverrides::default()
        },
        texture_size: 512,
        preprocess_image: true,
        preprocess_mesh: true,
    };

    // Synthetic condition image: an opaque red square on a transparent
    // 64x64 background (the alpha path skips RMBG-2.0).
    let side = 64_usize;
    let mut rgba = Vec::with_capacity(side * side);
    for y in 0..side {
        for x in 0..side {
            let inside = x >= 16 && x < 48 && y >= 16 && y < 48;
            rgba.push(if inside {
                [0.9_f32, 0.1, 0.1, 1.0]
            } else {
                [0.0, 0.0, 0.0, 0.0]
            });
        }
    }
    let image = Trellis2RawInput {
        original_size: lux3d_core::contracts::SpatialSize::new(64, 64),
        rgba: Some(rgba),
        rgb: None,
    };

    let vertices = cube_vertices();
    let faces = cube_faces();
    // Optional UV-reuse run (`LUX3D_TRELLIS2_TEX_E2E_UVS=1`): distinct
    // per-triangle UV rects in the trimesh-internal convention, passed
    // straight into the pipeline (no xatlas unwrap; the reference
    // `postprocess_mesh` UV branch).
    let corner_uvs = std::env::var_os("LUX3D_TRELLIS2_TEX_E2E_UVS")
        .is_some()
        .then(|| {
            let mut uvs = Vec::with_capacity(faces.len() / 3 * 6);
            for triangle in 0..faces.len() / 3 {
                let base = triangle as f32 * 0.07;
                uvs.extend([base, base, base + 0.06, base, base, base + 0.06]);
            }
            uvs
        });
    let output = pipeline
        .texture_mesh(
            &vertices,
            &faces,
            corner_uvs.as_deref(),
            &image,
            &options,
            &device,
        )
        .expect("texture_mesh e2e");

    // Structural output contract.
    assert_eq!(output.mesh.texture.size, options.texture_size);
    assert!(!output.mesh.uvs.is_empty());
    assert_eq!(output.mesh.vertices.len() % 3, 0);
    assert!(!output.mesh.faces.is_empty());
    assert_eq!(output.mesh.normals.len(), output.mesh.vertices.len());
    if let Some(expected_uvs) = &corner_uvs {
        assert_eq!(
            0, output.mesh.chart_count,
            "input UVs must skip the xatlas unwrap"
        );
        assert_eq!(
            expected_uvs, &output.mesh.uvs,
            "all corners have distinct UVs, so the atlas UV buffer is 1:1"
        );
        assert!(output.mesh.texture.stats.valid_texels > 0);
    }
    let latent_grid = (resolution / 16) as i32;
    assert!(output
        .shape_coords
        .iter()
        .all(|c| c.iter().all(|&v| v >= 0 && v < latent_grid)));
    assert!(!output.tex_coords.is_empty());
    assert_eq!(output.shape_latent.dims(), &[output.shape_coords.len(), 32]);
    // `tex_latent` lives on the latent `shape_coords` (see the output struct
    // docs); `tex_coords` is the decoded full-resolution voxelization, which
    // `texture_mesh` already verifies against the input voxelization.
    assert_eq!(output.tex_latent.dims(), &[output.shape_coords.len(), 32]);
    assert!(output.tex_coords.len() >= output.shape_coords.len());

    // GLB export smoke through the shared writer.
    let out_path = std::env::temp_dir().join(format!(
        "lux3d-trellis2-texture-e2e-{}.glb",
        std::process::id()
    ));
    pipeline
        .export
        .write_textured_glb(&output.mesh, &out_path)
        .expect("textured glb write");
    assert!(out_path.is_file());
    if let Some(expected_uvs) = &corner_uvs {
        let imported = lux3d_core::export::load_mesh(&out_path).expect("read e2e glb");
        let imported_uvs = imported.uvs.expect("e2e GLB must carry TEXCOORD_0");
        assert_eq!(
            expected_uvs.len(),
            imported_uvs.len(),
            "UV count unchanged through bake + GLB export"
        );
        for (expected, actual) in expected_uvs.iter().zip(imported_uvs.iter()) {
            assert!(
                (expected - actual).abs() <= 1e-5,
                "UV {expected} must round-trip, got {actual}"
            );
        }
    }
    let _ = std::fs::remove_file(&out_path);
}
