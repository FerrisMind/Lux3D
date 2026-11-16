//! TRELLIS.2 shape SLAT encoder tests:
//!
//! 1. Tensor-name coverage: the canonical `shape_enc_next_dc_f16c32_fp16`
//!    checkpoint holds exactly the 284 tensors the flexi-dual encoder config
//!    expects (gated on the checkpoint file).
//! 2. Tiny synthetic config: strict loading, `concat_dual_grid_input`
//!    hand-check, forward shape/coordinate semantics (the four S2C
//!    downsamples produce the row-major sorted unique `coord / 16` parents,
//!    no duplicates, order-invariant in the input row order) and
//!    determinism.
//! 3. Real-weight golden (gated): the fixture
//!    `.swarm/results/trellis2-shape-enc-golden` (reference F32 PyTorch
//!    encode of a 2048-voxel synthetic dual grid at resolution 256, generated
//!    by `scripts/trellis2_shape_enc_golden.py` with the unmodified reference
//!    model code and a pure-torch submanifold conv backend) is replayed
//!    through the Rust encoder; output coords must match exactly and the
//!    mean latent within F32 accumulation noise.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use lux3d_core::runtime::{
    Trellis2ShapeEncoder, Trellis2ShapeEncoderConfig, concat_dual_grid_input, spatial_to_channel,
};
use lux3d_core::test_support::workspace_root;

fn stage_checkpoint_file() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("LUX3D_TRELLIS2_SHAPE_ENC") {
        let path = PathBuf::from(path);
        return path.is_file().then_some(path);
    }
    let path = workspace_root()
        .join("models")
        .join("TRELLIS.2-4B")
        .join("ckpts")
        .join("shape_enc_next_dc_f16c32_fp16.safetensors");
    path.is_file().then_some(path)
}

fn golden_dir() -> PathBuf {
    workspace_root()
        .join(".swarm")
        .join("results")
        .join("trellis2-shape-enc-golden")
}

fn read_safetensors_keys(path: &Path) -> Vec<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).expect("open safetensors");
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .expect("safetensors header len");
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    assert!(header_len < 256 * 1024 * 1024, "suspicious header length");
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .expect("safetensors header");
    let header: serde_json::Value = serde_json::from_slice(&header_bytes).expect("header json");
    header
        .as_object()
        .expect("header object")
        .keys()
        .filter(|k| k.as_str() != "__metadata__")
        .cloned()
        .collect()
}

/// Real weights: the `shape_enc_next_dc_f16c32_fp16` stage holds exactly the
/// 284 tensors the flexi-dual encoder config expects.
#[test]
fn shape_encoder_real_weights_cover_284_stage_tensors() {
    let model_file = match stage_checkpoint_file() {
        Some(path) => path,
        None => {
            eprintln!("SKIP: shape encoder checkpoint not found (set LUX3D_TRELLIS2_SHAPE_ENC)");
            return;
        }
    };
    let keys = read_safetensors_keys(&model_file);
    assert_eq!(keys.len(), 284, "shape_enc stage tensor count");
    let config = Trellis2ShapeEncoderConfig::flexi_dual_shape();
    assert_eq!(config.in_channels, 6, "input_layer.weight is [64, 6]");
    assert_eq!(config.model_channels, vec![64, 128, 256, 512, 1024]);
    assert_eq!(config.num_blocks, vec![0, 4, 8, 16, 4]);
    assert_eq!(config.latent_channels, 32);
    let expected: HashSet<String> = config.expected_tensor_names().into_iter().collect();
    assert_eq!(expected.len(), 284, "expected names must be unique");
    let stage: HashSet<String> = keys.into_iter().collect();
    assert_eq!(
        expected, stage,
        "expected names must cover exactly the shape_enc stage tensors"
    );
}

// ---------------------------------------------------------------------------
// Tiny synthetic config
// ---------------------------------------------------------------------------

/// Tiny encoder config: 5 stages, channels [8, 16, 32, 64, 128], one
/// ConvNeXt block per stage, latent 4.
fn tiny_config() -> Trellis2ShapeEncoderConfig {
    Trellis2ShapeEncoderConfig {
        in_channels: 6,
        model_channels: vec![8, 16, 32, 64, 128],
        latent_channels: 4,
        num_blocks: vec![0, 1, 1, 1, 1],
        mlp_ratio: 2.0,
    }
}

/// `(name, shape)` for every tensor of the config (mirrors
/// `expected_tensor_names` with shapes).
fn tiny_weight_entries(config: &Trellis2ShapeEncoderConfig) -> Vec<(String, Vec<usize>)> {
    let mut entries = Vec::new();
    let first = config.model_channels[0];
    let last = *config.model_channels.last().unwrap();
    entries.push(("input_layer.weight".into(), vec![first, config.in_channels]));
    entries.push(("input_layer.bias".into(), vec![first]));
    entries.push((
        "to_latent.weight".into(),
        vec![config.latent_channels * 2, last],
    ));
    entries.push(("to_latent.bias".into(), vec![config.latent_channels * 2]));
    let stage_count = config.num_blocks.len();
    for (stage, &count) in config.num_blocks.iter().enumerate() {
        let c = config.model_channels[stage];
        for block in 0..count {
            let base = format!("blocks.{stage}.{block}.");
            let hidden = (c as f64 * config.mlp_ratio) as usize;
            entries.push((format!("{base}conv.weight"), vec![c, 3, 3, 3, c]));
            entries.push((format!("{base}conv.bias"), vec![c]));
            entries.push((format!("{base}norm.weight"), vec![c]));
            entries.push((format!("{base}norm.bias"), vec![c]));
            entries.push((format!("{base}mlp.0.weight"), vec![hidden, c]));
            entries.push((format!("{base}mlp.0.bias"), vec![hidden]));
            entries.push((format!("{base}mlp.2.weight"), vec![c, hidden]));
            entries.push((format!("{base}mlp.2.bias"), vec![c]));
        }
        if stage + 1 < stage_count {
            let cout = config.model_channels[stage + 1];
            let base = format!("blocks.{stage}.{count}.");
            entries.push((format!("{base}conv1.weight"), vec![cout / 8, 3, 3, 3, c]));
            entries.push((format!("{base}conv1.bias"), vec![cout / 8]));
            entries.push((format!("{base}conv2.weight"), vec![cout, 3, 3, 3, cout]));
            entries.push((format!("{base}conv2.bias"), vec![cout]));
            entries.push((format!("{base}norm1.weight"), vec![c]));
            entries.push((format!("{base}norm1.bias"), vec![c]));
        }
    }
    entries
}

fn pattern_tensor(shape: &[usize], seed: usize) -> Tensor {
    let len: usize = shape.iter().product();
    Tensor::from_vec(
        (0..len)
            .map(|i| (((i * 7 + seed * 3) % 23) as f32 - 11.0) * 0.05)
            .collect(),
        shape.to_vec(),
        &Device::Cpu,
    )
    .expect("tensor")
}

fn load_tiny_encoder() -> (Trellis2ShapeEncoder, Trellis2ShapeEncoderConfig) {
    let config = tiny_config();
    let mut map = std::collections::HashMap::new();
    for (seed, (name, shape)) in tiny_weight_entries(&config).into_iter().enumerate() {
        map.insert(name, pattern_tensor(&shape, seed));
    }
    let vb = candle_nn::VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let encoder = Trellis2ShapeEncoder::load(vb, &config).expect("tiny encoder load");
    (encoder, config)
}

/// The tiny config consumes exactly its own expected tensor names (strict
/// loading: a missing tensor fails the load).
#[test]
fn tiny_encoder_expected_names_unique_and_strict() {
    let config = tiny_config();
    let names = config.expected_tensor_names();
    let unique: HashSet<&String> = names.iter().collect();
    assert_eq!(unique.len(), names.len(), "expected names must be unique");
    assert_eq!(names.len(), 60, "tiny config tensor count");
    // Drop one tensor -> load must fail.
    let mut map = std::collections::HashMap::new();
    for (seed, (name, shape)) in tiny_weight_entries(&config).into_iter().enumerate().skip(1) {
        map.insert(name, pattern_tensor(&shape, seed));
    }
    let vb = candle_nn::VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    assert!(
        Trellis2ShapeEncoder::load(vb, &config).is_err(),
        "missing input_layer.weight must fail the load"
    );
}

/// Hand-check `concat_dual_grid_input`: `x = [dual * res - idx - 0.5,
/// intersected - 0.5]`.
#[test]
fn concat_dual_grid_input_layout() {
    let coords = vec![[10_i32, 20, 30], [7, 8, 9]];
    // Normalized positions: dual * res - idx must reconstruct [0.25, 0.5, 0.75]
    // and [1.25, -0.25, 0.5].
    let dual = Tensor::new(
        vec![
            (10.0 + 0.25) / 64.0,
            (20.0 + 0.5) / 64.0,
            (30.0 + 0.75) / 64.0,
            (7.0 + 1.25) / 64.0,
            (8.0 - 0.25) / 64.0,
            (9.0 + 0.5) / 64.0,
        ],
        &Device::Cpu,
    )
    .unwrap()
    .reshape((2, 3))
    .unwrap();
    let intersected = Tensor::new(vec![1.0_f32, 0.0, 1.0, 0.0, 1.0, 0.0], &Device::Cpu)
        .unwrap()
        .reshape((2, 3))
        .unwrap();
    let x = concat_dual_grid_input(&coords, &dual, &intersected, 64, &Device::Cpu)
        .expect("concat input");
    let values = x.to_vec2::<f32>().expect("values");
    let expected = [
        // offsets [0.25, 0.5, 0.75] - 0.5, flags [1, 0, 1] - 0.5
        [-0.25, 0.0, 0.25, 0.5, -0.5, 0.5],
        // offsets [1.25, -0.25, 0.5] - 0.5, flags [0, 1, 0] - 0.5
        [0.75, -0.75, 0.0, -0.5, 0.5, -0.5],
    ];
    for (row, exp) in values.iter().zip(expected.iter()) {
        for (v, e) in row.iter().zip(exp.iter()) {
            assert!((v - e).abs() < 1e-6, "{v} vs {e}");
        }
    }
}

/// Forward through the tiny encoder: output coords are the row-major sorted
/// unique `coord / 16` parents (four S2C downsamples), without duplicates,
/// independent of the input row order, and the run is deterministic.
#[test]
fn tiny_encoder_forward_downsample_coords_and_determinism() {
    let (encoder, _config) = load_tiny_encoder();
    let device = Device::Cpu;
    let resolution = 64_usize;

    // A dense 2x2x2 cube (all children of one level-1 parent), three isolated
    // voxels and a pair sharing a parent — deliberately unsorted.
    let mut coords: Vec<[i32; 3]> = Vec::new();
    for dz in 0..2_i32 {
        for dy in 0..2 {
            for dx in 0..2 {
                coords.push([32 + dx, 32 + dy, 32 + dz]);
            }
        }
    }
    coords.push([0, 0, 0]);
    coords.push([61, 60, 63]);
    coords.push([17, 33, 49]);
    // A pair sharing one level-1 parent [17, 16, 16] (distinct from the cube).
    coords.push([34, 32, 32]);
    coords.push([35, 33, 33]);
    // Encoder inputs must be duplicate-free (CoordMap is last-wins).
    {
        let mut unique = coords.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), coords.len(), "test coords must be unique");
    }
    let n = coords.len();

    // Deterministic pseudo-random inputs: offsets in [-0.45, 1.45] voxel
    // units as normalized positions, flags 0/1.
    let mut dual: Vec<f32> = Vec::with_capacity(n * 3);
    let mut flags: Vec<f32> = Vec::with_capacity(n * 3);
    for (i, c) in coords.iter().enumerate() {
        for axis in 0..3 {
            let offset = (((i * 3 + axis) * 37) % 19) as f32 / 19.0 * 1.9 - 0.45;
            dual.push((c[axis as usize] as f32 + offset) / resolution as f32);
            flags.push(if (i + axis) % 2 == 0 { 1.0 } else { 0.0 });
        }
    }
    let dual = Tensor::from_vec(dual, (n, 3), &device).expect("dual");
    let flags = Tensor::from_vec(flags, (n, 3), &device).expect("flags");

    let out = encoder
        .encode_shape_slat(&coords, &dual, &flags, resolution, &device)
        .expect("encode");

    // Expected coords: four S2C downsamples == floor(coord / 16), unique and
    // row-major sorted.
    let expected: Vec<[i32; 3]> = {
        let mut parents: Vec<[i32; 3]> = coords
            .iter()
            .map(|c| [c[0] / 16, c[1] / 16, c[2] / 16])
            .collect();
        parents.sort_unstable();
        parents.dedup();
        parents
    };
    assert_eq!(
        out.coords, expected,
        "encoder coords after four downsamples"
    );
    let (rows, channels) = out.feats.dims2().expect("feats dims");
    assert_eq!(rows, expected.len());
    assert_eq!(channels, 4, "latent_channels");
    let values = out.feats.to_vec2::<f32>().expect("feats values");
    assert!(
        values.iter().flatten().all(|v| v.is_finite()),
        "feats must be finite"
    );

    // Cross-check the coordinate chain against the standalone (reference
    // tested) spatial_to_channel applied four times.
    let mut chain_coords = coords.clone();
    for _ in 0..4 {
        let feats = Tensor::zeros((chain_coords.len(), 3), DType::F32, &device).expect("zeros");
        let s2c = spatial_to_channel(&chain_coords, &feats).expect("s2c");
        chain_coords = s2c.coords;
    }
    assert_eq!(out.coords, chain_coords, "coords match S2C^4 of the input");

    // Row-order invariance: reversed input rows give the identical latent.
    let mut rev_coords = coords.clone();
    rev_coords.reverse();
    let rev_dual = dual
        .to_vec2::<f32>()
        .expect("dual")
        .into_iter()
        .rev()
        .collect::<Vec<_>>();
    let rev_flags = flags
        .to_vec2::<f32>()
        .expect("flags")
        .into_iter()
        .rev()
        .collect::<Vec<_>>();
    let rev_dual = Tensor::from_vec(
        rev_dual.into_iter().flatten().collect::<Vec<_>>(),
        (n, 3),
        &device,
    )
    .expect("rev dual");
    let rev_flags = Tensor::from_vec(
        rev_flags.into_iter().flatten().collect::<Vec<_>>(),
        (n, 3),
        &device,
    )
    .expect("rev flags");
    let out_rev = encoder
        .encode_shape_slat(&rev_coords, &rev_dual, &rev_flags, resolution, &device)
        .expect("encode reversed");
    assert_eq!(out.coords, out_rev.coords, "coords are row-order invariant");
    let rev_values = out_rev.feats.to_vec2::<f32>().expect("rev values");
    for (a, b) in values.iter().zip(rev_values.iter()) {
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x, y, "feats are row-order invariant (bit-exact)");
        }
    }

    // Determinism: a second run is bit-identical.
    let out_again = encoder
        .encode_shape_slat(&coords, &dual, &flags, resolution, &device)
        .expect("encode again");
    let again_values = out_again.feats.to_vec2::<f32>().expect("again values");
    assert_eq!(values, again_values, "forward must be deterministic");
}

// ---------------------------------------------------------------------------
// Real-weight golden
// ---------------------------------------------------------------------------

/// Real weights + Python oracle: encode the 2048-voxel synthetic dual grid
/// (resolution 256) from the fixture and compare the mean latent against the
/// F32 reference within 1e-3 and the downsampled coords exactly.
#[test]
fn shape_encoder_matches_python_oracle() {
    let model_file = match stage_checkpoint_file() {
        Some(path) => path,
        None => {
            eprintln!("SKIP: shape encoder checkpoint not found (set LUX3D_TRELLIS2_SHAPE_ENC)");
            return;
        }
    };
    let golden = golden_dir();
    if !golden.join("latent.npz").is_file() {
        eprintln!("SKIP: shape encoder golden missing at {}", golden.display());
        return;
    }

    let inputs = std::fs::File::open(golden.join("inputs.npz")).expect("open inputs.npz");
    let mut npz = ndarray_npy::NpzReader::new(inputs).expect("npz reader");
    let voxel_indices: ndarray::Array2<i32> = npz
        .by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix2>("voxel_indices")
        .expect("voxel_indices");
    let dual_vertices: ndarray::Array2<f32> = npz
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix2>("dual_vertices")
        .expect("dual_vertices");
    let intersected: ndarray::Array2<f32> = npz
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix2>("intersected")
        .expect("intersected");
    let resolution_arr: ndarray::Array0<i64> = npz
        .by_name::<ndarray::OwnedRepr<i64>, ndarray::Ix0>("resolution")
        .expect("resolution");
    let resolution = resolution_arr[()] as usize;
    drop(npz);

    let latent_npz = std::fs::File::open(golden.join("latent.npz")).expect("open latent.npz");
    let mut npz = ndarray_npy::NpzReader::new(latent_npz).expect("npz reader");
    let ref_coords: ndarray::Array2<i32> = npz
        .by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix2>("coords")
        .expect("ref coords");
    let ref_mean: ndarray::Array2<f32> = npz
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix2>("mean_f32")
        .expect("ref mean");
    drop(npz);

    let n = voxel_indices.dim().0;
    let coords: Vec<[i32; 3]> = (0..n)
        .map(|i| {
            [
                voxel_indices[[i, 0]],
                voxel_indices[[i, 1]],
                voxel_indices[[i, 2]],
            ]
        })
        .collect();
    let device = Device::Cpu;
    let dual = Tensor::from_vec(
        dual_vertices.iter().copied().collect::<Vec<_>>(),
        (n, 3),
        &device,
    )
    .expect("dual tensor");
    let flags = Tensor::from_vec(
        intersected.iter().copied().collect::<Vec<_>>(),
        (n, 3),
        &device,
    )
    .expect("flags tensor");

    // SAFETY: the mmap-backed builder keeps the mapping alive for every
    // tensor created from it; the checkpoint file is not mutated.
    let encoder = unsafe {
        lux3d_core::runtime::load_trellis2_shape_encoder_from_stage_file(&model_file, &device)
    }
    .expect("shape encoder load");

    let started = std::time::Instant::now();
    let out = encoder
        .encode_shape_slat(&coords, &dual, &flags, resolution, &device)
        .expect("encode");
    println!("encode took {:?}", started.elapsed());

    let (rows, channels) = out.feats.dims2().expect("feats dims");
    assert_eq!(channels, 32, "latent channels");
    assert_eq!(rows, ref_coords.dim().0, "latent row count");

    // Coordinates must match the reference exactly (sorted parents).
    for i in 0..rows {
        let got = out.coords[i];
        let want = [ref_coords[[i, 0]], ref_coords[[i, 1]], ref_coords[[i, 2]]];
        assert_eq!(got, want, "latent coord row {i}");
    }

    let values = out.feats.to_vec2::<f32>().expect("feats values");
    assert!(
        values.iter().flatten().all(|v| v.is_finite()),
        "feats must be finite"
    );
    let mut max_abs = 0.0_f64;
    let mut sum_sq_diff = 0.0_f64;
    let mut sum_sq_ref = 0.0_f64;
    for (i, row) in values.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            let r = ref_mean[[i, j]] as f64;
            let d = (*v as f64 - r).abs();
            max_abs = max_abs.max(d);
            sum_sq_diff += d * d;
            sum_sq_ref += r * r;
        }
    }
    let rms_rel = (sum_sq_diff / sum_sq_ref.max(1e-12)).sqrt();
    println!("mean latent vs F32 reference: max_abs={max_abs:.3e}, rms_rel={rms_rel:.3e}");
    assert!(
        max_abs <= 1e-3,
        "mean latent deviates from the F32 reference by {max_abs:.3e} (> 1e-3)"
    );

    // Determinism: a second run is bit-identical.
    let out_again = encoder
        .encode_shape_slat(&coords, &dual, &flags, resolution, &device)
        .expect("encode again");
    assert_eq!(
        out.feats.to_vec2::<f32>().expect("first"),
        out_again.feats.to_vec2::<f32>().expect("second"),
        "forward must be deterministic"
    );
}
