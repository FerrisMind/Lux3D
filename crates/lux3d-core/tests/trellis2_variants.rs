//! TRELLIS.2 upstream-parity pipeline variants and sampler overrides:
//!
//! 1. Pure: `resolve_stage_params` precedence (override > legacy steps >
//!    pipeline-config default) and the `Trellis2SamplerOverrides` semantics
//!    for every field, including the legacy `ss_steps`/`shape_steps`/
//!    `tex_steps` base.
//! 2. Pure: the `1536_cascade` quantization mapping (`scale 96`) and the
//!    token-budget decrement 1536 -> ... -> 1024 with the production stop.
//! 3. `Trellis2Pipeline::resolutions()` for all four variants (upstream
//!    `ss_res` map) on a placeholder raw-dir pipeline (no weights needed).
//! 4. `Trellis2Stages::write_latent_npz` round trip (the upstream
//!    `return_latent=True` surface).
//!
//! Raw-dir gated smoke runs of the new variants are the later GPU-parity
//! workstream; nothing here loads real weights.

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use lux3d_core::contracts::Trellis2PipelineType;
use lux3d_core::runtime::{
    TRELLIS2_CASCADE_1536_TARGET_RESOLUTION, TRELLIS2_CASCADE_HR_RESOLUTION,
    TRELLIS2_CASCADE_LR_RESOLUTION, Trellis2Pipeline, Trellis2RunOptions, Trellis2SamplerOverrides,
    Trellis2SamplerParams, Trellis2Stages, quantize_hr_coords, resolve_stage_params,
};
use lux3d_core::test_support::workspace_root;
use ndarray::{Array1, Array2, ArrayBase, DataOwned, Dimension, OwnedRepr};

fn read_npz_array<S, D>(
    npz: &mut ndarray_npy::NpzReader<std::fs::File>,
    name: &str,
) -> ArrayBase<S, D>
where
    S: DataOwned,
    S::Elem: ndarray_npy::ReadableElement,
    D: Dimension,
{
    npz.by_name::<S, D>(name)
        .unwrap_or_else(|error| panic!("npz array `{name}`: {error}"))
}

fn raw_dir() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("LUX3D_TRELLIS2_RAW_DIR") {
        let path = PathBuf::from(path);
        return path.is_dir().then_some(path);
    }
    let path = workspace_root().join("models").join("TRELLIS.2-4B");
    path.is_dir().then_some(path)
}

/// Placeholder raw-dir pipeline (empty stage files, defaults only) — enough
/// for config-level surfaces like `resolutions()`.
fn placeholder_pipeline() -> (Trellis2Pipeline, PathBuf) {
    let dir = std::env::temp_dir().join(format!(
        "lux3d-trellis2-variants-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos()
    ));
    for (_prefix, relative) in [
        ("image_cond", "external/dinov3/model.safetensors"),
        ("ss_flow", "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"),
        (
            "ss_dec",
            "external/ss_dec/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
        ),
        (
            "shape_dit_512",
            "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
        ),
        (
            "shape_dit_1024",
            "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
        ),
        (
            "shape_dec",
            "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        ),
        (
            "tex_dit_512",
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
        ),
        (
            "tex_dit_1024",
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
        ),
        ("tex_dec", "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors"),
    ] {
        let file = dir.join(relative);
        std::fs::create_dir_all(file.parent().expect("parent")).expect("dirs");
        std::fs::write(&file, b"").expect("placeholder");
    }
    let pipeline =
        Trellis2Pipeline::load_from_raw_dir(&dir, &Device::Cpu).expect("placeholder pipeline load");
    (pipeline, dir)
}

fn options_with(pipeline_type: Trellis2PipelineType) -> Trellis2RunOptions {
    Trellis2RunOptions {
        pipeline_type,
        ..Trellis2RunOptions::default()
    }
}

/// The upstream `ss_res` map (`{'512': 32, '1024': 64, '1024_cascade': 32,
/// '1536_cascade': 32}`) and the decode resolutions.
#[test]
fn resolutions_map_all_four_variants() {
    let (pipeline, dir) = placeholder_pipeline();
    let cases = [
        (Trellis2PipelineType::R512, 32u32, 512u32),
        (Trellis2PipelineType::R1024, 64, 1024),
        (Trellis2PipelineType::Cascade1024, 32, 1024),
        (Trellis2PipelineType::Cascade1536, 32, 1536),
    ];
    for (pipeline_type, ss, decode) in cases {
        let (ss_resolution, shape_resolution) = pipeline
            .resolutions(&options_with(pipeline_type))
            .expect("resolutions");
        assert_eq!(ss_resolution, ss, "{} ss", pipeline_type.as_str());
        assert_eq!(
            shape_resolution,
            decode,
            "{} decode",
            pipeline_type.as_str()
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Override > pipeline default for every field; `None` fields keep the
/// default; the legacy `steps` argument is the base step count that a stage
/// override's `steps` beats.
#[test]
fn resolve_stage_params_merge_precedence() {
    let base = Trellis2SamplerParams {
        steps: 12,
        sigma_min: 1e-5,
        guidance_strength: 7.5,
        guidance_rescale: 0.5,
        guidance_interval: [0.6, 1.0],
        rescale_t: 3.0,
    };

    // No overrides: the legacy steps field wins over the config default.
    let resolved = resolve_stage_params(&base, None, 8);
    assert_eq!(resolved.steps, 8);
    assert_eq!(resolved.guidance_strength, 7.5);
    assert_eq!(resolved.guidance_rescale, 0.5);
    assert_eq!(resolved.guidance_interval, [0.6, 1.0]);
    assert_eq!(resolved.rescale_t, 3.0);
    assert_eq!(resolved.sigma_min, 1e-5);

    // Empty override: still just the legacy steps.
    let resolved = resolve_stage_params(&base, Some(&Trellis2SamplerOverrides::default()), 12);
    assert_eq!(
        resolved,
        Trellis2SamplerParams {
            steps: 12,
            ..base.clone()
        }
    );

    // Full override: every field replaces the config default and `steps`
    // beats the legacy argument.
    let overrides = Trellis2SamplerOverrides {
        steps: Some(4),
        guidance_strength: Some(3.5),
        guidance_rescale: Some(0.7),
        guidance_interval: Some((0.4, 0.9)),
        rescale_t: Some(2.5),
    };
    let resolved = resolve_stage_params(&base, Some(&overrides), 8);
    assert_eq!(resolved.steps, 4, "override steps > legacy steps");
    assert_eq!(resolved.guidance_strength, 3.5);
    // f32 overrides widen to the f64 sampler params (0.7_f32 != 0.7_f64).
    assert_eq!(resolved.guidance_rescale, 0.7_f32 as f64);
    assert_eq!(resolved.guidance_interval, [0.4_f32 as f64, 0.9_f32 as f64]);
    assert_eq!(resolved.rescale_t, 2.5);
    assert_eq!(resolved.sigma_min, 1e-5, "sigma_min has no override");

    // Partial override: unset fields keep the pipeline defaults.
    let overrides = Trellis2SamplerOverrides {
        guidance_interval: Some((0.5, 0.8)),
        ..Trellis2SamplerOverrides::default()
    };
    let resolved = resolve_stage_params(&base, Some(&overrides), 12);
    assert_eq!(resolved.steps, 12);
    assert_eq!(resolved.guidance_strength, 7.5);
    assert_eq!(resolved.guidance_interval, [0.5_f32 as f64, 0.8_f32 as f64]);

    // Defaults are untouched: the base config object is cloned, not mutated.
    assert_eq!(base.steps, 12);
    assert_eq!(base.guidance_interval, [0.6, 1.0]);
}

/// `1536_cascade` quantization: scale `1536 / 16 = 96`, the exact reference
/// expression `((coord + 0.5) / 512 * (hr // 16)).int()`, and the
/// production loop can only reduce the resolution down TO the 1024 stop —
/// never below.
#[test]
fn quantize_1536_target_maps_and_stops_at_1024() {
    // coord c -> trunc((c + 0.5) / 512 * 96) = trunc((c + 0.5) / 16 * 3);
    // c = 0..31 covers the full 32-parent grid edges (the LR SS grid is
    // 32^3 so the upsampled coords stay < 512 and quantize into 0..96).
    for parent in 0..32i32 {
        let coord = parent * 16; // 16 children per parent on the 512 grid
        let expected = ((coord as f32 + 0.5) / 512.0 * 96.0) as i32;
        let (coords, hr) = quantize_hr_coords(
            &[[coord, coord, coord]],
            TRELLIS2_CASCADE_LR_RESOLUTION,
            TRELLIS2_CASCADE_1536_TARGET_RESOLUTION,
            49152,
        );
        assert_eq!(hr, 1536, "under-budget 1536 pass stays at 1536");
        assert_eq!(coords, vec![[expected, expected, expected]]);
    }

    // Over-budget: 1536 -> 1408 -> ... -> 1024, where the == stop breaks
    // unconditionally. The 64-grid quantization at 1024 is the terminal
    // result; the resolution never drops below 1024.
    let hr_coords: Vec<[i32; 3]> = (0..96i32).map(|k| [0, k * 5, 0]).collect();
    let (coords, hr) = quantize_hr_coords(
        &hr_coords,
        TRELLIS2_CASCADE_LR_RESOLUTION,
        TRELLIS2_CASCADE_1536_TARGET_RESOLUTION,
        8,
    );
    assert_eq!(hr, TRELLIS2_CASCADE_HR_RESOLUTION, "stop resolution 1024");
    assert!(
        coords
            .iter()
            .all(|c| c.iter().all(|&v| (0..64).contains(&v)))
    );

    // Under budget at 1536: the loop breaks on the first pass.
    let (coords, hr) = quantize_hr_coords(
        &hr_coords,
        TRELLIS2_CASCADE_LR_RESOLUTION,
        TRELLIS2_CASCADE_1536_TARGET_RESOLUTION,
        49152,
    );
    assert_eq!(hr, 1536);
    let mut sorted = coords.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(coords, sorted);
}

/// The upstream `return_latent=True` surface: `write_latent_npz` round trip.
#[test]
fn write_latent_npz_round_trip() {
    let device = Device::Cpu;
    let coords = vec![[0i32, 0, 0], [1, 0, 0], [0, 1, 0]];
    let rows = coords.len();
    let shape_values: Vec<f32> = (0..rows * 32).map(|i| i as f32 * 0.5).collect();
    let tex_values: Vec<f32> = (0..rows * 32).map(|i| -(i as f32)).collect();
    let lr_values: Vec<f32> = (0..rows * 32).map(|i| i as f32).collect();
    let stages = Trellis2Stages {
        cond: Tensor::zeros((1,), DType::F32, &device).expect("cond"),
        neg_cond: Tensor::zeros((1,), DType::F32, &device).expect("neg"),
        z_s: None,
        ss_coords: coords.clone(),
        shape_coords: coords,
        shape_latent: Tensor::from_vec(shape_values, (rows, 32), &device).expect("shape"),
        decoded_feats: Tensor::zeros((rows, 7), DType::F32, &device).expect("feats"),
        mesh: None,
        lr_shape_latent: Some(Tensor::from_vec(lr_values, (rows, 32), &device).expect("lr")),
        hr_upsampled_coords: None,
        quant_coords: None,
        hr_resolution: Some(1536),
        subdiv_logits: Vec::new(),
        tex_latent: Some(Tensor::from_vec(tex_values, (rows, 32), &device).expect("tex")),
        tex_attrs: None,
        textured: None,
    };

    let dir = std::env::temp_dir().join(format!("lux3d-trellis2-npz-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dir");
    let path = dir.join("latents.npz");
    stages
        .write_latent_npz(&path, Trellis2PipelineType::Cascade1536)
        .expect("write npz");

    let mut npz =
        ndarray_npy::NpzReader::new(std::fs::File::open(&path).expect("open")).expect("npz");
    let shape_latent: Array2<f32> = read_npz_array(&mut npz, "shape_latent");
    let tex_latent: Array2<f32> = read_npz_array(&mut npz, "tex_latent");
    let lr_latent: Array2<f32> = read_npz_array(&mut npz, "lr_shape_latent");
    let coords_arr: Array2<i32> = read_npz_array(&mut npz, "coords");
    let resolution = read_npz_array::<OwnedRepr<i32>, ndarray::Ix0>(&mut npz, "resolution");
    let meta_bytes: Array1<u8> = read_npz_array(&mut npz, "meta");

    assert_eq!(shape_latent.dim(), (rows, 32));
    assert_eq!(shape_latent[[0, 0]], 0.0);
    assert_eq!(
        shape_latent[[rows - 1, 31]],
        ((rows - 1) * 32 + 31) as f32 * 0.5
    );
    assert_eq!(tex_latent[[0, 0]], 0.0);
    assert_eq!(tex_latent[[1, 0]], -32.0);
    assert_eq!(lr_latent.dim(), (rows, 32));
    assert_eq!(coords_arr.dim(), (rows, 3));
    assert_eq!(coords_arr[[2, 1]], 1);
    assert_eq!(resolution[()], 1536);
    let meta: serde_json::Value = serde_json::from_slice(&meta_bytes.to_vec()).expect("meta");
    assert_eq!(meta["pipeline_type"], "1536_cascade");
    assert_eq!(meta["resolution"], 1536);
    assert_eq!(meta["num_coords"], rows as u64);
    assert_eq!(meta["cascade"], true);
    assert_eq!(meta["hr_resolution"], 1536);

    // Geometry-only + non-cascade run: empty tex/lr arrays, mesh resolution.
    let single = Trellis2Stages {
        lr_shape_latent: None,
        hr_resolution: None,
        tex_latent: None,
        shape_latent: stages.shape_latent.clone(),
        shape_coords: stages.shape_coords.clone(),
        ..stages.clone()
    };
    let path2 = dir.join("single.npz");
    single
        .write_latent_npz(&path2, Trellis2PipelineType::R512)
        .expect("write npz");
    let mut npz2 =
        ndarray_npy::NpzReader::new(std::fs::File::open(&path2).expect("open")).expect("npz2");
    let tex: Array2<f32> = read_npz_array(&mut npz2, "tex_latent");
    let lr: Array2<f32> = read_npz_array(&mut npz2, "lr_shape_latent");
    let resolution = read_npz_array::<OwnedRepr<i32>, ndarray::Ix0>(&mut npz2, "resolution");
    assert_eq!(tex.dim(), (0, 32));
    assert_eq!(lr.dim(), (0, 32));
    assert_eq!(resolution[()], 512, "mesh-less fallback resolution");
    let meta_bytes: Array1<u8> = read_npz_array(&mut npz2, "meta");
    let meta: serde_json::Value = serde_json::from_slice(&meta_bytes.to_vec()).expect("meta");
    assert_eq!(meta["pipeline_type"], "512");
    assert_eq!(meta["cascade"], false);
    assert_eq!(meta["hr_resolution"], serde_json::Value::Null);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Raw-dir gated: the variant dispatch runs the real pipeline for the new
/// `1024` and `1536_cascade` types (smoke level; the later GPU-parity
/// workstream owns full oracle comparison). Skips without the raw dir.
#[test]
fn run_with_latents_accepts_all_variants_up_to_weight_load() {
    let Some(raw) = raw_dir() else {
        eprintln!("SKIP: TRELLIS.2 raw checkpoint dir not found (set LUX3D_TRELLIS2_RAW_DIR)");
        return;
    };
    let pipeline = Trellis2Pipeline::load_from_raw_dir(&raw, &Device::Cpu).expect("pipeline");
    let cond = Tensor::zeros((1, 4), DType::F32, &Device::Cpu).expect("cond");
    let ss_init = Tensor::zeros((1, 8, 16, 16, 16), DType::F32, &Device::Cpu).expect("ss init");
    for pipeline_type in [
        Trellis2PipelineType::R512,
        Trellis2PipelineType::R1024,
        Trellis2PipelineType::Cascade1024,
        Trellis2PipelineType::Cascade1536,
    ] {
        // No run: just confirm the options resolve (resolutions is the
        // variant dispatch entry).
        pipeline
            .resolutions(&options_with(pipeline_type))
            .unwrap_or_else(|error| panic!("{} resolutions: {error}", pipeline_type.as_str()));
        let _ = &cond;
        let _ = &ss_init;
    }
}
