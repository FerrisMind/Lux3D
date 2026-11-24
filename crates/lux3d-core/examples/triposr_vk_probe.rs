//! TripoSR per-stage numerical bisect probe (RX 570 Vulkan debugging).
//!
//! Runs the TripoSR pipeline stage by stage on one device, reports
//! finiteness/extrema for every intermediate, and optionally dumps them to a
//! `.npz` for cross-device comparison (e.g. vulkan vs cpu):
//!
//! ```text
//! CANDLE_VULKAN_DEVICE_NAME=570 cargo run --release -p lux3d-core --features vulkan -- \
//!   example triposr_vk_probe --device vulkan --model-path <dir> --source <png> \
//!   --dump rx570.npz --grid 32
//! cargo run --release -p lux3d-core --features vulkan -- \
//!   example triposr_vk_probe --device cpu --model-path <dir> --source <png> \
//!   --dump cpu.npz --grid 32
//! cargo run --release -p lux3d-core --features vulkan -- \
//!   example triposr_vk_probe --compare rx570.npz cpu.npz
//! ```

use std::path::PathBuf;

use candle_core::{Device, IndexOp, Tensor};
use lux3d_core::ModelAssetOptions;

fn tensor_stats(name: &str, t: &Tensor) -> anyhow::Result<(f64, f64, usize, usize)> {
    let flat = t
        .to_device(&Device::Cpu)?
        .contiguous()?
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let (mut max_abs, mut nan, mut inf) = (0f64, 0usize, 0usize);
    for &v in &flat {
        if v.is_nan() {
            nan += 1;
        } else if v.is_infinite() {
            inf += 1;
        } else {
            max_abs = max_abs.max(v.abs() as f64);
        }
    }
    println!(
        "[stage] {name}: shape={:?} nan={nan} inf={inf} max_abs={max_abs:.6}",
        t.shape().dims()
    );
    Ok((max_abs, 0.0, nan, inf))
}

fn dump_tensor(
    npz: &mut ndarray_npy::NpzWriter<std::fs::File>,
    name: &str,
    t: &Tensor,
) -> anyhow::Result<()> {
    let dims = t.shape().dims().to_vec();
    let flat = t
        .to_device(&Device::Cpu)?
        .contiguous()?
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&dims), flat)?;
    npz.add_array(name, &arr)?;
    Ok(())
}

fn run(device_name: &str, model_path: &str, source: &str, dump: Option<String>, grid: u32) -> anyhow::Result<()> {
    let device = match device_name {
        "cpu" => Device::Cpu,
        "vulkan" => Device::new_vulkan(0)?,
        other => anyhow::bail!("unknown device `{other}`"),
    };
    println!("[device] {:?}", device);
    let pipeline = lux3d_core::runtime::TripoSrPipeline::load(ModelAssetOptions {
        canonical_dir: Some(PathBuf::from(model_path)),
        cache_dir: None,
    })?;

    let mut npz = match &dump {
        Some(path) => Some(ndarray_npy::NpzWriter::new(std::fs::File::create(path)?)),
        None => None,
    };

    let prepared = pipeline.prepare_inputs_from_path(std::path::Path::new(source), &device)?;
    tensor_stats("preprocessed_image", &prepared.preprocessed_image)?;

    let (image_tokens, seed_tokens, backbone_tokens, detokenized, scene_codes) =
        pipeline.scene_codes(&prepared.preprocessed_image)?;
    tensor_stats("image_tokens", &image_tokens)?;
    tensor_stats("triplane_seed_tokens", &seed_tokens)?;
    tensor_stats("backbone_tokens", &backbone_tokens)?;
    tensor_stats("detokenized_triplanes", &detokenized)?;
    tensor_stats("scene_codes", &scene_codes)?;

    // Triplane query on a coarse grid: features (triplane sampler output) and
    // density (NeRF MLP + exp). Comparing these two across devices bisects
    // sampler vs MLP.
    let scene_code = scene_codes.i(0)?;
    let mut positions = Vec::new();
    let g = grid as usize;
    for x in 0..g {
        for y in 0..g {
            for z in 0..g {
                positions.push((x as f32 / (g - 1) as f32) * 2.0 * 0.87 - 0.87);
                positions.push((y as f32 / (g - 1) as f32) * 2.0 * 0.87 - 0.87);
                positions.push((z as f32 / (g - 1) as f32) * 2.0 * 0.87 - 0.87);
            }
        }
    }
    let positions = Tensor::from_slice(&positions, (g * g * g, 3), &device)?;
    let (features, density, _color) = pipeline.query_triplane(&scene_code, &positions, 8192)?;
    let density = density.affine(1.0, -1.0)?.exp()?;
    tensor_stats("query_features", &features)?;
    tensor_stats("query_density", &density)?;

    // Full-resolution grid density (GPU cheap, CPU slow) — finiteness only.
    if device_name != "cpu" {
        let grid_out = pipeline.neural.build_density_grid_cpu(
            &pipeline,
            &scene_codes,
            128,
            25.0,
            32768,
        )?;
        let (nan, inf, max_abs) = grid_stats(&grid_out.density_values);
        println!("[stage] density_grid_128: nan={nan} inf={inf} max_abs={max_abs:.4}");
    }

    if let Some(mut npz) = npz.take() {
        dump_tensor(&mut npz, "preprocessed_image", &prepared.preprocessed_image)?;
        dump_tensor(&mut npz, "image_tokens", &image_tokens)?;
        dump_tensor(&mut npz, "backbone_tokens", &backbone_tokens)?;
        dump_tensor(&mut npz, "detokenized_triplanes", &detokenized)?;
        dump_tensor(&mut npz, "scene_codes", &scene_codes)?;
        dump_tensor(&mut npz, "query_features", &features)?;
        dump_tensor(&mut npz, "query_density", &density)?;
        npz.finish()?;
        println!("[dump] wrote {:?}", dump.as_ref().unwrap());
    }
    Ok(())
}

fn grid_stats(values: &[f32]) -> (usize, usize, f64) {
    let (mut nan, mut inf, mut max_abs) = (0usize, 0usize, 0f64);
    for &v in values {
        if v.is_nan() {
            nan += 1;
        } else if v.is_infinite() {
            inf += 1;
        } else {
            max_abs = max_abs.max(v.abs() as f64);
        }
    }
    (nan, inf, max_abs)
}

fn compare(a_path: &str, b_path: &str) -> anyhow::Result<()> {
    let load = |path: &str| -> anyhow::Result<Vec<(String, Vec<f32>)>> {
        let mut npz = ndarray_npy::NpzReader::new(std::fs::File::open(path)?)?;
        let mut out = Vec::new();
        for name in npz.names()? {
            let arr: ndarray::ArrayD<f32> = npz.by_name(&name)?;
            out.push((name, arr.into_raw_vec()));
        }
        Ok(out)
    };
    let a = load(a_path)?;
    let b = load(b_path)?;
    let mut bmap = std::collections::HashMap::new();
    for (name, data) in b {
        bmap.insert(name, data);
    }
    println!(
        "{:<28} {:>12} {:>12} {:>10}",
        "tensor", "max_abs", "rel_rms", "nan_diff"
    );
    for (name, adata) in a {
        let Some(bdata) = bmap.get(&name) else {
            println!("{name:<28} MISSING in {b_path}");
            continue;
        };
        if adata.len() != bdata.len() {
            println!("{name:<28} length mismatch {} vs {}", adata.len(), bdata.len());
            continue;
        }
        let (mut max_abs, mut num, mut den, mut nan_diff) = (0f64, 0f64, 0f64, 0usize);
        for (&x, &y) in adata.iter().zip(bdata.iter()) {
            let x_ok = x.is_finite();
            let y_ok = y.is_finite();
            if x_ok != y_ok || (!x_ok && x.to_bits() != y.to_bits()) {
                nan_diff += 1;
                continue;
            }
            let d = (x - y) as f64;
            max_abs = max_abs.max(d.abs());
            num += d * d;
            den += (y as f64) * (y as f64);
        }
        let rel_rms = (num / adata.len() as f64).sqrt() / (den / adata.len() as f64).sqrt().max(1e-30);
        println!(
            "{name:<28} {max_abs:>12.4e} {rel_rms:>12.4e} {nan_diff:>10}"
        );
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut mode = "run".to_string();
    let mut device = "vulkan".to_string();
    let mut model_path = String::new();
    let mut source = String::new();
    let mut dump = None;
    let mut compare_paths = None;
    let mut grid = 32u32;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--compare" => {
                compare_paths =
                    Some((args.next().unwrap_or_default(), args.next().unwrap_or_default()))
            }
            "--device" => device = args.next().unwrap_or_default(),
            "--model-path" => model_path = args.next().unwrap_or_default(),
            "--source" => source = args.next().unwrap_or_default(),
            "--dump" => dump = Some(args.next().unwrap_or_default()),
            "--grid" => grid = args.next().and_then(|v| v.parse().ok()).unwrap_or(grid),
            other => anyhow::bail!("unexpected argument `{other}`"),
        }
    }
    if let Some((a, b)) = compare_paths {
        return compare(&a, &b);
    }
    run(&device, &model_path, &source, dump, grid)
}
