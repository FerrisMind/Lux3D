//! Isolated SS decoder timing on real TRELLIS.2 weights.
//!
//! ```text
//! cargo run --release --example ss_dec_bench -p lux3d-core --features vulkan -- \
//!   --device vulkan --model model.gguf --iters 2
//! ```

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use lux3d_core::runtime::{Trellis2SsDecoder, Trellis2SsDecoderConfig};

fn main() -> candle_core::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut device_name = "cpu".to_string();
    let mut model = None;
    let mut iters = 2usize;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device" => device_name = args.next().unwrap_or_default(),
            "--model" => model = args.next(),
            "--iters" => iters = args.next().and_then(|v| v.parse().ok()).unwrap_or(2),
            other => candle_core::bail!("unexpected argument `{other}`"),
        }
    }
    let device = match device_name.as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::new_cuda(0)?,
        "vulkan" => Device::new_vulkan(0)?,
        "wgpu" => Device::new_wgpu(0)?,
        other => candle_core::bail!("unknown device `{other}`"),
    };
    let model = model.ok_or_else(|| candle_core::Error::msg("--model is required"))?;
    let path = std::path::Path::new(&model);
    let vb = if path.extension().is_some_and(|ext| ext == "gguf") {
        VarBuilder::from_gguf(path, DType::F32, &device)?
    } else {
        unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device)? }
    };
    let decoder =
        Trellis2SsDecoder::load(vb.pp("ss_dec"), &Trellis2SsDecoderConfig::ss_dec_16l8())?;
    let values = (0..8 * 16 * 16 * 16)
        .map(|i| ((i as f32) * 0.001).sin() * 0.1)
        .collect::<Vec<_>>();
    let xs = Tensor::from_vec(values, (1, 8, 16, 16, 16), &device)?;
    for iter in 0..iters {
        let start = std::time::Instant::now();
        let out = decoder.forward(&xs)?;
        let enqueue = start.elapsed().as_secs_f64();
        let sum = out.flatten_all()?.sum_all()?.to_scalar::<f32>()?;
        device.synchronize()?;
        println!(
            "iter {iter}: {:.2}s (enqueue {enqueue:.2}s, wait {:.2}s) sum={sum:.4}",
            start.elapsed().as_secs_f64(),
            start.elapsed().as_secs_f64() - enqueue
        );
    }
    #[cfg(feature = "vulkan")]
    if matches!(device, Device::Vulkan(_)) {
        if let Some((wall, rows)) = candle_core::vulkan_gpu_profile_report() {
            println!("gpu-profile wall={wall:.2}s kernels={}", rows.len());
            for (name, count, total_ms) in rows.iter().take(40) {
                println!(
                    "gpu-profile {name:44} {count:>7}x {total_ms:>10.1}ms {:>8.1}us/ea",
                    total_ms * 1000.0 / (*count as f64)
                );
            }
        }
        if let Some(rows) = candle_core::vulkan_cpu_profile_report() {
            println!("cpu-profile:");
            for (name, count, total_ms) in rows {
                println!(
                    "cpu-profile {name:24} {count:>8}x {total_ms:>10.1}ms {:>8.1}us/ea",
                    total_ms * 1000.0 / (count as f64)
                );
            }
        }
        if let Some(rows) = candle_core::vulkan_flush_reason_report() {
            println!("flush-reasons:");
            for (name, count) in rows {
                println!("flush-reason {name:24} {count:>8}x");
            }
        }
    }
    Ok(())
}
