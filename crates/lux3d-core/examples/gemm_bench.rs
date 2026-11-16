//! Dense and packed GEMM timing probe per backend.
//!
//! ```text
//! cargo run --release --example gemm_bench -p lux3d-core --features wgpu -- \
//!   --device wgpu --rows 4096 --k 1024 --n 1024 --dtype f16 --iters 5
//! ```

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Tensor};
use candle_nn::Module;

fn main() -> candle_core::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut device_name = "cpu".to_string();
    let (mut rows, mut k, mut n, mut iters) = (4096usize, 1024usize, 1024usize, 3usize);
    let mut dtype = "f32".to_string();
    let mut quant = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device" => device_name = args.next().unwrap_or_default(),
            "--rows" => rows = args.next().and_then(|v| v.parse().ok()).unwrap_or(rows),
            "--k" => k = args.next().and_then(|v| v.parse().ok()).unwrap_or(k),
            "--n" => n = args.next().and_then(|v| v.parse().ok()).unwrap_or(n),
            "--iters" => iters = args.next().and_then(|v| v.parse().ok()).unwrap_or(iters),
            "--dtype" => dtype = args.next().unwrap_or(dtype),
            "--quant" => quant = args.next(),
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
    let dt = match dtype.as_str() {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        other => candle_core::bail!("unknown dtype `{other}`"),
    };
    let xs = Tensor::from_vec(
        (0..rows * k)
            .map(|i| ((i as f32) * 0.001).sin())
            .collect::<Vec<_>>(),
        (rows, k),
        &device,
    )?
    .to_dtype(dt)?;
    let w = Tensor::from_vec(
        (0..n * k)
            .map(|i| ((i as f32) * 0.0007).cos() * 0.05)
            .collect::<Vec<_>>(),
        (n, k),
        &device,
    )?
    .to_dtype(dt)?;
    let q = if let Some(q) = quant.as_deref() {
        let qd = match q {
            "q8_0" => GgmlDType::Q8_0,
            "q6_k" => GgmlDType::Q6K,
            "q4_k" => GgmlDType::Q4K,
            other => candle_core::bail!("unknown quant `{other}`"),
        };
        let w32 = w.to_dtype(DType::F32)?;
        Some(QMatMul::from_qtensor(QTensor::quantize(&w32, qd)?)?)
    } else {
        None
    };
    for iter in 0..iters {
        let start = std::time::Instant::now();
        let out = match &q {
            Some(qm) => qm.forward(&xs)?,
            None => xs.matmul(&w.t()?)?,
        };
        let sum = out
            .to_dtype(DType::F32)?
            .flatten_all()?
            .sum_all()?
            .to_scalar::<f32>()?;
        device.synchronize()?;
        println!(
            "iter {iter}: {:.3}s sum={sum:.3}",
            start.elapsed().as_secs_f64()
        );
    }
    #[cfg(feature = "vulkan")]
    if matches!(device, Device::Vulkan(_)) {
        if let Some((wall, rows)) = candle_core::vulkan_gpu_profile_report() {
            println!("gpu-profile wall={wall:.2}s kernels={}", rows.len());
            for (name, count, total_ms) in rows.iter().take(20) {
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
    }
    Ok(())
}
