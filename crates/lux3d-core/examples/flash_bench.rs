//! Per-op timing probe for the TripoSG FLASH SDF decode (RTX 3060, vulkan vs cuda).
//!
//! ```text
//! cargo run --release --example flash_bench -p lux3d-core --features vulkan -- \
//!   --device vulkan --iters 10
//! ```
//!
//! Shapes mirror one 16384-point outer chunk of the flash decode
//! (`effective_flash_query_chunk` cap): dim 1024, heads 8, head_dim 128,
//! kv_len = latent tokens (1024), ff 4096. Attention q-chunk follows the
//! runtime default (`attn_chunk_size`): 1024 on vulkan, 6144 otherwise.
//! The VAE runs Q8_0 GGUF weights with F32 activations, so the linear layers
//! are benched as QMatMul q8_0. The last section reproduces the per-level
//! grid elementwise work at the dims=505 flash resolution (505^3 f32).

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Tensor};
use candle_nn as nn;
use candle_nn::Module;

fn bench<F>(label: &str, device: &Device, iters: usize, f: F) -> candle_core::Result<()>
where
    F: Fn() -> candle_core::Result<Tensor>,
{
    drop(f()?);
    device.synchronize()?;
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let started = std::time::Instant::now();
        let out = f()?;
        device.synchronize()?;
        samples.push(started.elapsed().as_secs_f64() * 1e3);
        drop(out);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = samples[0];
    let med = samples[samples.len() / 2];
    println!("{label:44} min {min:8.3}ms  med {med:8.3}ms");
    Ok(())
}

fn qmatmul(w: &Tensor) -> candle_core::Result<QMatMul> {
    let w32 = w.to_dtype(DType::F32)?;
    Ok(QMatMul::from_qtensor(QTensor::quantize(&w32, GgmlDType::Q8_0)?)?)
}

fn main() -> candle_core::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut device_name = "cpu".to_string();
    let mut iters = 10usize;
    let mut attn_chunk = 0usize;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device" => device_name = args.next().unwrap_or_default(),
            "--iters" => iters = args.next().and_then(|v| v.parse().ok()).unwrap_or(iters),
            "--attn-chunk" => attn_chunk = args.next().and_then(|v| v.parse().ok()).unwrap_or(0),
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
    let rows = 16384usize;
    let dim = 1024usize;
    let heads = 8usize;
    let head_dim = 128usize;
    let kv_len = 1024usize; // --num-tokens 1024
    let ff = 4096usize;
    let attn_chunk = if attn_chunk > 0 {
        attn_chunk
    } else if device.is_vulkan() {
        1024
    } else {
        6144
    };
    println!(
        "device={device_name} rows={rows} kv_len={kv_len} attn_chunk={attn_chunk} iters={iters}"
    );

    // ---- Q8_0 linear layers (QMatMul, f32 activations) ----
    // proj_query [51 -> 1024] cannot be Q8_0 (K=51 not divisible by the 32-wide
    // quant block), so the GGUF package ships it as a dense F32 linear.
    let x51 = Tensor::randn(0f32, 1f32, (rows, 51), &device)?;
    let w51 = Tensor::randn(0f32, 1f32, (dim, 51), &device)?;
    let xs = Tensor::randn(0f32, 1f32, (rows, dim), &device)?;
    let xff = Tensor::randn(0f32, 1f32, (rows, ff), &device)?;
    let w_sq = qmatmul(&Tensor::randn(0f32, 1f32, (dim, dim), &device)?)?;
    let w_ff_in = qmatmul(&Tensor::randn(0f32, 1f32, (ff, dim), &device)?)?;
    let w_ff_out = qmatmul(&Tensor::randn(0f32, 1f32, (dim, ff), &device)?)?;
    let w_proj_out = qmatmul(&Tensor::randn(0f32, 1f32, (1, dim), &device)?)?;
    bench("f32 proj_query 16384x51@51x1024", &device, iters, || {
        x51.matmul(&w51.t()?)
    })?;
    bench("q8 to_q/to_out 16384x1024@1024x1024", &device, iters, || {
        w_sq.forward(&xs)
    })?;
    bench("q8 ff_in 16384x1024@1024x4096", &device, iters, || {
        w_ff_in.forward(&xs)
    })?;
    bench("q8 ff_out 16384x4096@4096x1024", &device, iters, || {
        w_ff_out.forward(&xff)
    })?;
    bench("q8 proj_out 16384x1024@1024x1", &device, iters, || {
        w_proj_out.forward(&xs)
    })?;

    // ---- SDPA pieces (f32), per attention sub-chunk and whole ----
    let qc = Tensor::randn(0f32, 1f32, (1, heads, attn_chunk, head_dim), &device)?;
    let k_t = Tensor::randn(0f32, 1f32, (1, heads, head_dim, kv_len), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, heads, kv_len, head_dim), &device)?;
    bench(
        &format!("f32 scores 1x{heads}x{attn_chunk}x{head_dim}@{head_dim}x{kv_len}"),
        &device,
        iters,
        || qc.matmul(&k_t),
    )?;
    bench(
        &format!("f32 softmax 1x{heads}x{attn_chunk}x{kv_len}"),
        &device,
        iters,
        || {
            let s = qc.matmul(&k_t)?;
            nn::ops::softmax_last_dim(&s)
        },
    )?;
    bench(
        &format!("f32 ctx 1x{heads}x{attn_chunk}x{kv_len}@{kv_len}x{head_dim}"),
        &device,
        iters,
        || {
            let s = qc.matmul(&k_t)?;
            let a = nn::ops::softmax_last_dim(&s)?;
            a.matmul(&v)
        },
    )?;

    // ---- copies / elementwise over one outer chunk ----
    let seq_major = Tensor::randn(0f32, 1f32, (1, rows, heads, head_dim), &device)?;
    let head_major = Tensor::randn(0f32, 1f32, (1, heads, rows, head_dim), &device)?;
    let alpha = Tensor::randn(0f32, 1f32, (dim,), &device)?;
    let beta = Tensor::randn(0f32, 1f32, (dim,), &device)?;
    let hidden = Tensor::randn(0f32, 1f32, (1, rows, dim), &device)?;
    let ff_act = Tensor::randn(0f32, 1f32, (rows, ff), &device)?;
    bench("transpose+contig q heads 67MB", &device, iters, || {
        seq_major.transpose(1, 2)?.contiguous()
    })?;
    bench("out transpose+reshape copy 67MB", &device, iters, || {
        head_major.transpose(1, 2)?.reshape((1, rows, dim))
    })?;
    bench("q affine scale 1x8x16384x128", &device, iters, || {
        head_major.affine(0.088, 0.0)
    })?;
    bench("add 1x16384x1024 (residual)", &device, iters, || {
        hidden.add(&hidden)
    })?;
    bench("layernorm 16384x1024", &device, iters, || {
        nn::ops::layer_norm(&hidden, &alpha, &beta, 1e-5)
    })?;
    bench("gelu_erf 16384x4096", &device, iters, || ff_act.gelu_erf())?;
    let n_parts = rows.div_ceil(attn_chunk);
    bench(
        &format!("cat {n_parts}x sdpa sub-outs dim2"),
        &device,
        iters,
        || {
            let mut parts = Vec::new();
            let mut start = 0usize;
            while start < rows {
                let len = (rows - start).min(attn_chunk);
                parts.push(head_major.narrow(2, start, len)?);
                start += len;
            }
            Tensor::cat(&parts, 2)
        },
    )?;
    // embed_points: broadcast_mul + sin + cos + cat -> [16384, 51]
    let pts = Tensor::randn(0f32, 1f32, (rows, 3), &device)?;
    let freqs = Tensor::randn(0f32, 1f32, (1, 1, 8), &device)?;
    bench("embed_points [16384,3]->[16384,51]", &device, iters, || {
        let products = pts
            .reshape((rows, 3, 1))?
            .broadcast_mul(&freqs)?
            .reshape((rows, 24))?;
        Tensor::cat(&[&pts, &products.sin()?, &products.cos()?], candle_core::D::Minus1)
    })?;

    // ---- dims=505 final-level grid elementwise (505^3 = 128.8M f32) ----
    let n505 = 505usize * 505 * 505;
    let grid = Tensor::randn(0f32, 1f32, (n505,), &device)?;
    bench("scale/copy affine 128.8M f32", &device, iters, || {
        grid.affine(1.0, 0.0)
    })?;
    bench("mul scalar 128.8M f32", &device, iters, || {
        grid.affine(-2.0, 0.0)
    })?;
    bench("add tensor 128.8M f32", &device, iters, || grid.add(&grid))?;
    bench("to_vec1 D2H 128.8M f32 (515MB)", &device, 3, || {
        let v = grid.to_vec1::<f32>();
        v.map(|_| grid.clone())
    })?;

    #[cfg(feature = "vulkan")]
    if matches!(device, Device::Vulkan(_)) {
        if let Some((wall, rows_)) = candle_core::vulkan_gpu_profile_report() {
            println!("gpu-profile wall={wall:.2}s kernels={}", rows_.len());
            for (name, count, total_ms) in rows_.iter().take(15) {
                println!(
                    "gpu-profile {name:44} {count:>7}x {total_ms:>10.1}ms {:>8.1}us/ea",
                    total_ms * 1000.0 / (*count as f64)
                );
            }
        }
    }
    Ok(())
}
