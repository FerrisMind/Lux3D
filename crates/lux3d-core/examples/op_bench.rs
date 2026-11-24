//! Per-op timing probe for the TripoSG VAE decode chunk (f32, RTX 3060).
//!
//! ```text
//! cargo run --release --example op_bench -p lux3d-core --features vulkan -- \
//!   --device vulkan --iters 10
//! ```
//!
//! Shapes mirror one 16384-point outer chunk of the flash SDF decode:
//! dim 1024, heads 8, head_dim 128, kv_len 2048, ff 4096, attn chunk 1024.

use candle_core::{DType, Device, Tensor};
use candle_nn as nn;

fn bench<F>(label: &str, device: &Device, iters: usize, f: F) -> candle_core::Result<()>
where
    F: Fn() -> candle_core::Result<Tensor>,
{
    drop(f()?);
    device.synchronize()?;
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let started = std::time::Instant::now();
        let out = f()?;
        device.synchronize()?;
        best = best.min(started.elapsed().as_secs_f64());
        drop(out);
    }
    println!("{label:34} {:.3}ms", best * 1e3);
    Ok(())
}

fn main() -> candle_core::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut device_name = "cpu".to_string();
    let mut iters = 5usize;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device" => device_name = args.next().unwrap_or_default(),
            "--iters" => iters = args.next().and_then(|v| v.parse().ok()).unwrap_or(iters),
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
    let kv_len = 2048usize;
    let ff = 4096usize;
    let chunk = 1024usize;

    let xs = Tensor::randn(0f32, 1f32, (rows, dim), &device)?;
    let ff_w = Tensor::randn(0f32, 1f32, (ff, dim), &device)?;
    let q = Tensor::randn(0f32, 1f32, (1, heads, chunk, head_dim), &device)?;
    let k_t = Tensor::randn(0f32, 1f32, (1, heads, head_dim, kv_len), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, heads, kv_len, head_dim), &device)?;
    let mut chunks = Vec::new();
    for _ in 0..16 {
        chunks.push(Tensor::randn(
            0f32,
            1f32,
            (1, heads, chunk, head_dim),
            &device,
        )?);
    }
    let seq_major = Tensor::randn(0f32, 1f32, (1, rows, heads, head_dim), &device)?;
    let alpha = Tensor::randn(0f32, 1f32, (dim,), &device)?;
    let beta = Tensor::randn(0f32, 1f32, (dim,), &device)?;

    bench("gelu_erf 16384x4096", &device, iters, || xs.gelu_erf())?;
    bench("add 16384x4096", &device, iters, || {
        let big = xs.matmul(&ff_w.t()?.contiguous()?)?;
        big.add(&big)
    })?;
    bench("softmax 8x1024x2048", &device, iters, || {
        let s = q.matmul(&k_t)?;
        nn::ops::softmax_last_dim(&s)
    })?;
    bench("scores batched 8x1024x128@128x2048", &device, iters, || {
        q.matmul(&k_t)
    })?;
    bench("ctx batched 8x1024x2048@2048x128", &device, iters, || {
        let s = q.matmul(&k_t)?;
        let attn = nn::ops::softmax_last_dim(&s)?;
        attn.matmul(&v)
    })?;
    bench("layernorm 16384x1024", &device, iters, || {
        nn::ops::layer_norm(&xs, &alpha, &beta, 1e-5)
    })?;
    bench("transpose+contig 1x1024x8x128", &device, iters, || {
        seq_major.narrow(1, 0, chunk)?.transpose(1, 2)?.contiguous()
    })?;
    bench("q narrow+contig 1x8x1024x128", &device, iters, || {
        let q_big = Tensor::randn(0f32, 1f32, (1, heads, rows, head_dim), &device)?;
        q_big.narrow(2, 0, chunk)?.contiguous()
    })?;
    bench("cat 16x 1x8x1024x128 dim2", &device, iters, || {
        let refs = chunks.iter().collect::<Vec<_>>();
        Tensor::cat(&refs, 2)
    })?;
    bench("affine 1x8x16384x128", &device, iters, || {
        seq_major.affine(0.088, 0.0)
    })?;
    bench("matmul 16384x1024@4096x1024", &device, iters, || {
        xs.matmul(&ff_w)
    })?;
    Ok(())
}
