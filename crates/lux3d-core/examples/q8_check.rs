//! Dump or compare Q8_0 matmul outputs across vulkan kernel selection modes
//! (CANDLE_VK_Q8_COOPMAT is read once per process, so each mode is a separate
//! invocation; `--compare` diff two dumps).
//!
//! ```text
//! cargo run --release --example q8_check -p lux3d-core --features vulkan -- \
//!   --device vulkan --rows 256 --k 512 --n 512 --dump dp4a.bin
//! CANDLE_VK_Q8_COOPMAT=1 cargo run --release ... --dump cm1.bin
//! cargo run --release --example q8_check -p lux3d-core --features vulkan -- \
//!   --compare dp4a.bin cm1.bin
//! ```

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};

fn main() -> candle_core::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut device_name = "cpu".to_string();
    let (mut rows, mut k, mut n) = (256usize, 512usize, 512usize);
    let mut dump = None;
    let mut compare: Option<String> = None;
    // Zero all 32-wide activation blocks except KB (or all weight blocks
    // except KB) so each dump isolates a single k-block's contribution.
    let mut act_block: Option<usize> = None;
    let mut w_block: Option<usize> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device" => device_name = args.next().unwrap_or_default(),
            "--rows" => rows = args.next().and_then(|v| v.parse().ok()).unwrap_or(rows),
            "--k" => k = args.next().and_then(|v| v.parse().ok()).unwrap_or(k),
            "--n" => n = args.next().and_then(|v| v.parse().ok()).unwrap_or(n),
            "--dump" => dump = Some(args.next().unwrap_or_default()),
            "--compare" => compare = Some(args.next().unwrap_or_default()),
            "--act-block" => act_block = args.next().and_then(|v| v.parse().ok()),
            "--w-block" => w_block = args.next().and_then(|v| v.parse().ok()),
            other => candle_core::bail!("unexpected argument `{other}`"),
        }
    }

    if let Some(path_b) = compare {
        let path_a = dump.expect("--compare needs the reference path in --dump");
        let a: Vec<f32> = std::fs::read(&path_a)
            .unwrap_or_else(|e| panic!("read {path_a}: {e}"))
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let b: Vec<f32> = std::fs::read(&path_b)
            .unwrap_or_else(|e| panic!("read {path_b}: {e}"))
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(a.len(), b.len(), "dump length mismatch");
        let (mut max_abs, mut num, mut den) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (*x - *y) as f64;
            max_abs = max_abs.max(d.abs());
            num += d * d;
            den += (*y as f64) * (*y as f64);
        }
        let rms_rel = (num / a.len() as f64).sqrt() / (den / b.len() as f64).sqrt().max(1e-30);
        println!(
            "compare {path_a} vs {path_b}: n={} max_abs={max_abs:.6e} rms_rel={rms_rel:.6e}",
            a.len(),
        );
        return Ok(());
    }

    let device = match device_name.as_str() {
        "cpu" => Device::Cpu,
        "vulkan" => Device::new_vulkan(0)?,
        other => candle_core::bail!("unknown device `{other}`"),
    };
    let xs = Tensor::from_vec(
        (0..rows * k)
            .map(|i| {
                let kk = i % k;
                if let Some(kb) = act_block {
                    if kk / 32 != kb {
                        0.0
                    } else {
                        ((i as f32) * 0.001).sin()
                    }
                } else {
                    ((i as f32) * 0.001).sin()
                }
            })
            .collect::<Vec<_>>(),
        (rows, k),
        &device,
    )?;
    let w = Tensor::from_vec(
        (0..n * k)
            .map(|i| {
                let kk = i % k;
                if let Some(kb) = w_block {
                    if kk / 32 != kb {
                        0.0
                    } else {
                        ((i as f32) * 0.0007).cos() * 0.05
                    }
                } else {
                    ((i as f32) * 0.0007).cos() * 0.05
                }
            })
            .collect::<Vec<_>>(),
        (n, k),
        &device,
    )?;
    let qm = QMatMul::from_qtensor(QTensor::quantize(&w, GgmlDType::Q8_0)?)?;
    let out = qm.forward(&xs)?.to_dtype(DType::F32)?;
    device.synchronize()?;
    let path = dump.expect("--dump path required");
    let mut bytes = Vec::with_capacity(out.elem_count() * 4);
    for v in out.flatten_all()?.to_vec1::<f32>()? {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&path, &bytes)?;
    let sum: f32 = out.sum_all()?.to_scalar::<f32>()?;
    println!(
        "dumped {} f32 to {path} (rows={rows} k={k} n={n}) sum={sum:.4}",
        bytes.len() / 4,
    );
    Ok(())
}
