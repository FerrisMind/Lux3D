//! Standalone primitive microbenchmarks (VK4): reproduce and isolate the
//! Vulkan `replication_pad2d` cost seen on Pi3X (~6 s per call on
//! [1,64,518,518], 4 calls per inference) and collect
//! GEMM/QMatMul/SDPA/sync/upload baselines per backend.
//!
//! One process runs exactly one backend and one group, selected by env:
//! - `LUX3D_BENCH_DEVICE=cuda|vulkan|wgpu` (default `cuda`; `cpu` also accepted)
//! - `LUX3D_BENCH_KIND=primitives` (dispatched from `bench_main.rs`)
//! - `LUX3D_BENCH_GROUP=pad|qmatmul|dense|sdpa|sync|upload` (default: all, in order)
//! - `LUX3D_BENCH_CASES=a,b,...` (exact case names; default: all)
//!
//! One line per case on stdout:
//! `[prim] backend=.. group=.. case=.. median_ms=.. min_ms=.. gflops=..`
//! (`gflops=0` for non-GEMM cases). Pad cases append `match=yes|no` from a
//! small CPU verification. Protocol: 3 warmup + 10 timed iterations, device
//! synchronized before and after each timed block; median + min reported.

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::ops;
use std::hint::black_box;
use std::time::Instant;

const WARMUP_ITERS: usize = 3;
const TIMED_ITERS: usize = 10;

pub(crate) fn run() {
    let backend = std::env::var("LUX3D_BENCH_DEVICE").unwrap_or_else(|_| "cuda".to_string());
    let device = match backend.as_str() {
        "vulkan" => Device::new_vulkan(0),
        "wgpu" => Device::new_wgpu(0),
        "cuda" => Device::new_cuda(0),
        "cpu" => Ok(Device::Cpu),
        other => Err(candle_core::Error::Msg(format!("unknown LUX3D_BENCH_DEVICE `{other}`")).bt()),
    }
    .expect("device init failed");
    let group = std::env::var("LUX3D_BENCH_GROUP").unwrap_or_default();
    let wanted: Vec<String> = std::env::var("LUX3D_BENCH_CASES")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    eprintln!("[prim] kind=primitives backend={backend} group={group:?} cases={wanted:?}");

    for g in ["pad", "qmatmul", "dense", "sdpa", "sync", "upload"] {
        if !group.is_empty() && group != g {
            continue;
        }
        match g {
            "pad" => run_pad(&device, &backend, &wanted),
            "qmatmul" => run_qmatmul(&device, &backend, &wanted),
            "dense" => run_dense(&device, &backend, &wanted),
            "sdpa" => run_sdpa(&device, &backend, &wanted),
            "sync" => run_sync(&device, &backend, &wanted),
            "upload" => run_upload(&device, &backend, &wanted),
            _ => unreachable!(),
        }
        let _ = device.synchronize();
    }
}

fn case_wanted(wanted: &[String], case: &str) -> bool {
    wanted.is_empty() || wanted.iter().any(|c| c == case)
}

/// Deterministic pseudo-random F32 payload (backend-independent; `randn` is
/// not implemented on every backend).
fn test_f32_data(elem_count: usize) -> Vec<f32> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    (0..elem_count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0
        })
        .collect()
}

fn median_min(mut samples: Vec<f64>) -> (f64, f64) {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    let min = samples[0];
    let n = samples.len();
    let median = if n % 2 == 1 {
        samples[n / 2]
    } else {
        (samples[n / 2 - 1] + samples[n / 2]) / 2.0
    };
    (median, min)
}

/// 3 warmup + 10 timed iterations; each timed block is synchronized before
/// and after, and its output tensor is kept alive until the closing sync.
fn time_case<F>(device: &Device, mut f: F) -> candle_core::Result<(f64, f64)>
where
    F: FnMut() -> candle_core::Result<Tensor>,
{
    for _ in 0..WARMUP_ITERS {
        black_box(f()?);
        device.synchronize()?;
    }
    let mut samples_ms = Vec::with_capacity(TIMED_ITERS);
    for _ in 0..TIMED_ITERS {
        device.synchronize()?;
        let start = Instant::now();
        let out = f()?;
        device.synchronize()?;
        samples_ms.push(start.elapsed().as_secs_f64() * 1e3);
        drop(out);
    }
    Ok(median_min(samples_ms))
}

fn print_case(
    backend: &str,
    group: &str,
    case: &str,
    median_ms: f64,
    min_ms: f64,
    gflops: f64,
    extra: &str,
) {
    println!(
        "[prim] backend={backend} group={group} case={case} median_ms={median_ms:.3} min_ms={min_ms:.3} gflops={gflops:.1}{extra}"
    );
}

fn print_error(backend: &str, group: &str, case: &str, err: &candle_core::Error) {
    println!(
        "[prim] backend={backend} group={group} case={case} error={:?}",
        err.to_string()
    );
}

fn gflops(flops: f64, median_ms: f64) -> f64 {
    if median_ms > 0.0 {
        flops / (median_ms * 1e6)
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// pad group
// ---------------------------------------------------------------------------

/// Manual replica of `candle_nn::ops::replication_pad2d(xs, 1)`.
fn pad_cat_chain(x: &Tensor) -> candle_core::Result<Tensor> {
    let (_b, _c, h, w) = x.dims4()?;
    let (first, last) = (x.narrow(3, 0, 1)?, x.narrow(3, w - 1, 1)?);
    let xs = Tensor::cat(&[&first, x, &last], 3)?;
    let (first, last) = (xs.narrow(2, 0, 1)?, xs.narrow(2, h - 1, 1)?);
    Tensor::cat(&[&first, &xs, &last], 2)
}

/// Same chain with the two narrow slices made contiguous before each cat.
fn pad_cat_chain_contig(x: &Tensor) -> candle_core::Result<Tensor> {
    let (_b, _c, h, w) = x.dims4()?;
    let (first, last) = (
        x.narrow(3, 0, 1)?.contiguous()?,
        x.narrow(3, w - 1, 1)?.contiguous()?,
    );
    let xs = Tensor::cat(&[&first, x, &last], 3)?;
    let (first, last) = (
        xs.narrow(2, 0, 1)?.contiguous()?,
        xs.narrow(2, h - 1, 1)?.contiguous()?,
    );
    Tensor::cat(&[&first, &xs, &last], 2)
}

/// Border indices `[0, 0..len, len-1]` for the `index_select`-based pad.
fn border_indices(len: u32) -> Vec<u32> {
    let mut idx = Vec::with_capacity(len as usize + 2);
    idx.push(0);
    idx.extend(0..len);
    idx.push(len - 1);
    idx
}

/// Scalar-loop reference for the full replication pad (`[b,c,h+2,w+2]`).
fn cpu_full_pad_ref(x: &[f32], b: usize, c: usize, h: usize, w: usize) -> Vec<f32> {
    let (oh, ow) = (h + 2, w + 2);
    let mut out = vec![0f32; b * c * oh * ow];
    for bi in 0..b {
        for ci in 0..c {
            for y in 0..oh {
                let sy = (y as isize - 1).clamp(0, h as isize - 1) as usize;
                for xx in 0..ow {
                    let sx = (xx as isize - 1).clamp(0, w as isize - 1) as usize;
                    out[((bi * c + ci) * oh + y) * ow + xx] = x[((bi * c + ci) * h + sy) * w + sx];
                }
            }
        }
    }
    out
}

/// Scalar-loop reference for the dim-3 stage alone (`[b,c,h,w+2]`).
fn cpu_stage3_ref(x: &[f32], b: usize, c: usize, h: usize, w: usize) -> Vec<f32> {
    let ow = w + 2;
    let mut out = vec![0f32; b * c * h * ow];
    for bi in 0..b {
        for ci in 0..c {
            for y in 0..h {
                for xx in 0..ow {
                    let sx = (xx as isize - 1).clamp(0, w as isize - 1) as usize;
                    out[((bi * c + ci) * h + y) * ow + xx] = x[((bi * c + ci) * h + y) * w + sx];
                }
            }
        }
    }
    out
}

fn tensors_equal(a: &Tensor, b: &Tensor) -> bool {
    if a.shape() != b.shape() || a.dtype() != b.dtype() {
        return false;
    }
    match (
        a.flatten_all().and_then(|t| t.to_vec1::<f32>()),
        b.flatten_all().and_then(|t| t.to_vec1::<f32>()),
    ) {
        (Ok(va), Ok(vb)) => va.iter().zip(vb.iter()).all(|(x, y)| (x - y).abs() <= 1e-5),
        _ => false,
    }
}

/// Verify every pad variant against the scalar-loop reference on CPU for a
/// small [1,2,8,8] input; the variant code paths are device-independent.
fn verify_pad_variants_on_cpu() -> Vec<(&'static str, bool)> {
    let (b, c, h, w) = (1usize, 2usize, 8usize, 8usize);
    let data = test_f32_data(b * c * h * w);
    let x = Tensor::from_slice(&data, (b, c, h, w), &Device::Cpu).expect("cpu verify tensor");
    let exp_full = Tensor::from_vec(
        cpu_full_pad_ref(&data, b, c, h, w),
        (b, c, h + 2, w + 2),
        &Device::Cpu,
    )
    .expect("cpu verify expected");
    let exp_stage3 = Tensor::from_vec(
        cpu_stage3_ref(&data, b, c, h, w),
        (b, c, h, w + 2),
        &Device::Cpu,
    )
    .expect("cpu verify expected");

    let idx3 =
        Tensor::from_vec(border_indices(w as u32), w + 2, &Device::Cpu).expect("cpu verify idx");
    let idx2 =
        Tensor::from_vec(border_indices(h as u32), h + 2, &Device::Cpu).expect("cpu verify idx");
    let index_select = x
        .index_select(&idx3, 3)
        .and_then(|xs| xs.index_select(&idx2, 2));

    let first3 = x.narrow(3, 0, 1).expect("narrow");
    let last3 = x.narrow(3, w - 1, 1).expect("narrow");
    let xs3 = Tensor::cat(&[&first3, &x, &last3], 3).expect("cat");
    let first2 = xs3.narrow(2, 0, 1).expect("narrow");
    let last2 = xs3.narrow(2, h - 1, 1).expect("narrow");
    let x_t = x.transpose(2, 3).expect("transpose");

    let checks: Vec<(&'static str, candle_core::Result<Tensor>, Tensor)> = vec![
        (
            "replication_pad2d",
            ops::replication_pad2d(&x, 1),
            exp_full.clone(),
        ),
        ("cat_chain", pad_cat_chain(&x), exp_full.clone()),
        (
            "cat_chain_contig",
            pad_cat_chain_contig(&x),
            exp_full.clone(),
        ),
        (
            "cat_dim3",
            Tensor::cat(&[&first3, &x, &last3], 3),
            exp_stage3,
        ),
        (
            "cat_dim2",
            Tensor::cat(&[&first2, &xs3, &last2], 2),
            exp_full.clone(),
        ),
        ("index_select", index_select, exp_full.clone()),
        ("contiguous_ref", x.contiguous(), x.clone()),
        // A strided contiguous() actually copies (contiguous() is a no-op
        // clone on an already-contiguous input).
        ("contiguous_copy_ref", x_t.contiguous(), x_t.clone()),
    ];
    checks
        .into_iter()
        .map(|(name, got, want)| {
            let ok = got.map(|g| tensors_equal(&g, &want)).unwrap_or(false);
            (name, ok)
        })
        .collect()
}

fn verify_get(verify: &[(&'static str, bool)], name: &str) -> bool {
    verify
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, ok)| *ok)
        .unwrap_or(false)
}

/// Server-shaped pad cases: the Pi3X hot path is
/// `candle_nn::ops::replication_pad2d` on [1,64,518,518] (and 259 for the
/// pyramid stages). Variants isolate cat cost (dim3/dim2 stage alone), the
/// contiguity of the narrow slices, an `index_select` gather alternative, and
/// `contiguous()` references. Note `contiguous()` on an already-contiguous
/// input is a no-op clone (`contiguous_ref`), so a strided view (`contiguous_copy_ref`)
/// provides the actual full-size copy bandwidth reference.
struct PadPrebuilt {
    idx3: Tensor,
    idx2: Tensor,
    first3: Tensor,
    last3: Tensor,
    xs3: Tensor,
    first2: Tensor,
    last2: Tensor,
    x_t: Tensor,
}

fn run_pad(device: &Device, backend: &str, wanted: &[String]) {
    let verify = verify_pad_variants_on_cpu();
    let summary = verify
        .iter()
        .map(|(k, ok)| format!("{k}={}", if *ok { "yes" } else { "no" }))
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!("[prim] pad cpu-verify [1,2,8,8] {summary}");

    let shapes = [
        (1usize, 64usize, 518usize, 518usize, "518x518"),
        (1, 64, 259, 259, "259x259"),
    ];
    for (b, c, h, w, tag) in shapes {
        let x = match Tensor::from_slice(&test_f32_data(b * c * h * w), (b, c, h, w), device) {
            Ok(x) => x,
            Err(err) => {
                print_error(backend, "pad", &format!("setup_{tag}"), &err);
                continue;
            }
        };
        let prebuilt = (|| -> candle_core::Result<PadPrebuilt> {
            let idx3 = Tensor::from_vec(border_indices(w as u32), w + 2, device)?;
            let idx2 = Tensor::from_vec(border_indices(h as u32), h + 2, device)?;
            let first3 = x.narrow(3, 0, 1)?;
            let last3 = x.narrow(3, w - 1, 1)?;
            let xs3 = Tensor::cat(&[&first3, &x, &last3], 3)?;
            let first2 = xs3.narrow(2, 0, 1)?;
            let last2 = xs3.narrow(2, h - 1, 1)?;
            let x_t = x.transpose(2, 3)?;
            Ok(PadPrebuilt {
                idx3,
                idx2,
                first3,
                last3,
                xs3,
                first2,
                last2,
                x_t,
            })
        })();
        let p = match prebuilt {
            Ok(p) => p,
            Err(err) => {
                print_error(backend, "pad", &format!("setup_{tag}"), &err);
                continue;
            }
        };

        let cases: Vec<(
            &'static str,
            Box<dyn Fn() -> candle_core::Result<Tensor> + '_>,
        )> = vec![
            (
                "replication_pad2d",
                Box::new(|| ops::replication_pad2d(&x, 1)),
            ),
            ("cat_chain", Box::new(|| pad_cat_chain(&x))),
            ("cat_chain_contig", Box::new(|| pad_cat_chain_contig(&x))),
            // The dim3 stage alone: three tensors with the narrow views the
            // real chain feeds it.
            (
                "cat_dim3",
                Box::new(|| Tensor::cat(&[&p.first3, &x, &p.last3], 3)),
            ),
            // The dim2 stage alone, with the stage-1 result prebuilt.
            (
                "cat_dim2",
                Box::new(|| Tensor::cat(&[&p.first2, &p.xs3, &p.last2], 2)),
            ),
            (
                "index_select",
                Box::new(|| {
                    let xs = x.index_select(&p.idx3, 3)?;
                    xs.index_select(&p.idx2, 2)
                }),
            ),
            // Bandwidth references (see the struct doc above).
            ("contiguous_ref", Box::new(|| x.contiguous())),
            ("contiguous_copy_ref", Box::new(|| p.x_t.contiguous())),
        ];

        for (name, f) in cases {
            let case = format!("{name}_{tag}");
            if !case_wanted(wanted, &case) {
                continue;
            }
            if !verify_get(&verify, name) {
                print_case(backend, "pad", &case, 0.0, 0.0, 0.0, " match=no");
                continue;
            }
            match time_case(device, || f()) {
                Ok((median, min)) => {
                    print_case(backend, "pad", &case, median, min, 0.0, " match=yes")
                }
                Err(err) => print_error(backend, "pad", &case, &err),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// qmatmul group
// ---------------------------------------------------------------------------

/// Packed GGML QMatMul baselines. The pinned fork has no `QTensor::to_device`;
/// `QTensor::quantize_onto` quantizes the CPU F32 weight and lands the packed
/// blocks directly on the target device, then `QMatMul::from_qtensor` wraps it
/// (the same load path `quantized_linear` uses in `src/runtime/nn_blocks.rs`).
fn run_qmatmul(device: &Device, backend: &str, wanted: &[String]) {
    // (m, k, n): weight is (n, k); k must divide the k-quant block size (256).
    const SHAPES: [(usize, usize, usize); 4] = [
        (1024, 4096, 1024),
        (4096, 1024, 1024),
        (896, 3584, 3584),
        (1, 4096, 1024),
    ];
    const DTYPES: [(GgmlDType, &str); 2] = [(GgmlDType::Q8_0, "q8_0"), (GgmlDType::Q4K, "q4_k")];
    for (dt, dt_name) in DTYPES {
        for (m, k, n) in SHAPES {
            let case = format!("{dt_name}_{m}x{k}x{n}");
            if !case_wanted(wanted, &case) {
                continue;
            }
            let outcome = (|| -> candle_core::Result<(f64, f64, f64)> {
                let w_cpu = Tensor::from_vec(test_f32_data(n * k), (n, k), &Device::Cpu)?;
                let qt = QTensor::quantize_onto(&w_cpu, dt, device)?;
                let qmm = QMatMul::from_qtensor(qt)?;
                let x = Tensor::from_vec(test_f32_data(m * k), (m, k), device)?;
                let flops = 2.0 * m as f64 * n as f64 * k as f64;
                let (median, min) = time_case(device, || qmm.forward(&x))?;
                Ok((median, min, flops))
            })();
            match outcome {
                Ok((median, min, flops)) => print_case(
                    backend,
                    "qmatmul",
                    &case,
                    median,
                    min,
                    gflops(flops, median),
                    "",
                ),
                Err(err) => print_error(backend, "qmatmul", &case, &err),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// dense group
// ---------------------------------------------------------------------------

/// Dense F32/F16 GEMM baselines: the qmatmul shapes plus one unaligned shape.
/// F16 weights/activations are cast on the CPU and uploaded; if a backend
/// lacks F16 GEMM the case prints the error and the run continues.
fn run_dense(device: &Device, backend: &str, wanted: &[String]) {
    const SHAPES: [(usize, usize, usize); 5] = [
        (1024, 4096, 1024),
        (4096, 1024, 1024),
        (896, 3584, 3584),
        (1, 4096, 1024),
        (1000, 4093, 1021),
    ];
    const DTYPES: [(DType, &str); 2] = [(DType::F32, "f32"), (DType::F16, "f16")];
    for (dt, dt_name) in DTYPES {
        for (m, k, n) in SHAPES {
            let case = format!("{dt_name}_{m}x{k}x{n}");
            if !case_wanted(wanted, &case) {
                continue;
            }
            let outcome = (|| -> candle_core::Result<(f64, f64, f64)> {
                let w = Tensor::from_vec(test_f32_data(n * k), (n, k), &Device::Cpu)?
                    .to_dtype(dt)?
                    .to_device(device)?;
                let x = Tensor::from_vec(test_f32_data(m * k), (m, k), &Device::Cpu)?
                    .to_dtype(dt)?
                    .to_device(device)?;
                // Contiguous transposed weight so the timed op is a plain 2D
                // GEMM (the layout attention code materializes per call).
                let w_t = w.t()?.contiguous()?;
                let flops = 2.0 * m as f64 * n as f64 * k as f64;
                let (median, min) = time_case(device, || x.matmul(&w_t))?;
                Ok((median, min, flops))
            })();
            match outcome {
                Ok((median, min, flops)) => print_case(
                    backend,
                    "dense",
                    &case,
                    median,
                    min,
                    gflops(flops, median),
                    "",
                ),
                Err(err) => print_error(backend, "dense", &case, &err),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// sdpa group
// ---------------------------------------------------------------------------

/// Mirrors `exact_query_chunked_sdpa` (src/runtime/attention_math.rs) with
/// `chunk_size = usize::MAX`: affine scale, k^T contiguous, q@k^T,
/// softmax_last_dim, attn@v.
fn sdpa_forward(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> candle_core::Result<Tensor> {
    let q = q.affine(scale as f64, 0.0)?;
    let k_t = k.transpose(2, 3)?.contiguous()?;
    let v = v.contiguous()?;
    let scores = q.matmul(&k_t)?;
    let attn = ops::softmax_last_dim(&scores)?;
    attn.matmul(&v)
}

/// F32 SDPA baselines, self-attention shapes (kv_len = s).
fn run_sdpa(device: &Device, backend: &str, wanted: &[String]) {
    const SHAPES: [(usize, usize, usize, usize); 3] =
        [(1, 16, 1024, 64), (1, 16, 4096, 64), (8, 16, 512, 64)];
    for (b, h, s, d) in SHAPES {
        let case = format!("qkv_{b}x{h}x{s}x{d}");
        if !case_wanted(wanted, &case) {
            continue;
        }
        let outcome = (|| -> candle_core::Result<(f64, f64, f64)> {
            let q = Tensor::from_vec(test_f32_data(b * h * s * d), (b, h, s, d), device)?;
            let k = Tensor::from_vec(test_f32_data(b * h * s * d), (b, h, s, d), device)?;
            let v = Tensor::from_vec(test_f32_data(b * h * s * d), (b, h, s, d), device)?;
            let scale = 1.0 / (d as f32).sqrt();
            // Two GEMMs per head: q@k^T (2*s*d*s) and attn@v (2*s*s*d).
            let flops = 4.0 * (b * h) as f64 * s as f64 * s as f64 * d as f64;
            let (median, min) = time_case(device, || sdpa_forward(&q, &k, &v, scale))?;
            Ok((median, min, flops))
        })();
        match outcome {
            Ok((median, min, flops)) => print_case(
                backend,
                "sdpa",
                &case,
                median,
                min,
                gflops(flops, median),
                "",
            ),
            Err(err) => print_error(backend, "sdpa", &case, &err),
        }
    }
}

// ---------------------------------------------------------------------------
// sync group
// ---------------------------------------------------------------------------

/// 1000x `device.synchronize()`; reports per-call median/min ms and the
/// average in microseconds.
fn run_sync(device: &Device, backend: &str, wanted: &[String]) {
    let case = "synchronize_1000x";
    if !case_wanted(wanted, case) {
        return;
    }
    let outcome = (|| -> candle_core::Result<(f64, f64, f64)> {
        for _ in 0..WARMUP_ITERS {
            device.synchronize()?;
        }
        // The timed op IS the synchronize call, so each sample stands alone.
        let mut samples_us = Vec::with_capacity(1000);
        for _ in 0..1000 {
            let start = Instant::now();
            device.synchronize()?;
            samples_us.push(start.elapsed().as_secs_f64() * 1e6);
        }
        let avg_us = samples_us.iter().sum::<f64>() / samples_us.len() as f64;
        let (median_us, min_us) = median_min(samples_us);
        Ok((median_us / 1e3, min_us / 1e3, avg_us))
    })();
    match outcome {
        Ok((median_ms, min_ms, avg_us)) => print_case(
            backend,
            "sync",
            case,
            median_ms,
            min_ms,
            0.0,
            &format!(" avg_us={avg_us:.2}"),
        ),
        Err(err) => print_error(backend, "sync", case, &err),
    }
}

// ---------------------------------------------------------------------------
// upload group
// ---------------------------------------------------------------------------

/// 30 tensors of 64 MiB F32 each (64 * 512 * 512 * 4 bytes) from a shared CPU
/// buffer: phase (a) synchronizes after every upload, phase (b) once at the
/// end. Extra `total_ms`/`avg_ms` fields report the phase totals.
fn run_upload(device: &Device, backend: &str, wanted: &[String]) {
    const TENSORS: usize = 30;
    const SHAPE: (usize, usize, usize) = (64, 512, 512); // 64 MiB F32
    let case_each = "upload_sync_each";
    let case_end = "upload_sync_end";
    if !wanted.is_empty() && !wanted.iter().any(|c| c == case_each || c == case_end) {
        return;
    }
    let data = test_f32_data(SHAPE.0 * SHAPE.1 * SHAPE.2);
    if let Err(err) = (|| -> candle_core::Result<()> {
        let warm = Tensor::from_slice(&data, SHAPE, device)?;
        device.synchronize()?;
        drop(warm);
        Ok(())
    })() {
        print_error(backend, "upload", "setup", &err);
        return;
    }

    // Phase (a): synchronize after each upload.
    if case_wanted(wanted, case_each) {
        let outcome = (|| -> candle_core::Result<(f64, f64, f64)> {
            let mut keep = Vec::with_capacity(TENSORS);
            let mut per_ms = Vec::with_capacity(TENSORS);
            let start = Instant::now();
            for _ in 0..TENSORS {
                let t0 = Instant::now();
                keep.push(Tensor::from_slice(&data, SHAPE, device)?);
                device.synchronize()?;
                per_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            let total_ms = start.elapsed().as_secs_f64() * 1e3;
            let (median, min) = median_min(per_ms);
            Ok((median, min, total_ms))
        })();
        match outcome {
            Ok((median, min, total_ms)) => print_case(
                backend,
                "upload",
                case_each,
                median,
                min,
                0.0,
                &format!(
                    " total_ms={total_ms:.1} avg_ms={:.1}",
                    total_ms / TENSORS as f64
                ),
            ),
            Err(err) => print_error(backend, "upload", case_each, &err),
        }
    }
    let _ = device.synchronize();

    // Phase (b): one synchronize at the end.
    if case_wanted(wanted, case_end) {
        let outcome = (|| -> candle_core::Result<(f64, f64, f64)> {
            let mut keep = Vec::with_capacity(TENSORS);
            device.synchronize()?;
            let start = Instant::now();
            for _ in 0..TENSORS {
                keep.push(Tensor::from_slice(&data, SHAPE, device)?);
            }
            device.synchronize()?;
            let total_ms = start.elapsed().as_secs_f64() * 1e3;
            // Single whole-phase sample: median/min collapse to the total.
            Ok((total_ms, total_ms, total_ms))
        })();
        match outcome {
            Ok((median, min, total_ms)) => print_case(
                backend,
                "upload",
                case_end,
                median,
                min,
                0.0,
                &format!(
                    " total_ms={total_ms:.1} avg_ms={:.1}",
                    total_ms / TENSORS as f64
                ),
            ),
            Err(err) => print_error(backend, "upload", case_end, &err),
        }
    }
    let _ = device.synchronize();
}
