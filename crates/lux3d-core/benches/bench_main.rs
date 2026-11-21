mod benchmarks;

// Custom main instead of criterion_main! so the selected model is known before
// the device is created and the matching GEMM opt-ins can be applied first —
// mirrors `lux3d-cli`'s per-family opt-in.
fn main() {
    let kind = std::env::var("LUX3D_BENCH_KIND").unwrap_or_else(|_| "models".to_string());
    match kind.as_str() {
        // Standalone primitive microbenchmarks (VK4); no criterion, no model assets.
        "primitives" => benchmarks::primitives::run(),
        _ => benchmarks::models::benches(),
    }
}
