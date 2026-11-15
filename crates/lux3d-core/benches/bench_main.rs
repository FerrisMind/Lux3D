mod benchmarks;

// Custom main instead of criterion_main! so the selected model is known before
// the device is created and the matching GEMM opt-ins can be applied first —
// mirrors `lux3d-cli`'s per-family opt-in.
fn main() {
    benchmarks::models::benches();
}
