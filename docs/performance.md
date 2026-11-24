# Performance

Measured on an NVIDIA GeForce RTX 3060 12 GiB (Windows, WDDM) unless noted.
For the full investigation log and remaining levers, see
[../PERF-NOTES.md](../PERF-NOTES.md).

## Inference benchmarks

Measured inference time (`[stage] infer`: preprocessing + neural pass, single iteration per process, weights excluded) on an NVIDIA GeForce RTX 3060 12 GiB, Windows (WDDM), candle `wgpu/vulkan` fork rev `efb506c9`, candle-3d (2026-09-06):

| model | CUDA | Vulkan | WGPU | Vulkan/CUDA | WGPU/CUDA |
|---|---:|---:|---:|---:|---:|
| pi3 (5 frames, 518×518) | **4.82 s** | 7.83 s | 13.96 s | 1.62× | 2.90× |
| pi3x (6 frames, 518×518) | **7.00 s** | 11.10 s | 31.84 s | 1.59× | 4.55× |
| triposr (single image) | **1.23 s** | 3.36 s | 4.89 s | 2.73× | 3.98× |

VGGT-1B was added later and is measured separately (fork rev `f94d26eb`, 2026-11-11): 3 frames at
518×518, including camera, DPT depth, and DPT point heads. RTX 3060 12 GiB: CUDA **6.76 s**,
Vulkan 10.96 s (1.62×), WGPU 19.34 s, CPU 107.66 s. The CUDA peak is 6.52 GiB at 3 frames
(Vulkan 6.1–7.4 GiB and WGPU 6.8 GiB across repeated runs); on an NVIDIA L4 23 GiB
the peak is 7.98 GiB at 3 frames (4.19 s infer) and 8.30 GiB at 12 frames (26.21 s infer).

TripoSG is measured per stage on an RTX 3060 12 GiB (Windows/WDDM, release build, single image
`test-assets/mesh-input.png`):

| Stage | CUDA | Vulkan | WGPU |
|---|---:|---:|---:|
| load | 7.12 s | 6.68 s | 6.60 s |
| infer (preprocess + DINO + denoise) | 6.84 s | 8.56 s | 13.13 s |
| mesh (flash SDF + DiffDMC + trimesh processing) | 0.73 s | 0.26 s | 0.45 s |
| export | 0.00 s | 0.00 s | 0.01 s |
| peak VRAM | 8.29 GiB | 8.36 GiB | 8.65 GiB |

All three columns use the same command (`--steps 2 --num-tokens 1024 --flash-octree-depth 5`).
TripoSG keeps all three submodels (DINOv2, DiT, VAE) resident — 7.4 GiB of weights — so memory is
dominated by the checkpoint. After the memory-default fixes (Vulkan transient pool 0 / inflight
grace 2, WGPU 2 MiB allocator blocks) Vulkan peaks within 0.1 GiB of CUDA and WGPU within ~0.4 GiB
(occasional ±0.5 GiB run-to-run variance from async staging).
12 GiB is enough for the 2048-token preview (Vulkan: load 6.56 s, infer 12.86 s, peak 9.13 GiB),
but 16 GiB is still recommended for the 50-step / depth-9 reference run.

Warm-loop criterion (10 samples, steady state — the candle-bench protocol, one back-to-back window): vulkan 3.14 s / 4.90 s per iter (1.16×/1.20× of CUDA 2.72 s / 4.09 s), wgpu 8.21 s / 24.0 s (3.0×/5.9×). The biggest single lever was a fused LayerNorm+RoPE kernel (`rope_layernorm.comp`/`.wgsl` in the fork): one dispatch per q/k replaces the slow-path LayerNorm, both surrounding copies, and the 8-op rope chain at every rope site, cutting wgpu pi3 from 54.9 s to 8.21 s warm.

Correctness: outputs of every backend × model combination were verified against the CUDA reference meshes (bounding box / center / mean delta within 1% of max extent, vertex/face counts within 0.5% — all PASS); the fused kernel is additionally unit-tested exact to 0.0 diff against the composed form on both backends.

Tracking stage-A detectors are tested on all four backends: the official ALIKED-n16 golden matches 735/735 keypoints and SuperPoint matches 415/415 (resize 882/882) on CPU, CUDA, Vulkan and WGPU, with worst keypoint drift 1.5e-5, score error ≤7.7e-6 and descriptor cosine ≥0.999999; the ALIKED+SuperPoint composition test passes on all four. Set `LUX3D_KEYPOINT_TEST_DEVICE=cpu|cuda|vulkan|wgpu` (default `cpu`) and enable the matching cargo feature; an unavailable requested backend is a hard test error, never a silent CPU pass. Remesh, O-Voxel export/fixtures, the server URL policy and the external-COLMAP BA path are host-side code independent of the candle device; their tests also pass in the `vulkan` and `wgpu` feature builds.

Memory behavior under repeated inference (10-iteration loop, same process):

- **CUDA** — flat; the decoder attention scores buffer is bounded to 768 MiB by query-axis chunking (`LUX3D_MAX_SDPA_SCORES_BYTES`), which removes a ~4 GiB transient that previously fragmented the CUDA memory pool and stalled a 12 GiB card ~100×. VGGT peaks at 6.5 GiB for 3 frames on the 12 GiB card, with DPT frames processed one at a time (`LUX3D_VGGT_DPT_CHUNK`, default 1).
- **Vulkan** — flat; retention is bounded by the inflight byte budget (256 MiB × grace band 2) and transients are not pooled by default (`CANDLE_VK_INFLIGHT_GRACE`, `CANDLE_VK_POOL_MAX_BYTES=0`); the pool can be re-enabled for latency-first runs. Batch caps are tuned from flush-reason profiling: transfer bytes 512 MiB (`CANDLE_VK_MAX_BATCH_TRANSFER_BYTES`), descriptor sets 8× dispatches — closing the batch per big activation copy cost ~4-9 ms of WDDM fence-signal latency per submission.
- **WGPU** — 10-iteration pi3x benchmark completes with a 10.4 GiB VRAM peak and zero errors (previously OOM'd by iteration ~3); free pool / recycle backlog / in-flight retention are each byte-capped (`CANDLE_WGPU_POOL_MAX_BYTES`, `CANDLE_WGPU_INFLIGHT_MAX_BYTES`) and 2 MiB allocator blocks remove most checkpoint slack (`CANDLE_WGPU_BLOCK_BYTES`).

Known gaps: the remaining Vulkan/CUDA warm gap (1.16–1.20×) is the structural WDDM submit tax (~4-9 ms per submission on a strictly dependent op chain) — a platform property code cannot reach; Linux without WDDM would close it. WGPU remains 3.0–5.9× behind warm (down from 13.9–19.5×), dominated by WGSL GEMM quality plus the same inter-pass WDDM gaps. A criterion harness for whole-model benches lives in `crates/lux3d-core/benches/` (run one `(backend, model)` pair per process: `LUX3D_BENCH_DEVICE=… LUX3D_BENCH_MODEL=… cargo bench -p lux3d-core --bench bench_main --features vulkan,wgpu`).

## System requirements and memory

- Rust 1.85 or newer
- Cargo with support for `edition = "2024"`
- Python 3.x for baseline tooling and normalization
- CUDA-capable NVIDIA hardware for verified runtime inference
- **Vulkan and WGPU backends** (via [FerrisMind/candle](https://github.com/FerrisMind/candle) `wgpu/vulkan` fork): verified for correctness and memory stability on RTX 3060 (see [inference benchmarks](#inference-benchmarks)); slower than CUDA, WGPU substantially so. TripoSG peaks within ~0.4 GiB of CUDA on both backends, so the 7.9 GB weight set leaves ~3 GiB of headroom on a 12 GiB card.
- Metal backend on macOS is build-checked in CI (`.github/workflows/metal.yml` builds and runs the lib/quant-probe checks); no model inference parity is validated in this repository.
- Canonical model package directories supplied with `--model-path`, or downloadable from Hugging Face into the user cache
- TripoSG needs roughly 8.3–8.7 GiB at the 1024-token preview (9.1 GiB at 2048 tokens) on all three backends; 16 GiB+ is recommended for the 50-step reference run.
- TRELLIS.2 loads its stage checkpoints sequentially and peaks at ~7.8 GiB with the texture stages resident on the `1024_cascade` run (geometry-only runs are lighter); a 12 GiB GPU fits with ~4 GiB headroom, and CPU inference works but is slow. `--no-low-vram` keeps every stage resident instead and therefore needs more peak VRAM (it fails with an explicit memory-limit error rather than silently falling back). The server loads the canonical package only; the raw `microsoft/TRELLIS.2-4B` checkpoint directory is a CLI-only loader. `--texture-mesh` loads the shape encoder from the raw checkpoint, from `external/shape_enc/model.safetensors` beside a canonical/GGUF package, or from `shape_enc.*` tensors embedded in a standalone GGUF, and runs the xatlas unwrap + PBR bake on the CPU (cost scales with face count, chart count and `texture_size²`; duplicate-position vertices are welded before charting so topology-repair splits do not turn every split sheet into its own chart).
