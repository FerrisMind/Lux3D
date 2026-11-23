# Performance notes: measured results and remaining levers (RTX 3060 12 GiB, Windows WDDM)

Status: 2026-11-22, candle rev `9c74af75`, RTX 3060 12 GiB / Windows 11 WDDM.
The 2026-11-22 section below supersedes the 2026-11-21 e2e numbers (which
supersede the whole-model criterion numbers from 2026-09-07); the older
sections stay as measurement history.

## Decision (2026-11-22)

The 90% target is **not met**: same-session Vulkan/CUDA is **1.81x**
(1444.53 s vs 796.39 s). The measured cause is the missing INT8 tensor-core
GEMM in candle Vulkan: dp4a reaches 8.37 TFLOPS, CUDA's `mma.sync` reaches
26.29. The validated INT8-coopmat handoff below reaches 31.3 TFLOPS in a
standalone microbenchmark, so the path exists, but it requires candle-fork
changes. The user reserved those changes ("candle я сам все сделау. Та мне
трогай"), so **1.81x is the accepted limit for this repository** until the
candle kernel lands. After it lands, re-run the 18-scenario suite.

## E2E Vulkan/CUDA 1.81x: TripoSG decode sync reduction (2026-11-22)

Full 18-scenario suite, same binary and weights, `--test-threads=1`, one
session: CUDA **796.39 s**, Vulkan **1444.53 s** (18/18 both). Vulkan/CUDA
**1.81x**. Both absolute walls dropped ~16% vs 2026-11-21 (955.13 /
1717.21 s), but the ratio is unchanged: the fixes removed host-side sync and
repeated per-chunk work that both backends paid. Full per-scenario table:
[`E2E-RESULTS.md`](E2E-RESULTS.md).

Fixes, all measurement-gated (numbers are e2e wall time):

| Fix | Before -> after |
|---|---|
| P1: accumulate decode chunks on-device per level, one blocking D2H (was 512 syncs/level; batch ~2048 -> ~8) | scribble 141.01 -> 92.00 s; surface 59.92 -> 47.87 s; batch 646.21 -> 545.86 s; image 28.02 -> 24.87 s |
| P2: build `k_t = k.transpose(2,3).contiguous()` once in `prepare_cross_kv` (was one 4 MiB copy + dispatch per chunk) | included above |
| P3: cache the decoder freqs table at load (was one H2D per chunk) | included above |
| P5: fold the output negate into `proj_out` weights at load (packed GGUF keeps the runtime `affine(-1)`) | included above |

CPU parity is bit-identical for dense and refinement levels; the CUDA numbers
improved by the same mechanism, which is why the ratio did not move.

### Attribution: what still separates Vulkan from CUDA

| scenario group | Vk/CUDA | dominant bound | 90% reachable? |
|---|---:|---|---|
| `ovox_roundtrip` | 0.98x | CPU/IO only | **yes** (already) |
| `trellis2_texture_mesh` | 1.23x | CPU xatlas/bake floor | no (GPU part >1.11) |
| `trellis2_512_textured_glb` | 1.57x | CPU xatlas 182 s + q4_k decode | no |
| VGGT (4 point/scene + 2 track) | 1.81-2.15x | q8 aggregator GEMM (3.14x ceiling) | no |
| TripoSG image | 1.93x | q8 VAE SDF decode GEMM | no |
| TripoSG scribble | 2.11x | q8 VAE SDF decode GEMM | no |
| TripoSG batch | 2.24x | q8 VAE SDF decode GEMM | no |
| TripoSG surface | 2.25x | q8 VAE SDF decode GEMM | no |
| Pi3 | 2.34x | q8 GEMM | no |
| TripoSR | 2.64x | q8 GEMM + triplane sampler | no |
| TRELLIS.2 geometry | 2.67x | q4_k GEMM (5.17x ceiling) + sparse conv | no |

After P1, the TripoSG decode stage is no longer host-sync-bound: the
remaining time scales with the per-chunk q8 GEMMs, and the primitive ceiling
(8.37 vs 26.29 TFLOPS) bounds those scenarios near 3x before any Lux3D
overhead. An earlier note cited "7.4 s of actual GPU kernel time" for the
scribble decode; that figure predates P1 and was incomplete (it contradicts
the ~42 s FLOP floor at the measured Vulkan Q8 rate), so it is retired.

### Remaining Rust-level levers (measured risk, not landed)

* **P6: batch the decode across samples** (batch scenario only). All 4 samples
  share identical dense query grids; only the kv cache differs. Remove the
  batch-1 bail in `prepare_cross_kv`, reshape points to `[B,N,3]`, one
  hierarchical pass per level: 2048 -> 512 chunks. Medium-large risk; per
  sample parity required. Does not help single-sample scenarios.
* **Grouped TRELLIS.2 offsets** (Patch C): stack the 26 gather tables into one
  `[26,N]` select, one batched GEMM, one ones-GEMM reduce, row-chunked to
  512 MiB-1 GiB. ~191 -> ~5 dispatches per chunk but bandwidth-bound; needs
  A/B. Do not use `index_add` scatter on Vulkan (two permute
  materializations) or `narrow` slices for per-offset matmuls (nonzero
  offset forces a full copy).
* **Candle Vulkan Q8_0/Q4_K GEMM kernels** are the only path to 1.11x on the
  GEMM-bound scenarios; that work lives in the candle fork and is explicitly
  out of scope here.

### Candle-side handoff: the Q8_0/Q4_K Vulkan GEMM gap

Measured at rev `9c74af75` on the RTX 3060: CUDA Q8_0 **26.29 TFLOPS** vs
Vulkan **8.37** (3.14x); Q4_K **25.45** vs **4.92** (5.17x). The cause is a
missing kernel path, not a Lux3D scheduling issue:

* **CUDA** uses INT8 tensor-core MMA. `mmq_instance_q8_0.cu` feeds
  `mmq_mma.cuh:857,879`:
  `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` on Ampere (m16n8k16 on
  Turing). 26.29 TFLOPS is near the 3060 INT8 tensor peak.
* **Vulkan** uses only `GL_EXT_integer_dot_product` (dp4a) in
  `candle-vulkan-kernels/src/shaders/mul_mmq.comp` (64x64 tile, 128 threads).
  `vulkan-shaders-gen.cpp:619-620` generates the `_q8_1` variant only under
  `!f16acc && !coopmat && !coopmat2`, so no cooperative-matrix quantized GEMM
  exists. 8.37 TFLOPS is ~33% of the 3060 dp4a peak (~25.4 TOPS).
* Upstream llama.cpp b10455 has the same gate
  (`ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp:626-627`), so this is
  not a candle regression; the Vulkan backend never had an INT8 tensor-core
  GEMM.

Proposed candle change (one kernel + one generator entry):

1. Generate `matmul_q8_0_q8_1_cm1` (and the Q4_K/Q5_K/Q6_K siblings) from a
   coopmat variant of `mul_mmq.comp`: `VK_KHR_cooperative_matrix` with A/B
   `int8_t`, C `int32_t`; `coopMatMulAdd` per K=32 block, then apply the
   Q8_0 `d_a*d_b` block scale into the float accumulator in the epilogue.
   Mirror `mmq_mma.cuh`'s per-block scaling; do not accumulate across block
   boundaries without applying the scale.
2. Gate on `VK_KHR_cooperative_matrix` int8 component support and glslc
   coopmat support; keep the dp4a `mul_mmq` as the fallback.
3. Numerical gate: CPU `QMatMul` parity for Q8_0/Q4_K at the existing
   Vulkan q8_1 test tolerance.

Compiler and device validation (2026-11-22): glslc 1.4.350 compiles
`coopmat<int8_t, gl_ScopeSubgroup, 16,32, A>` / `<32,16,B>` /
`<int32_t,16,16,C>` with `coopMatMulAdd` to
`OpCooperativeMatrixMulAddKHR` (signed int8 components). The RTX 3060
exposes `VK_KHR_cooperative_matrix` revision 2 and reports the required
configuration: M=16 N=16 K=32, A=SINT8, B=SINT8, C=SINT32, Result=SINT32,
scope=SUBGROUP.

Standalone microbenchmark (not candle): 16x16x32 SINT8 coopmat, 64x64
workgroup tile (4x4 accumulators), 32 K-blocks, 1024 workgroups,
device-local buffers, 100 dispatches: **31.25-31.34 TFLOPS** (three runs,
0.0274 s). CUDA mmq Q8_0 is 26.29, Vulkan dp4a 8.37. Caveat: the benchmark
reuses the same 128 KB A/B tiles across workgroups (L2-resident), so it
measures the MMA ceiling; it omits Q8_0 block scales and GGUF unpacking.
A production kernel needs 128x128 tiling or L2-aware scheduling to sustain
it. Even at 60-70% of this ceiling, Vulkan Q8_0 GEMM matches CUDA.

Expected: matching the CUDA INT8 MMA rate removes the 3.1-5.2x kernel
ceiling; GEMM-bound scenarios then land at the non-GEMM overhead
(estimated ~1.1-1.3x). Without it, 1.81x is the measured floor.

### Measured dead ends (2026-11-22)

* TRELLIS.2 F32 sparse kernels on Vulkan (halves dispatches/conv, 191 -> 84):
  geometry+texture decode F32 104.65 s vs F16 86.95 s warm. The F32 gather
  and GEMM bandwidth loss outweighs the dispatch saving; reverted.
* TripoSG 64k decode chunk: surface mesh 344.64 s vs ~45 s. The 32k transients
  (FF 512 MiB x2, SDPA outputs 1 GiB + cat) cross the allocator/VRAM edge.
* Decode SDPA `attn_chunk` 6144 -> 24576: surface 49.13 vs 48.59 s, neutral.
* `CANDLE_VK_POOL_MAX_BYTES=512M`: surface 45.3 vs 46.3-47.8 s, scribble
  90.9 vs 90.5 s — within noise; not enabled by default.

## WGPU TripoSG field-path regression and fix (2026-11-22)

The TripoSG dense/refine grid builders (`triposg_field.rs`) keep every chunk
logit tensor on the device and download once per 256 MiB. On WGPU that policy
is much slower than on CUDA/Vulkan: the measured cost of `Tensor::cat` grows
with the pending window, so a large device-side buffer costs more than the
per-chunk D2H it replaces. Measured on the two field-path scenarios (same
binary, `--test-threads=1`):

| pending policy | surface | batch |
|---|---:|---:|
| per-chunk D2H (no cat) | **106.68 s** | 1244.63 s |
| 16-chunk cat window | 198.91 s | **1128.52 s** |
| 256 MiB byte cap | 384.72 s | **1116.14 s** |

WGPU now flushes every 16 chunks and does not use the byte cap; CUDA/Vulkan
keep the 256 MiB cap. The 16-chunk window minimizes the surface+batch sum
(1327 s vs 1351 s per-chunk and 1501 s byte-cap): surface alone prefers
per-chunk D2H, batch alone prefers the byte cap, and 16 is the crossover.
The flash decoder path keeps the 256 MiB cap on every backend because its
chunk tensors are large enough that one D2H per level wins (image 72.74 s vs
75.23 s per-chunk).

The chunk-count flush exposed a latent indexing bug in `refine_level`: the
flush window was tracked as a batch index but used to slice `marked_coords`
(a cell vector). On CUDA/Vulkan no scenario crossed the 256 MiB cap
mid-level, so the window was always `[0..]` and the bug stayed dormant; the
WGPU 16-chunk flush made the two index spaces diverge and panicked
(`len 33632, index 33632`). The window now tracks the first pending cell
(`pending_first_cell`) and derives it from
`batch_index * cells_per_batch`; surface and batch both pass with the fix.

## E2E Vulkan/CUDA: 2.62x -> 1.80x (2026-11-21)

Full 18-scenario suite, same binary and weights, `--test-threads=1`:
CUDA **955.13 s**, Vulkan **1717.21 s**, WGPU **3479.83 s** (18/18 on all
three). Vulkan/CUDA 2.62x -> **1.80x**; WGPU/CUDA **3.64x**, neutral vs its
previous 3466.58 s run. Full per-scenario table:
[`E2E-RESULTS.md`](E2E-RESULTS.md).

Fixes, all measurement-gated (numbers are e2e wall time):

| Fix | Before -> after |
|---|---|
| TripoSG-scribble GGUF: quantize `vae.` in addition to `transformer.` (the F16 SDF decoder ran 3.4 TFLOPS; packed Q8 runs 8.5) | 764.56 -> 141.01 s |
| Pi3X: `.contiguous()` before `replication_pad2d` (strided `cat` on the channel-last upsample output hit `copy_strided_src`, ~8 s per pad) | 33.36 -> 9.53 s |
| TripoSR SDF query chunk 8192 -> 32768 (~50 fixed dispatches per chunk; 128^3 = 256 chunks at 8192) | 13.58 -> 5.31 s |
| TripoSG/Scribble query chunk 16384 -> 32768 (dense/hierarchical only), K/V projected once per grid, VAE attn chunk 1024 -> 6144 | surface 69.20 -> 59.92 s; image 28.87 -> 28.02 s |
| VGGT `fmap_norm`: replace the NHWC transpose + native LayerNorm with a ones-GEMM channel norm on the contiguous NCHW feature map | track head 7.14 -> 0.57 s; queries 19.39 -> 11.27 s; auto-FPS 36.43 -> 13.43 s |

### Root causes worth remembering

* **Vulkan `copy_strided_src` is the recurring trap.** A permute/NHWC view
  materializes at ~16 MB/s. In VGGT it turned one `[1,3,128,259,259]` LayerNorm
  into 6.47 s; in Pi3X a padded `cat` cost ~8 s. Always check `.is_contiguous()`
  before pad/cat/layer_norm on Vulkan, or avoid the layout change entirely.
* **Tiny-tensor dispatch cost dominates.** VGGT track adds ~4,885 dispatches
  per pass; `CANDLE_VULKAN_CPU_PROFILE=1` measured 472 us average submit and
  68 ms average flush drain. The tracker loop itself is only 0.01-0.02 s per
  op once the transpose is gone.
* **Kernel ceiling is real.** Primitive benchmark (`LUX3D_BENCH_KIND=primitives`):

  | kernel | Vulkan | CUDA | ratio |
  |---|---:|---:|---:|
  | Q8_0 GEMM 4096x1024x1024 | 8.37 TFLOPS | 26.29 TFLOPS | 3.14x |
  | Q8_0 GEMM 1024x4096x1024 | 7.53 TFLOPS | 26.81 TFLOPS | 3.56x |
  | Q4_K GEMM 4096x1024x1024 | 4.92 TFLOPS | 25.45 TFLOPS | 5.17x |
  | dense F16 4096x1024x1024 | 3.06 TFLOPS | 23.47 TFLOPS | 7.67x |
  | dense F32 4096x1024x1024 | 4.35 TFLOPS | 3.32 TFLOPS | 0.76x |

  Every q8 scenario is bounded near 2-3x by this alone. Dense F32 is the one
  kernel where Vulkan wins, which is why TRELLIS.2 F32 sparse-conv stages have
  lower ratios despite being dispatch-bound.
* **CPU floor dilutes backend ratios.** TRELLIS.2 `uv_unwrap` (xatlas, 26k
  charts) is 182 s of the 193 s texture-export stage on CUDA; the bake adds
  8.9 s. `trellis2_texture_mesh` is 292.11 s on Vulkan vs 258.78 s on CUDA
  (1.13x) because both backends wait on the same CPU work.

### Remaining dispatch-bound work

TripoSG batch / scribble dense decode (16.7M-67M SDF queries, ~40 fixed
dispatches per 32k chunk) and TRELLIS.2 sparse conv (27 offsets x
index_select + matmul + add) still pay Vulkan's per-dispatch submit cost.
VGGT tracking is no longer in this group.

### Measured dead ends (2026-11-21)

* Reusing the VGGT DPT feature maps across auto-FPS passes: measured neutral
  before the norm fix (20.99 vs 21.28 s), because the 6.47 s norm ran per
  tracker call. Not revisited after the fix: the extractor is now 0.26-1.42 s
  per pass, and the API change would only remove that.
* TripoSG flash decoder with a 32k chunk: image mesh regressed 19.4 -> 22.95 s;
  kept the 16384 cap.
* `LUX3D_VULKAN_DEQUANT_WEIGHTS` for dense F16: image 7.47 -> 13.43 s infer.
* TripoSR 65536 query chunk: slower than 32768 through allocator churn.
* VGGT inflight budget 512 MB vs 1 GB: 17.57 s vs 24.34 s.

## Whole-model criterion on candle b2522dd6 (2026-09-07)

Fresh 10-sample criterion medians, sequential runs, weights resident. This is
the first full 9-cell sweep on the post-quick-win candle (wgpu conv_transpose2d
gather `a1644634`, vulkan padded-B ALIGNED GEMM `5f5eb526`):

| bench | cuda | vulkan | wgpu | vs efb506c9 base: cuda / vulkan / wgpu |
|---|---:|---:|---:|---|
| pi3/iter | 2.72s | 3.24s | 5.79s | 1.00x / 0.97x / 1.42x |
| pi3x/iter | 4.07s | 5.0-7.2s (see note) | 10.56s | 1.01x / ~1.0x / 2.27x |
| triposr/iter | 0.93s | 1.44s | 2.53s | 1.33x / 2.33x / 1.93x |

(efb506c9 baselines: pi3 2.72/3.14/8.21s, pi3x 4.09/4.90/24.0s, triposr
1.23/3.36/4.89s. wgpu pi3x 24.0 -> 10.6s is mostly the conv_tr gather +
padded GEMM series; wgpu triposr base ran cold-pipeline and warm cache
accounts for part of the further 3.64 -> 2.53s drop in this sweep.)

**vulkan_pi3x timing noise, and the false regression.** An earlier sweep
recorded 7.55s for vulkan_pi3x (+54% vs the 4.90s efb506c9 baseline) and
looked like a pad-commit regression. Attribution work (per-shader dispatch
histogram via a RAII wall-time guard, back-to-back on worktrees 65b33be2 vs
HEAD): dispatch wall 9.0s (HEAD) vs 9.7s (65b33be2) over 2 iters — HEAD is
FASTER; the pad path moved 644 scalar `matmul_f32_f32_fp32` calls (~1ms
each) onto `matmul_f32_f32_aligned_fp32` (~23us each). Criterion sample
spread on vulkan_pi3x is enormous: {5.03..9.85}s within ONE 10-sample run
(cuda reproduces to +-0.5%). Conclusion: no code regression; the 7.55s was
WDDM run-to-run variance. Report vulkan_pi3x as "min ~5.0s, median 6-7s
depending on ambient". cuda_pi3x back-to-back re-run during the sweep
reproduced exactly (4.068 vs 4.087s), confirming the noise is
backend-specific (WDDM queue/sync), not machine drift.

## Verified state

Inference wall (`[stage] infer`, single iteration per process):

| model | cuda | vulkan | wgpu | vulkan/cuda | wgpu/cuda |
|---|---:|---:|---:|---:|---:|
| pi3 (5 frames, 518x518) | 4.82s | 7.83s | 13.96s | 1.62x | 2.90x |
| pi3x (6 frames, 518x518) | 7.00s | 11.10s | 31.84s | 1.59x | 4.55x |
| triposr (single image) | 1.23s | 3.36s | 4.89s | 2.73x | 3.98x |

(cuda columns from the earlier same-day window; vulkan/wgpu measured after
the fused layernorm+rope kernel. wgpu improved 82.9s -> 13.96s on pi3 and
88.7s -> 31.84s on pi3x.)

Memory is flat under repeated inference on all three backends (wgpu
10-iteration bench after the fused kernel: rc=0, VRAM peak 10.4 GiB, zero
errors; cuda pool step eliminated by SDPA chunking; vulkan bounded by
grace-gated inflight drain + 2 GiB pool).

All mesh comparisons vs the cuda reference PASS on every backend (3 models
x vulkan+wgpu after the fused kernel; unit test exact to 0.0 on both).

## Fused layernorm+rope (2026-09-06, rev efb506c9) — the lever that moved

`rope_layernorm.comp/.wgsl`: one 64-lane workgroup per (b, heads, n, 64)
row reads the STRIDED qkv head view directly, does biased-var LayerNorm +
affine, then `out[j] = y[j]*cos[j] + y[j^quarter]*sin_alt[j]` (negations
pre-baked in the lux3d sin tables; partner index is a XOR because quarter
is a power of two). Routed via `candle_nn::ops::layernorm_rope_fused` /
`rope_fused` (CustomOp3, vulkan_fwd + wgpu_fwd) at every rope site: 72
per pi3 decode, plus pi3x core/cross attention. This collapses the
per-head-norm slow tensor-op chain (forced by the non-contiguous qkv
view), both surrounding `.contiguous()` copies, and the 8-op rope chain —
~12 dispatches per q/k — into one kernel.

Note: `sin_alt_full` comes back NON-contiguous from the cat/affine table
construction on GPU backends; the old broadcast path never cared, the
fused kernel's flat table indexing does — tables are materialized
contiguous in `Rope2d::embeddings`.

Warm criterion, one back-to-back window (10 samples/cell, patch build =
committed code):

| bench | cuda | vulkan | wgpu | prev vulkan | prev wgpu |
|---|---:|---:|---:|---:|---:|
| pi3/iter | 2.72s | 3.14s (1.16x) | 8.21s (3.0x) | 3.59s (1.27x) | 54.9s (19.5x) |
| pi3x/iter | 4.09s | 4.90s (1.20x) | 24.0s (5.9x) | 5.13s (1.22x) | 60.4s (13.9x) |

The wgpu wall was inter-pass driver gap on the strictly dependent chain;
removing ~860 of ~9.5k passes (plus their GPU time) cut pi3 by 6.7x.
Vulkan gains are smaller (submissions batch many dispatches), but the
warm ratio dropped to 1.16x/1.20x. Remaining warm gap: 0.42s/0.81s per
iter — dominated by the structural WDDM submit tax, not kernel count.

## Measured dead ends (do not retry on this hardware)

- single-submission graph (huge batch caps): 27.3s — VRAM thrash on 12 GiB.
- CANDLE_VK_MAX_BATCH_DISPATCHES=512 + compute 4 GiB: 14.9s; 1024/8 GiB: 18.0s.
- wgpu batch 64: 6x WORSE (uniform ring reuse waits); grace band on wgpu: worse
  (retirement only advances inside device.poll).
- scalar flash_attn on vulkan: 29.9s; coopmat FA2 (`flash_attn_f32_f16_f32_cm1`,
  enabled for the measurement): 12.83s vs 11.43s chunked — thousands of small
  per-tile dispatches pay the WDDM fence-signal tax and f32 K/V locality is
  worse than two large coopmat GEMMs. Chunked two-pass stays the default.

## Warm-loop criterion results (2026-09-06, rev e0e24758)

10-sample criterion runs (steady state, weights resident — the candle-bench
protocol):

| bench | cuda | vulkan | ratio |
|---|---:|---:|---:|
| pi3/iter | 2.80s | 3.34s | 1.19x |
| pi3x/iter | 4.21s | 5.13s | 1.22x |

The warm gap is much smaller than the single-shot gap (cold allocator/first
iteration dominates the single-shot numbers).

## MUL_MAT_ADD — implemented (guarded)

mul_mm.comp BIAS_ADD epilogue + `_bias` variants from vulkan-shaders-gen +
`VulkanStorage::matmul_bias` + `candle_nn::ops::mul_mat_add` (CustomOp3,
vulkan-routed from lux3d `linear_fwd`). Fused ONLY for verified shapes
(m%64==0 && n%64==0 && k%32==0, m>8, f32 -> matmul_f32_f32_aligned_cm1_bias);
the unaligned cm1 staged-store epilogue writes bias without the matmul
contribution on some tiles, so those shapes stay unfused (llama.cpp upstream
does not fuse bias into coopmat either). Unit test: fused vs unfused on the
same device.

## Warm-loop wgpu (2026-09-06, rev bba156a7, post-FA2)

| bench | cuda | wgpu | ratio |
|---|---:|---:|---:|
| pi3/iter | 2.80s | 76.9s | 27.5x |
| pi3x/iter | 4.21s | 76.3s | 18.1x |

Zero OOM/invalid across the 10-sample runs. The remaining gap is
GPU-kernel-bound (WGSL GEMM + elementwise quality vs cuBLAS); a
competitive WGSL GEMM remains the one open lever, sized as its own
project (see plan below).

## MUL_MAT_ADD batch>1 failure — ROOT-CAUSED (2026-09-06, rev 64bef917)

Not a barrier or stage-aliasing issue: presenting the weight with a
single real batch made the kernel's `batch_idx_a = i03*ne02 + i02` walk
past the 1-batch weight buffer for batches 1..n-1 — the (3,64,128,256)
probe lost exactly two full batch planes (bias-only output). Fix:
`candle_nn::ops::mul_mat_add` now feeds the weight as a stride-0
broadcast view (`unsqueeze(0).broadcast_as(batch,k,n)`), matching
candle's own broadcast_matmul; `batch_stride_a` becomes 0 and every
batch reads weight batch 0. The shape-matrix unit test is exact on ALL
shapes (aligned, unaligned cm1, batch>1); the alignment/batch gates are
removed.

Fusion remains opt-in (CANDLE_LUX3D_FUSED_LINEAR=1): on the 12 GiB WDDM
card fusing the dense pi3x linears is net-negative (warm criterion
5.13s -> 8.85s median + OOM risk — removing the add dispatches shifts
batch/allocator dynamics over the VRAM edge).

Measurement caveat learned the hard way: warm vulkan numbers drift
strongly with ambient WDDM load (identical binary: 5.13s and 9.7s
within one day; cuda stable at 4.21s). vulkan/cuda ratios are only
meaningful within a single back-to-back measurement window.

The forced staged coopmat epilogue writes bias without the matmul
contribution on some tiles when batch > 1 (RTX 3060, current driver;
batch=1 identical shapes are exact; the unaligned scalar variant is
exact within accumulation-order noise). Shape-matrix unit test added
(candle-nn `mul_mat_add_tests`); the tensor-level gate requires
batch == 1 on top of m/n/k alignment. Production multi-view linears
(batch = num_views, m = 8244) stay unfused until this is root-caused;
candidates: subgroup-scope barrier semantics after coopMatStore to
shared, or stage slot aliasing across cms_per_row/cms_per_col fragments.

## Back-to-back warm criterion, one window (2026-09-06, rev e69fe16d)

| bench | cuda | vulkan | wgpu |
|---|---:|---:|---:|
| pi3/iter | 2.82s | 3.59s (1.27x) | 54.9s (19.5x) |
| pi3x/iter | 4.34s | 9.36s (2.16x)* | 60.4s (13.9x) |

*The vulkan_pi3x window hit the slow side of the ambient-WDDM drift (the
same build measured 1.22x earlier; cuda is stable). wgpu improved from
76.3s/76.9s to 60.4s/54.9s after enabling coop64 for unaligned shapes.

## WGSL GEMM attribution (wgpu GPU-timestamp profiler, rev 3241f1a5)

pi3x GPU time by kernel (before the coop fix): matmul-warptile 515x
20783ms (95% of GPU, 40.4ms/ea ~ 0.43 TFLOPS); the coop64 gate required
m/n/k % 16 while pi3x linears are m=3903/1301. The kernel already
zero-fills ragged shared-tile loads and clips stores, so the host
%16 checks were removed: matmul-coop64 521x 2127ms (4.1ms/ea, 10x).
GPU total ~5.9s of the ~70s wall — the remaining wgpu gap is CPU-side
dispatch/pass overhead (~9.5k passes x ~6ms WDDM), the same structural
tax as vulkan; pass-folding for provably independent dispatches is the
next wgpu lever.

## Pass folding (rev 029c5acc) — implemented, NEUTRAL on dense models

encode_pending_dispatches folds consecutive deferred dispatches whose
storage-buffer sets are pairwise disjoint into one compute pass (uniform
ring buffers excluded from the dependency set). pi3x wgpu: 70.5s vs
69.9s, mesh compare PASS — the dense-model dispatch chain is almost
strictly dependent (consecutive ops share the activation tensor), so
the fold rate is near zero. Conclusion: the ~64s of wall over ~5.9s GPU
is inter-pass driver gap on a strictly dependent chain — removable only
by dispatch-count reduction (op fusion in the model graph), not by
pass-level folding. vulkan_pi3x drift (2.16x evening vs 1.22x morning
at identical builds): cuda stays 4.21s across windows while vulkan
swings 5.1-12.2s — ambient WDDM/driver state, not code; A/B on the
same build confirmed it.

## Dispatch-reduction levers — final state (2026-09-06)

Every reduction candidate has been measured on the strictly dependent
chain; the fused layernorm+rope kernel above is the one that paid:

| lever | result |
|---|---|
| FUSED LN+ROPE (new kernel, both backends) | LANDED — wgpu pi3 54.9s -> 8.21s warm, vulkan 1.16x/1.20x warm |
| MUL_MAT_ADD bias epilogue (unaligned cm1, exact after broadcast fix) | NET-NEGATIVE on 12 GiB: warm 5.13s -> 8.85s median + OOM risk; stays behind CANDLE_LUX3D_FUSED_LINEAR |
| SDPA single-shot on vulkan (no chunking; allocator keeps transients flat) | warm NEUTRAL (5.25s vs 5.13s) with a 4 GiB scores transient pushing VRAM peak to 11.4/12.3 GiB — reverted, chunking stays |
| pass folding (wgpu) | NEUTRAL alone — the dispatch chain is almost strictly dependent; the win came from removing dispatches at the source (fused kernel) |
| rope whole-tensor reformulation | 17 -> 8 ops, wall-neutral alone; superseded by the fused kernel |

What remains after the fused kernel is the structural WDDM submit tax:
~4-9 ms per submission on vulkan and inter-pass gaps on wgpu, on a chain
that is still strictly dependent (consecutive ops consume the previous
activation). Closing it in code would require fusing across the large
GEMMs themselves (FlashAttention-style monolith per block), which the
coopmat FA2 measurement already showed loses on this 12 GiB card
(12.83s vs 11.43s chunked). On this hardware the remaining gap is a
platform property: Linux (no WDDM fence-signaling latency) or a larger
VRAM card would close it without code changes.

## Remaining levers (integration plans)

### 1. MUL_MAT_ADD — GEMM with bias epilogue (vulkan)

Expected gain: <=5-10% of vulkan wall. Batches close on compute_bytes (big GEMM
ops), so removing ~500 bias-add dispatches out of 9477 reduces submissions only
indirectly; the direct saving is one activation-size read+write per linear.

Plan:
1. mul_mm.comp: add `#ifdef BIAS_ADD` binding 5 (`readonly buffer BIAS
   {float data_bias[];}`); at each store site (coopMatStore paths ~484/496,
   scalar ~526/529) add `data_bias[col]` where col is the output column index
   already computed at that site.
2. build.rs: register `matmul_f32_f32_fp32_bias` / `..._cm1_bias` variants
   (copy the non-bias entries + BIAS_ADD define) — coopmat glslc support is
   already probed at line 80.
3. vulkan_backend.rs: extend the matmul descriptor layout with the 5th
   binding only for bias pipelines; new `pub fn matmul_bias(...)` that mirrors
   `run_matmul_f32` and passes the bias storage.
4. lux3d-core: `CustomOp3` FusedLinear (cpu/cuda fwd = matmul+add fallback;
   vulkan_fwd = matmul_bias) applied only on vulkan devices at Linear call
   sites (pi3_encoder.rs, pi3_decoder.rs, pi3x.rs).
Risk: descriptor-layout plumbing in vulkan_backend.rs (pipeline registry) is
the only unfamiliar part; everything else is mechanical.

### 2. WGSL GEMM (wgpu)

Before writing a third kernel: build a wgpu GPU-timestamp profiler (mirror the
vulkan one) to attribute the remaining ~80s of pi3x wall between GPU and CPU.
The backend already ships matmul_warptile (64x64 tile, BK=32) and coop64
(0.53ms/1024^3 measured); wgpu-llm's gemm.wgsl (16x16 tile) is simpler than
either and unlikely to win. If attribution shows GEMM time dominates, port the
born matmul tile scheme with subgroup ops instead.

### 3. Attention on vulkan

Chunked two-pass with coopmat GEMMs is the measured optimum for F32 vision
attention on WDDM. Revisit only after MUL_MAT_ADD reduces the dispatch count.

## Vulkan quantized GEMM: why 90% of CUDA is unreachable (2026-11-22, rev 9c74af75)

**Correction (same day, after the CPU profile):** the quantized GEMM is not what
sets the Vulkan wall time. On TripoSG image-to-mesh the GPU kernels sum to
1.9 s of a 24-26 s wall, and on triposr to ~0.1 s of 4 s. What dominates is
candle's blocking `flush_drain` on WDDM: 486 drains x 31 ms = 15.2 s for
TripoSG and 86 x 24 ms = 2.1 s for triposr, versus zero on CUDA. The GEMM
measurements below still bound how much a better kernel could win -- very
little, because GEMM is 320 ms of the TripoSG wall -- and they are the reason
the f16 coopmat work was stopped.

GPU kernel totals from `LUX3D_GPU_PROFILE=1 CANDLE_VULKAN_GPU_PROFILE=1`:

| scenario | wall | sum of kernels | blocking drains |
|---|---|---|---|
| `triposg_image_to_mesh` | 24-26 s | 1.9 s | 486 x 31 ms = 15.2 s |
| `triposr_image_to_mesh` | 4-5 s | ~0.1 s | 86 x 24 ms = 2.1 s (0 in one run -- state dependent) |

The flush reasons are `batch_limit` and `cap_compute_bytes` (the per-batch
512 MiB compute cap), so each batch is submitted and then the 256 MiB
in-flight budget forces a blocking fence wait. The wait costs ~31 ms on this
driver (the comment in `vulkan_inflight_byte_budget` assumes 5-10 ms).
Knob sweep on TripoSG (wall / peak VRAM, idle 714 MiB):

| setting | wall | peak VRAM |
|---|---|---|
| default (budget 256 MiB, grace 2) | 26 s | 6.83 GiB |
| `CANDLE_VK_INFLIGHT_GRACE=1` | 30 s | 5.29 GiB |
| `CANDLE_VULKAN_INFLIGHT_MAX_BYTES=128MiB` | 30 s | 5.29 GiB |
| `CANDLE_VULKAN_INFLIGHT_MAX_BYTES=512MiB` | 25 s | 9.07 GiB |
| `CANDLE_VULKAN_INFLIGHT_MAX_BYTES=1GiB` | 55 s | 11.35 GiB |
| `CANDLE_VK_POOL_MAX_BYTES=256MiB` | 25 s | 8.11 GiB |
| `CANDLE_VK_MAX_BATCH_COMPUTE_BYTES=128MiB` | 84 s | 11.38 GiB |
| CUDA reference | 13.8 s | 3.91 GiB |

So the speed-optimal defaults also maximise memory, and the memory-reducing
settings cost 15-20% wall. Memory parity with CUDA is not reachable through
these knobs: Vulkan keeps ~2.9 GiB more resident during sampling (timeline:
CUDA is flat at 3.9 GiB from load to exit; Vulkan climbs to 6.8 GiB during the
sampling phase and drops to 4.7 GiB after). Both the drain cost and the
retained-bytes accounting are candle-internal.

Primitive ceilings measured on the RTX 3060 (4096^3, candle `QMatMul`, one
process per configuration) -- useful as an upper bound, not as the wall-time
explanation:

| path | TFLOPS |
|---|---|
| CUDA Q8_0 (`mma.sync m16n8k32 s8.s8.s32`) | 26.29 |
| Vulkan Q8_0 dp4a (`mul_mmq.comp`, production) | 8.37 |
| Vulkan Q8_0 INT8 coopmat, 4 accumulators | 4.52 |
| Vulkan Q8_0 INT8 coopmat, ILP-16 | 5.17 |
| Vulkan Q8_0 INT8 coopmat, no per-block epilogue (invalid numerics) | 8.30 |
| Vulkan Q8_0 f16 coopmat, dequant folded into staging | 9.73 |
| ... same, MMAs only (staging hoisted out, invalid numerics) | 19.02 |

Three findings, each measured rather than inferred:

1. **INT8 cooperative matrices cannot replace dp4a.** CUDA applies the
   per-32-block `d_a*d_b` scale to the int32 accumulator in registers.
   `GL_KHR_cooperative_matrix` has no element accessor: glslc rejects
   `coopMatGet` for both int and float accumulators, so the only way to scale
   a block is `coopMatStore` to shared, barrier, read back, scale, accumulate.
   That epilogue costs more than the MMA it scales -- removing it (invalid
   numerics) takes the same kernel from 5.17 to 8.30 TFLOPS, still only at
   dp4a parity.

2. **f16 cooperative matrices with the scale folded into the staged operand
   work, but are staging bound.** Staging `q*d` as f16 lets a single f32
   accumulator span all of K, so there is no epilogue at all. The MMA path
   alone runs at 19.02 TFLOPS (75% of the 25.5 TFLOPS f16 tensor peak); the
   full kernel reaches 9.73 TFLOPS at `K_STEP=32` (14 KB shared, three
   workgroups per SM). `K_STEP=64` (7.45) and 8 warps over a 64x128 tile
   (5.64) are both worse: occupancy, not instruction count, sets the staging
   rate. Replacing the spec-constant divisors with literals (glslc emits
   `OpUDiv`/`OpUMod` when it cannot prove they are powers of two) was worth
   2.3x on its own at `K_STEP=32`.

3. **A better GEMM kernel cannot deliver the target, and neither can the
   GEMM ceiling alone explain the gap.** Vulkan Q8_0 would need 23.7 TFLOPS to
   be 90% of CUDA's 26.29; the f16 tensor-core peak is 25.5 and the best MMA
   rate measured here is 19.02, before any data movement. But even an
   infinitely fast GEMM would only remove 0.32 s from the 24-26 s TripoSG
   wall. The wall is set by the blocking WDDM drains above, so a candle-side
   fix has to target submission retirement and retained-buffer accounting
   first, and the GEMM second.

Design notes if a candle-side implementation is ever picked up:

- llama.cpp's Vulkan coopmat path (`mul_mm.comp` + `dequant_funcs.glsl`) folds
  the scale into the shared f16 tile exactly this way and is the reference to
  mirror.
- Staging must use literal tiling constants, not specialization constants.
- The remaining 2x on staging is global-load latency: A's 34-byte Q8_0 blocks
  only allow 2-byte-aligned loads, and one thread currently issues two
  dependent loads per four values. Wider per-thread copies (one full 32-value
  block for A, one full 16-byte Q8_1 word for B) and a uniform
  `full_m`/`full_n` fast path that drops the per-row bounds check are the next
  things to try.
- Numerics of the f16 path: f16 operands, f32 accumulate, nmse 1.1e-8 and
  max_rel 1.9e-2 against the CPU reference at 1024^3 -- the same arithmetic as
  dequantizing to f16 and running a dense f16 matmul.

The earlier experiments lived outside this repo, in the scratch candle clone
`G:\jobs\candle-based-projects\candle-int8-cm`. The work now happens directly in
the user's candle checkout (`G:\jobs\candle-based-projects\candle`):
`candle-vulkan-kernels/src/shaders/mul_mmq_cm.comp` plus the
`CANDLE_VK_Q8_COOPMAT` selector in `candle-core/src/vulkan_backend.rs`. That
checkout is left uncommitted — the user commits candle themselves.

## Where the Vulkan/CUDA gap actually is (2026-11-23, rev 9c74af75 + local patch)

Same-session per-op comparison on the TripoSG VAE flash shapes
(`cargo run --release --example flash_bench -p lux3d-core --features vulkan`,
`--iters 6..10`, min shown; CUDA from `target-cuda`, CANDLE_VULKAN_DEVICE_NAME
pinned to the RTX 3060). Shapes: attention/ff at a 16384-point outer chunk,
dim 1024, ff 4096, kv_len 1024.

| op | CUDA | VK dp4a (default) | VK f16 coopmat | VK int8 coopmat (old) |
|---|---|---|---|---|
| q8 to_q/to_out `16384x1024@1024x1024` | 1.220 ms | 3.617 ms | 1.971 ms | 6.398 ms |
| q8 ff_in `16384x1024@1024x4096` | 4.058 ms | 13.392 ms | 8.417 ms | 23.530 ms |
| q8 ff_out `16384x4096@4096x1024` | 4.474 ms | 13.255 ms | 10.519 ms | 27.789 ms |
| f32 softmax `1x8x1024x1024` (CUDA at its 6144 q-chunk) | 8.289 ms | 0.706 ms | 0.749 ms | 0.709 ms |
| layernorm `16384x1024` | 1.123 ms | 0.615 ms | 0.623 ms | 0.616 ms |

So the exact-mode gap is one op: the Q8_0 GEMM (dp4a 9.5-10.3 TFLOPS vs CUDA's
int8 tensor-core mmq at ~34 TOPS), i.e. 3.3x. Everything else in the VAE
decoder is at parity or faster on Vulkan (the CUDA-side softmax in candle is
what is slow there, ~11x the Vulkan kernel on the same logical work).

Measured dead ends on this hardware (do not retry):

- dp4a tile sweep (`CANDLE_VK_Q8_TILE`): 64x128 = 21.66 ms, 128x64 = 14.46 ms,
  default 64x64 = 13.84 ms on q8 ff_in.
- Deeper submission queues (`CANDLE_VK_MAX_PENDING_SUBMISSIONS` 96/128/160/256):
  `VK_ERROR_DEVICE_LOST` at the SDF decode even with the default batch size.
  WDDM resets the GPU past a queue depth of 64 (already documented in the
  backend); `CANDLE_VK_MAX_BATCH_DISPATCHES=256` with pending 256 crashes the
  same way, and batch 128 alone is neutral.
- f16 coopmat as the default: 1.45-2.35x faster than dp4a, but the f16 operand
  rounding moves the isosurface (8738 vs 10528 vertices, -17%), so it stays
  opt-in.
- The old int8 coopmat kernel (0.47-0.80x of dp4a): the opaque MMA register
  layout forced a shared spill + subgroup barrier per 32-wide block and a
  1152-byte-strided SSBO read per MMA tile.

Why a bigger workgroup tile is the lever, not another staging tweak: with a
64x64 tile the weight matrix is re-read `n/64 = 256` times for ff_in
(4.25 MiB x 256 = 1.1 GiB) and the q8_1 activations `m/64 = 64` times
(18.9 MiB x 64 = 1.2 GiB). The DP4A kernel is at ~2.3 GiB of traffic for
68.7 GMAC, i.e. roughly half memory-bound; CUDA's MMQ uses 128-wide tiles
(and stream-k) which halves both counts.

### Exact int8 tensor-core mmq: rewritten, bit-exact, still slower (measured)

`mul_mmq_cm.comp` was rewritten to amortize everything around the mandatory
per-32-block spill: one staging group covers FOUR blocks (KG = 128) into dense
byte panels (A quants stripped of scales through the existing
`block_q8_0_packed16` view, B quants as 128-bit `ivec4` words of a
`block_q8_1_x4` record), the panels are padded to a 144-byte stride, and all
four cm tiles of a block spill into a warp-private 4 KiB stage between two
subgroup barriers. Shared usage 23.5 KiB (2 workgroups/SM); the host gate for
the path now also requires `subgroup_size == 32`, because the staging hands
each panel row/column to two threads and 2 * BM must equal the workgroup size.

Bit-exactness against dp4a holds: `q8_check` dumps of a 256x512@512x512 Q8_0
matmul compare at `max_abs = 0`, `rms_rel = 0`.

Timings (RTX 3060, flash_bench, min of 6):

| variant | to_q | ff_in | ff_out |
|---|---|---|---|
| dp4a (default) | 3.60 ms | 13.39 ms | 13.23 ms |
| int8 cm, unpadded 128 B panel stride | 12.81 ms | 48.29 ms | 50.87 ms |
| int8 cm, 144 B padded stride, 1-tile stage (8 barriers/block) | 6.00 ms | 21.52 ms | 24.10 ms |
| int8 cm, 144 B padded stride, 4-tile stage (2 barriers/block) | 5.86 ms | 21.23 ms | 23.57 ms |
| f16 cm (`CANDLE_VK_Q8_COOPMAT=f16`), inexact | 1.83 ms | 8.29 ms | 10.46 ms |

Two findings worth keeping:

- The 128-byte panel stride was a 16-way shared bank conflict: every row of a
  tile started on bank 0. Padding to 144 B (36 words, still a 16-byte multiple)
  took ff_in from 48.3 to 21.5 ms.
- Barrier count is not the limiter: the tile-serialized (8 barriers/block) and
  the four-tile (2 barriers/block) forms are within noise of each other. The
  int8 path is instruction-bound in the **epilogue**: the opaque MMA register
  layout cannot be scaled in place, so every output element costs a shared
  store + load + FMA per 32-wide block (32 outputs/thread for a 64x64 tile =>
  ~384 epilogue instructions per 128-K group against 16 MMA instructions).
  dp4a needs no spill, so exact mode stays on dp4a. Tensor cores only pay off
  when the block scale folds into the operand, which is the f16 path (and its
  operand rounding is what shifts the isosurface).

## Harnesses

- bench: `LUX3D_BENCH_DEVICE=... LUX3D_BENCH_MODEL=... cargo bench -p
  lux3d-core --bench bench_main --features vulkan,wgpu -- <backend>_<model>`
  (one process per pair; `.swarm/tools/bench_one.sh` adds VRAM/RAM sampling).
- profiles: `CANDLE_VULKAN_CPU_PROFILE=1` (per-phase CPU + flush reasons),
  `CANDLE_VULKAN_GPU_PROFILE=1` + `LUX3D_GPU_PROFILE` (kernel report).
- correctness: `python .swarm/tools/compare_mesh.py compare <ref> <out>`
  against `.swarm/out/perfprof/*_cuda_ref.*`.
