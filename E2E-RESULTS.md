# Lux3D real-weight e2e results

Manual run of the scenario suite in [`E2E.md`](E2E.md) against the weights that
are already on this machine.

## Environment

| Item | Value |
|---|---|
| Date | 2026-11-21 |
| Host | Windows 10.0.26200 x64, Git Bash |
| GPU | NVIDIA GeForce RTX 3060, 12 GiB, compute 8.6 |
| CUDA | 12.8 |
| Rust | 1.97 stable, MSVC |
| Build | `cargo build --release -p lux3d-cli --features cuda,vulkan,wgpu` |
| Test binary | `target/release/deps/e2e_scenarios-e61b5cb6fa13cc66.exe` |
| Serialization | `GpuTestLock` + `--test-threads=1` |

## Command

```bash
cargo build --release -p lux3d-cli --features cuda,vulkan,wgpu
cargo test --release -p lux3d-cli --features cuda,vulkan,wgpu --test e2e_scenarios --no-run
LUX3D_E2E=1 \
LUX3D_E2E_REQUIRE=1 \
LUX3D_E2E_DEVICE=cuda \
./target/release/deps/e2e_scenarios-e61b5cb6fa13cc66.exe --test-threads=1 --nocapture
```

Every weight and asset path resolved locally; `LUX3D_E2E_REQUIRE=1` means a
missing file is a failure, not a skip.

## Backend matrix

All 18 scenarios were run end to end on each backend with the same test binary
and weights, changing only `LUX3D_E2E_DEVICE`. The TripoSG tests select
`LUX3D_WGPU_DEQUANT_WEIGHTS=0` on WGPU (packed Q8 weights); other WGPU tests
keep the runtime default.

| Backend | Result | Suite wall time | Notes |
|---|---|---|---|
| CUDA | **18/18 passed** | 955.13 s (15m55s) | Same-binary reference run. |
| Vulkan | **18/18 passed** | 1717.21 s (28m37s) | Same-binary run after the fixes below; 1.54x faster than the 2638.52 s pre-fix baseline. |
| WGPU | 18/18 on the previous build | 3466.58 s (57m46s) | Not re-run after the 2026-11-21 changes; TripoSG still needs packed Q8 (`LUX3D_WGPU_DEQUANT_WEIGHTS=0`). |

Overall Vulkan/CUDA suite ratio: **1.80x** (was 2.62x on the pre-fix baseline).
Per-scenario ratios are in the table below.

## Automated scenario results

| Test | Family / scenario | Weights | CUDA | Vulkan | Vk/CUDA | Artifact |
|---|---|---|---|---|---|---|
| `pi3_frames_to_point_cloud` | Pi3 multi-frame point cloud | `3d/standalone/pi3-q8_0` | 3.57 s | 6.99 s | 1.96x | `cloud.ply` 25 MiB |
| `pi3x_frames_to_point_cloud` | Pi3X point cloud | `3d/standalone/pi3x-q8_0` | 5.27 s | 9.53 s | 1.81x | `cloud.ply` 37 MiB |
| `pi3x_vo_inject_to_point_cloud` | Pi3X VO + pose/depth/ray injection | `3d/standalone/pi3x-q8_0` | 6.51 s | 10.84 s | 1.67x | `vo.ply` 37 MiB |
| `triposr_image_to_mesh` | TripoSR image-to-mesh | `3d/standalone/triposr-q8_0` | 2.37 s | 5.31 s | 2.24x | `mesh.glb` 628 KiB |
| `triposg_image_to_mesh` | TripoSG image-to-mesh, RMBG-1.4 RGB fallback | `3d/standalone/triposg-q8_0` + `rmbg-1.4` | 14.42 s | 28.02 s | 1.94x | `mesh.glb` 308 KiB |
| `triposg_batch_to_meshes` | 2 images x 2 shapes in one batch | `3d/standalone/triposg-q8_0` | 272.66 s | 646.21 s | 2.37x | `batch_0..3.glb` |
| `triposg_surface_to_mesh` | VAE surface encode/decode | `3d/standalone/triposg-q8_0` | 23.95 s | 59.92 s | 2.50x | `surface.glb` 18 MiB |
| `triposg_scribble_to_mesh` | Scribble image + text prompt | `3d/huggingface/TripoSG-scribble-GGUF/triposg-scribble-q8_0.gguf` | 49.00 s | 141.01 s | 2.88x | `scribble.glb` 9.2 MiB |
| `vggt_frames_to_point_cloud` | VGGT multi-frame point cloud | `3d/standalone/vggt-q8_0` | 5.54 s | 10.22 s | 1.84x | `cloud.ply` 11 MiB |
| `vggt_depth_points` | VGGT depth+confidence unprojection | `3d/standalone/vggt-q8_0` | 5.80 s | 11.00 s | 1.90x | `cloud.ply`, `depth.ply` 36 MiB |
| `vggt_glb_scene` | VGGT GLB scene with cameras | `3d/standalone/vggt-q8_0` | 5.65 s | 10.58 s | 1.87x | `scene.glb` 6.2 MiB |
| `vggt_colmap_feedforward` | VGGT feedforward COLMAP export | `3d/standalone/vggt-q8_0` | 5.58 s | 10.64 s | 1.91x | `colmap/` binary model |
| `vggt_track_queries` | Explicit query tracking | `3d/standalone/vggt-q8_0` | 6.34 s | 11.27 s | 1.78x | `tracks.json` |
| `vggt_track_auto_fps` | DINOv2 + FPS automatic queries | `vggt-q8_0` + `models/dinov2_reg4` | 7.72 s | 13.43 s | 1.74x | `tracks.json` 37 KiB |
| `trellis2_512_geometry` | TRELLIS.2 512 geometry, decimation | `3d/standalone/trellis2-q4_k` + `rmbg-2.0` | 35.48 s | 86.85 s | 2.45x | `geometry.glb` 1.4 MiB |
| `trellis2_512_textured_glb` | TRELLIS.2 512 PBR GLB | `3d/standalone/trellis2-q4_k` + `rmbg-2.0` | 245.56 s | 362.40 s | 1.48x | `textured.glb` 3.6 MiB + PBR sidecars |
| `trellis2_texture_mesh` | PBR bake onto an existing cube mesh | `3d/standalone/trellis2-q4_k` + `rmbg-2.0` | 258.78 s | 292.11 s | 1.13x | `cube_textured.glb` 116 KiB |
| `ovox_roundtrip` | mesh2ovox -> ovox2mesh/ovox2glb | none | 0.64 s | 0.61 s | 0.95x | `cube.vxz`, `cube.ply`, `cube.glb` |

`ovox_roundtrip` is the sum of its three subcommands.

CUDA full-suite result: **18/18 passed** in **955.13 s** (15m55s), serialized
on one GPU. Vulkan is **18/18 in 1717.21 s** (28m37s) with the same binary and
weights. `target/e2e-scenarios/` keeps every artifact for manual inspection.

## Vulkan vs CUDA: root causes and fixes (2026-11-21)

The pre-fix Vulkan suite was 2638.52 s (2.62x CUDA). The changes below cut it
to 1717.21 s (1.80x). Each item is measurement-gated.

| Fix | Effect |
|---|---|
| TripoSG-scribble standalone GGUF now quantizes the VAE decoder as well as the DiT (`transformer.` + `vae.` default). The old package left the SDF VAE in F16, so the dense 16.7M-point decode ran F16 GEMM (3.4 TFLOPS) instead of packed Q8 (8.5 TFLOPS). | Scribble e2e 764.56 s -> 141.01 s. Quality unchanged: new q8/q6/q5/q4 packages reproduce the DiT-only drift table (q8 CD rel 0.4609% vs 0.461%, F@1% 0.907 vs 0.907). |
| `replication_pad2d` materializes a contiguous input first. Candle's pad builds borders with `narrow` + `cat`; on a channel-last (strided) upsample output the Vulkan `cat` falls back to `copy_strided_src` at single-digit MB/s (~8 s for one [1,64,518,518] pad). | Pi3X 33.36 s -> 9.53 s; Pi3X VO 50.40 s -> 10.84 s. |
| TripoSR SDF query chunk default 8192 -> 32768. The triplane sampler + density MLP issue ~50 tiny dispatches per chunk; 128^3 has 256 chunks at 8192. | TripoSR mesh 8.50 s -> 2.41 s; e2e 13.58 s -> 5.31 s. 65536 regressed through allocator churn. |
| TripoSG/Scribble SDF query chunk default 16384 -> 32768 for the dense/hierarchical grid, and cross-attention K/V projected once per grid instead of once per decoder call. The flash decoder keeps the measured 16384 cap (32k regressed the standard image mesh 19.4 s -> 22.95 s). | Surface 69.20 s -> 59.92 s; image 28.87 s -> 28.02 s; scribble contributes to the 113 s standalone profile (16 steps). Batch is neutral (646.21 s): the 4-shape batch is already memory-bound. |
| VAE self-attention chunk default 1024 -> 6144, matching the DiT knob. | ~4% on the scribble dense decode; no CUDA regression. |
| VGGT `fmap_norm` no longer transposes to channel-last on Vulkan. Candle's `layer_norm` spans only the last axis, so the upstream NHWC permute forced a `copy_strided_src` transpose of the 103 MB `[1,3,128,259,259]` feature map at ~16 MB/s. The replacement reduces over C with a ones-vector GEMM and keeps every operand contiguous (`channel_layer_norm_nchw`, unit-tested against the native op). | `track.fmap_norm` 6.47 s -> 0.01 s; track head 7.14 s -> 0.57 s; `vggt_track_queries` 19.39 s -> 11.27 s; `vggt_track_auto_fps` 36.43 s -> 13.43 s (three passes). |

### Remaining gap is kernel-level, not Lux3D scheduling

The standalone primitive benchmark (`LUX3D_BENCH_KIND=primitives`,
`target/release/deps/bench_main-*.exe`) gives the hard ceiling:

| kernel | Vulkan | CUDA | ratio |
|---|---:|---:|---:|
| Q8_0 GEMM 4096x1024x1024 | 8.37 TFLOPS | 26.29 TFLOPS | 3.14x |
| Q8_0 GEMM 1024x4096x1024 | 7.53 TFLOPS | 26.81 TFLOPS | 3.56x |
| Q4_K GEMM 4096x1024x1024 | 4.92 TFLOPS | 25.45 TFLOPS | 5.17x |
| dense F16 4096x1024x1024 | 3.06 TFLOPS | 23.47 TFLOPS | 7.67x |
| dense F32 4096x1024x1024 | 4.35 TFLOPS | 3.32 TFLOPS | 0.76x |

Every q8-based scenario is therefore bounded near 2-3x before any Lux3D
overhead: Pi3 1.96x, VGGT 1.84-1.91x, TripoSG image 1.94x, surface 2.50x,
scribble 2.88x. Dense F32 is the one kernel where Vulkan is faster, which is
why the TRELLIS.2 sparse-conv stages (F32) sit lower (geometry 2.45x).

The two scenarios that used to sit far above the ceiling are now in range:

* **VGGT track / auto-FPS** dropped from 3.06x / 4.72x to **1.78x / 1.74x**
  after the `fmap_norm` transpose fix. The tracker loop was never the
  problem: `corr` 0.01 s, `corr_mlp` ~0 s and `updateformer` 0.01-0.02 s per
  iteration; the single channel LayerNorm on the 103 MB feature map was 6.47 s.
  The remaining gap is the q8 aggregator (5.0 s of the 11.3 s query run) plus
  the DINOv2/detector passes for auto-FPS (three track passes, 3.1 s total).
* **TripoSG batch / scribble dense decode.** 16.7M-67M SDF queries at ~40
  fixed dispatches per 32k chunk; Vulkan's per-dispatch submit cost dominates
  the 7.4 s of actual GPU kernel time measured in the scribble profile.

TRELLIS.2 textured scenarios are CPU-bound in both backends: `uv_unwrap`
(xatlas, 26k charts) takes 182 s of the 193 s texture-export stage on CUDA,
and the backend-independent bake is another 8.9 s. That is why their ratios
are the lowest (1.13-1.48x) despite the slow sparse-conv decode.

### What would close the remaining gap

The Q8_0/Q4_K Vulkan GEMM kernels are 3.1-5.2x slower than CUDA; no Lux3D
change can recover that. The candle-side levers, in measured order:

1. Q8_0/Q4_K Vulkan GEMM: integer-dot / coopmat path with subgroup reduction,
   matching the CUDA dp4a throughput. This alone bounds every q8 scenario.
2. Vulkan dispatch/flush overhead on the remaining dispatch-bound paths:
   TripoSG dense SDF decode (~40 fixed dispatches per chunk) and TRELLIS.2
   sparse conv. The VGGT tracker is no longer in this group.
3. Sparse-conv dispatch count in TRELLIS.2 (27 offsets x index_select +
   matmul + add per conv) — needs a grouped/offset-fused kernel.

## Manual probes and findings

### Pi3X VO ray injection

The initial automated run failed with:

```text
Error: canonical weights validation failed: point-cloud export rejected a non-finite point
```

`--inject-condition pose` and `--inject-condition depth` passed; `ray` failed.
The next VO chunk re-embeds the previous chunk's normalized rays. The official
Pi3X implementation divides `xy` by `z + 1e-6`; Lux3D divided by `z` directly.
A normalized ray whose squared norm overflowed/underflowed to zero therefore
produced `0 / 0 = NaN`, which reached point export.

Fix: `crates/lux3d-core/src/runtime/pi3x.rs` now uses the reference
`z + 1e-6`. The exact CLI command in the README (`pose,depth,ray`, chunk 2,
overlap 1) now completes and writes a 37 MiB PLY.

### VGGT track head with quantized weights

The first real q8 track run failed with:

```text
Error: canonical weights validation failed: VGGT track head forward failed:
unexpected shape for input [1, 1, 3, 384]
```

The track head's `flow_head` receives a rank-4 `[B, N, S, hidden]` activation.
Dense `candle_nn::Linear` broadcasts over leading dims, but `QMatMul` only
accepts rank 2/3, so the q8 package failed. Fix: the quantized branch of
`nn_blocks::Linear` flattens `[..., in]` to `[rows, in]` and restores the
leading dims. Both explicit and automatic tracking now pass on `vggt-q8_0`.

### VGGT `--vggt-dtype f16` with q8

`--vggt-dtype f16` on the q8 package fails in the camera head with
`dtype mismatch in matmul, lhs: F16, rhs: F32`. The suite uses the documented
reference mode (`--vggt-dtype f32`), which passes. This is a remaining
combination limitation, not covered by the passing scenario.

### TRELLIS.2 default narrow-band remesh

The shipped default (`--remesh` on, band 1.0) was attempted on the 512
pipeline. It spent **1020 s of CPU** on the 1.5M-voxel decode and was stopped
before export. `--no-remesh` completes the same geometry in ~43 s. The
automated tests use `--no-remesh` (plus `--decimation-target 50000`) and the
remesh implementation keeps its unit tests. This is a CPU performance
limitation of the current narrow-band BVH path, not a missing scenario.

### TRELLIS.2 mesh texturing from the standalone GGUF

`--texture-mesh` needs the shape encoder
(`shape_enc_next_dc_f16c32_fp16.safetensors`). Standalone TRELLIS.2 GGUF
packages now embed it as 284 `shape_enc.*` tensors via
`weights quantize --merge-source ... --merge-prefix shape_enc`, so the suite
textures the cube mesh from the same standalone package as the image-to-3D
tests (passed on CUDA, Vulkan and WGPU with `trellis2-q4_k`; the regenerated
`q8_0` package was validated separately on CUDA in 568.11 s, producing the
same 122 KiB textured GLB with two PBR textures). All five variants (`f16`,
`q8_0`, `q6_k`, `q5_k`, `q4_k`) now carry the encoder. The raw
`candle-3d/models/TRELLIS.2-4B` tree and the
`external/shape_enc/model.safetensors` sidecar layout remain supported.

### Surface fixture

The reference surface input is a pickled dict with two independent NumPy
arrays. The test fixture originally stored the same array object under both
keys; the restricted pickle reader does not resolve a shared memo entry twice.
The fixture now writes `surface_normals = points.copy()`.

### WGPU TripoSG VAE at 512 latent tokens

With `LUX3D_WGPU_DEQUANT_WEIGHTS=f16` the TripoSG VAE decoder produced tensors
of exactly 1 MiB (512 tokens × 1024 channels × F16). Candle's WGPU hot-ring
recycles buffers in that size class while one is still bound as a copy source;
the batch then fails validation with

```text
In a CommandEncoder, label = 'candle-wgpu-batch'
  Source and destination cannot be the same buffer
```

The same run passes at 256 and 1024 tokens (different size classes). Lux3D now
loads the TripoSG VAE (decoder and surface encoder) as dense F32 on WGPU:
`nn_blocks::with_force_dense_f32` scopes dequantization to F32 during the VAE
load, keeping the 512-token tensors out of the 1 MiB class. The F32 VAE also
matches the reference decoder precision and fits 12 GiB because the
DINOv2/CLIP/DiT bundle is dropped before the staged decode.

### WGPU dense-F16 batched DiT

Dense F16 weights (`LUX3D_WGPU_DEQUANT_WEIGHTS=f16`) are fine for single-sample
denoise but corrupt a real batch on WGPU: `--num-shapes 2 --steps 4
--num-tokens 1024` produced one 608k-vertex noisy mesh and one empty mesh,
while the same run at batch 1 produced a normal ~10k-vertex mesh. The packed Q8
path (`LUX3D_WGPU_DEQUANT_WEIGHTS=0`) is batch-correct and memory-lean, so the
WGPU TripoSG e2e scenarios use it. The VAE still dequantizes to dense F32.

### WGPU VAE surface encode OOM

The surface-input encoder attends a 2048-point query against the 204800-point
cloud. With the shared 768 MiB SDPA score budget that is a ~760 MiB score chunk
plus softmax temporaries, which OOMs a 12 GiB card once the dense TripoSG
weights are resident. `max_sdpa_scores_bytes` now defaults to 128 MiB on WGPU
(768 MiB on CUDA/Vulkan; `LUX3D_MAX_SDPA_SCORES_BYTES` still overrides).

## Coverage boundary

Implemented and exercised: Pi3, Pi3X (including VO), TripoSR, TripoSG
(image/batch/surface/scribble), VGGT (points/depth/GLB/COLMAP/tracking), and
TRELLIS.2 (geometry/PBR/texture existing mesh), plus RMBG preprocessing and
O-Voxel conversion.

The reference trees for Hunyuan3D-2, Hunyuan3D-2.1, Pixal3D, idol-sdk and
`diso` have no Lux3D runtime family and no local checkpoints; they remain
documented in `E2E.md` as unimplemented coverage. `diso` is an algorithm
reference only.

## Known CLI surfaces not in this suite

- VGGT COLMAP track mode + external bundle adjustment: `colmap` is not
  installed on this host. The manual command is documented in `E2E.md`.
- TRELLIS.2 1024/cascade pipelines, `--num-samples`, and `--return-latent`:
  configured but not run at full weight size; they reuse the sampler code
  covered by the 512 tests.
- HTTP server with real weights: the existing `http_contract.rs` tests are
  weightless; a real-weight server job was not added here.
