# Lux3D real-weight e2e results

Manual run of the scenario suite in [`E2E.md`](E2E.md) against the weights that
are already on this machine.

## Environment

| Item | Value |
|---|---|
| Date | 2026-11-21 initial sweep; CUDA/Vulkan rerun 2026-11-22 (same session); WGPU rerun 2026-11-22 with the field-path flush fix |
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
| CUDA | **18/18 passed** | 796.39 s (13m16s) | Same-session reference run (2026-11-22). |
| Vulkan | **18/18 passed** | 1444.53 s (24m05s) | Same-session run (2026-11-22); 1.83x faster than the 2638.52 s pre-fix baseline. |
| WGPU | **18/18 passed** | 2769.87 s (46m10s) | 2026-11-22 run with the field-path flush fix (16-chunk WGPU window); was 2908.76 s with the 256 MiB device buffer. TripoSG uses packed Q8 (`LUX3D_WGPU_DEQUANT_WEIGHTS=0`). |

Overall same-session ratio: Vulkan/CUDA **1.81x** (was 2.62x on the pre-fix
baseline). The absolute suite times move together between sessions (the
2026-11-21 pair was 955.13/1717.21 s); compare ratios only within one session.
Per-scenario ratios are in the table below. WGPU was rerun later the same day
with the field-path flush fix; its 2769.87 s is 3.48x the CUDA reference, but
that is a cross-session comparison (the WGPU suite ran after a rebuild), so
read the WGPU column as absolute scenario times, not as a session ratio.

## Automated scenario results

| Test | Family / scenario | Weights | CUDA | Vulkan | Vk/CUDA | WGPU | Wg/CUDA | Artifact |
|---|---|---|---|---|---|---|---|---|
| `pi3_frames_to_point_cloud` | Pi3 multi-frame point cloud | `3d/standalone/pi3-q8_0` | 3.16 s | 7.38 s | 2.34x | 8.04 s | 2.54x | `cloud.ply` 25 MiB |
| `pi3x_frames_to_point_cloud` | Pi3X point cloud | `3d/standalone/pi3x-q8_0` | 4.69 s | 10.29 s | 2.19x | 12.89 s | 2.75x | `cloud.ply` 37 MiB |
| `pi3x_vo_inject_to_point_cloud` | Pi3X VO + pose/depth/ray injection | `3d/standalone/pi3x-q8_0` | 5.76 s | 10.80 s | 1.88x | 17.54 s | 3.05x | `vo.ply` 37 MiB |
| `triposr_image_to_mesh` | TripoSR image-to-mesh | `3d/standalone/triposr-q8_0` | 2.04 s | 5.39 s | 2.64x | 5.68 s | 2.78x | `mesh.glb` 628 KiB |
| `triposg_image_to_mesh` | TripoSG image-to-mesh, RMBG-1.4 RGB fallback | `3d/standalone/triposg-q8_0` + `rmbg-1.4` | 12.90 s | 24.87 s | 1.93x | 69.97 s | 5.42x | `mesh.glb` 308 KiB |
| `triposg_batch_to_meshes` | 2 images x 2 shapes in one batch | `3d/standalone/triposg-q8_0` | 244.20 s | 545.86 s | 2.24x | 1128.97 s | 4.62x | `batch_0..3.glb` |
| `triposg_surface_to_mesh` | VAE surface encode/decode | `3d/standalone/triposg-q8_0` | 21.29 s | 47.87 s | 2.25x | 195.27 s | 9.17x | `surface.glb` 18 MiB |
| `triposg_scribble_to_mesh` | Scribble image + text prompt | `3d/huggingface/TripoSG-scribble-GGUF/triposg-scribble-q8_0.gguf` | 43.53 s | 92.00 s | 2.11x | 218.61 s | 5.02x | `scribble.glb` 9.2 MiB |
| `vggt_frames_to_point_cloud` | VGGT multi-frame point cloud | `3d/standalone/vggt-q8_0` | 4.95 s | 10.00 s | 2.02x | 19.57 s | 3.95x | `cloud.ply` 11 MiB |
| `vggt_depth_points` | VGGT depth+confidence unprojection | `3d/standalone/vggt-q8_0` | 5.12 s | 10.23 s | 2.00x | 19.59 s | 3.83x | `cloud.ply`, `depth.ply` 36 MiB |
| `vggt_glb_scene` | VGGT GLB scene with cameras | `3d/standalone/vggt-q8_0` | 4.98 s | 10.18 s | 2.04x | 19.65 s | 3.95x | `scene.glb` 6.2 MiB |
| `vggt_colmap_feedforward` | VGGT feedforward COLMAP export | `3d/standalone/vggt-q8_0` | 4.89 s | 10.51 s | 2.15x | 19.31 s | 3.95x | `colmap/` binary model |
| `vggt_track_queries` | Explicit query tracking | `3d/standalone/vggt-q8_0` | 5.58 s | 10.44 s | 1.87x | 21.34 s | 3.82x | `tracks.json` |
| `vggt_track_auto_fps` | DINOv2 + FPS automatic queries | `vggt-q8_0` + `models/dinov2_reg4` | 6.72 s | 12.15 s | 1.81x | 30.66 s | 4.56x | `tracks.json` 37 KiB |
| `trellis2_512_geometry` | TRELLIS.2 512 geometry, decimation | `3d/standalone/trellis2-q4_k` + `rmbg-2.0` | 31.89 s | 85.08 s | 2.67x | 147.34 s | 4.62x | `geometry.glb` 1.4 MiB |
| `trellis2_512_textured_glb` | TRELLIS.2 512 PBR GLB | `3d/standalone/trellis2-q4_k` + `rmbg-2.0` | 195.08 s | 305.98 s | 1.57x | 543.56 s | 2.79x | `textured.glb` 3.6 MiB + PBR sidecars |
| `trellis2_texture_mesh` | PBR bake onto an existing cube mesh | `3d/standalone/trellis2-q4_k` + `rmbg-2.0` | 198.75 s | 244.69 s | 1.23x | 291.01 s | 1.46x | `cube_textured.glb` 116 KiB |
| `ovox_roundtrip` | mesh2ovox -> ovox2mesh/ovox2glb | none | 0.60 s | 0.59 s | 0.98x | 0.59 s | 0.98x | `cube.vxz`, `cube.ply`, `cube.glb` |

`ovox_roundtrip` is the sum of its three subcommands.

CUDA full-suite result: **18/18 passed** in **796.39 s** (13m16s), serialized
on one GPU. Vulkan is **18/18 in 1444.53 s** (24m05s) with the same binary and
weights. `target/e2e-scenarios/` keeps every artifact for manual inspection.

## 2026-11-22: TripoSG decode sync reduction

The SDF grid builders downloaded every chunk's logits with a blocking
`to_vec1`: 512 WDDM flushes per level for a 16.7M-point dense grid, and 2048+
in the 4-sample batch. The chunk tensors now stay on the device and are
downloaded once per level (256 MiB pending cap). The cross-attention `k_t`
transpose is hoisted out of the chunk loop, the query freqs table is cached
per decoder, and the output negate is folded into `proj_out` for non-quantized
weights. CPU parity: `decode_latents` and both grid modes are bit-identical.

Same-session effect on the slow scenarios (Vulkan / CUDA):

| scenario | before | after | ratio before -> after |
|---|---:|---:|---:|
| `triposg_scribble_to_mesh` | 141.01 / 49.00 | **92.00 / 43.53** | 2.88x -> 2.11x |
| `triposg_surface_to_mesh` | 59.92 / 23.95 | **47.87 / 21.29** | 2.50x -> 2.25x |
| `triposg_batch_to_meshes` | 646.21 / 272.66 | **545.86 / 244.20** | 2.37x -> 2.24x |
| `triposg_image_to_mesh` | 28.02 / 14.42 | **24.87 / 12.90** | 1.94x -> 1.93x |

The CUDA side also improved (~11%), so part of the absolute gain is
session-level; the architectural change (512 blocking syncs -> 1 per level)
and the bit-identical CPU parity are backend-independent.

WGPU keeps the device-side accumulation only in the flash path: its
`Tensor::cat` makes a large pending buffer slower than the per-chunk D2H it
replaces, so the field builders flush every 16 chunks there. Same-session
effect: surface 384.72 -> **195.27 s**, batch 1116.14 -> 1128.97 s (the
16-chunk window minimizes the sum; per-chunk D2H reaches surface 106.68 s but
costs batch 116 s). The full WGPU suite is **18/18 in 2769.87 s**, down from
2908.76 s. The chunk-count flush also exposed a latent `refine_level` bug
where a batch index was used to slice the cell vector; the window now tracks
the first pending cell. Details in
[`PERF-NOTES.md`](PERF-NOTES.md#wgpu-triposg-field-path-regression-and-fix-2026-11-22).

Measured dead ends in this round, reverted or left opt-in:

* TRELLIS.2 sparse conv with F32 kernels on Vulkan (removes the F16
  gather/matmul conversion hub, 7 -> 3 dispatches per offset): **slower**
  (F16 warm 86.95 s vs F32 104.65 s on `trellis2_512_geometry`), reverted.
* TripoSG decode chunk 65536 on Vulkan: `surface` mesh 344.64 s vs ~45 s at
  32768 (allocator/VRAM churn), left at 32768.
* `LUX3D_TRIPOSG_ATTN_CHUNK=24576`: 49.13 s vs 48.59 s on surface, neutral.
* `CANDLE_VK_POOL_MAX_BYTES=512M`: surface 45.3 s vs 46.3-47.8 s, scribble
  90.9 s vs 90.5 s — within noise, not enabled by default (VRAM risk on the
  large checkpoints).

## Vulkan vs CUDA: root causes and fixes (2026-11-21)

The pre-fix Vulkan suite was 2638.52 s (2.62x CUDA). The 2026-11-21 changes
cut it to 1717.21 s (1.80x); with the 2026-11-22 TripoSG work the same-session
pair is 1444.53 / 796.39 s (1.81x). Each item is measurement-gated.

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

Every q8-based scenario is bounded near 2-3x before any Lux3D overhead. The
same-session ratios are: Pi3 2.34x, VGGT 1.81-2.15x, TripoSR 2.64x, TripoSG
image 1.93x, surface 2.25x, scribble 2.11x, batch 2.24x. Dense F32 is the one
kernel where Vulkan is faster, so the TRELLIS.2 F32 sparse-conv decode is not
the geometry bottleneck; its 2.67x comes from the q4_k structure generator
(5.17x ceiling). `trellis2_textured_glb` is lower (1.57x) because the
backend-independent CPU xatlas/bake floor dilutes the GPU part.

### Attribution: which scenarios can reach 90% without candle work

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

The q8/Q4_K kernel ceiling alone is 3.14x/5.17x, so no amount of Rust-level
dispatch reduction can bring a GEMM-bound scenario to 1.11x. The two
exceptions that already meet the target are CPU/IO-only (`ovox`) or
CPU-floor-dominated (`texture_mesh` is 1.23x and the GPU share is small).

### Remaining dispatch/sync-bound work

* **TripoSG decode.** The 2026-11-22 change removed the per-chunk blocking
  D2H syncs (512 -> 1 per level). What remains per 32k chunk is ~45 fixed
  dispatches plus the q8 GEMMs; the decode stage is now the q8 kernel ceiling,
  not host sync.
* **TRELLIS.2 sparse conv.** 27 offsets x `index_select` + matmul + add per
  conv; 40 convs per decode, ~51 s for 80 convs on Vulkan. The F32-kernel
  variant was measured slower and reverted. Grouped-offset batching remains a
  possible Rust-level lever but is bandwidth-bound and was not kept.
* **VGGT tracker** is no longer in this group: `fmap_norm` 6.47 -> 0.01 s,
  track head 7.14 -> 0.57 s.

TRELLIS.2 textured scenarios are CPU-bound in both backends: `uv_unwrap`
(xatlas, 26k charts) takes 182 s of the 193 s texture-export stage on CUDA,
and the backend-independent bake is another 8.9 s. That is why their ratios
are the lowest (1.23-1.57x) despite the slow sparse-conv decode.

WGPU was **3.64x** CUDA in the 2026-11-21 session (3479.83 vs 955.13 s) and
is **3.48x** the 2026-11-22 CUDA reference after the field-path flush fix
(2769.87 s, cross-session comparison). Its largest gaps remain TripoSG
(4.6-9.2x), TRELLIS.2 geometry (4.6x) and the VGGT q8 aggregator (3.8-4.0x),
consistent with its dense-F16/conv kernel throughput. The TripoSG surface
number is the outlier: WGPU `Tensor::cat` makes device-side chunk
accumulation expensive, so the field path now flushes every 16 chunks
(surface 195.27 s vs 384.72 s with the 256 MiB buffer; per-chunk D2H reaches
106.68 s but costs batch 116 s).

### What would close the remaining gap

The Q8_0/Q4_K Vulkan GEMM kernels are 3.1-5.2x slower than CUDA; no Lux3D
change can recover that. The candle-side levers, in measured order:

1. Q8_0/Q4_K Vulkan GEMM: add a `VK_KHR_cooperative_matrix` INT8 path
   (A/B int8, C int32, per-block scale in the epilogue). CUDA reaches 26.29
   TFLOPS with `mma.sync...s8.s8.s32`; Vulkan's `mul_mmq.comp` only has dp4a
   and reaches 8.37. Concrete file/line handoff and the generator gate are in
   [`PERF-NOTES.md`](PERF-NOTES.md#candle-side-handoff-the-q8_0q4_k-vulkan-gemm-gap).
   This alone bounds every q8 scenario.
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
