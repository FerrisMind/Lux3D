# Lux3D real-weight e2e results

Manual run of the scenario suite in [`E2E.md`](E2E.md) against the weights that
are already on this machine.

## Environment

| Item | Value |
|---|---|
| Date | 2026-11-20 |
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
| CUDA | **18/18 passed** | 1016.13 s (16m56s) | Reference run; detailed per-test timings below. |
| Vulkan | **18/18 passed** | 2665.44 s (44m25s) | Clean full-suite re-run with the Pi3X/VGGT view-materialization fixes. |
| WGPU | **18/18 passed** | 3466.58 s (57m46s) | TripoSG needs packed Q8 (`LUX3D_WGPU_DEQUANT_WEIGHTS=0`); dense F16 corrupts batched denoise. |

WGPU per-test wall times (seconds): `pi3` 8.7, `pi3x` 14.0, `pi3x` VO 19.5,
`trellis2_512_geometry` 249.2, `trellis2_512_textured_glb` 687.0,
`trellis2_texture_mesh` 536.5, `triposg_image` 79.1, `triposg_batch` 1338.3,
`triposg_surface` 120.6, `triposg_scribble` 257.8, `triposr` 8.8,
VGGT 21.2–34.5 each, `ovox_roundtrip` 0.6 total.

## Automated scenario results

| Test | Family / scenario | Weights | Wall time | Artifact |
|---|---|---|---|---|
| `pi3_frames_to_point_cloud` | Pi3 multi-frame point cloud | `3d/standalone/pi3-q8_0` | 4.42 s | `cloud.ply` 25 MiB |
| `pi3x_frames_to_point_cloud` | Pi3X point cloud | `3d/standalone/pi3x-q8_0` | 6.45 s | `cloud.ply` 37 MiB |
| `pi3x_vo_inject_to_point_cloud` | Pi3X VO + pose/depth/ray injection | `3d/standalone/pi3x-q8_0` | 6.33 s | `vo.ply` 37 MiB |
| `triposr_image_to_mesh` | TripoSR image-to-mesh | `3d/standalone/triposr-q8_0` | 2.98 s | `mesh.glb` 625 KiB |
| `triposg_image_to_mesh` | TripoSG image-to-mesh, RMBG-1.4 RGB fallback | `3d/standalone/triposg-q8_0` + `rmbg-1.4` | 14.61 s | `mesh.glb` 230 KiB |
| `triposg_batch_to_meshes` | 2 images x 2 shapes in one batch | `3d/standalone/triposg-q8_0` | 276.25 s | `batch_0..3.glb` |
| `triposg_surface_to_mesh` | VAE surface encode/decode | `3d/standalone/triposg-q8_0` | 23.69 s | `surface.glb` 18 MiB |
| `triposg_scribble_to_mesh` | Scribble image + text prompt | `3d/huggingface/TripoSG-scribble-GGUF/triposg-scribble-q8_0.gguf` | 80.07 s | `scribble.glb` 8.9 MiB |
| `vggt_frames_to_point_cloud` | VGGT multi-frame point cloud | `3d/standalone/vggt-q8_0` | 5.34 s | `cloud.ply` 11 MiB |
| `vggt_depth_points` | VGGT depth+confidence unprojection | `3d/standalone/vggt-q8_0` | 5.86 s | `cloud.ply`, `depth.ply` 36 MiB |
| `vggt_glb_scene` | VGGT GLB scene with cameras | `3d/standalone/vggt-q8_0` | 5.43 s | `scene.glb` 6.2 MiB |
| `vggt_colmap_feedforward` | VGGT feedforward COLMAP export | `3d/standalone/vggt-q8_0` | 5.54 s | `colmap/` binary model |
| `vggt_track_queries` | Explicit query tracking | `3d/standalone/vggt-q8_0` | 6.08 s | `tracks.json` |
| `vggt_track_auto_fps` | DINOv2 + FPS automatic queries | `vggt-q8_0` + `models/dinov2_reg4` | 7.56 s | `tracks.json` 37 KiB |
| `trellis2_512_geometry` | TRELLIS.2 512 geometry, decimation | `3d/standalone/trellis2-q4_k` + `rmbg-2.0` | 37.57 s | `geometry.glb` 1.4 MiB |
| `trellis2_512_textured_glb` | TRELLIS.2 512 PBR GLB | `3d/standalone/trellis2-q4_k` + `rmbg-2.0` | 257.90 s | `textured.glb` 3.6 MiB + PBR sidecars |
| `trellis2_texture_mesh` | PBR bake onto an existing cube mesh | `3d/standalone/trellis2-q4_k` + `rmbg-2.0` | 269.17 s | `cube_textured.glb` 118 KiB |
| `ovox_roundtrip` | mesh2ovox -> ovox2mesh/ovox2glb | none | 0.94 s | `cube.vxz`, `cube.ply`, `cube.glb` |

CUDA full-suite result: **18/18 passed** in **1016.13 s** (16m56s), serialized
on one GPU. The same suite is **18/18 on Vulkan and WGPU** (see the backend
matrix above). `target/e2e-scenarios/` keeps every artifact for manual
inspection.

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
packages now embed it as `shape_enc.*` tensors via
`weights quantize --merge-source ... --merge-prefix shape_enc`, so the suite
textures the cube mesh from the same `trellis2-q4_k` standalone package as the
image-to-3D tests (passed on CUDA, Vulkan and WGPU). The raw
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
