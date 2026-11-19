# Lux3D end-to-end scenario coverage

This document maps the user-facing scenarios exposed by the official model
sources in `../candle-3d-refs` to the Lux3D CLI, the real-weight e2e tests in
`crates/lux3d-cli/tests/e2e_scenarios.rs`, and the local weight packages used
for the manual runs.

## Scope

Reference trees audited:

| Source | Kind | Lux3D family | Local weights |
|---|---|---|---|
| `vggt` | Multi-view geometry model | `vggt` | canonical + standalone GGUF |
| `TripoSG` | Image/scribble-to-3D + VAE | `triposg`, `triposg --mode scribble`, `--surface-input` | canonical + standalone GGUF |
| `TRELLIS.2` | Image-to-3D + PBR texturing | `trellis2`, `trellis2 --texture-mesh` | canonical + standalone GGUF + raw checkpoints |
| `RMBG-2.0` | Background matte | `trellis2` preprocessing | canonical + GGUF |
| `Hunyuan3D-2` | Image/multiview/text-to-3D | not implemented | none |
| `Hunyuan3D-2.1` | Image-to-shape + PBR paint | not implemented | none |
| `Pixal3D` | Image/multiview-to-PBR GLB | not implemented (TRELLIS.2-derived code is a separate model) | none |
| `idol-sdk` | Image-to-3D Gaussian splats | not implemented | none |
| `diso` | Differentiable iso-surfacing utility | algorithm reference only | none |
| `diffusers` | Library dependency/reference | not a model family | n/a |

The five unimplemented sources have no weights in the local workspace. They are
listed here so the coverage boundary is explicit; no Lux3D e2e test can run
them without a new model family implementation and checkpoints.

## Scenario matrix

### VGGT

Official source: `candle-3d-refs/vggt` (`vggt/models/vggt.py`,
`demo_gradio.py`, `demo_viser.py`, `demo_colmap.py`).

| User scenario | Official surface | Lux3D CLI | E2E test |
|---|---|---|---|
| Multi-view point cloud | `VGGT.forward`, `world_points`, `depth` | `run vggt --source frames --output cloud.ply` | `vggt_frames_to_point_cloud` |
| Depth+confidence point cloud | `unproject_depth_map_to_point_map` | `--depth-points out.ply` | `vggt_depth_points` |
| Downloadable GLB scene | `predictions_to_glb` | `--glb scene.glb [--glb-show-cam]` | `vggt_glb_scene` |
| COLMAP feedforward reconstruction | `demo_colmap.py` feedforward | `--colmap-dir dir --colmap-mode feedforward` | `vggt_colmap_feedforward` |
| Track user query points | `track_head(query_points=...)` | `--track-queries JSON --track-output tracks.json` | `vggt_track_queries` |
| Automatic query selection + tracks | `track_predict.py`, DINOv2 + ALIKED/SuperPoint | `--track-auto --track-keypoints aliked-sp` | `vggt_track_auto_fps` (FPS selection; ALIKED+SuperPoint need separate converted weights) |
| COLMAP track export + external BA | `demo_colmap.py --use_ba` | `--colmap-mode track --colmap-ba` | not automated (external COLMAP is not installed); manual command documented |
| Single-view reconstruction | one image in the official demo | `run vggt --source image.png` | covered by the same point-cloud path; no separate test |

### TripoSG

Official source: `candle-3d-refs/TripoSG` (`scripts/inference_triposg.py`,
`scripts/inference_triposg_scribble.py`, `scripts/inference_vae.py`).

| User scenario | Official surface | Lux3D CLI | E2E test |
|---|---|---|---|
| Image-to-3D mesh | `run_triposg` / image pipeline | `run triposg --source image.png --output mesh.glb` | `triposg_image_to_mesh` |
| RGB input with RMBG fallback | `prepare_image` + BriaRMBG-1.4 | `--rmbg-model-path rmbg-1.4` | same test uses `mesh-input-rgb.png` |
| Batch images / N shapes per prompt | `batch_size`, `num_shapes_per_prompt` | repeated `--source`, `--num-shapes N` | `triposg_batch_to_meshes` |
| VAE surface encode/decode | `scripts/inference_vae.py` | `--surface-input surface.npy` | `triposg_surface_to_mesh` (generates the fixture with Python/numpy) |
| Scribble image + text prompt | `inference_triposg_scribble.py` | `--mode scribble --prompt ...` | `triposg_scribble_to_mesh` |
| Face-count simplification | `--faces` | `--faces N` | not separately tested (same export path) |

### TRELLIS.2

Official source: `candle-3d-refs/TRELLIS.2` (`example.py`, `app.py`,
`example_texturing.py`, `app_texturing.py`, `trellis2/pipelines/...`).

| User scenario | Official surface | Lux3D CLI | E2E test |
|---|---|---|---|
| Image-to-3D geometry | `Trellis2ImageTo3DPipeline.run` | `run trellis2 --pipeline-type 512 --texture false` | `trellis2_512_geometry` |
| Image-to-3D PBR GLB | `example.py` + `to_glb` | `run trellis2 --pipeline-type 512 --texture true` | `trellis2_512_textured_glb` |
| Textured mesh from existing mesh | `Trellis2TexturingPipeline.run` | `--texture-mesh mesh.obj` | `trellis2_texture_mesh` |
| 1024 / cascade variants | `pipeline_type` 1024/1024_cascade/1536_cascade | `--pipeline-type ...` | unit/cascade tests exist; heavy full-weight variant not run by default |
| Multiple samples | `run(num_samples=N)` | `--num-samples N` | not separately tested (same sampler path) |
| Latent export | `return_latent=True` | `--return-latent out.npz` | not separately tested |
| Remesh / decimation / hole fill | `o_voxel.postprocess.to_glb` | `--no-remesh`, `--remesh-band`, `--decimation-target`, `--fill-hole-perimeter` | `trellis2_512_geometry` uses `--no-remesh --decimation-target 50000`; the default narrow-band remesh is unit-tested but exceeded 17 min CPU on the 1.5M-voxel decode (see `E2E-RESULTS.md`) |
| Low-VRAM stage residency | `low_vram=True/False` | `--no-low-vram` | not separately tested |
| RMBG-2.0 preprocessing | `preprocess_image` + BiRefNet | `--rmbg-model-path rmbg-2.0` | covered by image-to-3D tests |

Both image-to-3D tests pass `--no-remesh` to avoid the CPU-only narrow-band
remesh bottleneck (same reasoning as the matrix row above). The standalone
`trellis2-q4_k` GGUF package has no `shape_enc` stage, so `--texture-mesh` uses
the raw checkpoint tree at
`candle-3d/models/TRELLIS.2-4B` (override with
`LUX3D_E2E_TRELLIS2_TEXTURE_MODEL`). The image-to-3D tests exercise the
standalone package.

### RMBG-2.0

| User scenario | Official surface | Lux3D CLI | E2E test |
|---|---|---|---|
| Foreground matte | `BiRefNet.forward` + sigmoid | used internally by `run trellis2` | covered by `trellis2_512_*` with RGB input |
| Standalone matte output | official README saves a grayscale mask | no standalone CLI (only pipeline preprocessing) | not applicable |

## Non-model CLI scenarios

These are user-facing Lux3D surfaces that do not depend on a reference model
family and are covered separately:

| Scenario | CLI | E2E test |
|---|---|---|
| O-Voxel mesh conversion round-trip | `ovox mesh2ovox`, `ovox ovox2mesh`, `ovox ovox2glb` | `ovox_roundtrip` |
| Model inspection | `inspect <family>` | contract tests |
| Weight normalization | `weights normalize` | separate Python-tool tests |
| Weight quantization | `weights quantize` | `weights.rs` tests |
| HTTP job server | `lux3d-server` | `http_contract.rs` (weightless contract) |

## Running the real-weight suite

Prerequisites:

- local weights under `../3d/standalone` and/or `../3d/canonical-weights`;
- CUDA build for the default GPU run:
  `cargo test --release -p lux3d-cli --features cuda --test e2e_scenarios -- --test-threads=1`;
- environment:

```powershell
$env:LUX3D_E2E = "1"
$env:LUX3D_E2E_DEVICE = "cuda"          # or cpu
$env:LUX3D_E2E_WEIGHTS = "G:\jobs\candle-based-projects\3d\standalone"
$env:LUX3D_E2E_CANONICAL = "G:\jobs\candle-based-projects\3d\canonical-weights"
$env:LUX3D_E2E_ASSETS = "G:\jobs\candle-based-projects\candle-3d\test-assets"
# Optional overrides:
$env:LUX3D_E2E_SCRIBBLE_MODEL = "G:\jobs\candle-based-projects\3d\huggingface\TripoSG-scribble-GGUF\triposg-scribble-q8_0.gguf"
$env:LUX3D_E2E_RMBG2 = "G:\jobs\candle-based-projects\candle-3d\3d\canonical-weights\rmbg-2.0"
$env:LUX3D_E2E_TRELLIS2_TEXTURE_MODEL = "G:\jobs\candle-based-projects\candle-3d\models\TRELLIS.2-4B"
$env:LUX3D_E2E_DINOV2_REG = "G:\jobs\candle-based-projects\candle-3d\models\dinov2_reg4\model.safetensors"
```

The full CUDA suite serialized on one RTX 3060 takes about 17.5 minutes; the
TripoSG batch and TRELLIS.2 textured/texturing cases dominate.

`LUX3D_E2E=1` enables the suite. Without it each test prints a `SKIP` line.
Set `LUX3D_E2E_REQUIRE=1` to turn missing weights/assets into failures instead
of skips.

The suite writes each run's artifacts under
`target/e2e-scenarios/<test-name>/`, so outputs can be inspected manually.

## Manual run record

Filled after the real-weight manual pass; see `E2E-RESULTS.md`.
