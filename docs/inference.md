# Inference: running the CLI

Every `run` command accepts `--device cpu|cuda|vulkan|wgpu` (default `cuda`).
CUDA is the verified production backend; see [../PERF-NOTES.md](../PERF-NOTES.md)
and [performance.md](performance.md) for measured numbers.

## Run inference

GPU backends are Cargo features (`cuda` / `vulkan` / `wgpu` / `metal`). Default builds are CPU-only. The CLI `--device` enum exposes `cpu`, `cuda`, `wgpu`, and `vulkan` only: a Metal build cannot be selected from the CLI. The server library builder can select Metal (`with_device_backend`), but Metal inference is untested in this repository (see
[system requirements](performance.md#system-requirements-and-memory)).

```powershell
# Pi3 -> PLY with an explicit canonical package directory (layout matches 3d/canonical-weights/pi3)
cargo run -p lux3d-cli --features cuda -- run pi3 --model-path <canonical-pi3-dir> --source <input-sequence> --output <output-file.ply>

# Pi3X core -> PLY with an explicit canonical package directory (layout matches 3d/canonical-weights/pi3x)
cargo run -p lux3d-cli --features cuda -- run pi3x --model-path <canonical-pi3x-dir> --source <input-sequence> --conditions <conditions-file> --output <output-file.ply>

# Pi3X VO -> PLY with an explicit canonical package directory (layout matches 3d/canonical-weights/pi3x)
cargo run -p lux3d-cli --features cuda -- run pi3x --model-path <canonical-pi3x-dir> --source <input-video> --vo --chunk-size 8 --overlap 4 --conf-threshold 0.05 --inject-condition pose,depth,ray --output <output-file.ply>

# TripoSR -> GLB with an explicit canonical package directory (layout matches 3d/canonical-weights/triposr)
cargo run -p lux3d-cli --features cuda -- run triposr --model-path <canonical-triposr-dir> --source <input-image> --mc-resolution 256 --mc-threshold 25.0 --output <output-file.glb>

# VGGT -> PLY (world points filtered by confidence > 1.0 by default)
cargo run -p lux3d-cli -- run vggt --model-path <canonical-vggt-dir> --source <input-sequence> --output <output-file.ply>

# VGGT canonical package auto-download (oxide-lab/VGGT-1B)
cargo run -p lux3d-cli -- run vggt --source <input-sequence> --output <output-file.ply>

# VGGT pad preprocessing, f16 weights, sequential 2-frame chunks, COLMAP feedforward
# export + depth-unprojected point cloud next to the default confidence-filtered PLY
cargo run -p lux3d-cli -- run vggt --model-path <canonical-vggt-dir> --source <input-sequence> --preprocess-mode pad --vggt-dtype f16 --batch-size 2 --colmap-dir <sparse-dir> --depth-points <depth.ply> --output <output-file.ply>

# VGGT square-1024 preprocessing (the official COLMAP demo path), feedforward COLMAP
# export with the point-head point source and a seeded random point cap
cargo run -p lux3d-cli -- run vggt --model-path <canonical-vggt-dir> --source <input-sequence> --preprocess-mode square --colmap-dir <sparse-dir> --colmap-point-source pointmap --colmap-seed 42 --output <output-file.ply>

# VGGT from a single image file (one-frame scene)
cargo run -p lux3d-cli -- run vggt --source <image.png> --output <output-file.ply>

# VGGT explicit track queries -> tracks JSON
cargo run -p lux3d-cli -- run vggt --source <input-sequence> --track-queries '[{"frame":0,"x":259,"y":259}]' --track-output tracks.json --output <output-file.ply>

# VGGT automatic stage-A query selection (DINOv2-reg ranking + official
# ALIKED-n16/SuperPoint keypoints) and a track-based COLMAP export without BA
# (COLMAP-ready input for external BA)
cargo run -p lux3d-cli -- run vggt --model-path <canonical-vggt-dir> --source <input-sequence> --track-auto --query-frame-num 8 --max-query-pts 4096 --colmap-dir <sparse-dir> --colmap-mode track --output <output-file.ply>

# Weight-free stage-A opt-out (deterministic FPS grid instead of ALIKED+SuperPoint)
cargo run -p lux3d-cli -- run vggt --source <input-sequence> --track-auto --track-keypoints fps --output <output-file.ply>

# Same track export + external COLMAP bundle adjustment (colmap from PATH or LUX3D_COLMAP_BIN).
# On success cameras.bin/images.bin/points3D.bin (plus rigs.bin/frames.bin when that COLMAP
# release writes them) in <sparse-dir> are replaced in place.
cargo run -p lux3d-cli -- run vggt --model-path <canonical-vggt-dir> --source <input-sequence> --track-auto --colmap-dir <sparse-dir> --colmap-mode track --colmap-ba --output <output-file.ply>

# Stage-A keypoint checkpoints (one-time): convert the official .pth files to safetensors.
# ALIKED:    https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth
# SuperPoint: https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth
python scripts/convert_keypoint_weights.py --input aliked-n16.pth --output models/aliked-superpoint/aliked-n16.safetensors
python scripts/convert_keypoint_weights.py --input superpoint_v1.pth --output models/aliked-superpoint/superpoint_v1.safetensors

# VGGT demo-style GLB scene (confidence-filtered point cloud + optional camera cones)
cargo run -p lux3d-cli -- run vggt --model-path <canonical-vggt-dir> --source <input-sequence> --glb <scene.glb> --glb-conf-percentile 50 --glb-show-cam --output <output-file.ply>

# TripoSG -> GLB (defaults: 50 rectified-flow steps, CFG 7.0, seed 42, flash decoder depth 9)
cargo run -p lux3d-cli --features cuda -- run triposg --rmbg-model-path <canonical-rmbg-dir> --model-path <canonical-triposg-dir> --source <input-image> --output <output-file.glb>
# Flash decoder is the reference default; disable it for dense/hierarchical marching cubes:
cargo run -p lux3d-cli --features cuda -- run triposg --model-path <canonical-triposg-dir> --source <input-image> --flash-decoder false --dense-octree-depth 8 --hierarchical-octree-depth 9 --output <output-file.glb>
# Faster preview: 2 steps, flash depth 5
cargo run -p lux3d-cli --features cuda -- run triposg --model-path <canonical-triposg-dir> --source <input-image> --steps 2 --flash-decoder true --flash-octree-depth 5 --output <preview.glb>

# TripoSG canonical package auto-download (oxide-lab/TripoSG); RMBG 1.4 is a separate package
cargo run -p lux3d-cli --features cuda -- run triposg --source <input-image> --output <output-file.glb>

# Batched TripoSG: repeatable --source (one image per source) and --num-shapes per prompt;
# total batch = sources * --num-shapes, outputs get _0, _1, ... suffixes
cargo run -p lux3d-cli --features cuda -- run triposg --model-path <canonical-triposg-dir> --source <image-a.png> --source <image-b.png> --num-shapes 2 --output <output.glb>

# TripoSG reference VAE surface encode (scripts/inference_vae.py): a .npy keyed dict
# {"surface_points": [N,3], "surface_normals": [N,3]}; skips image/DINO/denoise.
# N must be at least 204800 points (the reference takes a fixed 204800-point subset;
# smaller inputs error instead of being resampled).
cargo run -p lux3d-cli --features cuda -- run triposg --model-path <canonical-triposg-dir> --surface-input <surface.npy> --output <output.glb>

# TripoSG-scribble: CLIP prompt + scribble image dual conditioning. Pass a standalone
# package (.gguf file or directory containing model.gguf), the raw VAST-AI/TripoSG-scribble
# checkpoint, or a legacy DiT-only .gguf; --prompt is required, --scribble-conf/--prompt-conf
# override the reference branch weights (0.4 / 1.0)
cargo run -p lux3d-cli --features cuda -- run triposg --mode scribble --prompt "a cat with wings" --model-path triposg-scribble-q8_0.gguf --source <scribble.png> --scribble-conf 0.4 --prompt-conf 1.0 --output <output.glb>

# Reference post-process/load knobs: `--faces <n>` (target face count after merge-close +
# meshopt quadric simplification — a clean-room approximation of upstream pymeshlab; 0 = off,
# the upstream `-1` equivalent; no-op at/above the current face count), `--timesteps` (explicit
# schedule override, replaces `--steps`), `--bounds` (SDF query window `min_x,...,max_z` in the
# normalized model frame; a single scalar `b` means the symmetric cube `[-b, b]`), `--dtype` (`f32`
# reference; `f16` has load/DINO/VAE smoke coverage;
# `bf16` is accepted but unverified)
cargo run -p lux3d-cli --features cuda -- run triposg --model-path <canonical-triposg-dir> --source <input-image> --faces 50000 --timesteps 999,777,500,250,0 --bounds "-1.1,-1.1,-1.1,1.1,1.1,1.1" --dtype f16 --output <output.glb>

# TRELLIS.2 -> textured GLB + PBR sidecar from a canonical package
# (defaults: 1024_cascade pipeline, 12 steps per sampler, SS/shape guidance 7.5,
# texture guidance 1.0, seed 42, texture on).
# RGB inputs need the separate RMBG-2.0 package; a sibling `rmbg-2.0`/`RMBG-2.0`
# directory next to --model-path is discovered automatically.
cargo run -p lux3d-cli --features cuda -- run trellis2 --model-path <canonical-trellis2-dir> --rmbg-model-path <canonical-rmbg2-dir> --source <input-image> --output <output-file.glb>
# Textured-GLB bake edge length override (default 2048; the CLI accepts 64..=8192, multiples of 8;
# `--no-texture-export` skips the UV unwrap + bake entirely, `--no-uv-unwrap` skips just the unwrap):
cargo run -p lux3d-cli --features cuda -- run trellis2 --model-path <canonical-trellis2-dir> --rmbg-model-path <canonical-rmbg2-dir> --source <input-image> --texture-size 2048 --output <output-file.glb>
# Geometry-only run (no texture SLAT / tex VAE / PBR sidecar):
cargo run -p lux3d-cli --features cuda -- run trellis2 --model-path <canonical-trellis2-dir> --rmbg-model-path <canonical-rmbg2-dir> --source <input-image> --texture false --output <output-file.glb>
# Faster 512-preview pipeline with overrides:
cargo run -p lux3d-cli --features cuda -- run trellis2 --model-path <canonical-trellis2-dir> --rmbg-model-path <canonical-rmbg2-dir> --source <input-image> --pipeline-type 512 --steps 12 --seed 42 --output <preview.glb>
# The raw microsoft/TRELLIS.2-4B checkpoint directory (pipeline.json + ckpts/) is accepted as-is:
cargo run -p lux3d-cli --features cuda -- run trellis2 --model-path <raw-TRELLIS.2-4B-dir> --rmbg-model-path <raw-RMBG-2.0-dir> --source <input-image> --output <output-file.glb>
# Resident stage mode (upstream low_vram=False): keep every loaded stage on the device for the
# whole run instead of the default per-stage load/drop; same outputs, faster, higher peak VRAM
cargo run -p lux3d-cli --features cuda -- run trellis2 --model-path <canonical-trellis2-dir> --rmbg-model-path <canonical-rmbg2-dir> --source <input-image> --no-low-vram --output <output-file.glb>
# Restore the pre-remesh cleanup chain, or tune the default narrow-band remesh band/project:
cargo run -p lux3d-cli --features cuda -- run trellis2 --model-path <canonical-trellis2-dir> --rmbg-model-path <canonical-rmbg2-dir> --source <input-image> --no-remesh --output <output-file.glb>
cargo run -p lux3d-cli --features cuda -- run trellis2 --model-path <canonical-trellis2-dir> --rmbg-model-path <canonical-rmbg2-dir> --source <input-image> --remesh-band 1 --remesh-project 0.5 --output <output-file.glb>

# TRELLIS.2 texturing of an existing mesh (upstream Trellis2TexturingPipeline):
# --source is the condition image, --texture-mesh accepts .glb/.obj/.ply/.stl, output must be .glb.
# The shape encoder comes from the raw `ckpts/shape_enc_next_dc_f16c32_fp16.safetensors`,
# from `external/shape_enc/model.safetensors` beside a canonical/GGUF package, or from
# `shape_enc.*` tensors embedded in a standalone GGUF (`weights quantize --merge-source`).
# `--texture-resolution` 512|1024|1536 (default 1024; 1536 shares the 1024 conditioning/DiT).
# A standalone GGUF package with the embedded encoder is fully self-contained:
cargo run -p lux3d-cli --features cuda -- run trellis2 --model-path <standalone-trellis2-gguf-dir> --rmbg-model-path <canonical-rmbg2-dir> --source <condition-image> --texture-mesh <input-mesh.glb> --texture-resolution 512 --output textured.glb
cargo run -p lux3d-cli --features cuda -- run trellis2 --model-path <raw-TRELLIS.2-4B-dir> --rmbg-model-path <raw-RMBG-2.0-dir> --source <condition-image> --texture-mesh <input-mesh.glb> --texture-resolution 1536 --output textured.glb

# Auto-download canonical packages from Hugging Face into the user cache
cargo run -p lux3d-cli --features cuda -- run pi3 --source <input-sequence> --output <output-file.ply>
```

## Output formats

Mesh commands (`triposr`, `triposg`, `trellis2`) choose the writer from the
output extension:

| Extension | Behavior |
|-----------|----------|
| `.glb` | Primary/default. Required for textured output. |
| `.obj` | Positions-only for `trellis2`. |
| `.ply` | Per-vertex colors kept for `trellis2`. |
| `.stl` | Binary and colorless; cannot carry vertex colors. |

Point-cloud commands (`pi3`, `pi3x`, `vggt`) write `.ply` (primary) or `.xyz`.

### Textured TRELLIS.2 output

A textured `trellis2` `.glb` run:

- embeds the baked base-color + metallic-roughness PNG textures with UV
  coordinates and converted normals (upstream `to_glb` textured-GLB parity);
- writes a `<stem>.pbrvox.bin` + `<stem>.pbr.json` sidecar pair next to the
  output. The sparse-voxel PBR vertex attributes (6 channels) live outside the
  GLB/OBJ/PLY/STL mesh formats, so the sidecar preserves them losslessly.

`.obj`/`.ply`/`.stl` `trellis2` outputs never embed textures — they keep the
per-vertex-color PLY, positions-only OBJ, or color-less binary STL mesh (the
textured GLB writer requires a `.glb` output).

### HTTP output

- The mesh API defaults to GLB and accepts `options.output_format`; the
  point-cloud API defaults to PLY and accepts `options.output_format`
  (`ply`/`xyz`).
- The server maps the TRELLIS.2 postprocess and texture-export fields
  (`postprocess`, `decimation_target`, `fill_hole_perimeter`,
  `min_component_area`, `remesh`, `remesh_band`, `remesh_project`, `texture`,
  `texture_size`). `trellis2` mesh jobs with a `.glb` output write the
  embedded-texture GLB plus the PBR sidecars in the job workspace; the sidecars
  are not served through `/content`.
- `POST /v1/meshes/texturings` textures an existing mesh and always writes
  `trellis2-textured.glb` (no PBR sidecar). The CLI `--texture-mesh` behaves the
  same way and requires a `.glb` output.
- `--return-latent` and the CLI-only TripoSG surfaces (`--surface-input`,
  `--mode scribble`, repeatable `--source`, `--num-shapes`) have no HTTP fields.

> VGGT flags (`--preprocess-mode`, `--colmap-*`, `--track-*`, `--glb`,
> `--dump-npz`) are documented in [vggt.md](vggt.md). TRELLIS.2 sampling,
> post-processing, and texturing flags are documented in [trellis2.md](trellis2.md).

## Experimental backends: Vulkan and WGPU

Select a non-CUDA device with `--device`:

```powershell
cargo run -p lux3d-cli --features vulkan -- run triposr --device vulkan --source <input-image> --output <output-file.glb>
cargo run -p lux3d-cli --features wgpu -- run pi3 --device wgpu --source <input-sequence> --output <output-file.ply>
cargo run -p lux3d-cli --features vulkan -- run triposg --device vulkan --source <input-image> --steps 2 --num-tokens 1024 --flash-octree-depth 5 --output <preview.glb>
```

These paths are experimental and not verified in CI. Inference may be unstable,
produce incorrect output, or fail entirely depending on your GPU, driver, and
OS. Use CUDA for production workloads.

Real-weight coverage: all 18 e2e scenarios pass on CUDA, Vulkan and WGPU
(RTX 3060 12 GiB); see [../E2E-RESULTS.md](../E2E-RESULTS.md).

- **TripoSG on WGPU:** keep packed Q8 weights
  (`LUX3D_WGPU_DEQUANT_WEIGHTS=0`). The default dense-F32 dequant OOMs a 12 GiB
  card, and dense F16 (`=f16`) mishandles batched DiT denoise (one noisy and one
  empty mesh for `--num-shapes 2`).
- **TripoSG VAE decode** always runs dense F32 on WGPU, which also avoids a
  candle hot-ring alias at 512 latent tokens.

## O-Voxel volume commands

The `ovox` subcommand group converts between textured meshes and the sparse PBR O-Voxel volume used by TRELLIS.2 (the upstream `o-voxel` examples). It is CLI-only (no HTTP route) and prints a JSON stats report per run.

```powershell
# Textured mesh -> sparse PBR .vxz (512^3 default; .npz/.ply are also accepted as outputs)
cargo run -p lux3d-cli -- ovox mesh2ovox <input.glb> <output.vxz>

# .vxz/.npz/.ply -> reconstructed .ply with base-color vertex colors
cargo run -p lux3d-cli -- ovox ovox2mesh <input.vxz> <output.ply> --grid-size 512 --aabb -0.5,-0.5,-0.5,0.5,0.5,0.5

# .vxz/.npz/.ply -> cleaned, UV-unwrapped, textured .glb (official example defaults:
# decimation 100000, texture 2048)
cargo run -p lux3d-cli -- ovox ovox2glb <input.vxz> <output.glb> --grid-size 512 --texture-size 2048
```

### `mesh2ovox` flags

- `--grid-size N[,N,N]` (default 512)
- `--voxel-size S[,S,S]`
- `--aabb MINX,...,MAXZ`
- `--no-normalize` — skip the official center + `0.99999/max_extent` unit-cube
  normalization
- `--mip-level-offset` (0.0)
- `--face-weight` (1.0)
- `--boundary-weight` (0.2)
- `--regularization-weight` (1e-2)
- `--compression none|deflate` (default `deflate`)

`.vxz` is written with `none` or `deflate` only: LZMA files can be read (the
official writer defaults to LZMA), but LZMA writing is not supported, and
zstd-compressed files are rejected with a clear error because this build has no
zstd decoder.

### `ovox2mesh` / `ovox2glb`

Both need `--grid-size` and `--aabb` because `.vxz` does not store the grid size
or bounds (defaults 512 and `[-0.5, 0.5]^3`).

`ovox2glb` additionally accepts:

- `--decimation-target` (100000)
- `--texture-size` (2048)
- `--fill-hole-perimeter` (3e-2)
- `--min-component-area` (1e-5)
