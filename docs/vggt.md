# VGGT: multi-view geometry and COLMAP export

VGGT-1B predicts point maps, depth maps, camera poses, and multi-frame tracks from
one or more images. Lux3D implements the feedforward model plus the official
COLMAP export paths; the fine tracker stages B/C (LightGlue matching and
refinement) are not included.

Run commands live in [inference.md](inference.md). Weight formats and
quantization behavior live in [weights.md](weights.md); the Q8_0 K-quant caveat
for VGGT is covered under [quantization fidelity](weights.md#quantization-fidelity).

## Inputs

`--source` accepts:

| Input | Formats |
|-------|---------|
| Frame directory | `.png`, `.jpg`, `.jpeg` |
| Video file | `.mp4`, `.mov`, `.avi`, `.mkv` |
| Single image (one-frame scene) | `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff` |

## Preprocessing

`--preprocess-mode` selects the resize policy:

| Mode | Behavior |
|------|----------|
| `crop` (default) | Resize to width 518, center-crop rows taller than 518, white-pad to the batch maximum. |
| `pad` | Largest side 518, center-padded white 518×518 square. No pixels are discarded; use it when framing matters. |
| `square` | Official `demo_colmap.py` path: black-pad to 1024×1024, model input is its 518×518 bilinear downsample, and the official `x1/y1/x2/y2/width/height` coordinates drive the original-resolution COLMAP export. |

## Weights and batching

- `--vggt-dtype f32|f16|bf16` selects the network load dtype. `f32` is the
  reference mode; public outputs stay F32. This is separate from the TripoSG
  `--dtype` flag. F16/BF16 are wired, but the automated parity suite is
  F32/CUDA-only and BF16 is unverified.
- `--batch-size N` splits the frame list into sequential chunks of N frames run
  as independent B=1 passes. It bounds peak VRAM but is **not** one attention
  pass over all frames: cross-frame attention only sees frames inside their
  chunk, so chunked results differ from a single full-scene pass whenever the
  scene spans multiple chunks.
- Default is 1; `0` is rejected; a value at or above the frame count is a plain
  single pass.

## Point-cloud output

- The run writes a PLY point cloud filtered by confidence (`conf > 1.0` by
  default).
- `--depth-points <path>` additionally unprojects the depth maps to a
  world-space PLY/XYZ cloud (`depth_conf > --depth-conf-threshold`, default 1.0).

## COLMAP export

`--colmap-dir <dir>` writes the COLMAP binary sparse model
(`cameras.bin`/`images.bin`/`points3D.bin`) plus a sibling `points.ply`.

### Feedforward mode (default)

`--colmap-mode feedforward` reproduces the upstream `demo_colmap.py` branch:

- One PINHOLE camera per image.
- `--colmap-point-source depth` (default) keeps `depth_conf >= --colmap-conf-threshold`
  (default 5.0); `pointmap` uses the point-head `world_points` instead.
- Points are capped at `--colmap-max-points` (default 100000).
- `--colmap-world-conf-threshold <v>` adds a `world_points_conf >= v` gate on
  top of the depth-confidence gate (unset = off).
- Without `--colmap-seed` the cap keeps a deterministic evenly-spaced subsample;
  with a seed it draws randomly (distribution-equivalent to numpy's
  `randomly_limit_trues`, not bit-identical).
- Intrinsics are rescaled to the original resolution. Directory sources keep
  their frame file names; video and single-image sources write `image_N`.
- The run prints the export stats.

### Track mode

`--colmap-mode track` writes the track-based `batch_np_matrix_to_pycolmap`
reconstruction instead:

- SIMPLE_PINHOLE cameras, multi-frame tracks.
- `vis > --vis-thresh` (default 0.2), reprojection error
  `< --max-reproj-error` (default 8.0), `--shared-camera`, and the per-frame
  `--colmap-min-inlier-per-frame` floor (default 64).
- Requires `--colmap-dir` and `--track-auto`.
- Bundle adjustment is **not** run by default (upstream `--use_ba` defaults to
  false). The output is COLMAP-ready input for external BA or gsplat.

### External bundle adjustment

`--colmap-ba` runs `colmap bundle_adjuster --input_path <dir> --output_path <fresh temp dir>`
immediately after the track export.

- Binary from `LUX3D_COLMAP_BIN` or `PATH`.
- Track mode only; `--colmap-dir` is required.
- On success, `cameras.bin`/`images.bin`/`points3D.bin` are replaced in place
  and `points.ply` is preserved.
- On failure, the unadjusted model is left untouched and the error includes the
  captured stderr tail.

## Tracking

### Explicit queries

`--track-queries '<json>'` tracks explicit query points.

- Accepts `{"frame":0,"x":F,"y":F}` objects or `[0,x,y]` arrays.
- Pixel coordinates are in the model input resolution, `x`/`y` in `[0, 518)`.
- The query frame must be `0`; rotate the clip so the desired query frame comes
  first.
- Writes `{"coords": [[[x,y],...],...], "visibility": [[...],...], "confidence": [[...],...]|null}`
  to `--track-output` (default `<source-stem>_tracks.json`), plus `"query_frames"`
  in auto mode.
- `--track-iters` sets refinement iterations (default 4; `0` is rejected).

### Automatic stage-A query selection

`--track-auto` replaces explicit queries with tracking stage A:

1. DINOv2 ViT-B/14-with-registers CLS similarity + FPS ranks
   `--query-frame-num` frames (default 8; frame 0 is prepended when missing).
2. Per-frame query points come from the official ALIKED-n16 + SuperPoint
   extractors (`--track-keypoints aliked-sp`, matching
   `keypoint_extractor="aliked+sp"`: ALIKED keypoints followed by SuperPoint,
   per-detector `max_num_keypoints = --max-query-pts` default 4096,
   `det_thres = 0.005`).
3. The concatenated points are confidence-gated like `track_predict.py`
   (`conf > 1.2` when more than 512 survive).

`--track-keypoints fps` selects the deterministic weight-free FPS grid instead.

Required checkpoints:

| Component | Default path | Override |
|-----------|--------------|----------|
| ALIKED-n16 | `models/aliked-superpoint/aliked-n16.safetensors` | `LUX3D_ALIKED_WEIGHTS` |
| SuperPoint | `models/aliked-superpoint/superpoint_v1.safetensors` | `LUX3D_SUPERPOINT_WEIGHTS` |
| DINOv2-reg | `models/dinov2_reg4/model.safetensors` | `LUX3D_DINOV2_REG_MODEL_PATH` |

Convert the official `.pth` files once:

```powershell
python scripts/convert_keypoint_weights.py --input aliked-n16.pth --output models/aliked-superpoint/aliked-n16.safetensors
python scripts/convert_keypoint_weights.py --input superpoint_v1.pth --output models/aliked-superpoint/superpoint_v1.safetensors
```

Missing checkpoints are an error; there is never a silent FPS fallback.

### Tracking caveats

- Detection runs on the prepared model-resolution frame (the official demo
  detects on its 1024 px loaded images because its tracker is separate); points
  are scaled back before the track head.
- The official `torch.randperm` query shuffle is not reproduced, so query order
  is deterministic.
- Fine tracker stages B/C (LightGlue matching/refinement) are not included, so
  auto tracks are coarser than the official VGGSfM input.
- `--track-queries` and `--track-auto` are mutually exclusive.
- `--batch-size` chunking greater than 1 is rejected together with any tracking
  (the track pass runs over all frames at once).
- `--track-points-ply <path>` writes one colored vertex per query (frame-0 world
  point at the query pixel).

## GLB scene export

`--glb <scene.glb>` writes the demo-gradio GLB scene from `predictions_to_glb`:

- Confidence-filtered colored points from `--glb-point-source depth` (default)
  or `pointmap`.
- `--glb-conf-percentile` drops that percentage of the lowest-confidence points
  (default 50; `0` disables the percentile gate and keeps every point with
  `conf > 1e-5`).
- Per-frame camera cones with `--glb-show-cam`.
- The scene is aligned to the first camera with the official OpenGL ±y/z
  convention.

## Raw tensor dump

`--dump-npz <path>` keeps the raw inference tensors:

- Historical keys: `world_points`, `world_points_conf`, `depth`, `depth_conf`,
  `extrinsics`, `intrinsics`, `pose_enc`, `pose_enc_list` (`[K, S, 9]`, one
  entry per camera refinement iteration).
- Official demo keys: `images`, `extrinsic`, `intrinsic`,
  `world_points_from_depth`.
- Works together with the track/COLMAP/depth/GLB flags.

Use the dump with `scripts/compare_geometry.py vggt` to measure quantization
drift; see [weights.md](weights.md#quantization-fidelity).

## Licensing

VGGT-1B code and weights are CC-BY-NC-4.0; commercial use requires the separate
`facebook/VGGT-1B-Commercial` checkpoint. ALIKED and SuperPoint are separate
third-party checkpoints. See [models.md](models.md#vggt-1b) for details.
