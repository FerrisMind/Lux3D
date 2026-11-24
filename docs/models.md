# Supported models and licensing

Lux3D ships six model families. Each family has a canonical package layout, a
runtime pipeline, and its own upstream license terms. The CLI can inspect the
expected contract and the recorded license metadata before redistribution.

| Family | Task | Output | Canonical weights | Upstream |
|--------|------|--------|-------------------|----------|
| `pi3` | Multi-view point cloud | PLY / XYZ | `oxide-lab/Pi3` | [yyfz/Pi3](https://github.com/yyfz/Pi3), [`yyfz233/Pi3`](https://huggingface.co/yyfz233/Pi3) |
| `pi3x` | Multi-view point cloud + VO | PLY / XYZ | `oxide-lab/Pi3X` | [yyfz/Pi3](https://github.com/yyfz/Pi3), [`yyfz233/Pi3X`](https://huggingface.co/yyfz233/Pi3X) |
| `triposr` | Single-image mesh | GLB / OBJ / PLY / STL | `oxide-lab/TripoSR` | [VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR), [`stabilityai/TripoSR`](https://huggingface.co/stabilityai/TripoSR) |
| `vggt` | Multi-view geometry + COLMAP | PLY / XYZ / COLMAP / GLB | `oxide-lab/VGGT-1B` | [facebookresearch/vggt](https://github.com/facebookresearch/vggt), [`facebook/VGGT-1B`](https://huggingface.co/facebook/VGGT-1B) |
| `triposg` | Image-to-mesh, scribble, surface encode | GLB / OBJ / PLY / STL | `oxide-lab/TripoSG` | [VAST-AI-Research/TripoSG](https://github.com/VAST-AI-Research/TripoSG), [`VAST-AI/TripoSG`](https://huggingface.co/VAST-AI/TripoSG) |
| `trellis2` | Image-to-3D, mesh texturing, O-Voxel | GLB / OBJ / PLY / STL | `oxide-lab/TRELLIS.2-4B` (reserved) | [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2), [`microsoft/TRELLIS.2-4B`](https://huggingface.co/microsoft/TRELLIS.2-4B) |

`oxide-lab/TRELLIS.2-4B` is reserved but not published; pass an explicit
`--model-path` for TRELLIS.2.

## Pipeline summaries

### Pi3 and Pi3X

Pi3 produces point clouds from unordered image sequences. Pi3X adds ray/depth
conditioning, visual odometry chunking, and a metric head. Both use a DINOv2
image encoder and a permutation-equivariant decoder.

### TripoSR

Single-image mesh reconstruction with a transformer encoder, triplane
representation, and marching cubes. The CLI exposes `--mc-resolution` and
`--mc-threshold`.

### VGGT-1B

Multi-view point/depth/camera inference with feedforward or track-based COLMAP
export, depth unprojection, stage-A tracking, and the demo-style GLB scene.
See [vggt.md](vggt.md).

### TripoSG

Image-to-mesh inference with a DINOv2-large encoder, a 21-block rectified-flow
DiT, and an SDF VAE decoder/encoder. Additional surfaces:

- Reference-default flash decoder plus a clean-room DiffDMC fallback.
- TripoSG-scribble dual conditioning (`--mode scribble`), including a standalone
  one-file GGUF package.
- Batched multi-image runs (`--num-shapes`) and the reference VAE surface-encode
  path (`--surface-input`).
- Both canonical and standalone GGUF distribution formats; see
  [weights.md](weights.md).

### TRELLIS.2-4B

Image-to-3D inference (DINOv3 ViT-L/16, SS flow DiT, shape/texture SLAT flow
transformers, FlexiDualGrid VAE, O-Voxel dual-grid mesh with PBR vertex
attributes) plus texturing of existing meshes (`--texture-mesh`). See
[trellis2.md](trellis2.md).

## Licensing

The code and documentation in this repository are Apache-2.0. Upstream model
assets and canonicalized artifacts keep their original licenses and usage
restrictions. Review model-family-specific terms before redistribution; the CLI
`inspect` command exposes the recorded contract.

### Pi3 / Pi3X

The upstream Pi3 code is BSD-2-Clause for academic use; commercial use requires
contacting the authors. Pi3X model weights are CC-BY-NC-4.0 ("strictly
non-commercial"), and its code is BSD-3-Clause. Verify the current upstream
terms before redistribution.

### TripoSR

Code and weights are MIT (`stabilityai/TripoSR`).

### VGGT-1B

The original code and `facebook/VGGT-1B` checkpoint are CC-BY-NC-4.0
(non-commercial). Commercial use requires the separately distributed
`facebook/VGGT-1B-Commercial` checkpoint. The canonical package used by
`--model-path`/auto-download is published as `oxide-lab/VGGT-1B`; the raw
upstream checkpoint stays at `facebook/VGGT-1B`.

The native track head supports explicit query points and automatic stage-A
query selection (DINOv2-reg frame ranking + ALIKED-n16/SuperPoint keypoints).
The official fine tracker stages B/C (LightGlue matching/refinement) are not
included. ALIKED and SuperPoint are separate third-party checkpoints, not part
of the VGGT canonical package: the LightGlue reference implementation is
Apache-2.0, while the downloaded ALIKED/SuperPoint weights retain their upstream
licenses and terms.

### TripoSG

The upstream code and weights are MIT. The raw checkpoint lives at
`VAST-AI/TripoSG`; the canonical package used by `--model-path`/auto-download is
published as `oxide-lab/TripoSG`.

Upstream `load_image` uses BriaRMBG 1.4 when a valid alpha channel is missing;
Lux3D implements that fallback fully. The RMBG weights do not join the MIT
TripoSG package: they are normalized into the separate `rmbg-1.4` canonical
package under the BRIA source-available/non-commercial license (commercial use
requires a BRIA license), resolved via `--rmbg-model-path` or a sibling
`rmbg-1.4` directory.

Upstream's pipeline defaults to the flash decoder (`use_flash_decoder=True`,
`flash_octree_depth=9`); Lux3D implements the same flash SDF schedule plus a
clean-room Rust DiffDMC. Diso (CC BY-NC 4.0) source and tables are not copied,
so ambiguous checkerboard cells may resolve differently; the non-flash
dense/hierarchical path remains available via `--flash-decoder false`. The full
VAE encode API (FPS sampling, `vae.quant`, mean/logvar/std distribution) is also
implemented even though the image-to-3D pipeline only calls `vae.decode`.

### TRELLIS.2-4B

The upstream code (`microsoft/TRELLIS.2`) and the `microsoft/TRELLIS.2-4B` model
card are MIT licensed; the reference is mirrored under
`candle-3d-refs/TRELLIS.2`.

- Obtain the weights from the upstream repository and normalize them with
  `weights normalize ... trellis2` before running.
- The bundled DINOv3 ViT-L/16 encoder is a separately licensed gated Meta
  model. Verify its terms independently; DINOv3 access may be required
  separately from TRELLIS.2.
- The reference background remover, RMBG-2.0 (BiRefNet/Swin-L), is implemented
  and used for RGB inputs. Its weights do not join the MIT TRELLIS.2 package.
  Normalize them under the `rmbg2` auxiliary family and pass `--rmbg-model-path`
  (or place `rmbg-2.0` next to `--model-path`). RMBG-2.0 is CC-BY-NC-4.0
  (non-commercial), with commercial use subject to BRIA's terms.

## Acknowledgements

- [FerrisMind/candle](https://github.com/FerrisMind/candle) — Vulkan/WGPU backend
  fork used by this workspace.
- [yyfz/Pi3](https://github.com/yyfz/Pi3) — Pi3 and Pi3X.
- [VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR) — TripoSR.
- [facebookresearch/vggt](https://github.com/facebookresearch/vggt) — VGGT.
- [VAST-AI-Research/TripoSG](https://github.com/VAST-AI-Research/TripoSG) — TripoSG.
- [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) — TRELLIS.2 and O-Voxel.
- [Shiaoming/ALIKED](https://github.com/Shiaoming/ALIKED) and
  [cvg/LightGlue](https://github.com/cvg/LightGlue) — keypoint extractors used by
  VGGT stage-A tracking.
