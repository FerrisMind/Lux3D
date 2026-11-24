# TRELLIS.2: image-to-3D, texturing, and mesh post-processing

TRELLIS.2-4B generates textured meshes from a single image and can also texture
an existing mesh. The pipeline is DINOv3 ViT-L/16 → sparse-structure flow DiT →
shape/texture SLAT flow transformers → FlexiDualGrid VAE → O-Voxel dual-grid
mesh with PBR vertex attributes.

Run commands live in [inference.md](inference.md). Weight packing and
quantization live in [weights.md](weights.md). Backend coverage and measured
timings are in [performance.md](performance.md).

## Pipelines and sampling

### Pipeline variants

`--pipeline-type` accepts:

| Value | Meaning |
|-------|---------|
| `512` | 512 preview pipeline |
| `1024` | 1024 pipeline (decodes at 1024) |
| `1024_cascade` | Reference default: cascade that always decodes at 1024 |
| `1536_cascade` | Cascade targeting the 1536 grid with the hardcoded 1024 token-budget stop |

Variant dispatch and the 1536→…→1024 token-budget loop are covered by tests,
but no real-weight end-to-end `1536_cascade` parity run is recorded in this
tree.

### Sampler parameters

- `--steps` feeds all three samplers.
- Per-stage overrides replace the per-stage defaults when set:
  - `--ss-steps` / `--shape-steps` / `--tex-steps`
  - `--ss-guidance` / `--shape-guidance` / `--tex-guidance`
  - `--ss-rescale` / `--shape-rescale` / `--tex-rescale`
  - `--ss-interval` / `--shape-interval` / `--tex-interval` (`lo,hi`)
- Resolution order follows `resolve_stage_params`: a stage `steps` override
  wins over `--steps`.
- Reference defaults: 12 steps per sampler, SS/shape guidance 7.5, texture
  guidance 1.0, seed 42.

### Multiple samples

`--num-samples N` follows the upstream `run(num_samples)` draw structure
(`trellis2_image_to_3d.py:208,253`):

- One joint `[N, 8, R, R, R]` sparse-structure noise draw and one joint SLAT
  noise draw per stage over all samples' concatenated coordinates, both from
  `--seed`.
- One shared conditioning pass, then per-sample SLAT sampling/decode/export.
- Outputs and latent files are suffixed `_0`, `_1`, ...
- `N = 1` keeps the single-run path bit-identically.

The Lux3D RNG is not bit-identical to `torch.randn`, so draws are
distribution-equivalent, not value-identical, to the reference.

### Latent export

`--return-latent <path>` writes `shape_latent`, `tex_latent`, `coords`,
`resolution`, `lr_shape_latent`, and `meta` as `.npz` (the upstream
`return_latent` surface).

## Preprocessing

The reference preprocessing is reproduced exactly:

1. LANCZOS max-side downscale.
2. Alpha path when the RGBA input carries non-opaque alpha; otherwise
   RMBG-2.0/BiRefNet produces the mask. Missing RMBG weights are an error, never
   an opaque fallback.
3. Alpha > 0.8 bounding box.
4. Pillow-rounded square crop.
5. Premultiply and resize.

`--no-preprocess-image` skips RMBG/crop and feeds the square source image
directly; it no longer requires an RMBG model.

## Post-processing and mesh export

Post-processing runs by default and mirrors the shipped `to_glb` entry points
(`example.py:43`, `app.py:503`):

1. Shared hole fill: `--fill-hole-perimeter` (default `3e-2`; `0` = off).
2. Narrow-band dual-contouring remesh on the canonical `[-0.5, 0.5]^3` frame
   with `scale = (resolution + 3 * band) / resolution`:
   - `--remesh-band` (default 1 grid cell)
   - `--remesh-project` (default `0` = off)
3. A single simplify pass toward `--decimation-target` (default 1000000; `0`
   keeps the remeshed topology).

### Cleanup-chain fallback

`--no-remesh` restores the previous cleanup chain byte-for-byte:

- Aggressive 3× simplify.
- Duplicate-face and non-manifold repair.
- Small-component removal via `--min-component-area` (default `1e-5`; `0` keeps
  all).
- Orientation unification.
- Final simplify.

`--no-postprocess` disables the whole pass.

### Remesh fidelity

- Remeshed textured GLBs set `doubleSided = false` (upstream
  `postprocess.py:303`); non-remeshed/pass-through meshes keep `true`.
- The remesh branch is a direct port of the official CuMesh narrow-band
  dual-contouring algorithm: UDF−eps level set, base resolution halved to 32
  then adaptive octree expansion, mean edge intersections without QEF, official
  adaptive diagonal and triangle topology.
- It is validated against official CuMesh golden meshes at resolutions 32/64.
- Hole fill, cleanup, and decimation remain clean-room ports on the meshopt
  simplifier, so post-remesh simplification can still differ from the reference
  CuMesh pipeline.

## Texture baking and GLB export

When `--texture` is on (default), the textured-GLB export reproduces the
upstream `to_glb` UV unwrap and PBR bake:

- `--texture-size <n>` sets the bake edge length (default 2048). The CLI accepts
  any value in `64..=8192`; the upstream reference only allows 1024..4096 in
  steps of 1024, and the core adds an nvdiffrast-parity multiple-of-8 rule.
- The `.glb` embeds the baked base-color and metallic-roughness PNGs with
  TEXCOORD UVs, area-weighted normals, and the upstream glTF axis conversion
  `(x, y, z) -> (x, z, -y)`.
- `--no-texture-export` skips the whole stage; `--no-uv-unwrap` skips just the
  xatlas unwrap. Both fall back to the vertex-color mesh.
- `.obj`/`.ply`/`.stl` outputs never embed textures.

The xatlas unwrap and bake run on the CPU: measured ~42 s unwrap + ~9 s bake for
an 80k-triangle sphere at 2048 (release build, this tree), scaling with
`--texture-size` squared.

Uncovered atlas texels are filled with a TELEA-approximation inpaint (iterative
nearest-valid onion-peel), not the reference `cv2.inpaint`. The run prints
`[bake]` stats: charts, atlases, atlas size, texture size, valid/inpainted texel
counts.

The mesh stays the O-Voxel dual-grid surface, and the `<stem>.pbrvox.bin` +
`<stem>.pbr.json` sidecar pair remains the lossless per-voxel PBR record.

## Memory behavior

- Weight stages load sequentially: ~7.8 GiB measured peak on `1024_cascade`
  with the texture stages resident.
- CPU inference works but is slow.
- `--no-low-vram` maps to the upstream `low_vram=False` mode: every loaded stage
  stays resident on the device (one up-front load set, no per-stage drop).
  Outputs are identical to the default for the same seed, peak VRAM is higher,
  and a resident set that does not fit fails with an error naming the flag, the
  stage, and the device memory limit instead of silently falling back.

## Model resolution

Auto-download from `oxide-lab/TRELLIS.2-4B` is reserved but not yet published,
so pass an explicit `--model-path` (canonical package or raw checkpoint
directory). The server loads the canonical package only; the raw
`microsoft/TRELLIS.2-4B` checkpoint directory is a CLI-only loader.

## Texturing an existing mesh

`--texture-mesh <file>` textures an existing mesh from the `--source` condition
image (upstream `Trellis2TexturingPipeline`) instead of running image-to-3D.

### Pipeline

1. Preprocess the mesh into `[-0.5, 0.5]^3`: center/scale to 0.99999 plus y/z
   swap. `--no-preprocess-mesh` requires the input already in that frame.
2. Voxelize to the flexible dual grid at `--texture-resolution`:
   - `512`, `1024` (default), or `1536`.
   - `1536` voxelizes/encodes at the full 1536 grid but shares the 1024
     conditioning and `tex_dit_1024`.
3. Encode a raw shape SLAT with `shape_enc_next_dc_f16c32_fp16`.
4. Sample `tex_dit_512`/`tex_dit_1024` with the `--tex-*` overrides.
5. Decode with synthesized occupancy guides.
6. Bake the UV atlas.

### UV handling

- The core `texture_mesh` API bakes directly on caller-supplied per-corner UVs
  (the reference `postprocess_mesh` UV-reuse branch, no xatlas run).
- The CLI `--texture-mesh` path and the HTTP texturing job pass the imported
  mesh's per-corner UVs through: an input that carries UVs keeps them (no
  xatlas), while a mesh without UVs still unwraps.
- Cleanup/decimation/remeshing are not run (UV seams duplicate vertices and
  zero-area faces are dropped).
- UV charting welds duplicate-position vertices first, so topology-repair vertex
  splits do not multiply chart count; chart count tracks connected surface
  components.

### Inputs and outputs

- `--texture-mesh` accepts `.glb`, `.obj`, `.ply`, or `.stl`.
- The output must be `.glb`.
- It writes only the textured GLB (no PBR sidecar).

### Flag matrix

Rejected together with `--texture-mesh`:

- `--pipeline-type`
- `--max-num-tokens`
- `--num-samples`
- `--return-latent`
- `--no-postprocess`
- `--decimation-target`
- `--no-texture-export`
- `--no-uv-unwrap`
- `--texture false`

Honored in texturing mode:

- `--tex-*`
- `--seed`
- `--texture-size`
- `--texture-resolution`
- `--no-preprocess-image`
- `--no-preprocess-mesh`

`--steps`/`--ss-*`/`--shape-*` are not rejected but have no effect in this mode.

### Shape encoder resolution

The shape encoder is resolved from the first available source:

1. `<model-dir>/ckpts/shape_enc_next_dc_f16c32_fp16.safetensors` (raw checkpoint
   layout).
2. `<model-dir>/external/shape_enc/model.safetensors` (sidecar next to a merged
   canonical/GGUF package).
3. `shape_enc.*` keys in a merged safetensors package.
4. `shape_enc.*` tensors embedded in a standalone GGUF by
   `weights quantize --merge-source`.

The raw `TRELLIS.2-4B` directory, a canonical `model.gguf` + external
encoder/`external/dinov3` layout, and a self-contained standalone TRELLIS.2 GGUF
all work.

### Quantized texturing smoke test

Quantized Q8_0 texturing was smoke-tested on CUDA with
`--texture-resolution 512 --texture-size 512` on the Q8 512 shape output
(507,540 vertices / 1,000,200 faces):

- 3,241 charts, 2 atlases, 110,789 valid texels.
- 30 MB PBR GLB with base-color and metallic-roughness PNGs.

## Licensing

TRELLIS.2 code and model card are MIT. The bundled DINOv3 encoder is a
separately licensed gated Meta model; the RMBG-2.0 background remover is
CC-BY-NC-4.0. See [models.md](models.md#trellis2-4b).
