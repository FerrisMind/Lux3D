# Weights: canonical packages, normalization, and GGUF quantization

This guide covers the canonical package layout, the `inspect` and
`weights normalize` commands, GGUF quantization, and the measured fidelity of
every variant. For the CLI `run` surface, see [inference.md](inference.md).

## Canonical package layout

A canonical package is a self-describing directory consumed by `--model-path`:

| File | Purpose |
|------|---------|
| `model.safetensors` | Weights with family-specific normalized names |
| `resolved_config.json` | Architecture and preprocessing parameters |
| `manifest.json` | Tensor inventory and family identity |
| `checksums.json` | SHA-256 checksums validated on load |

Inspect the expected contract for a family before preparing weights:

```powershell
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> triposg
```

## Normalize canonical weights

```powershell
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-pi3-dir> --output-dir <canonical-pi3-dir> pi3
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-pi3x-dir> --output-dir <canonical-pi3x-dir> pi3x
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-triposr-dir> --output-dir <canonical-triposr-dir> triposr
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-vggt-dir> --output-dir <canonical-vggt-dir> vggt
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-triposg-dir> --output-dir <canonical-triposg-dir> triposg
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-trellis2-dir> --output-dir <canonical-trellis2-dir> trellis2

# Auxiliary RMBG-2.0 package (CC-BY-NC): use the packer directly.
python -I tools/python_baseline/normalize_weights.py --family rmbg2 `
  --raw-model-dir <raw-RMBG-2.0-dir> --output-dir <canonical-rmbg2-dir>
```

TRELLIS.2 normalization packs the flat `microsoft/TRELLIS.2-4B` layout (seven
`ckpts/*.json` + `*.safetensors` checkpoint pairs, the external `ss_dec` sparse
structure decoder, and the DINOv3 config/model files) into the canonical package
layout that `Trellis2Pipeline::load` and the server expect. The texturing-only shape encoder is not packed into
the merged canonical weight file. `--texture-mesh` loads it from the raw
`ckpts/shape_enc_next_dc_f16c32_fp16.safetensors`, from
`external/shape_enc/model.safetensors` beside a canonical `model.gguf`, or
from `shape_enc.*` tensors embedded in a standalone GGUF with
`weights quantize --merge-source` (see [TRELLIS.2](trellis2.md)).

## Quantize weights to GGUF

`weights quantize` packs the canonical `model.safetensors` into `model.gguf`.
Only rank-2 `*.weight` matrices become GGUF blocks; norms, biases, embeddings,
convolutions, and rows that do not divide into the dtype block stay in the
fallback dtype. Quantized linears run through `QMatMul` (integer dot-product
kernels), everything else is dequantized on demand.

The converter embeds the sibling `resolved_config.json` in GGUF metadata, so
`model.gguf` can be used as a **standalone package**: put it alone in a
directory and pass that directory as `--model-path`. The loader synthesizes
the manifest/checksums and reads the config from the GGUF. If `model.gguf`
sits next to a full canonical package, the package files are still validated
and the GGUF supplies the weights. Rank-0 tensors (training-only buffers such
as BatchNorm `num_batches_tracked`) are omitted; they are not used at
inference. `--model-path` also accepts a standalone `.gguf` file directly, so
quantized files can be distributed flat (`TripoSG-1.5B-Q8_0.gguf`,
`VGGT-1B-Q8_0.gguf`, ...).

`general.architecture` is derived from the resolved config: top-level
`architecture`, then `family`, then the model name (`triposg`, `vggt`,
`rmbg`). For example RMBG writes `BriaRMBG`, TripoSG writes `triposg`, and
VGGT (whose config has neither key) writes `vggt`. `lux3d.family` always
records the runtime package family.

```powershell
# Q8_0 weights, F16 fallback (reference-quality default)
cargo run -p lux3d-cli -- weights quantize triposg `
  --source <canonical-triposg-dir>/model.safetensors `
  --output <gguf-triposg-dir>/model.gguf --dtype q8_0 --fallback-dtype f16

# K-quants: smaller files, 256-wide blocks
cargo run -p lux3d-cli -- weights quantize vggt `
  --source <canonical-vggt-dir>/model.safetensors `
  --output <gguf-vggt-dir>/model.gguf --dtype q4_k --fallback-dtype f16

# Pi3, Pi3X and TripoSR use the same converter.
cargo run -p lux3d-cli -- weights quantize pi3 `
  --source <canonical-pi3-dir>/model.safetensors `
  --output <gguf-pi3-dir>/model.gguf --dtype q8_0 --fallback-dtype f16
cargo run -p lux3d-cli -- weights quantize pi3x `
  --source <canonical-pi3x-dir>/model.safetensors `
  --output <gguf-pi3x-dir>/model.gguf --dtype q8_0 --fallback-dtype f16
cargo run -p lux3d-cli -- weights quantize triposr `
  --source <canonical-triposr-dir>/model.safetensors `
  --output <gguf-triposr-dir>/model.gguf --dtype q8_0 --fallback-dtype f16

# TRELLIS.2: quantize only the flow-DiT linear weights (`QMatMul` path);
# DINOv3/VAE/decoder tensors, 3D conv kernels, norms and biases stay F16
# (their loaders read dense tensors). The F16 build drops the prefixes.
# The shape encoder lives outside the canonical package; `--merge-source`
# embeds it in the standalone GGUF under `shape_enc.*` so `--texture-mesh`
# works without the raw TRELLIS.2-4B checkpoint tree.
cargo run -p lux3d-cli -- weights quantize trellis2 `
  --source <canonical-trellis2-dir>/model.safetensors `
  --output trellis2-q8_0.gguf --dtype q8_0 --fallback-dtype f16 `
  --prefix ss_flow. --prefix shape_dit_512. --prefix shape_dit_1024. `
  --prefix tex_dit_512. --prefix tex_dit_1024. `
  --merge-source <raw-TRELLIS.2-4B-dir>/ckpts/shape_enc_next_dc_f16c32_fp16.safetensors `
  --merge-prefix shape_enc

# TripoSG-scribble canonical package: the normalizer detects the extra
# `text_encoder/model.safetensors` shard, merges transformer/vae/image_encoder/
# text_encoder under one safetensors file, and embeds the configs, pipeline
# defaults and CLIP tokenizer. `--model-path <dir>` loads it directly.
cargo run -p lux3d-cli -- weights normalize `
  --repo-root <runtime-root> `
  --raw-model-dir <raw-TripoSG-scribble-dir> `
  --output-dir <canonical-triposg-scribble-dir> triposg

# TripoSG-scribble standalone package: one GGUF with the DINOv2 image encoder,
# dual-cross-attention DiT, SDF VAE decoder, CLIP text encoder and CLIP
# tokenizer. --source is the raw `VAST-AI/TripoSG-scribble` checkpoint
# directory. Rank-2 linear weights in the DiT and VAE are quantized by
# default (the VAE Q8_0 path is measured near-lossless); DINOv2/CLIP stay in
# the fallback dtype. Pass `--prefix` to override the set.
cargo run -p lux3d-cli -- weights quantize triposg --scribble --standalone `
  --source <raw-TripoSG-scribble-dir> `
  --output triposg-scribble-q8_0.gguf `
  --dtype q8_0 --fallback-dtype f16

# Legacy DiT-only override: quantizes only the dual-cross-attention DiT; the
# text encoder, VAE and DINOv2 stay in the raw checkpoint. Write to
# `<checkpoint>/transformer/model.gguf` for automatic pickup, or pass the
# `.gguf` file directly as `--model-path`.
cargo run -p lux3d-cli -- weights quantize triposg --scribble `
  --source <raw-TripoSG-scribble-dir> `
  --output <raw-TripoSG-scribble-dir>/transformer/model.gguf `
  --dtype q8_0 --fallback-dtype f16

# RMBG-1.4 is an auxiliary package: separate GGUF, F16, same standalone rules.
cargo run -p lux3d-cli -- weights quantize --aux-family rmbg `
  --source <canonical-rmbg-dir>/model.safetensors `
  --output <gguf-rmbg-dir>/model.gguf --dtype f16 --fallback-dtype f16

# RMBG-2.0/BiRefNet for TRELLIS.2 is a separate auxiliary package too.
# F16 is the reference-fidelity default (alpha max |diff| = 1 on the validation
# image); Q8_0 shrinks 0.82 GiB to 0.24 GiB with alpha max |diff| = 7.
cargo run -p lux3d-cli -- weights quantize --aux-family rmbg2 `
  --source <canonical-rmbg2-dir>/model.safetensors `
  --output <gguf-rmbg2-f16-dir>/model.gguf --dtype f16 --fallback-dtype f16
cargo run -p lux3d-cli -- weights quantize --aux-family rmbg2 `
  --source <canonical-rmbg2-dir>/model.safetensors `
  --output <gguf-rmbg2-q8_0-dir>/model.gguf --dtype q8_0 --fallback-dtype f16

# Run with only the GGUF directories; no safetensors/manifest/checksums needed.
cargo run -p lux3d-cli -- run triposg --device cuda `
  --model-path <gguf-triposg-dir> --rmbg-model-path <gguf-rmbg-dir> `
  --source <image> --output out.obj
cargo run -p lux3d-cli -- run trellis2 --device cuda `
  --model-path <gguf-trellis2-dir> --rmbg-model-path <gguf-rmbg2-f16-dir> `
  --source <image> --output out.glb

# TripoSG-scribble canonical package: pass the directory containing
# `model.safetensors`, `resolved_config.json`, `manifest.json` and
# `checksums.json`. No raw checkpoint is needed.
cargo run -p lux3d-cli -- run triposg --device cuda --mode scribble `
  --prompt "a cat with wings" --model-path <canonical-triposg-scribble-dir> `
  --source <scribble.png> --output out.glb

# TripoSG-scribble standalone package: pass the `.gguf` file or a directory
# containing only `model.gguf`. No raw checkpoint, safetensors or tokenizer
# files are needed.
cargo run -p lux3d-cli -- run triposg --device cuda --mode scribble `
  --prompt "a cat with wings" --model-path triposg-scribble-q8_0.gguf `
  --source <scribble.png> --output out.glb
cargo run -p lux3d-cli -- run triposg --device cuda --mode scribble `
  --prompt "a cat with wings" --model-path <dir-containing-model.gguf> `
  --source <scribble.png> --output out.glb

# Legacy DiT-only override: the raw checkpoint supplies the float text
# encoder/VAE/DINOv2 and tokenizer. A sibling `transformer/model.gguf` is
# picked up automatically; the `.gguf` can also be passed directly.
cargo run -p lux3d-cli -- run triposg --device cuda --mode scribble `
  --prompt "a cat with wings" --model-path models/TripoSG-scribble `
  --source <scribble.png> --output out.glb
cargo run -p lux3d-cli -- run triposg --device cuda --mode scribble `
  --prompt "a cat with wings" --model-path <scribble-dit-q8_0.gguf> `
  --source <scribble.png> --output out.glb

# Optional: `--all-2d` also quantizes non-`*.weight` rank-2 tensors;
# `--prefix transformer.` (repeatable) restricts quantization for bisection.
```

Supported `--dtype` values: `f16`, `f32`, `bf16`, `q8_0`, `q4_0`, `q4_1`,
`q5_0`, `q5_1`, `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`. `q8_1`/`q8_k` are
activation-side GGML types and are rejected as weight targets. A GGUF created
without an embedded config (no sibling `resolved_config.json` at conversion
time) can still sit next to a full canonical package, but is rejected as a
standalone package.

Measured on the bundled canonical packages (`RTX 3060 12 GiB`; TripoSG
`--steps 2 --num-tokens 1024 --flash-octree-depth 5`, VGGT/Pi3/Pi3X 3 frames,
TripoSR 1 image at `--mc-resolution 256`):

| model | variant | package size | weights kept |
|---|---|---|---|
| TripoSG | canonical F32 | 7.40 GiB | — |
| TripoSG | q8_0 | 1.97 GiB | 527 matrices |
| TripoSG | q6_k | 1.52 GiB | 525 |
| TripoSG | q5_k | 1.28 GiB | 525 |
| TripoSG | q4_k | 1.05 GiB | 525 |
| VGGT | canonical F32 | 4.68 GiB | — |
| VGGT | q8_0 | 1.33 GiB | 384 |
| VGGT | q6_k | 1.09 GiB | 331 |
| VGGT | q5_k | 0.95 GiB | 331 |
| VGGT | q4_k | 0.82 GiB | 331 |
| Pi3 | canonical F32 | 3.57 GiB | — |
| Pi3 | f16 | 1.79 GiB | all tensors |
| Pi3 | q8_0 | 0.95 GiB | 318 |
| Pi3 | q6_k | 0.74 GiB | 318 |
| Pi3 | q5_k | 0.62 GiB | 318 |
| Pi3 | q4_k | 0.51 GiB | 318 |
| Pi3X | canonical F32 | 5.07 GiB | — |
| Pi3X | f16 | 2.53 GiB | all tensors |
| Pi3X | q8_0 | 1.37 GiB | 466 |
| Pi3X | q6_k | 1.06 GiB | 466 |
| Pi3X | q5_k | 0.90 GiB | 466 |
| Pi3X | q4_k | 0.74 GiB | 466 |
| TripoSR | canonical F32 | 1.56 GiB | — |
| TripoSR | f16 | 0.78 GiB | all tensors |
| TripoSR | q8_0 | 0.42 GiB | 244 |
| TripoSR | q6_k | 0.33 GiB | 235 |
| TripoSR | q5_k | 0.27 GiB | 235 |
| TripoSR | q4_k | 0.23 GiB | 235 |
| TRELLIS.2-4B | canonical (BF16/F16/F32 mix) | 15.07 GiB | — |
| TRELLIS.2-4B | f16 | 15.16 GiB | all tensors + 284 shape_enc |
| TRELLIS.2-4B | q8_0 | 9.53 GiB | 1074 DiT linears + 284 shape_enc |
| TRELLIS.2-4B | q6_k | 8.07 GiB | 1070 + 284 shape_enc |
| TRELLIS.2-4B | q5_k | 7.27 GiB | 1070 + 284 shape_enc |
| TRELLIS.2-4B | q4_k | 6.52 GiB | 1070 + 284 shape_enc |
| TripoSG-scribble | raw checkpoint F16 | 4.38 GiB | — |
| TripoSG-scribble | canonical F16 | 4.38 GiB | — |
| TripoSG-scribble | q8_0 standalone | 2.70 GiB | 467 matrices |
| TripoSG-scribble | q6_k standalone | 2.27 GiB | 465 |
| TripoSG-scribble | q5_k standalone | 2.03 GiB | 465 |
| TripoSG-scribble | q4_k standalone | 1.81 GiB | 465 |

F16 is a straight precision conversion, not block quantization: every tensor is
stored at 16-bit and the linear layers use the regular path without `QMatMul`.
It costs about twice the Q8_0 size but preserves the F32 result: on
CUDA/Vulkan/WGPU/CPU the three builds drift at most 0.034% / 0.042% / 0.0014%
CD (Pi3/Pi3X/TripoSR) with F@1% = 1.000 on every backend. Build it with
`--dtype f16 --fallback-dtype f16`.

TRELLIS.2-4B quantization is DiT-only by policy: the five flow transformers
(`ss_flow`, `shape_dit_512/1024`, `tex_dit_512/1024`) load linears through the
`QMatMul` path, while DINOv3 (`image_cond`), the sparse VAE decoders and the
dense structure decoder read dense tensors (3D conv kernels, norms), so those
stages stay F16 in every variant. Standalone builds pass the raw
`ckpts/shape_enc_next_dc_f16c32_fp16.safetensors` stage through
`--merge-source ... --merge-prefix shape_enc`; it is stored F16 (its 3D
convolution kernels flattened) and makes `--texture-mesh` self-contained.
Candle's GGUF container caps tensor rank at four, so the 3D convolution
kernels are stored flattened and reshaped by the loader. The K-quants skip
four `input_layer` matrices whose row length does not divide into 256-wide
blocks; Q8_0 additionally skips one (row length 8).

TripoSR Q8_0 quantizes 244 matrices; the K-quants use 256-wide blocks, so nine
matrices whose row length is not divisible by 256 stay in F16.

RMBG-1.4 (auxiliary): F32 0.16 GiB → F16 0.08 GiB; 686 tensors written,
112 rank-0 training buffers omitted.

TripoSG-scribble quantizes the dual-cross-attention DiT and the SDF VAE
decoder; with `--standalone` it merges all four networks into one GGUF. The
CLIP text encoder and the DINOv2 scribble encoder stay in the F16 fallback
because their Q8_0 conditioning drift reaches the mesh.
The embedded `lux3d.resolved_config` carries every component config, the
pipeline defaults and the CLIP `vocab.json`/`merges.txt`/tokenizer configs, so
the resulting file is a complete model package. The canonical safetensors
package uses the same merge and embedded config/tokenizer, preserves the raw
F16 weights, and is loaded with `--model-path <canonical-triposg-scribble-dir>`.
The legacy DiT-only override still works with the raw checkpoint and is smaller
(1.67 / 1.29 / 1.08 / 0.88 GiB for q8_0/q6_k/q5_k/q4_k).

Other components can be added with repeatable `--prefix` flags. Measured on
CUDA against the transformer-only q8 mesh (same `mesh-input.png`, prompt
"a cat"):

| quantized prefixes | package size | CD rel | F@1% | voxel IoU | note |
|---|---|---|---|---|---|
| `transformer.` | 2.92 GiB | — | — | — | old DiT-only baseline |
| `transformer.` + `vae.` (default) | 2.70 GiB | 0.006% | 1.000 | 0.991 | near-lossless |
| `transformer.` + `image_encoder.` | 2.65 GiB | 0.184% | 0.939 | 0.700 | noticeable drift |
| `transformer.` + `text_encoder.` | 2.81 GiB | 0.159% | 0.960 | 0.710 | noticeable drift |
| all four | 2.33 GiB | 1.017% | 0.812 | 0.558 | topology drift |

The VAE decoder is the only extra component that stays effectively lossless
under Q8_0, so it is part of the default package. DINOv2 and CLIP Q8_0 are
supported (CLIP dequantizes at load), but their conditioning drift propagates
into the mesh, so they stay F16 unless `--prefix` opts them in.

The reference run is the raw checkpoint's native F16 DiT promoted to F32
(`--dtype` default `f32`), with the same seed/steps/image. Standalone and
DiT-only builds produce byte-identical meshes for the same quantized DiT and
input (verified on CUDA for all four variants); the table below is therefore
shared. Scribble drift (CUDA, RTX 3060; `cat_with_wings.png`, prompt
"a cat with wings", 16 steps, 512 tokens, dense/hierarchical depth 8, seed 42;
reference 80,109 vertices / 159,976 faces):

| variant | CD rel | F@0.5% | F@1% | voxel IoU | normal | vertices | faces | max drift | mean drift |
|---|---|---|---|---|---|---|---|---|---|
| q8_0 | 0.461% | 0.626 | 0.907 | 0.268 | 0.925 | 82,145 | 164,044 | 0.0669 | 0.0131 |
| q6_k | 0.603% | 0.515 | 0.812 | 0.204 | 0.908 | 84,865 | 169,440 | 0.1043 | 0.0171 |
| q5_k | 0.557% | 0.523 | 0.857 | 0.203 | 0.920 | 76,732 | 153,204 | 0.0654 | 0.0158 |
| q4_k | 0.744% | 0.444 | 0.742 | 0.167 | 0.893 | 83,340 | 166,368 | 0.1421 | 0.0211 |

Max/mean drift are the existing `compare_geometry.py` nearest-neighbour
metrics (`max`, `chamfer_l1_mean` in mesh units); voxel IoU captures the
topology change (marching-cubes vertex/face counts move by up to 6%). The
CFG-distilled scribble DiT is far more quantization-sensitive than the image
TripoSG model: only Q8_0 keeps F@1% above 0.90, and every K-quant changes the
surface topology substantially. Prefer Q8_0 for scribble; Q6_K is a compact
fallback, Q5_K/Q4_K are preview-grade. The standalone package is the default
distribution format; the DiT-only GGUF remains available for users who already
keep the raw checkpoint.

## Quantization fidelity

`scripts/compare_geometry.py` computes drift against the float reference:
symmetric Chamfer-L1 (relative to the reference bbox diagonal), F-score@tau,
surface voxel IoU, normal consistency for meshes; pointmap/depth PSNR,
AbsRel, Chamfer and camera-pose AUC for VGGT tensor dumps.
`scripts/eval_quantization.py` runs the reference plus every quantized package
and prints the table. For TripoSG, the reference run refuses a raw checkpoint
directory that contains a sibling `transformer/model.gguf` (the loader would
prefer it and silently evaluate a quantized reference); pass
`--allow-reference-gguf` only if that is intentional.

```powershell
# Mesh drift (TripoSG/TripoSR)
python scripts/compare_geometry.py mesh <reference.obj> <quantized.obj>

# VGGT: dump raw tensors next to the PLY, then compare
cargo run -p lux3d-cli -- run vggt --model-path <pkg> --source <frames> --output <out.ply> --dump-npz <out.npz>
python scripts/compare_geometry.py vggt <reference.npz> <quantized.npz>

# One-shot sweep over several quantized packages
python scripts/eval_quantization.py triposg --binary target/release/lux3d-cli.exe `
  --reference <canonical-triposg-dir> --variant q8_0=<dir-q8_0> --variant q4_k=<dir-q4_k> `
  --source <input-image> --device cuda --output-dir out/eval -- --steps 2 --num-tokens 1024 --flash-octree-depth 5

# The same loop covers the point-cloud and TripoSR families.
python scripts/eval_quantization.py pi3x --binary target/release/lux3d-cli.exe `
  --reference <canonical-pi3x-dir> --variant q8_0=<dir-q8_0> `
  --source <frames-dir> --device cuda --output-dir out/eval-pi3x
python scripts/eval_quantization.py triposr --binary target/release/lux3d-cli.exe `
  --reference <canonical-triposr-dir> --variant q4_k=<dir-q4_k> `
  --source <input-image> --device cuda --output-dir out/eval-triposr `
  --mc-resolution 256 --mc-threshold 25.0

# TRELLIS.2 (variants are standalone .gguf files, not directories)
python scripts/eval_quantization.py trellis2 --binary target/release/lux3d-cli.exe `
  --reference <canonical-trellis2-dir> --variant q8_0=trellis2-q8_0.gguf `
  --source <input-image> --device cuda --output-dir out/eval-trellis2 -- `
  --pipeline-type 512 --steps 12 --texture false --seed 42

# TripoSG-scribble (variants are standalone or DiT-only .gguf files; the raw
# reference dir is exported as LUX3D_TRIPOSG_SCRIBBLE_MODEL_DIR for the
# reference run and any legacy DiT-only variants)
python scripts/eval_quantization.py triposg --scribble `
  --reference <raw-TripoSG-scribble-dir> --variant q8_0=triposg-scribble-q8_0.gguf `
  --source <scribble.png> --device cuda --output-dir out/eval-scribble -- `
  --mode scribble --prompt "a cat with wings"
```

TripoSG mesh drift (2 steps, 1024 tokens, depth 5; reference is the same
backend's F32 mesh):

| variant | backend | CD rel | F@1% | voxel IoU | VRAM (delta) | wall |
|---|---|---|---|---|---|---|
| q8_0 | CUDA | 0.026% | 0.996 | 0.994 | 3.1 GiB | 10.4 s |
| q8_0 | Vulkan | 0.184% | 0.978 | 0.970 | 3.9 GiB | 13.3 s |
| q6_k | CUDA | 0.135% | 0.988 | 0.988 | — | — |
| q6_k | Vulkan | 0.134% | 0.995 | 0.995 | — | — |
| q5_k | CUDA | 0.155% | 0.993 | 0.994 | — | — |
| q5_k | Vulkan | 0.224% | 0.980 | 0.984 | — | — |
| q4_k | CUDA | 0.253% | 0.973 | 0.953 | 2.2 GiB | 10.0 s |
| q4_k | Vulkan | 0.180% | 0.982 | 0.972 | 2.3 GiB | 12.7 s |
| F32 | CUDA | — | — | — | 8.6 GiB | 14.9 s |
| F32 | Vulkan | — | — | — | 8.8 GiB | 17.1 s |

Two-step sampling is input-sensitive. On a deliberately harsh RGB input
(`mesh-input-rgb.png`, exercising the RMBG path) Q5_K and Q4_0 drifted to
~1.4% Chamfer while Q4_K stayed at 0.23% and Q8_0 at 0.06%. The outliers are
repeatable on that input but do not reproduce on the alpha input above, so
treat Q5_K/Q4_0 as input-dependent and prefer Q8_0/Q6_K/Q4_K for deployment.

VGGT (3 frames; reference is the same backend's F32 tensors; pose AUC on the
2 relative frame pairs):

| variant | backend | pointmap PSNR | depth PSNR | depth AbsRel | CD rel | pose AUC@5 | VRAM (delta) |
|---|---|---|---|---|---|---|---|
| q8_0 | CUDA | 53.1 dB | 48.1 dB | 0.0038 | 0.274% | 1.0 | 4.7 GiB |
| q8_0 | Vulkan | 55.9 dB | 53.5 dB | 0.0020 | 0.182% | 1.0 | 4.4 GiB |
| q8_0 | WGPU | 55.4 dB | 51.0 dB | 0.0027 | 0.193% | 1.0 | — |
| q6_k | CUDA | 45.7 dB | 44.8 dB | 0.0053 | 0.571% | 1.0 | — |
| q6_k | Vulkan | 46.5 dB | 49.5 dB | 0.0026 | 0.413% | 0.5 | — |
| q5_k | CUDA | 34.7 dB | 35.9 dB | 0.0144 | 1.54% | 0.0 | — |
| q5_k | Vulkan | 36.4 dB | 35.0 dB | 0.0180 | 1.39% | 1.0 | — |
| q4_k | CUDA | 24.7 dB | 23.5 dB | 0.0683 | 3.50% | 0.0 | 4.2 GiB |
| q4_k | Vulkan | 26.9 dB | 25.8 dB | 0.0511 | 2.84% | 0.0 | 3.7 GiB |
| F32 | CUDA | — | — | — | — | — | 6.7 GiB |
| F32 | Vulkan | — | — | — | — | — | 7.3 GiB |

Pi3/Pi3X point clouds and TripoSR mesh (CUDA detail; reference is the same
backend's F32 run; Pi3/Pi3X 3 frames, TripoSR `mesh-input.png` at 256):

| family | variant | CD rel | F@0.5% | F@1% | voxel IoU | normal |
|---|---|---|---|---|---|---|
| Pi3 | f16 | 0.034% | 1.000 | 1.000 | — | — |
| Pi3 | q8_0 | 0.166% | 0.990 | 0.99995 | — | — |
| Pi3 | q6_k | 0.139% | 0.985 | 0.99907 | — | — |
| Pi3 | q5_k | 0.851% | 0.469 | 0.706 | — | — |
| Pi3 | q4_k | 0.813% | 0.475 | 0.696 | — | — |
| Pi3X | f16 | 0.037% | 1.000 | 1.000 | — | — |
| Pi3X | q8_0 | 0.081% | 0.99999 | 1.000 | — | — |
| Pi3X | q6_k | 0.873% | 0.129 | 0.598 | — | — |
| Pi3X | q5_k | 2.279% | 0.000 | 0.000 | — | — |
| Pi3X | q4_k | 0.789% | 0.274 | 0.719 | — | — |
| TripoSR | f16 | 0.001% | 1.000 | 1.000 | 0.998 | 1.000 |
| TripoSR | q8_0 | 0.036% | 0.9997 | 0.99994 | 0.920 | 0.993 |
| TripoSR | q6_k | 0.050% | 0.9998 | 0.99995 | 0.886 | 0.991 |
| TripoSR | q5_k | 0.058% | 0.9996 | 0.99985 | 0.863 | 0.990 |
| TripoSR | q4_k | 0.129% | 0.9964 | 0.99985 | 0.682 | 0.978 |

The quantized packages on every backend, each compared against that backend's
own F32 run. F32 outputs agree across backends within 0.04% CD with
F@1% = 1.000, so the spread below is quantization behavior, not baseline
noise:

| family | variant | CD rel CUDA / Vulkan / WGPU / CPU | F@1% CUDA / Vulkan / WGPU / CPU |
|---|---|---|---|
| Pi3 | q8_0 | 0.166 / 0.172 / 0.259 / 0.251% | 0.99995 / 1.000 / 0.9999 / 0.9999 |
| Pi3 | q6_k | 0.139 / 0.455 / 0.546 / 0.555% | 0.99907 / 0.966 / 0.919 / 0.895 |
| Pi3 | q5_k | 0.851 / 0.190 / 0.240 / 0.209% | 0.706 / 0.995 / 0.986 / 0.990 |
| Pi3 | q4_k | 0.813 / 1.191 / 0.964 / 1.016% | 0.696 / 0.535 / 0.663 / 0.644 |
| Pi3X | q8_0 | 0.081 / 0.071 / 0.086 / 0.155% | 1.000 / 1.000 / 1.000 / 1.000 |
| Pi3X | q6_k | 0.873 / 1.818 / 1.181 / 1.613% | 0.598 / 0.070 / 0.237 / 0.089 |
| Pi3X | q5_k | 2.279 / 3.383 / 2.819 / 2.749% | 0.000 / 0.000 / 0.000 / 0.000 |
| Pi3X | q4_k | 0.789 / 0.328 / 0.383 / 0.370% | 0.719 / 0.947 / 0.930 / 0.938 |
| TripoSR | q8_0 | 0.036 / 0.025 / 0.032 / 0.028% | 0.9999 / 1.000 / 1.000 / 0.9999 |
| TripoSR | q6_k | 0.050 / 0.052 / 0.056 / 0.038% | 0.9999 / 1.000 / 0.9999 / 1.000 |
| TripoSR | q5_k | 0.058 / 0.055 / 0.066 / 0.061% | 0.9998 / 0.9999 / 0.9999 / 0.9999 |
| TripoSR | q4_k | 0.129 / 0.131 / 0.132 / 0.136% | 0.9998 / 0.9997 / 0.9997 / 0.9997 |

TripoSR voxel IoU is also backend-independent: 0.920/0.944/0.930/0.939 for
Q8_0 and 0.682/0.675/0.675/0.663 for Q4_K (CUDA/Vulkan/WGPU/CPU), so the
topology warning below holds on every backend. F16 is not in the table because
it is not block-quantized, but it keeps F@1% = 1.000 on all four backends with
CD ≤ 0.034% (Pi3), ≤ 0.042% (Pi3X) and ≤ 0.0014% (TripoSR); TripoSR voxel IoU
is 0.997–0.998 and normal consistency ≥ 0.9996.

Q8_0 is the only uniformly safe variant: F@1% ≥ 0.9999 for all three families
on all four backends. Pi3 Q6_K is excellent on CUDA (0.99907) but drops to
0.895–0.966 on CPU/Vulkan/WGPU, so treat it as a CUDA-only choice; Pi3
Q5_K/Q4_K are backend-sensitive, so keep them out of production.
Pi3X needs Q8_0 — Q6_K/Q5_K/Q4_K collapse despite the usual monotonic storage
error. The image `encoder.` is the sensitive module: keeping it F16 restores
Q6_K F@1% from 0.598 to 0.99996, but the resulting package is 1.40 GiB versus
1.37 GiB for all-Q8_0, so Q8_0 is strictly better. TripoSR is the most
quantization-tolerant family here: Q8_0 through Q4_K keep F@1% ≥ 0.9997 on
every backend. Q4_K still loses surface topology (voxel IoU 0.66–0.68 versus
0.92–0.94 at Q8_0), so prefer Q6_K/Q8_0 when topology fidelity matters.

TripoSG and VGGT: TripoSG stays within ~0.25% Chamfer even at Q4_K, so
Q4_K/Q5_K are practical there. VGGT is pose-sensitive: Q8_0 is near-lossless,
Q6_K is an acceptable fallback, Q5_K is marginal, and Q4_K is not viable
(pointmap and camera pose collapse), matching the published finding that naive
4-bit quantization destroys VGGT-style reconstruction. Use `--dump-npz` and the
script above on your own validation set before shipping a lower-bit variant.
RMBG-1.4 is a small matting network and is kept as a separate F16 GGUF, which
halves its footprint with no measurable mask drift.

TRELLIS.2-4B mesh drift (CUDA, RTX 3060 12 GiB; 512 pipeline, 12 steps per
sampler, seed 42, texture off, `mesh-input.png`; reference is the canonical
F32 package on the same backend). The F16 build is essentially exact; the
O-Voxel dual-grid output is discrete, so quantization flips some voxel
occupancies — voxel IoU measures that topology drift, not positional error:

| variant | CD rel | F@0.5% | F@1% | voxel IoU | normal | vertices (ref 914,470) | load | infer wall |
|---|---|---|---|---|---|---|---|---|
| f16 | 0.0027% | 0.9999 | 1.000 | 0.996 | 0.993 | 914,157 | 0.03 s | 142.8 s |
| q8_0 | 0.187% | 0.9139 | 0.9835 | 0.623 | 0.904 | 943,514 | 0.03 s | 98.7 s |
| q6_k | 0.199% | 0.9181 | 0.9813 | 0.609 | 0.905 | 952,678 | 0.03 s | 100.5 s |
| q5_k | 0.274% | 0.8839 | 0.9729 | 0.445 | 0.890 | 893,909 | 0.03 s | 96.3 s |
| q4_k | 0.311% | 0.8331 | 0.9625 | 0.400 | 0.885 | 887,078 | 0.03 s | 95.7 s |

Quantized DiT linears are also faster than F32: Q8_0 and below run the 12-step
512 pipeline in ~96–101 s versus 135 s for F32 and 143 s for F16, because
`QMatMul` kernels skip the F32 dequantize pass. (WGPU is the exception: its
packed q8 matmul shader is ~10x slower than the dense f32 GEMM, so Lux3D
dequantizes q8 weights once at load on WGPU; `LUX3D_WGPU_DEQUANT_WEIGHTS=0`
restores the packed path. See the backend coverage note below.) Q8_0/Q6_K are
the production choices (F@1% ≥ 0.98, normals ≥ 0.90); Q5_K/Q4_K are
preview-grade. The `1024_cascade` path was verified end-to-end with Q4_K
quantized weights (2-step run with texture: SS DiT, shape DiTs at 512/1024 and
texture DiT all load through `QMatMul` and decode).

Backend coverage with real weights (same input, seed 42): the geometry-only
512 run passes on CPU, CUDA, Vulkan and WGPU; the full `1024_cascade` run with
texture passes on CUDA in 438 s with a **7.8 GiB peak** on a 12 GiB RTX 3060.
F16 and Q8_0 GGUF variants run on all four testable backends. K-quants
(Q4_K/Q5_K/Q6_K) run on CPU, CUDA and Vulkan but fail on WGPU at this model
size because candle's WGPU Q8K activation-roundtrip dispatch exceeds the
65,535-workgroup limit for the 8,192-wide MLP; that limit applies to the packed
path (`LUX3D_WGPU_DEQUANT_WEIGHTS=0`), because by default WGPU dequantizes
quantized weights at load (see below). Q8_0 is the tested quantized choice on
WGPU until the candle kernel is fixed. Vulkan additionally needed a
Lux3D-side workaround for candle's unclamped `index_select` dispatch above
65,535 rows (the sparse VAE gathers reach ~500k rows at resolution 512); the
runtime splits those gathers into limit-sized chunks, after which Vulkan
matches CPU/CUDA geometry and is run-to-run deterministic.

Measured q8_0 512 geometry-only wall time (RTX 3060 12 GiB, release CLI, one
sampler step): CUDA 22.0 s, Vulkan 55.9 s, WGPU 97.6 s (down from 97 / 129 /
303 s). WGPU now routes ragged F16 sparse-VAE GEMMs through the register-tiled
kernel and keeps its GEMM accumulators in registers (Naga had lowered the
dynamically indexed arrays to local memory), and its 256 MiB large-allocation
flush threshold lets normal intermediates batch. The remaining WGPU cost is
~7 ms of WDDM latency per compute pass plus the exact-F32 DiT path (CUDA runs
q8 GEMMs on int8 kernels). The remaining Vulkan gap is candle-side
submission/drain latency: sparse VAE and DiT issue tens of thousands of small
dispatches and blocking drains account for ~32 s while GPU kernels sum to a few
seconds; larger batches/budgets and disabled drains all regressed. The Vulkan
SS 3D decoder itself is 0.6–1.8 s after the layout/kernel fixes.
