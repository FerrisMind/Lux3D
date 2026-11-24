<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-5B7CFA" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-232323" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-232323" alt="Português"></a>
</p>

---

<p align="center">
  <b>Rust workspace for clean-room 3D inference, contract inspection, and weight canonicalization.</b><br>
  CUDA-first runtime for Pi3, Pi3X, TripoSR, VGGT, TripoSG, and TRELLIS.2.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-Apache--2.0-2ea44f" alt="Apache 2.0">
  <img src="https://img.shields.io/badge/Rust-1.85+-93450a?logo=rust" alt="Rust 1.85+">
  <img src="https://img.shields.io/badge/Runtime-CUDA%20first-76B900" alt="CUDA first runtime">
</p>

<h1 align="center">Lux3D / candle-3d</h1>

## Demo

https://github.com/user-attachments/assets/0a9f766f-68d3-482a-9c94-c55d0c355808

## What is this?

Lux3D / candle-3d is a Rust 2024 workspace that runs 3D reconstruction and
generation models on [Candle](https://github.com/FerrisMind/candle) with a
CUDA-first runtime and Vulkan/WGPU fallbacks. It provides:

- a single CLI for contract inspection, weight normalization, GGUF
  quantization, inference, and O-Voxel conversion;
- reproducible canonical weight packages with manifests and checksums;
- clean-room ports of the reference preprocessing, sampling, and mesh export
  paths;
- an OpenAI-style asynchronous HTTP API for point-cloud and mesh generation.

Every model family keeps its upstream license. Review the per-family terms in
[docs/models.md](docs/models.md) before redistributing weights.

## Supported models

| Family | Task | Output | Upstream |
|--------|------|--------|----------|
| [`pi3`](docs/models.md#pi3-and-pi3x) | Multi-view point cloud | PLY / XYZ | [yyfz/Pi3](https://github.com/yyfz/Pi3) |
| [`pi3x`](docs/models.md#pi3-and-pi3x) | Multi-view point cloud + VO | PLY / XYZ | [yyfz/Pi3](https://github.com/yyfz/Pi3) |
| [`triposr`](docs/models.md#triposr) | Single-image mesh | GLB / OBJ / PLY / STL | [VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR) |
| [`vggt`](docs/vggt.md) | Multi-view geometry + COLMAP | PLY / XYZ / COLMAP / GLB | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) |
| [`triposg`](docs/models.md#triposg) | Image-to-mesh, scribble, surface encode | GLB / OBJ / PLY / STL | [VAST-AI-Research/TripoSG](https://github.com/VAST-AI-Research/TripoSG) |
| [`trellis2`](docs/trellis2.md) | Image-to-3D, mesh texturing, O-Voxel | GLB / OBJ / PLY / STL | [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) |

Canonical packages are published under `oxide-lab/*` on Hugging Face and
auto-download into the user cache. `oxide-lab/TRELLIS.2-4B` is reserved but not
yet published, so TRELLIS.2 needs an explicit `--model-path`.

## Quick start

### Prerequisites

- Rust 1.85+ with `edition = "2024"` support
- Python 3.x for normalization and evaluation scripts
- An NVIDIA GPU for verified CUDA inference; Vulkan and WGPU are supported but
  experimental (see [experimental backends](docs/inference.md#experimental-backends-vulkan-and-wgpu))

### Build

```powershell
cargo build --release -p lux3d-cli --features cuda
```

Default builds are CPU-only. GPU backends are Cargo features: `cuda`, `vulkan`,
`wgpu`, `metal`.

### Inspect and prepare weights

```powershell
# Show the contract expected by a family
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> triposg

# Normalize a raw checkpoint into the canonical package
cargo run -p lux3d-cli -- weights normalize `
  --repo-root <runtime-root> `
  --raw-model-dir <raw-triposg-dir> `
  --output-dir <canonical-triposg-dir> triposg
```

Full weight guides: [canonical packages and quantization](docs/weights.md).

### Generate

```powershell
# Image -> mesh (auto-downloads the canonical TripoSG package)
cargo run -p lux3d-cli --features cuda -- run triposg `
  --source .\input.png --output .\out.glb

# Image -> textured GLB (explicit TRELLIS.2 package; RMBG-2.0 is separate)
cargo run -p lux3d-cli --features cuda -- run trellis2 `
  --model-path <canonical-trellis2-dir> `
  --rmbg-model-path <canonical-rmbg2-dir> `
  --source .\input.png --output .\out.glb
```

All family commands, flags, and output-format rules:
[docs/inference.md](docs/inference.md).

### Serve

```powershell
cargo run -p lux3d-server --features cuda -- --host 127.0.0.1 --port 8080
curl http://127.0.0.1:8080/health
```

Endpoints, request fields, and SSRF policy: [docs/server.md](docs/server.md).

## Documentation

| Document | Contents |
|----------|----------|
| [docs/models.md](docs/models.md) | Supported families, pipeline summaries, upstream attribution, licensing |
| [docs/inference.md](docs/inference.md) | `run` commands for every family, output formats, experimental backends, O-Voxel commands |
| [docs/vggt.md](docs/vggt.md) | VGGT preprocessing, COLMAP export, tracking, GLB scene, tensor dumps |
| [docs/trellis2.md](docs/trellis2.md) | TRELLIS.2 pipeline variants, sampling, post-processing, texture baking, mesh texturing |
| [docs/weights.md](docs/weights.md) | Canonical package layout, normalization, GGUF quantization, measured fidelity |
| [docs/server.md](docs/server.md) | HTTP API, server flags, job workflow, embedding the router |
| [docs/performance.md](docs/performance.md) | Backend benchmarks, warm-loop results, memory behavior |
| [FEATURES.md](FEATURES.md) | Feature inventory and user stories |
| [E2E.md](E2E.md) | End-to-end scenario matrix and how to run it |
| [E2E-RESULTS.md](E2E-RESULTS.md) | Latest real-weight e2e results |
| [PERF-NOTES.md](PERF-NOTES.md) | Performance investigation log and remaining levers |

## Performance

Measured inference time (`[stage] infer`: preprocessing + neural pass, weights
excluded) on an NVIDIA GeForce RTX 3060 12 GiB, Windows (WDDM):

| Model | CUDA | Vulkan | WGPU | Vulkan/CUDA | WGPU/CUDA |
|---|---:|---:|---:|---:|---:|
| pi3 (5 frames, 518×518) | **4.82 s** | 7.83 s | 13.96 s | 1.62× | 2.90× |
| pi3x (6 frames, 518×518) | **7.00 s** | 11.10 s | 31.84 s | 1.59× | 4.55× |
| triposr (single image) | **1.23 s** | 3.36 s | 4.89 s | 2.73× | 3.98× |
| vggt (3 frames, 518×518) | **6.76 s** | 10.96 s | 19.34 s | 1.62× | 2.86× |

TripoSG preview (`--steps 2 --num-tokens 1024 --flash-octree-depth 5`): infer
6.84 s CUDA / 8.56 s Vulkan / 13.13 s WGPU at a ~8.3–8.7 GiB peak. TRELLIS.2
`1024_cascade` with texture runs in 438 s on CUDA with a 7.8 GiB peak.

Full tables, VRAM numbers, warm-loop results, and known gaps:
[docs/performance.md](docs/performance.md).

## Repository layout

| Path | Purpose |
|------|---------|
| `crates/lux3d-core` | Core runtime, contracts, geometry/export code, and weight validation |
| `crates/lux3d-cli` | CLI front-end for inspection, normalization, and inference runs |
| `crates/lux3d-server-core` | Embeddable Axum router and HTTP API for external applications |
| `crates/lux3d-server` | Standalone HTTP server binary |
| `docs/` | User guides: models, inference, VGGT, TRELLIS.2, weights, server, performance |
| `scripts/` | Quantization evaluation, geometry comparison, keypoint conversion |
| [`tools/python_baseline`](tools/python_baseline/README.md) | Python parity tooling and canonical-weight normalization scripts |
| `.github/assets` | Project documentation assets |

## System requirements

- Rust 1.85 or newer; Cargo with `edition = "2024"` support.
- Python 3.x for baseline tooling and normalization.
- CUDA-capable NVIDIA hardware for verified runtime inference. The 12 GiB
  RTX 3060 is the reference test card.
- Vulkan and WGPU backends via the [FerrisMind/candle](https://github.com/FerrisMind/candle)
  fork; verified for correctness and memory stability but slower than CUDA.
- Metal is build-checked in CI; no model inference parity is validated.
- 16 GiB+ VRAM is recommended for the full TripoSG 50-step reference run.
  TRELLIS.2 peaks at ~7.8 GiB on `1024_cascade` with texture stages resident.

## License

The code and documentation in this repository are Apache-2.0. Upstream model
assets and canonicalized artifacts keep their original licenses and usage
restrictions; review [docs/models.md](docs/models.md#licensing) before
redistribution.

## Acknowledgements

Built on [FerrisMind/candle](https://github.com/FerrisMind/candle). Model
implementations follow [Pi3](https://github.com/yyfz/Pi3),
[TripoSR](https://github.com/VAST-AI-Research/TripoSR),
[VGGT](https://github.com/facebookresearch/vggt),
[TripoSG](https://github.com/VAST-AI-Research/TripoSG), and
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2).
