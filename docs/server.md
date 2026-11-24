# HTTP server

The `lux3d-server` binary and the embeddable `lux3d-server-core` router expose an
OpenAI-style asynchronous job API. This document lists the flags, routes, request
fields, SSRF policy, and the embedding example.

## Run the HTTP server

```powershell
cargo run -p lux3d-server --features cuda -- --host 127.0.0.1 --port 8080
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
```

TRELLIS.2 RGB requests need the separate RMBG-2.0 package. Pass it with
`--trellis2-rmbg-model-path <dir>` / `LUX3D_TRELLIS2_RMBG_MODEL_PATH`, or place
`rmbg-2.0` (or raw `RMBG-2.0`) next to the explicit `--model-path`. TripoSG's
`--rmbg-model-path` / `LUX3D_RMBG_MODEL_PATH` remains the separate RMBG-1.4
package; the two families deliberately do not share weights.

### Server flags

| Flag | Default / environment |
|------|-----------------------|
| `--host` / `--port` | `127.0.0.1:8080` |
| `--model-path` | `LUX3D_MODEL_PATH` |
| `--cache-dir` | `LUX3D_CACHE_DIR` |
| `--rmbg-model-path` | `LUX3D_RMBG_MODEL_PATH` |
| `--trellis2-rmbg-model-path` | `LUX3D_TRELLIS2_RMBG_MODEL_PATH` |
| `--cuda-device` | `0` |
| `--upload-dir` | unset |
| `--allow-private-urls` | `LUX3D_ALLOW_PRIVATE_URLS` (off) |
| `--cors-origin` | repeatable |
| `--job-ttl-secs` | `86400` |
| `--cleanup-interval-secs` | `300` |
| `--access-log` | `true` |
| `--access-log-health` | `false` |
| `--access-log-format` | `text` or `json` |
| `--request-id-header` | `true` |
| `--metrics` | `true` |

The binary always builds with the platform-default device (CUDA off macOS, Metal
on macOS metal builds); Vulkan/WGPU (and an explicit Metal choice) are reachable
only through the embeddable router builder's `with_device_backend`.

### URL fetch policy

`source_url` / `mesh_url` fetches are SSRF-hardened:

- Only `http`/`https`, with no embedded credentials.
- Loopback, RFC1918, link-local, ULA, unspecified, multicast, broadcast, CGNAT,
  and IPv4-mapped IPv6 addresses are rejected.
- DNS answers are validated as returned to the connector (no rebinding window).
- Redirects are revalidated per hop (max 5, no `https`→`http` downgrade).
- Fetch limits: 10 s connect / 30 s total timeout, 64 MiB streaming cap.
- Media fetches ignore proxy environment variables (a proxy would bypass the
  address policy) and reject responses that are still redirects after the
  policy.
- `--allow-private-urls` (or `LUX3D_ALLOW_PRIVATE_URLS=1`) disables only the
  non-public address filter for LAN deployments; every other rule still applies.
- `source_file_id` / `source_base64` inputs are not affected.

## Prebuilt releases

Prebuilt releases (`.github/workflows/release.yml`): Windows + Ubuntu × cpu/cuda/vulkan.
Names follow llama.cpp (`lux3d-<tag>-bin-…`). CUDA builds do not bundle NVIDIA libraries:
download the matching `cudart-lux3d-<tag>-…` archive and extract it next to the binaries
(same directory as the `.exe`; `$ORIGIN` on Linux). The host only needs the NVIDIA driver.

## Generation endpoints

Generation endpoints accept `multipart/form-data` (or JSON) and return `202 Accepted` with an async job object:

| Endpoint | Models | Output |
|----------|--------|--------|
| `POST /v1/point-clouds/generations` | `pi3`, `pi3x`, `vggt` | PLY |
| `POST /v1/meshes/generations` | `triposr`, `triposg`, `trellis2` | GLB |
| `POST /v1/meshes/texturings` | `trellis2` | GLB (embedded textures) |

### Mesh texturing

`POST /v1/meshes/texturings` textures an existing mesh from one condition image
(TRELLIS.2 only).

**Multipart fields:** `source` (condition image), `mesh`
(`.glb`/`.obj`/`.ply`/`.stl`), optional `model` (default `trellis2`), and
`options`.

**JSON fields:** `source_url`/`source_file_id`/`source_base64`/`source_filename`,
`mesh_url`/`mesh_file_id`/`mesh_base64`/`mesh_filename`, optional `model`,
`options`, `webhook_url`. At least one `source*` variant and one `mesh*` variant
is required.

**`options`:**

- `output_format` — must be `glb` (default `glb`)
- `seed` (42)
- `texture_resolution` (512/1024/1536; default 1024)
- `texture_size` (2048)
- `tex_steps`
- `tex_guidance_strength`
- `tex_guidance_rescale`
- `tex_guidance_interval` (`[lo, hi]`)
- `tex_rescale_t`
- `preprocess_image`, `preprocess_mesh` (both default true)

The job writes exactly one `trellis2-textured.glb` (baked PBR atlas embedded, no
PBR sidecar).

**Request budgets** are enforced before job insertion (HTTP 400 when exceeded):

- `num_samples <= 64` (TRELLIS.2)
- `timesteps` length `<= 1000` (TripoSG)
- `steps`/`ss_steps`/`shape_steps`/`tex_steps` `<= 1000` per stage

Mesh payloads must carry a supported extension (`.glb`/`.obj`/`.ply`/`.stl`),
and texturing JSON validates model/options before resolving media.

### Other endpoints

| Route | Purpose |
|-------|---------|
| `GET /` , `GET /health` | Liveness |
| `GET /metrics` | Prometheus; enabled by `--metrics`, returns 503 when disabled |
| `GET`/`POST /v1/files`, `GET`/`DELETE /v1/files/{id}`, `GET /v1/files/{id}/content` | Upload once and pass `source_file_id`/`mesh_file_id` |
| `GET /v1/generations`, `GET`/`DELETE /v1/generations/{id}`, `POST /v1/generations/{id}/cancel` | Job listing, status, cancel |
| `GET /v1/generations/{id}/stream` | SSE status updates |
| `POST /v1/models/reload`, `POST /v1/models/unload`, `POST /v1/models/status` | Model lifecycle |
| `GET /v1/system/info`, `POST /v1/system/doctor` | Runtime diagnostics |

Swagger UI is served at `/docs` and the schema at `/api-doc/openapi.json` (the
`swagger-ui` feature is on by default).

## Common fields

Common fields:

- `model` — required for point clouds (`pi3`, `pi3x`, or `vggt`), optional for meshes (`triposr` default; set `triposg` for TripoSG or `trellis2` for TRELLIS.2)
- `source` — required upload (`.zip` frame archive, a video file (`.mp4`/`.mov`/`.avi`/`.mkv`), an image (`.png`/`.jpg`/`.jpeg`/`.webp`), or an already-extracted frame directory)
- `conditions` — optional Pi3X conditions file
- `options` — optional JSON string (or JSON object in a JSON request) with family-specific parameters.
  - TripoSG: `steps`, `guidance_scale`, `seed`, `num_tokens`, `use_flash_decoder`, `flash_octree_depth`, `dense_octree_depth`, `hierarchical_octree_depth`, `query_chunk_size`, `timesteps`, `bounds` (`[min_x,min_y,min_z,max_x,max_y,max_z]`), `faces`, `output_format` (`glb` default, `obj`, `ply`, `stl`). The CLI-only TripoSG surfaces (`--surface-input`, `--mode scribble`, repeatable `--source`, `--num-shapes`) have no request fields.
  - TRELLIS.2: `steps` (all three Euler samplers), `seed`, `pipeline_type` (`512`/`1024`/`1024_cascade`/`1536_cascade`), `max_num_tokens`, `texture`, `output_format` (`glb` default, `obj`, `ply`), `num_samples` (values > 1 use the same joint-draw structure as the CLI and write `_0`, `_1`, ... outputs; `/content` serves sample 0), `texture_size`, the postprocess fields (`postprocess`, `decimation_target`, `fill_hole_perimeter`, `min_component_area`, `remesh`, `remesh_band`, `remesh_project`), and the per-stage overrides `ss_steps`/`ss_guidance_strength`/`ss_guidance_rescale`/`ss_guidance_interval`/`ss_rescale_t` and the matching `shape_*`/`tex_*` fields. `--return-latent` stays CLI-only; existing-mesh texturing uses `POST /v1/meshes/texturings`.
  - Point clouds: `output_format` (`ply` default, `xyz`), plus `interval`, `vo`, `chunk_size`, `overlap`, `conf_threshold`, `inject_condition`. The VGGT `--preprocess-mode`/`--batch-size`/`--colmap-dir`/`--depth-points`/`--track-queries`/`--track-auto`/`--glb` surface is CLI-only.

## Workflow

Workflow:

```powershell
# 1. Create a mesh generation job
curl -X POST http://127.0.0.1:8080/v1/meshes/generations `
  -F "model=triposr" `
  -F "source=@input.png" `
  -F "options={\"mc_resolution\":256,\"mc_threshold\":25.0}"

# TripoSG job (completed downloads as triposg.glb)
curl -X POST http://127.0.0.1:8080/v1/meshes/generations `
  -F "model=triposg" `
  -F "source=@input.png" `
  -F "options={\"steps\":50,\"use_flash_decoder\":true,\"flash_octree_depth\":9}"

# TRELLIS.2 job (completed downloads as trellis2.glb)
curl -X POST http://127.0.0.1:8080/v1/meshes/generations `
  -F "model=trellis2" `
  -F "source=@input.png" `
  -F "options={\"steps\":12,\"seed\":42,\"pipeline_type\":\"1024_cascade\",\"texture\":true}"

# Texture an existing mesh (completed downloads as trellis2-textured.glb):
# source = condition image, mesh = the .glb/.obj/.ply/.stl to texture
curl -X POST http://127.0.0.1:8080/v1/meshes/texturings `
  -F "model=trellis2" `
  -F "source=@condition.png" `
  -F "mesh=@input.glb" `
  -F "options={\"texture_resolution\":1024,\"texture_size\":2048,\"tex_steps\":12}"

# 2. Poll job status
curl http://127.0.0.1:8080/v1/generations/gen_...

# 3. Download completed asset
curl http://127.0.0.1:8080/v1/generations/gen_.../content --output mesh.glb
```

## Embedding the router

Embed the router in another Axum app:

```rust
use lux3d_server_core::{
    lux3d_for_server_builder::Lux3dForServerBuilder,
    lux3d_server_router_builder::Lux3dServerRouterBuilder,
};

let lux3d = Lux3dForServerBuilder::new().build()?;
let routes = Lux3dServerRouterBuilder::new()
    .with_lux3d(lux3d)
    .build()
    .await?;
let app = axum::Router::new().nest("/api/lux3d", routes);
```
