"""Generate an official `o_voxel.postprocess.to_glb` fixture.

Runs the real TRELLIS.2 `to_glb` (CuMesh + nvdiffrast + FlexGEMM + trimesh) on
a deterministic unit cube and an 8^3 dense attribute volume, then exports the
returned textured mesh as GLB.

The fixture exercises `postprocess.py:278-336` end to end: PBR channel packing
(`base_color` + `alpha`, `roughness`/`metallic` into G/B), TELEA inpainting,
UV V flip and the Y/Z axis swap.

Usage (Git Bash, oracle venv; all built extension sources on PYTHONPATH):
    PYTHONPATH=".../o-voxel-src;.../CuMesh-src;.../flexgemm-src" \
      G:/jobs/candle-based-projects/candle_refs/.venv-oracle/Scripts/python.exe \
      scripts/generate_ovox_to_glb_fixture.py \
      --out crates/lux3d-core/tests/fixtures/ovox

The output is `official_to_glb.glb` plus `to_glb_params.json`.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

from o_voxel.postprocess import to_glb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="crates/lux3d-core/tests/fixtures/ovox",
        help="fixture output directory (relative to the Lux3D repo root)",
    )
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--texture-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("official to_glb requires CUDA; torch.cuda.is_available() is False")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    vertices = torch.tensor(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=torch.int32,
    )

    grid = args.grid_size
    coords = (
        torch.stack(
            torch.meshgrid(
                torch.arange(grid),
                torch.arange(grid),
                torch.arange(grid),
                indexing="ij",
            ),
            dim=-1,
        )
        .int()
        .reshape(-1, 3)
        .to(args.device)
    )
    channels = 6
    generator = torch.Generator(device="cpu").manual_seed(20261118)
    attr_volume = torch.rand((coords.shape[0], channels), generator=generator, dtype=torch.float32).to(
        args.device
    )
    aabb = torch.tensor(
        [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32, device=args.device
    )
    attr_layout = {
        "base_color": slice(0, 3),
        "metallic": slice(3, 4),
        "roughness": slice(4, 5),
        "alpha": slice(5, 6),
    }

    mesh = to_glb(
        vertices,
        faces,
        attr_volume,
        coords,
        attr_layout,
        aabb,
        grid_size=[grid, grid, grid],
        decimation_target=1000,
        texture_size=args.texture_size,
        remesh=False,
    )

    glb_path = out_dir / "official_to_glb.glb"
    glb_bytes = mesh.export(file_type="glb")
    glb_path.write_bytes(glb_bytes)

    reloaded = trimesh.load(glb_path, force="mesh")
    material = reloaded.visual.material
    base_color = material.baseColorTexture
    metallic_roughness = material.metallicRoughnessTexture
    params = {
        "generator": "scripts/generate_ovox_to_glb_fixture.py",
        "official_function": "o_voxel.postprocess.to_glb",
        "official_lines": "postprocess.py:278-336 (material assembly) and 100-336",
        "remesh": False,
        "decimation_target": 1000,
        "texture_size": args.texture_size,
        "grid_size": [grid, grid, grid],
        "attr_schema": [["base_color", 3], ["metallic", 1], ["roughness", 1], ["alpha", 1]],
        "attr_seed": 20261118,
        "aabb": aabb.detach().cpu().tolist(),
        "attr_volume_sha256": hashlib.sha256(
            attr_volume.detach().cpu().numpy().astype(np.float32).tobytes()
        ).hexdigest(),
        "output_vertices": int(len(reloaded.vertices)),
        "output_faces": int(len(reloaded.faces)),
        "base_color_texture_size": list(base_color.size),
        "metallic_roughness_texture_size": list(metallic_roughness.size),
        "base_color_mode": base_color.mode,
        "metallic_roughness_mode": metallic_roughness.mode,
        "double_sided": bool(material.doubleSided),
        "alpha_mode": material.alphaMode,
        "glb_bytes": len(glb_bytes),
        "glb_sha256": hashlib.sha256(glb_bytes).hexdigest(),
        "torch_version": torch.__version__,
    }
    params_path = out_dir / "to_glb_params.json"
    params_path.write_text(json.dumps(params, indent=1), encoding="utf-8")
    print(f"{glb_path.name}: {len(glb_bytes)} bytes")
    print(json.dumps(params, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
