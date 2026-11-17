"""Generate official CuMesh remesh goldens for the Lux3D parity tests.

Uses the real `cumesh.remeshing.remesh_narrow_band_dc` (CUDA) with the call
semantics from `o_voxel/postprocess.py:165-180`:

    center = aabb.mean(dim=0)
    scale  = (resolution + 3 * band) / resolution * (aabb[1] - aabb[0]).max()
    remesh_narrow_band_dc(v, f, center, scale, resolution, band, project_back)

`aabb` is the tight AABB of the deterministic input mesh (the official pipeline
passes the voxel-volume AABB; synthetic remesh fixtures do not have one, so the
mesh AABB is used and recorded in each params.json). `project_back=0` keeps the
raw dual-contouring output that RM1 must reproduce.

Usage (Git Bash, oracle venv):
    PYTHONPATH=G:/jobs/candle-based-projects/candle_refs/.build-oracle/CuMesh-src \
      G:/jobs/candle-based-projects/candle_refs/.venv-oracle/Scripts/python.exe \
      scripts/generate_cumesh_remesh_goldens.py \
      --out crates/lux3d-core/tests/fixtures/remesh

Each `<shape>_res<resolution>/` directory stores `vertices.npy` (float32 Nx3),
`faces.npy` (int32 Mx3), the deterministic input as `input_vertices.npy` /
`input_faces.npy`, and `params.json`. The arrays are gzip-compressed straight
from the in-memory official tensors; `.npy` is the uncompressed container.
"""

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

import cumesh
from cumesh.remeshing import remesh_narrow_band_dc


def unit_cube():
    vertices = np.array(
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
        dtype=np.float32,
    )
    faces = np.array(
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
        dtype=np.int32,
    )
    return vertices, faces


def icosphere(radius=0.5, subdivisions=2):
    """Deterministic watertight icosphere with outward winding."""
    t = (1.0 + 5.0**0.5) / 2.0
    base = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]

    vertices = [np.asarray(p, dtype=np.float64) for p in base]
    midpoint_cache = {}

    def midpoint(a, b):
        key = (a, b) if a < b else (b, a)
        if key in midpoint_cache:
            return midpoint_cache[key]
        point = (vertices[a] + vertices[b]) * 0.5
        vertices.append(point)
        index = len(vertices) - 1
        midpoint_cache[key] = index
        return index

    for _ in range(subdivisions):
        next_faces = []
        for a, b, c in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            next_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = next_faces

    points = np.asarray(vertices, dtype=np.float64)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points *= radius
    return points.astype(np.float32), np.asarray(faces, dtype=np.int32)


def save_npy_gz(path, array):
    """Write a gzip-compressed .npy container (no pickled objects)."""
    import io

    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    payload = buffer.getvalue()
    with open(path, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as handle:
            handle.write(payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="crates/lux3d-core/tests/fixtures/remesh",
        help="fixture output directory (relative to the Lux3D repo root)",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("official remesh requires CUDA; torch.cuda.is_available() is False")

    out_root = Path(args.out)
    shapes = {"cube": unit_cube(), "sphere": icosphere()}
    resolutions = [32, 64]
    band = 1
    project_back = 0

    for shape_name, (in_vertices, in_faces) in shapes.items():
        vertices = torch.as_tensor(in_vertices, dtype=torch.float32, device=args.device)
        faces = torch.as_tensor(in_faces, dtype=torch.int32, device=args.device)
        bvh = cumesh.cuBVH(vertices, faces)
        aabb = torch.stack([vertices.min(dim=0).values, vertices.max(dim=0).values])
        center = aabb.mean(dim=0)
        max_extent = (aabb[1] - aabb[0]).max().item()

        for resolution in resolutions:
            scale = (resolution + 3 * band) / resolution * max_extent
            out_vertices, out_faces = remesh_narrow_band_dc(
                vertices,
                faces,
                center=center,
                scale=scale,
                resolution=resolution,
                band=band,
                project_back=project_back,
                bvh=bvh,
            )
            out_vertices = out_vertices.detach().cpu().numpy().astype(np.float32, copy=False)
            out_faces = out_faces.detach().cpu().numpy().astype(np.int32, copy=False)

            target = out_root / f"{shape_name}_res{resolution}"
            target.mkdir(parents=True, exist_ok=True)
            save_npy_gz(target / "vertices.npy.gz", out_vertices)
            save_npy_gz(target / "faces.npy.gz", out_faces)
            save_npy_gz(target / "input_vertices.npy.gz", in_vertices)
            save_npy_gz(target / "input_faces.npy.gz", in_faces)

            params = {
                "generator": "scripts/generate_cumesh_remesh_goldens.py",
                "official_function": "cumesh.remeshing.remesh_narrow_band_dc",
                "official_call_semantics": "o_voxel/postprocess.py:165-180",
                "shape": shape_name,
                "resolution": resolution,
                "band": band,
                "project_back": project_back,
                "center": center.detach().cpu().tolist(),
                "scale": scale,
                "max_extent": max_extent,
                "aabb": aabb.detach().cpu().tolist(),
                "input_vertices": int(in_vertices.shape[0]),
                "input_faces": int(in_faces.shape[0]),
                "output_vertices": int(out_vertices.shape[0]),
                "output_faces": int(out_faces.shape[0]),
                "vertex_dtype": "float32",
                "face_dtype": "int32",
                "storage": "gzip(npy)",
                "torch_version": torch.__version__,
                "cumesh_module": str(Path(cumesh.__file__).resolve()),
                "sha256": {
                    "vertices": hashlib.sha256(out_vertices.tobytes()).hexdigest(),
                    "faces": hashlib.sha256(out_faces.tobytes()).hexdigest(),
                },
            }
            (target / "params.json").write_text(json.dumps(params, indent=1), encoding="utf-8")
            print(
                f"{shape_name} res {resolution}: "
                f"V={out_vertices.shape[0]} F={out_faces.shape[0]} "
                f"scale={scale:.9f}"
            )

    total = sum(path.stat().st_size for path in out_root.rglob("*") if path.is_file())
    print(f"fixture bytes: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
