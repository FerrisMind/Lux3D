"""Generate official `.vxz` fixtures with the real TRELLIS.2 `o_voxel` writer.

This script must run against a locally compiled `o_voxel._C` extension (see
`.swarm/results/OR1.md` for the Windows build recipe). It imports the official
`o_voxel.io.write_vxz` / `read_vxz` unchanged and records every input voxel so
the Rust reader test can compare per-coordinate attributes.

Usage (Git Bash, oracle venv):
    PYTHONPATH=G:/jobs/candle-based-projects/candle_refs/.build-oracle/o-voxel-src \
      G:/jobs/candle-based-projects/candle_refs/.venv-oracle/Scripts/python.exe \
      scripts/generate_ovox_official_fixtures.py \
      --out crates/lux3d-core/tests/fixtures/ovox

The fixture grid is deterministic: a 7x7x7 block minus its 8 corners (335
voxels), coordinates -3..=3, chunk_size 8 so negative chunk indices and
multiple chunks are exercised. One file per official compression codec, all
with the official default compression level and `as_is` interleave.
"""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

import o_voxel
from o_voxel.io import read_vxz, write_vxz

COMPRESSIONS = ["none", "deflate", "lzma", "zstd"]


def build_grid():
    coords = []
    for x in range(-3, 4):
        for y in range(-3, 4):
            for z in range(-3, 4):
                # Drop the 8 cube corners so the surface is not a perfect
                # rectangular block; keeps every value deterministic.
                if abs(x) == 3 and abs(y) == 3 and abs(z) == 3:
                    continue
                coords.append([x, y, z])
    coords = torch.tensor(coords, dtype=torch.int32)

    base_color = torch.zeros((coords.shape[0], 3), dtype=torch.uint8)
    metallic = torch.zeros((coords.shape[0], 1), dtype=torch.uint8)
    roughness = torch.zeros((coords.shape[0], 1), dtype=torch.uint8)
    alpha = torch.zeros((coords.shape[0], 1), dtype=torch.uint8)
    for row, (x, y, z) in enumerate(coords.tolist()):
        seed = x * 31 + y * 17 + z * 7
        base_color[row, 0] = (seed * 3 + 7) % 256
        base_color[row, 1] = (seed * 5 + 11) % 256
        base_color[row, 2] = (seed * 7 + 13) % 256
        metallic[row, 0] = (seed * 11 + 19) % 256
        roughness[row, 0] = (seed * 13 + 23) % 256
        alpha[row, 0] = 255 - ((seed * 17 + 29) % 64)
    attrs = {
        "base_color": base_color,
        "metallic": metallic,
        "roughness": roughness,
        "alpha": alpha,
    }
    return coords, attrs


def canonical(grid_coords, attrs):
    """Map coordinate -> attribute rows for order-independent comparison."""
    rows = {}
    for row, coord in enumerate(grid_coords.tolist()):
        rows[tuple(coord)] = {
            name: values[row].tolist() for name, values in attrs.items()
        }
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="crates/lux3d-core/tests/fixtures/ovox",
        help="fixture output directory (relative to the Lux3D repo root)",
    )
    parser.add_argument("--chunk-size", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    coords, attrs = build_grid()
    expected = canonical(coords, attrs)

    record = {
        "generator": "scripts/generate_ovox_official_fixtures.py",
        "official_writer": "o_voxel.io.write_vxz",
        "o_voxel_module": str(Path(o_voxel.__file__).resolve()),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "chunk_size": args.chunk_size,
        "compression_level": None,
        "attr_interleave": "as_is",
        "filter": "none",
        "attr_schema": [[name, int(values.shape[1])] for name, values in attrs.items()],
        "coord": coords.tolist(),
        "attrs": {name: values.tolist() for name, values in attrs.items()},
        "files": {},
    }

    for compression in COMPRESSIONS:
        path = out_dir / f"official_{compression}_as_is.vxz"
        write_vxz(
            str(path),
            coords,
            attrs,
            chunk_size=args.chunk_size,
            filter="none",
            compression=compression,
            compression_level=None,
            attr_interleave="as_is",
        )
        read_coords, read_attrs = read_vxz(str(path))
        got = canonical(read_coords, read_attrs)
        if got != expected:
            raise SystemExit(f"{path.name}: official round-trip mismatch")
        data = path.read_bytes()
        record["files"][path.name] = {
            "compression": compression,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "num_voxel": int(read_coords.shape[0]),
        }
        print(f"{path.name}: {len(data)} bytes, {read_coords.shape[0]} voxels")

    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"{metadata_path.name}: {metadata_path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
