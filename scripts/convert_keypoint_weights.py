#!/usr/bin/env python3
"""Convert the official ALIKED / SuperPoint `.pth` checkpoints to safetensors.

The LightGlue reference loads ordinary `state_dict` files with unchanged
tensor names. Lux3D loads the same names from safetensors, so this is a
lossless container conversion. BatchNorm `num_batches_tracked` counters are
dropped: candle's BatchNorm loader does not consume them.

Usage:
  python scripts/convert_keypoint_weights.py \
      --input models/aliked-superpoint/aliked-n16.pth \
      --output models/aliked-superpoint/aliked-n16.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert(input_path: Path, output_path: Path) -> None:
    state = torch.load(input_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise SystemExit(f"{input_path}: expected a state_dict, got {type(state)!r}")
    tensors = {
        name: tensor.contiguous()
        for name, tensor in state.items()
        if not name.endswith("num_batches_tracked")
    }
    if not tensors:
        raise SystemExit(f"{input_path}: no tensors found")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_path))
    print(f"wrote {output_path} ({len(tensors)} tensors)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    convert(args.input, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
