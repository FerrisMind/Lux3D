#!/usr/bin/env python3
"""Generate the official LightGlue ALIKED-n16 CPU golden fixture.

The fixture is produced by the reference implementation in
``candle_refs/LightGlue`` (``lightglue/aliked.py``), not by Lux3D:

    ALIKED(model_name="aliked-n16", max_num_keypoints=4096,
           detection_threshold=0.005).eval().extract(image, resize=None)

The checkpoint download is short-circuited to the local converted source
``models/aliked-superpoint/aliked-n16.pth`` so the script works offline.

The input is a deterministic uint8 synthetic RGB image (saved in the fixture
and re-divided by 255 in the Rust test), which yields a few hundred keypoints
and stays well below ``max_num_keypoints``, so the golden order is unambiguous.

Usage (from the repo root, no PYTHONPATH needed):

    python -I scripts/gen_aliked_golden.py

LightGlue is a local reference checkout; ``-I`` keeps the interpreter from
importing anything from the working directory while the script inserts the
reference path explicitly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIGHTGLUE = ROOT.parent / "candle_refs/LightGlue"
DEFAULT_WEIGHTS = ROOT / "models/aliked-superpoint/aliked-n16.pth"
DEFAULT_OUT = ROOT / "crates/lux3d-core/tests/fixtures/keypoints/aliked_n16_256.npz"


def synthetic_image(size: int = 256) -> np.ndarray:
    """Deterministic textured RGB image with hard edges for the score head."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = 0.5 + 0.2 * np.sin(xx / 7.0) * np.cos(yy / 9.0)
    image = np.stack(
        [
            base,
            0.5 + 0.2 * np.cos((xx + yy) / 13.0),
            0.5 + 0.2 * np.sin((xx - yy) / 11.0),
        ],
        axis=0,
    )
    half = size // 2
    # Checkerboard in the top-left quadrant.
    checker = (((xx // 8) + (yy // 8)) % 2).astype(np.float32)
    image[:, :, :half] = image[:, :, :half] * 0.3 + checker[None, :, :half] * 0.7
    # Dot grid in the bottom-right quadrant.
    dots = (
        (((xx % 16) - 8) ** 2 + ((yy % 16) - 8) ** 2) < 9
    ).astype(np.float32)
    image[:, half:, half:] = image[:, half:, half:] * 0.4 + dots[None, half:, half:] * 0.6
    # Two saturated rectangles with strong corners.
    image[:, 40:80, 140:170] = np.array([0.95, 0.15, 0.25], dtype=np.float32)[:, None, None]
    image[:, 160:200, 20:60] = np.array([0.15, 0.95, 0.35], dtype=np.float32)[:, None, None]
    return np.clip(image, 0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lightglue", type=Path, default=DEFAULT_LIGHTGLUE)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    if not args.lightglue.is_dir():
        raise SystemExit(f"LightGlue checkout not found: {args.lightglue}")
    if not args.weights.is_file():
        raise SystemExit(f"ALIKED checkpoint not found: {args.weights}")

    import torch

    sys.path.insert(0, str(args.lightglue))
    from lightglue import ALIKED  # noqa: E402

    original_loader = torch.hub.load_state_dict_from_url
    torch.hub.load_state_dict_from_url = lambda *a, **k: torch.load(  # noqa: ARG005
        args.weights, map_location="cpu", weights_only=True
    )
    try:
        model = ALIKED(
            model_name="aliked-n16",
            max_num_keypoints=4096,
            detection_threshold=0.005,
        ).eval()
    finally:
        torch.hub.load_state_dict_from_url = original_loader

    image = synthetic_image(args.size)
    image_u8 = (image * 255.0).round().astype(np.uint8)
    image_t = torch.from_numpy(image_u8).float() / 255.0
    with torch.no_grad():
        out = model.extract(image_t, resize=None)

    keypoints = out["keypoints"][0].numpy().astype(np.float32)
    scores = out["keypoint_scores"][0].numpy().astype(np.float32)
    descriptors = out["descriptors"][0].numpy().astype(np.float32)
    if keypoints.shape[0] >= 4096:
        raise SystemExit(
            "golden image hit max_num_keypoints; selection order would be ambiguous"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        image=image_u8,
        keypoints=keypoints,
        keypoint_scores=scores,
        descriptors=descriptors,
    )
    print(
        f"wrote {args.output} ({args.output.stat().st_size / 1024:.1f} KiB): "
        f"{keypoints.shape[0]} keypoints, descriptor dim {descriptors.shape[1]}, "
        f"score range {scores.min():.6f}..{scores.max():.6f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
