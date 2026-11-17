#!/usr/bin/env python3
"""Generate the official LightGlue SuperPoint golden fixture.

The Rust parity test in `crates/lux3d-core/src/runtime/superpoint.rs` compares
against the reference `lightglue.superpoint.SuperPoint` run on CPU. This script
reproduces the fixture:

  PYTHONPATH=G:/jobs/candle-based-projects/candle_refs/LightGlue \
  python scripts/gen_superpoint_golden.py \
      --weights models/aliked-superpoint/superpoint_v1.pth \
      --output crates/lux3d-core/tests/fixtures/keypoints/superpoint_golden_256.npz

The input is a deterministic 256x256 synthetic RGB image (fixed RNG seed plus
fixed sinusoid patterns), stored in the archive as uint8 CHW so the Rust test
does not have to reproduce the generator. The archive stores:

* `image` (uint8, 3xHxW),
* `keypoints` / `scores` / `descriptors` for `extract(image, resize=None)`,
* `keypoints_resize512` / `scores_resize512` for `extract(image, resize=512)`.

All keypoints are returned in the original input coordinate system by
`Extractor.extract` (origin="center"), exactly what the Rust port emits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def synthetic_image(height: int, width: int, seed: int = 1234) -> np.ndarray:
    """Deterministic textured RGB image, uint8 HxWx3."""
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    base = rng.rand(16, 16, 3) * 255.0
    upsampled = np.repeat(np.repeat(base, height // 16, axis=0), width // 16, axis=1)
    wave = np.stack(
        [
            128.0 + 100.0 * np.sin(x / 7.0 + y / 11.0),
            128.0 + 100.0 * np.sin((x + 2.0 * y) / 5.0),
            128.0 + 100.0 * np.cos((2.0 * x - y) / 9.0),
        ],
        axis=-1,
    )
    image = 0.5 * upsampled + 0.5 * wave
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def build_extractor(weights: Path):
    # LightGlue's constructor fetches `superpoint_v1.pth` through
    # `torch.hub.load_state_dict_from_url`; substitute the local checkpoint so
    # the script works offline.
    from lightglue import SuperPoint

    state = torch.load(weights, map_location="cpu", weights_only=True)
    original = torch.hub.load_state_dict_from_url
    torch.hub.load_state_dict_from_url = lambda *args, **kwargs: state
    try:
        extractor = SuperPoint(max_num_keypoints=4096, detection_threshold=0.005)
    finally:
        torch.hub.load_state_dict_from_url = original
    return extractor.eval()


def extract(extractor, image: np.ndarray, resize):
    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
    with torch.no_grad():
        output = extractor.extract(tensor, resize=resize)
    keypoints = output["keypoints"][0].to(torch.float32).numpy()
    scores = output["keypoint_scores"][0].to(torch.float32).numpy()
    return keypoints, scores, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/aliked-superpoint/superpoint_v1.pth"),
        help="official superpoint_v1.pth checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("crates/lux3d-core/tests/fixtures/keypoints/superpoint_golden_256.npz"),
        help="destination .npz",
    )
    parser.add_argument("--size", type=int, default=256, help="square image side")
    parser.add_argument("--resize", type=int, default=512, help="resize-path long side")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"missing checkpoint: {args.weights}")
    if args.size % 16 != 0:
        raise SystemExit("--size must be a multiple of 16")

    image = synthetic_image(args.size, args.size, args.seed)
    extractor = build_extractor(args.weights)

    keypoints, scores, output = extract(extractor, image, resize=None)
    descriptors = output["descriptors"][0].to(torch.float32).numpy()
    print(
        f"resize=None: {len(keypoints)} keypoints, "
        f"score range [{scores.min():.6f}, {scores.max():.6f}], "
        f"descriptors {descriptors.shape}"
    )

    resized_keypoints, resized_scores, _ = extract(
        extractor, image, resize=args.resize
    )
    print(
        f"resize={args.resize}: {len(resized_keypoints)} keypoints, "
        f"score range [{resized_scores.min():.6f}, {resized_scores.max():.6f}]"
    )

    # Determinism check: the fixture must be reproducible.
    again_keypoints, again_scores, again_output = extract(extractor, image, resize=None)
    if not (
        np.array_equal(keypoints, again_keypoints)
        and np.array_equal(scores, again_scores)
        and np.array_equal(descriptors, again_output["descriptors"][0].numpy())
    ):
        raise SystemExit("official extraction is not deterministic on this input")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        image=np.ascontiguousarray(image.transpose(2, 0, 1)),
        keypoints=keypoints,
        scores=scores,
        descriptors=descriptors,
        keypoints_resize512=resized_keypoints,
        scores_resize512=resized_scores,
    )
    total = sum(path.stat().st_size for path in args.output.parent.glob("*"))
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024.0:.1f} KiB, "
          f"fixture dir {total / 1024.0:.1f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
