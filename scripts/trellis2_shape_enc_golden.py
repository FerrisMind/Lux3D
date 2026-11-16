#!/usr/bin/env python3
"""Reference oracle for the TRELLIS.2 shape SLAT encoder (FlexiDualGridVaeEncoder).

Runs the *unmodified* reference `FlexiDualGridVaeEncoder` /
`SparseUnetVaeEncoder` code from candle-3d-refs/TRELLIS.2 with only a
pure-PyTorch submanifold 3x3x3 sparse conv backend injected (same validated
adapter as `.swarm/scripts/trellis2_vae_oracle.py`; no flex_gemm needed).

Outputs `.swarm/results/trellis2-shape-enc-golden/`:
  - `inputs.npz`: voxel indices, normalized dual vertices, intersected flags,
    resolution (the synthetic flexible dual grid the encoder consumes)
  - `block_<s>_<b>_coords.npz` for every block (coords always; feats for the
    four S2C down blocks + input layer)
  - `latent.npz`: final mean latent (F32) + fp16-torso variant + out coords
  - `manifest.json`

The F32 run mirrors the Rust implementation (all-F32 compute): the reference
`self.dtype` override skips the torso fp16 casts. The fp16 run keeps upstream
`use_fp16` inference semantics (blocks F16, input layer/norm/to_latent F32).

Usage (repo root):
  python -I scripts/trellis2_shape_enc_golden.py [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]  # candle-3d repo root
REF = ROOT.parent / "candle-3d-refs" / "TRELLIS.2"


# ---------------------------------------------------------------------------
# Pure-PyTorch submanifold sparse conv backend (matches flex_gemm weight
# layout [Co, Kd, Kh, Kw, Ci] and cross-correlation semantics) — verbatim from
# the validated `.swarm/scripts/trellis2_vae_oracle.py` adapter.
# ---------------------------------------------------------------------------

def _pack(coords: torch.Tensor) -> torch.Tensor:
    c = coords.to(torch.int64)
    return (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]


class _CoordLookup:
    def __init__(self, coords: torch.Tensor):
        self.keys = _pack(coords)
        self.sorted_keys, self.order = torch.sort(self.keys)
        self.n = coords.shape[0]
        self.device = coords.device

    def find(self, query: torch.Tensor) -> torch.Tensor:
        q = _pack(query)
        idx = torch.searchsorted(self.sorted_keys, q)
        in_range = idx < self.n
        idx_clamped = idx.clamp(max=max(self.n - 1, 0))
        found = in_range & (self.sorted_keys[idx_clamped] == q)
        out = torch.full_like(idx, -1)
        out[found] = self.order[idx_clamped[found]]
        return out


def subm_conv(coords: torch.Tensor, feats: torch.Tensor, weight: torch.Tensor,
              bias: torch.Tensor | None) -> torch.Tensor:
    """out[p] = bias + sum_k W[:, k, :] @ feats[p + offset_k], missing = 0."""
    if coords.shape[1] == 4:  # drop batch column
        coords = coords[:, 1:]
    lookup = _CoordLookup(coords)
    weight = weight.to(feats.dtype)
    if bias is not None:
        bias = bias.to(feats.dtype)
    cout, kd, kh, kw, cin = weight.shape
    assert (kd, kh, kw) == (3, 3, 3)
    w = weight.permute(1, 2, 3, 4, 0).reshape(27, cin, cout)
    out = torch.zeros((coords.shape[0], cout), dtype=feats.dtype, device=feats.device)
    for kd_i in range(3):
        for kh_i in range(3):
            for kw_i in range(3):
                off = torch.tensor([kd_i - 1, kh_i - 1, kw_i - 1], dtype=torch.long,
                                   device=coords.device)
                q = coords + off
                src = lookup.find(q)
                valid = src >= 0
                if not bool(valid.any()):
                    continue
                dst = torch.nonzero(valid, as_tuple=False).squeeze(1)
                src_idx = src[valid]
                contrib = feats.index_select(0, src_idx) @ w[kd_i * 9 + kh_i * 3 + kw_i]
                out.index_add_(0, dst, contrib)
    if bias is not None:
        out = out + bias
    return out


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1,
                       dilation=1, padding=None, bias=True, indice_key=None):
    assert stride == 1 and padding is None
    k = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
    assert k == (3, 3, 3), k
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = k
    self.stride = (1, 1, 1)
    self.dilation = (1, 1, 1)
    # flex_gemm layout: [Co, Kd, Kh, Kw, Ci]
    self.weight = nn.Parameter(torch.empty((out_channels, 3, 3, 3, in_channels)))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)
    nn.init.kaiming_uniform_(self.weight.reshape(out_channels, -1), a=5 ** 0.5)
    if self.bias is not None:
        self.bias.data.uniform_(-0.1, 0.1)


def sparse_conv3d_forward(self, x):
    return x.replace(subm_conv(x.coords, x.feats, self.weight, self.bias))


def install_adapters():
    if str(REF) not in sys.path:
        sys.path.insert(0, str(REF))
    ov = types.ModuleType("o_voxel")
    ov.__path__ = []  # package marker
    conv = types.ModuleType("o_voxel.convert")

    def _missing(*args, **kwargs):
        raise RuntimeError("o_voxel stub: encoder path never calls this")

    conv.flexible_dual_grid_to_mesh = _missing
    ov.convert = conv
    sys.modules["o_voxel"] = ov
    sys.modules["o_voxel.convert"] = conv

    import trellis2.modules.sparse.config as sp_config
    sp_config.CONV = "none"
    import trellis2.modules.sparse.conv.conv as conv_mod
    backend = types.ModuleType("trellis2.modules.sparse.conv.conv_none")
    backend.sparse_conv3d_init = sparse_conv3d_init
    backend.sparse_conv3d_forward = sparse_conv3d_forward
    conv_mod._backends["none"] = backend

    import trellis2.representations as reps

    class SimpleMesh:
        def __init__(self, vertices, faces, vertex_attrs=None):
            self.vertices = vertices.float()
            self.faces = faces.int()
            self.vertex_attrs = vertex_attrs

    reps.Mesh = SimpleMesh


# ---------------------------------------------------------------------------
# Synthetic flexible dual grid
# ---------------------------------------------------------------------------

def build_inputs(resolution: int, num_voxels: int, seed: int):
    """Spherical shell of voxels with deterministic dual-vertex offsets and
    intersection flags. Rows are shuffled: the reference ops are
    order-invariant and the S2C output must come out parent-sorted regardless
    of the input row order."""
    rng = np.random.default_rng(seed)
    res = resolution
    center = np.array([res / 2.0, res / 2.0, res / 2.0])
    axis = np.arange(res // 4, res - res // 4)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    grid = grid.reshape(-1, 3).astype(np.float64)
    dist = np.linalg.norm(grid - center, axis=1)
    mask = (dist > res * 0.16) & (dist < res * 0.26)
    voxels = grid[mask]
    sel = rng.permutation(len(voxels))[:num_voxels]
    voxel_indices = voxels[np.sort(sel)].astype(np.int64)
    n = len(voxel_indices)
    # Dual vertex offsets live in [-0.5, 1.5] voxel units (the decoder head's
    # dual range); `dual_vertices` are the normalized positions the upstream
    # voxelization returns, so feats = dual*res - idx recovers the offsets.
    offsets = rng.uniform(-0.45, 1.45, size=(n, 3))
    dual_vertices = ((voxel_indices + offsets) / res).astype(np.float32)
    intersected = (rng.uniform(size=(n, 3)) < 0.5).astype(np.float32)
    perm = rng.permutation(n)
    return {
        "voxel_indices": voxel_indices[perm].astype(np.int32),
        "dual_vertices": dual_vertices[perm],
        "intersected": intersected[perm],
        "resolution": np.int64(res),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "TRELLIS.2-4B"))
    ap.add_argument("--out-dir", default=str(ROOT / ".swarm" / "results" / "trellis2-shape-enc-golden"))
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--num-voxels", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    install_adapters()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from trellis2.models import from_pretrained
    from trellis2.modules.sparse import SparseTensor

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    enc = from_pretrained(str(model_dir / "ckpts" / "shape_enc_next_dc_f16c32_fp16"))
    enc = enc.float().to(device).eval()
    enc.dtype = torch.float32  # skip the torso fp16 casts (Rust parity run)

    data = build_inputs(args.resolution, args.num_voxels, args.seed)
    res = int(data["resolution"])
    voxel_indices = torch.from_numpy(data["voxel_indices"].astype(np.int64))
    n = voxel_indices.shape[0]
    coords4 = torch.cat(
        [torch.zeros((n, 1), dtype=torch.long), voxel_indices], dim=-1
    ).to(device)
    vertices = SparseTensor(
        feats=torch.from_numpy(data["dual_vertices"]).float().to(device) * res
        - voxel_indices.to(device),
        coords=coords4,
    )
    intersected = vertices.replace(
        torch.from_numpy(data["intersected"]).float().to(device)
    )
    print(f"input voxels: {n} resolution: {res}")

    captured = {}

    def make_hook(name):
        def hook(_m, _inp, out):
            captured[name + ".coords"] = out.coords.detach().cpu().numpy().astype(np.int32)
            captured[name + ".feats"] = out.feats.detach().float().cpu().numpy().astype(np.float32)
        return hook

    def coords_hook(name):
        def hook(_m, _inp, out):
            captured[name + ".coords"] = out.coords.detach().cpu().numpy().astype(np.int32)
        return hook

    handles = [enc.input_layer.register_forward_hook(make_hook("input_layer"))]
    for si, stage in enumerate(enc.blocks):
        for bi, block in enumerate(stage):
            is_down = si < len(enc.blocks) - 1 and bi == len(stage) - 1
            handles.append(block.register_forward_hook(
                make_hook(f"blocks.{si}.{bi}") if is_down else coords_hook(f"blocks.{si}.{bi}")
            ))

    with torch.no_grad():
        z = enc(vertices, intersected, sample_posterior=False, return_raw=False)
    for h in handles:
        h.remove()
    out_coords = z.coords[:, 1:].cpu().numpy().astype(np.int32)
    mean_f32 = z.feats.detach().float().cpu().numpy().astype(np.float32)
    print("latent:", mean_f32.shape, "out coords:", out_coords.shape)

    # Upstream fp16 inference (blocks F16; input layer / norm / to_latent F32).
    enc.convert_to_fp16()
    enc.dtype = torch.float16
    with torch.no_grad():
        z16 = enc(vertices, intersected, sample_posterior=False, return_raw=False)
    mean_f16 = z16.feats.detach().float().cpu().numpy().astype(np.float32)
    delta = np.abs(mean_f16.astype(np.float64) - mean_f32.astype(np.float64))
    print(f"fp16-vs-f32 mean latent: max_abs={delta.max():.3e} rms={np.sqrt((delta ** 2).mean()):.3e}")
    print(
        "f32 stats: min={:.3e} max={:.3e} absmean={:.3e}".format(
            mean_f32.min(), mean_f32.max(), np.abs(mean_f32).mean()
        )
    )

    np.savez_compressed(
        out_dir / "inputs.npz",
        voxel_indices=data["voxel_indices"],
        dual_vertices=data["dual_vertices"],
        intersected=data["intersected"],
        resolution=data["resolution"],
    )
    np.savez_compressed(
        out_dir / "latent.npz",
        coords=out_coords,
        mean_f32=mean_f32,
        mean_f16=mean_f16,
    )
    for key, value in captured.items():
        np.savez_compressed(out_dir / (key.replace(".", "_") + ".npz"), values=value)
    manifest = {
        "seed": args.seed,
        "resolution": res,
        "num_voxels": n,
        "device": str(device),
        "latent_shape": list(mean_f32.shape),
        "out_coords_shape": list(out_coords.shape),
        "tensors": sorted(captured.keys()),
        "fp16_vs_f32_max_abs": float(delta.max()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
