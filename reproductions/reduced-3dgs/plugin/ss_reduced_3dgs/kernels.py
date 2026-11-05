"""JIT-built CUDA kernels (csrc/) and their Python wrappers.

* `transmittance_stats`: per-Gaussian sum of transmittance and number of blended pixels for
  one view, re-traversing gsplat 1.5.3's own tile intersection lists (official Reduced-3DGS
  rasterizer `calculate_mean_transmittance` branch).
* `knn_index`: exact K nearest neighbours excluding self (official simple-knn `distIndex2`).
* `kmeans_assign`: 1-D nearest codebook centre, lowest index on ties (official `updateIds`).
"""

import functools
from pathlib import Path

import torch

CSRC = Path(__file__).resolve().parent / "csrc"


@functools.cache
def ext():
    """Build (first call, cached by torch in ~/.cache/torch_extensions) and import the module."""
    from torch.utils.cpp_extension import load

    return load(
        name="ss_reduced_3dgs_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "transmittance.cu"), str(CSRC / "knn.cu"),
                 str(CSRC / "kmeans.cu")],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        verbose=False,
    )


@torch.no_grad()
def transmittance_stats(info, opacities=None, alpha_max=0.999):
    """(sum_T [N], touched [N] int32) for a single-camera, non-packed gsplat render meta.

    `opacities` overrides the per-Gaussian alpha multiplier (default: the meta's activated
    opacities); alpha = min(alpha_max, opacity * exp(-sigma)).
    """
    op = info["opacities"][0] if opacities is None else opacities
    if info["means2d"].dim() != 3 or info["means2d"].shape[0] != 1:
        raise ValueError("transmittance statistics expect one non-packed camera")
    return ext().transmittance_stats(
        info["means2d"][0].detach().float().contiguous(),
        info["conics"][0].detach().float().contiguous(),
        op.detach().float().contiguous(),
        info["isect_offsets"][0].int().contiguous(),
        info["flatten_ids"].int().contiguous(),
        int(info["width"]),
        int(info["height"]),
        int(info["tile_size"]),
        float(alpha_max),
    )


@torch.no_grad()
def knn_index(points, k):
    """Exact k-NN (squared distances [N,k], int64 indices [N,k], -1 where fewer than k exist)."""
    d2, idx = ext().knn_index(points.detach().float().contiguous(), int(k))
    return d2, idx.long()


@torch.no_grad()
def kmeans_assign(values, centers):
    """int64 index of the nearest centre for each value (first index on ties)."""
    return ext().kmeans_assign(values.float().contiguous(), centers.float().contiguous()).long()
