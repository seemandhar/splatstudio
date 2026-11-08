"""JIT-built CUDA statistics kernel (csrc/bogauss_stats.cu) and its Python wrapper.

gsplat 1.5.3 exposes the per-view tile lists and projected 2D Gaussians in `rasterization`'s
meta but not per-Gaussian sums of blending weights. The kernel re-traverses gsplat's tile lists
with gsplat's alpha numerics (build pattern of the SplatStudio Taming 3DGS port).
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
        name="ss_bogauss_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "bogauss_stats.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


@torch.no_grad()
def pixel_stats(meta, pixel_error):
    """Per-Gaussian sums for one non-packed camera: `blend` = sum T*alpha, `se` = sum
    T*alpha*E(p), `count` = number of contributing pixels (float)."""
    if meta["means2d"].dim() != 3 or meta["means2d"].shape[0] != 1:
        raise ValueError("BOGausS statistics expect one non-packed camera")
    H, W = int(meta["height"]), int(meta["width"])
    if tuple(pixel_error.shape) != (H, W):
        raise ValueError("pixel_error must be [H, W]")
    blend, se, cnt = ext().bogauss_stats(
        meta["means2d"][0].detach().float().contiguous(),
        meta["conics"][0].detach().float().contiguous(),
        meta["opacities"][0].detach().float().contiguous(),
        meta["isect_offsets"][0].int().contiguous(),
        meta["flatten_ids"].int().contiguous(),
        pixel_error.detach().float().contiguous(),
        W,
        H,
        int(meta["tile_size"]),
    )
    return {"blend": blend, "se": se, "count": cnt.float()}
