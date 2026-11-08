"""JIT-built CUDA kernels reused from the Mini-Splatting native plugin
(reproductions/mini-splatting/plugin/ss_mini_splatting/csrc, adapted from the official
Mini-Splatting rasterizer, Inria 3DGS licence in csrc/LICENSE-3DGS.md).

`max_contrib_counts` re-traverses gsplat's own tile lists with gsplat's alpha numerics and returns,
per rasterized Gaussian, |S_i| = #pixels whose maximum-weight Gaussian (argmax_i w_i, HGS §5.2)
is i. `dist_cuda2` is the vendored simple-knn distCUDA2 used by Scaffold-GS's anchor init.
"""

import functools
from pathlib import Path

import torch

CSRC = Path(__file__).resolve().parent / "csrc"


@functools.cache
def ext():
    from torch.utils.cpp_extension import load

    return load(
        name="ss_hard_gs_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "ms_raster.cu"), str(CSRC / "simple_knn.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


@torch.no_grad()
def max_contrib_counts(info):
    """|S_i| for every Gaussian of a single-camera, non-packed gsplat render (int32 [M]) and the
    per-pixel argmax id map ([H,W], -1 where nothing is blended)."""
    if info["means2d"].dim() != 3 or info["means2d"].shape[0] != 1:
        raise ValueError("expects one non-packed camera")
    _, _, area_max, argmax, _ = ext().ms_stats(
        info["means2d"][0].detach().float().contiguous(),
        info["conics"][0].detach().float().contiguous(),
        info["opacities"][0].detach().float().contiguous(),
        info["isect_offsets"][0].int().contiguous(),
        info["flatten_ids"].int().contiguous(),
        int(info["width"]),
        int(info["height"]),
        int(info["tile_size"]),
        False,
    )
    return area_max, argmax


@torch.no_grad()
def dist_cuda2(points):
    return ext().dist_cuda2(points.detach().float().contiguous())
