"""JIT-built CUDA blend-statistics kernel (csrc/cad_stats.cu) and its Python wrapper.

gsplat 1.5.3 returns per-view tile lists (`isect_offsets`, `flatten_ids`) and projected 2D
Gaussians in the rasterization meta; the kernel re-traverses them with gsplat's alpha numerics
and returns, per primitive, max_p alpha*T (paper Sec. 3.2, w_i for one view) and
sum_p (1-SSIM(p)) alpha*T (paper Sec. 3.1, SSIM term). Build pattern follows the SplatStudio
Revising-Densification port.
"""

import functools
from pathlib import Path

import torch

CSRC = Path(__file__).resolve().parent / "csrc"


@functools.cache
def ext():
    from torch.utils.cpp_extension import load

    return load(
        name="ss_confidence_densify_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "cad_stats.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


@torch.no_grad()
def blend_stats(meta, error):
    """(max blend weight [N], sum of error*blend weight [N]) for one non-packed camera."""
    if meta["means2d"].dim() != 3 or meta["means2d"].shape[0] != 1:
        raise ValueError("confidence-densify statistics expect one non-packed camera")
    return ext().blend_stats(
        meta["means2d"][0].detach().float().contiguous(),
        meta["conics"][0].detach().float().contiguous(),
        meta["opacities"][0].detach().float().contiguous(),
        meta["isect_offsets"][0].int().contiguous(),
        meta["flatten_ids"].int().contiguous(),
        error.detach().float().contiguous(),
        int(meta["width"]),
        int(meta["height"]),
        int(meta["tile_size"]),
    )
