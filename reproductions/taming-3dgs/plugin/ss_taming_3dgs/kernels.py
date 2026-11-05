"""JIT-built CUDA statistics kernel (csrc/taming_stats.cu) and its Python wrapper.

gsplat 1.5.3 exposes the per-view tile lists (`isect_offsets`, `flatten_ids`) and projected 2D
Gaussians in `rasterization`'s meta, but not the per-Gaussian pixel accumulations that the
official Taming 3DGS rasterizer returns for scoring (accum_weights / accum_count / accum_blend /
accum_dist). The kernel re-traverses gsplat's own tile lists with gsplat's alpha numerics.
Build pattern follows the SplatStudio Mini-Splatting port (plan #28).
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
        name="ss_taming_3dgs_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "taming_stats.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


@torch.no_grad()
def pixel_stats(meta, pixel_weights):
    """Per-Gaussian sums over contributing pixels for one non-packed camera.

    Returns dict of [N] tensors: `saliency` = sum S_v(p), `count` = #pixels (float), `blend` =
    sum alpha*T, `dist` = sum ||mu2d - pixel centre||.
    """
    if meta["means2d"].dim() != 3 or meta["means2d"].shape[0] != 1:
        raise ValueError("Taming statistics expect one non-packed camera")
    H, W = int(meta["height"]), int(meta["width"])
    if tuple(pixel_weights.shape) != (H, W):
        raise ValueError("pixel_weights must be [H, W]")
    w, cnt, blend, dist = ext().taming_stats(
        meta["means2d"][0].detach().float().contiguous(),
        meta["conics"][0].detach().float().contiguous(),
        meta["opacities"][0].detach().float().contiguous(),
        meta["isect_offsets"][0].int().contiguous(),
        meta["flatten_ids"].int().contiguous(),
        pixel_weights.detach().float().contiguous(),
        W,
        H,
        int(meta["tile_size"]),
    )
    return {"saliency": w, "count": cnt.float(), "blend": blend, "dist": dist}
