"""JIT-built CUDA kernel (csrc/litegs_stats.cu) for LiteGS's per-pixel opacity-gradient moments.

gsplat 1.5.3 only returns the per-Gaussian *sum* of opacity gradients; the variance metric needs
the per-pixel squares as well, so the kernel re-traverses gsplat's own tile lists.
"""

import functools
from pathlib import Path

import torch

CSRC = Path(__file__).resolve().parent / "csrc"


@functools.cache
def ext():
    from torch.utils.cpp_extension import load

    return load(
        name="ss_litegs_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "litegs_stats.cu")],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        verbose=False,
    )


@torch.no_grad()
def opacity_grad_moments(meta, colors, final_rgb, v_rgb):
    """Per-Gaussian moments of dL/do over the fragments of one non-packed camera.

    meta: gsplat `rasterization` meta (one camera); colors [N,3] post-activation colours used for
    that render; final_rgb [H,W,3] rendered colour incl. background; v_rgb [H,W,3] = dL/d final_rgb.
    Returns dict of [N] tensors: sq_sum (S), sum (M), count (C, float), weight (sum T*alpha).
    """
    if meta["means2d"].dim() != 3 or meta["means2d"].shape[0] != 1:
        raise ValueError("LiteGS statistics expect one non-packed camera")
    H, W = int(meta["height"]), int(meta["width"])
    if tuple(final_rgb.shape) != (H, W, 3) or tuple(v_rgb.shape) != (H, W, 3):
        raise ValueError("final_rgb and v_rgb must be [H, W, 3]")
    sq, sm, cnt, wt = ext().litegs_stats(
        meta["means2d"][0].detach().float().contiguous(),
        meta["conics"][0].detach().float().contiguous(),
        meta["opacities"][0].detach().float().contiguous(),
        colors.detach().float().contiguous(),
        meta["isect_offsets"][0].int().contiguous(),
        meta["flatten_ids"].int().contiguous(),
        final_rgb.detach().float().contiguous(),
        v_rgb.detach().float().contiguous(),
        W,
        H,
        int(meta["tile_size"]),
    )
    return {"sq_sum": sq, "sum": sm, "count": cnt.float(), "weight": wt}
