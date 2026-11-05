"""JIT-built CUDA statistics kernels (csrc/) and their Python wrappers.

gsplat 1.5.3 exposes the per-view tile lists (`isect_offsets`, `flatten_ids`) and projected
2D Gaussians in `rasterization`'s meta, but neither per-pixel gradient norms (EfficientGS
Eq. 6), top-K blending-weight flags (Eq. 8) nor colour-error weights (Eq. 9).
`csrc/egs_raster.cu` re-traverses gsplat's tile lists with gsplat's alpha numerics.
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
        name="ss_efficientgs_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "egs_raster.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


def _flat(meta):
    """Single-camera, non-packed gsplat meta -> contiguous kernel inputs."""
    if meta["means2d"].dim() != 3 or meta["means2d"].shape[0] != 1:
        raise ValueError("EfficientGS statistics expect one non-packed camera")
    return (
        meta["means2d"][0].detach().float().contiguous(),
        meta["conics"][0].detach().float().contiguous(),
        meta["opacities"][0].detach().float().contiguous(),
    ), (
        meta["isect_offsets"][0].int().contiguous(),
        meta["flatten_ids"].int().contiguous(),
        int(meta["width"]),
        int(meta["height"]),
        int(meta["tile_size"]),
    )


@torch.no_grad()
def gradnorm(meta, colors, v_colors, v_alphas):
    """Eq. 6 for one view: per Gaussian, sum over pixels of the norm of the pixel's NDC-space
    mean2d gradient; plus the summed per-pixel vectors in pixel units (= means2d.grad).

    colors [N,3] per-Gaussian RGB used by the rasterizer; v_colors [H,W,3] and v_alphas [H,W]
    are the upstream gradients of gsplat's raw rendered colours / alphas.
    """
    g, t = _flat(meta)
    return ext().egs_gradnorm(
        *g,
        colors.detach().float().reshape(-1, 3).contiguous(),
        *t,
        v_colors.detach().float().reshape(-1, 3).contiguous(),
        v_alphas.detach().float().reshape(-1).contiguous(),
    )


@torch.no_grad()
def blend_stats(meta, top_k=1, weight_after_T=True, pix_loss=None):
    """Eq. 8 dominant flags [N] (int32 0/1) and Eq. 9 loss weights [N] for one view."""
    g, t = _flat(meta)
    pl = None if pix_loss is None else pix_loss.detach().float().contiguous()
    dominant, lw = ext().egs_stats(*g, *t, int(top_k), bool(weight_after_T), pl)
    return {"dominant": dominant, "loss_weight": lw}
