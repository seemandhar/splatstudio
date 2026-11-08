"""JIT-built RadSplat max-contribution score kernel (csrc/lp_score.cu) and wrappers.

gsplat 1.5.3 exposes per-view tile lists (`isect_offsets`, `flatten_ids`) and the projected 2D
Gaussians in `rasterization`'s meta but no per-Gaussian max blending weight. The kernel
re-traverses gsplat's tile lists with gsplat's alpha numerics (official LP-3DGS `mw_score`).
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
        name="ss_lp_3dgs_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "lp_score.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


@torch.no_grad()
def project(params, camera, opacities):
    """gsplat projection + tile intersection with the renderer's settings (classic, eps2d 0.3).

    ``opacities`` are the activated (possibly masked) opacities the training renderer uses.
    """
    from gsplat import rasterization

    _, _, meta = rasterization(
        means=params["means"].detach(),
        quats=params["quats"].detach(),
        scales=params["scales"].detach().exp(),
        opacities=opacities.detach(),
        colors=params["sh0"].detach(),
        viewmats=camera["viewmat"][None],
        Ks=camera["K"][None],
        width=int(camera["width"]),
        height=int(camera["height"]),
        sh_degree=0,
        packed=False,
        render_mode="RGB",
        rasterize_mode="classic",
    )
    return meta


@torch.no_grad()
def max_contribution_(meta, score):
    """In place: score_i = max(score_i, max over this view's pixels of alpha_i T_i)."""
    if meta["means2d"].dim() != 3 or meta["means2d"].shape[0] != 1:
        raise ValueError("LP-3DGS score expects one non-packed camera")
    return ext().max_contribution(
        meta["means2d"][0].detach().float().contiguous(),
        meta["conics"][0].detach().float().contiguous(),
        meta["opacities"][0].detach().float().contiguous(),
        meta["isect_offsets"][0].int().contiguous(),
        meta["flatten_ids"].int().contiguous(),
        int(meta["width"]),
        int(meta["height"]),
        int(meta["tile_size"]),
        score,
    )
