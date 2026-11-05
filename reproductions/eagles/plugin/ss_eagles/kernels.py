"""JIT-built CUDA influence kernel (csrc/influence.cu) and its Python wrapper."""

import functools
from pathlib import Path

import torch

CSRC = Path(__file__).resolve().parent / "csrc"
GSPLAT_ALPHA_MAX = 0.999


@functools.cache
def ext():
    """Build (first call, cached by torch in ~/.cache/torch_extensions) and import the module."""
    from torch.utils.cpp_extension import load

    return load(
        name="ss_eagles_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "influence.cu")],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        verbose=False,
    )


@torch.no_grad()
def influence(info, alpha_max=GSPLAT_ALPHA_MAX):
    """W_i = sum_p alpha_i T_i [N] (EAGLES Eq. 3) for a single-camera, non-packed gsplat render."""
    if info["means2d"].dim() != 3 or info["means2d"].shape[0] != 1:
        raise ValueError("influence statistics expect one non-packed camera")
    return ext().influence_stats(
        info["means2d"][0].detach().float().contiguous(),
        info["conics"][0].detach().float().contiguous(),
        info["opacities"][0].detach().float().contiguous(),
        info["isect_offsets"][0].int().contiguous(),
        info["flatten_ids"].int().contiguous(),
        int(info["width"]),
        int(info["height"]),
        int(info["tile_size"]),
        float(alpha_max),
    )
