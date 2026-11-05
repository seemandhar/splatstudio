"""JIT-built CUDA error-attribution kernel (csrc/err_stats.cu) and its Python wrapper.

gsplat 1.5.3 exposes per-view tile lists (`isect_offsets`, `flatten_ids`) and the projected 2D
Gaussians in `rasterization`'s meta; the kernel re-traverses them with gsplat's alpha numerics
and accumulates E^pi_k = sum_u Err(u) alpha_k(u) T_k(u) (paper Sec. 3.2). Build pattern follows
the SplatStudio Improving-ADC / Taming-3DGS ports.
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
        name="ss_revising_densification_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "err_stats.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


@torch.no_grad()
def error_stats(meta, error):
    """Per-Gaussian (E^pi_k, sum_u w_k) for one non-packed camera; `error` is [H, W]."""
    if meta["means2d"].dim() != 3 or meta["means2d"].shape[0] != 1:
        raise ValueError("Revising-Densification statistics expect one non-packed camera")
    err, blend = ext().err_stats(
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
    return err, blend
