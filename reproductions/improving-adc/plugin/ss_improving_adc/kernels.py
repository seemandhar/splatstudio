"""JIT-built CUDA statistics kernel (csrc/adc_stats.cu) and its Python wrapper.

gsplat 1.5.3 exposes the per-view tile lists (`isect_offsets`, `flatten_ids`) and the projected
2D Gaussians in `rasterization`'s meta, but not the per-Gaussian pixel count (PixelGS) or the
blend-weight sum sigma_k (paper Eq. 15). The kernel re-traverses gsplat's tile lists with gsplat's
alpha numerics. Build pattern follows the SplatStudio Taming-3DGS / Mini-Splatting ports.
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
        name="ss_improving_adc_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "adc_stats.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


@torch.no_grad()
def pixel_stats(meta):
    """Per-Gaussian (count, blend) for one non-packed camera: #contributing pixels (float) and
    sum over pixels of alpha*T."""
    if meta["means2d"].dim() != 3 or meta["means2d"].shape[0] != 1:
        raise ValueError("Improving-ADC statistics expect one non-packed camera")
    cnt, blend = ext().adc_stats(
        meta["means2d"][0].detach().float().contiguous(),
        meta["conics"][0].detach().float().contiguous(),
        meta["opacities"][0].detach().float().contiguous(),
        meta["isect_offsets"][0].int().contiguous(),
        meta["flatten_ids"].int().contiguous(),
        int(meta["width"]),
        int(meta["height"]),
        int(meta["tile_size"]),
    )
    return cnt.float(), blend
