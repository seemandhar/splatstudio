"""JIT-built forward-only per-Gaussian pixel counts on gsplat 1.5.3 tile lists.

Kernel `csrc/adc_stats.cu` is reused unchanged from the SplatStudio Improving-ADC port (validated
there against a brute-force per-pixel loop). Pixel-GS needs only the count m_k^i (paper Eq. 8;
official forward.cu:364 `atomicAdd(&pixels[id], 1)` per blended pixel).
"""

import functools
from pathlib import Path

import torch

CSRC = Path(__file__).resolve().parent / "csrc"


@functools.cache
def ext():
    """Build on first use (cached by torch in ~/.cache/torch_extensions) and import."""
    from torch.utils.cpp_extension import load

    return load(
        name="ss_pixel_gs_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "adc_stats.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


@torch.no_grad()
def pixel_counts(meta):
    """Float [N] number of pixels each Gaussian is blended into, for one non-packed camera
    (gsplat numerics: alpha = min(0.999, o exp(-sigma)) >= 1/255, stop when next T <= 1e-4)."""
    if meta["means2d"].dim() != 3 or meta["means2d"].shape[0] != 1:
        raise ValueError("Pixel-GS statistics expect one non-packed camera")
    cnt, _ = ext().adc_stats(
        meta["means2d"][0].detach().float().contiguous(),
        meta["conics"][0].detach().float().contiguous(),
        meta["opacities"][0].detach().float().contiguous(),
        meta["isect_offsets"][0].int().contiguous(),
        meta["flatten_ids"].int().contiguous(),
        int(meta["width"]),
        int(meta["height"]),
        int(meta["tile_size"]),
    )
    return cnt.float()
