"""JIT-built forward-only blending statistics (csrc/contrib_stats.cu, adapted from gsplat 1.5.3
RasterizeToPixels3DGSFwd.cu via the linear-splatting plugin)."""

import functools
import math
from pathlib import Path

import torch

CSRC = Path(__file__).resolve().parent / "csrc"


@functools.cache
def ext():
    """Build on first use (cached by torch in ~/.cache/torch_extensions) and import."""
    from torch.utils.cpp_extension import load

    return load(
        name="ss_augs_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "contrib_stats.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


def tile_lists(means2d, radii, depths, width, height, tile_size=16):
    """gsplat isect_tiles + isect_offset_encode for one camera (non-packed).

    means2d [N,2], radii [N,2] int32, depths [N] -> (offsets [th,tw] int32, flatten_ids int32)."""
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    tw, th = math.ceil(width / tile_size), math.ceil(height / tile_size)
    _, isect_ids, flatten_ids = isect_tiles(
        means2d[None], radii[None], depths[None], tile_size, tw, th, packed=False, n_images=1
    )
    offsets = isect_offset_encode(isect_ids, 1, tw, th).reshape(th, tw)
    return offsets.int().contiguous(), flatten_ids.int().contiguous()


@torch.no_grad()
def contrib_stats(means2d, conics, opacities, offsets, flatten_ids, width, height, tile_size=16,
                  t_threshold=0.7, alpha_max=0.999, inclusive_stop=False):
    """Per-pixel T-threshold Gaussian index [H,W] (-1: none) and per-Gaussian contributing-pixel
    counts [N] (int32). gsplat pixel centres (j + 0.5, i + 0.5)."""
    depth_idx, counts = ext().contrib_stats(
        means2d.float().contiguous(), conics.float().contiguous(), opacities.float().contiguous(),
        int(width), int(height), int(tile_size), offsets, flatten_ids, float(t_threshold),
        float(alpha_max), bool(inclusive_stop))
    return depth_idx, counts


@functools.cache
def ssim_ext():
    """Vendored fused-ssim (github.com/rahul-goel/fused-ssim, MIT): the official AuGS SSIM."""
    from torch.utils.cpp_extension import load

    src = CSRC / "fused_ssim"
    return load(
        name="ss_augs_fused_ssim",
        sources=[str(src / "ext.cpp"), str(src / "ssim.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


class _FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img1, img2):
        c1, c2 = 0.01**2, 0.03**2
        img1 = img1.contiguous()
        img2 = img2.contiguous()
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ssim_ext().fusedssim(c1, c2, img1, img2, True)
        ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        return ssim_map

    @staticmethod
    def backward(ctx, grad):
        img1, img2, a, b, c = ctx.saved_tensors
        g = ssim_ext().fusedssim_backward(0.01**2, 0.03**2, img1, img2, grad.contiguous(), a, b, c)
        return g, None


def fused_ssim_map(img1, img2):
    """'same'-padded SSIM map, NCHW CUDA float32 (fused_ssim(..., padding='same'))."""
    return _FusedSSIMMap.apply(img1, img2)
