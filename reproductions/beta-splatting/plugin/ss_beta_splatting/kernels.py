"""JIT-built DBS kernels (csrc/beta_raster.cu, ported from the official beta-splatting gsplat fork)
and their autograd wrappers."""

import functools
from pathlib import Path

import torch

CSRC = Path(__file__).resolve().parent / "csrc"


@functools.cache
def ext():
    """Build on first use (cached by torch in ~/.cache/torch_extensions) and import."""
    from torch.utils.cpp_extension import load

    return load(
        name="ss_beta_splatting_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "beta_raster.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


class _BetaRasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means2d, conics, colors, opacities, betas, backgrounds, tile_offsets,
                flatten_ids, width, height, tile_size):
        args = tuple(t.contiguous() for t in (means2d, conics, colors, opacities, betas))
        bg = None if backgrounds is None else backgrounds.contiguous()
        lists = (tile_offsets.int().contiguous(), flatten_ids.int().contiguous())
        renders, alphas, last_ids = ext().beta_forward(*args, bg, width, height, tile_size, *lists)
        ctx.save_for_backward(*args, *lists, alphas, last_ids)
        ctx.bg = bg
        ctx.meta = (width, height, tile_size)
        ctx.mark_non_differentiable(last_ids)
        return renders, alphas, last_ids

    @staticmethod
    def backward(ctx, v_renders, v_alphas, _v_last):
        means2d, conics, colors, opacities, betas, offsets, ids, alphas, last_ids = ctx.saved_tensors
        width, height, tile_size = ctx.meta
        if v_renders is None:
            v_renders = torch.zeros(height, width, colors.shape[-1], device=means2d.device)
        if v_alphas is None:
            v_alphas = torch.zeros(height, width, 1, device=means2d.device)
        vm, vc, vcol, vo, vb = ext().beta_backward(
            means2d, conics, colors, opacities, betas, ctx.bg, width, height, tile_size, offsets,
            ids, alphas, last_ids, v_renders.contiguous(), v_alphas.contiguous())
        return vm, vc, vcol, vo, vb, None, None, None, None, None, None


def rasterize_beta(means2d, conics, colors, opacities, betas, tile_offsets, flatten_ids, width,
                   height, tile_size=16, backgrounds=None):
    """One camera, non-packed: means2d [N,2], conics [N,3], colors [N,C] (C in 1,3,4),
    opacities [N], betas [N] (activated, 4 e^b), gsplat tile lists.

    alpha = min(0.999, o (1 - r^2)^beta) for r^2 = d^T conic d < 1 (paper Sec. 4.2).
    Returns colours [H,W,C] (+ T*bg if backgrounds given) and alpha [H,W,1].
    """
    renders, alphas, _ = _BetaRasterize.apply(
        means2d, conics, colors, opacities, betas, backgrounds, tile_offsets, flatten_ids,
        int(width), int(height), int(tile_size))
    return renders, alphas


class _SphericalBeta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dirs, c0, coeffs, masks):
        dirs, c0, coeffs = dirs.contiguous(), c0.contiguous(), coeffs.contiguous()
        masks = None if masks is None else masks.contiguous()
        colors = ext().sb_forward(dirs, c0, coeffs, masks)
        ctx.save_for_backward(dirs, coeffs)
        ctx.masks = masks
        return colors

    @staticmethod
    def backward(ctx, v_colors):
        dirs, coeffs = ctx.saved_tensors
        v_c0, v_coeffs = ext().sb_backward(dirs, coeffs, ctx.masks, v_colors.contiguous())
        # the official kernel provides no view-direction gradient
        return None, v_c0, v_coeffs, None


def spherical_beta(dirs, c0, coeffs, masks=None):
    """dirs [N,3] (unnormalised), c0 [N,3], coeffs [N,M,6] = (r,g,b activated, theta, phi, b_m)
    -> c0 + sum_m [dot>0] dot^(4 e^{b_m}) (r,g,b)_m; zero where masks is False."""
    return _SphericalBeta.apply(dirs, c0, coeffs, masks)
