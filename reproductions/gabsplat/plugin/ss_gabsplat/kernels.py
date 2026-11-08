"""JIT-built 3DGabSplat rasterizer (csrc/gabor_raster.cu, adapted from the linear-splatting
kernel in this repo, itself from gsplat 1.5.3 RasterizeToPixels3DGS{Fwd,Bwd}.cu) and its
autograd wrapper."""

import functools
from pathlib import Path

import torch

CSRC = Path(__file__).resolve().parent / "csrc"


@functools.cache
def ext():
    """Build on first use (cached by torch in ~/.cache/torch_extensions) and import."""
    from torch.utils.cpp_extension import load

    return load(
        name="ss_gabsplat_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "gabor_raster.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


class _GaborRasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means2d, conics, colors, opacities, gabs, backgrounds, tile_offsets,
                flatten_ids, width, height, tile_size):
        args = (means2d.contiguous(), conics.contiguous(), colors.contiguous(),
                opacities.contiguous(), gabs.contiguous())
        bg = None if backgrounds is None else backgrounds.contiguous()
        lists = (tile_offsets.int().contiguous(), flatten_ids.int().contiguous())
        renders, alphas, last_ids = ext().gabor_forward(*args, bg, width, height, tile_size,
                                                        *lists)
        ctx.save_for_backward(*args, *lists, alphas, last_ids)
        ctx.bg = bg
        ctx.meta = (width, height, tile_size)
        ctx.mark_non_differentiable(last_ids)
        return renders, alphas, last_ids

    @staticmethod
    def backward(ctx, v_renders, v_alphas, _v_last):
        means2d, conics, colors, opacities, gabs, offsets, ids, alphas, last_ids = \
            ctx.saved_tensors
        width, height, tile_size = ctx.meta
        if v_renders is None:
            v_renders = torch.zeros(height, width, colors.shape[-1], device=means2d.device)
        if v_alphas is None:
            v_alphas = torch.zeros(height, width, 1, device=means2d.device)
        vm, vc, vcol, vo, vg = ext().gabor_backward(
            means2d, conics, colors, opacities, gabs, ctx.bg, width, height, tile_size, offsets,
            ids, alphas, last_ids, v_renders.contiguous(), v_alphas.contiguous())
        return vm, vc, vcol, vo, vg, None, None, None, None, None, None


def rasterize_gabor(means2d, conics, colors, opacities, gabs, tile_offsets, flatten_ids, width,
                    height, tile_size=16, backgrounds=None):
    """One camera, non-packed: means2d [N,2], conics [N,3], colors [N,C] (C in 3,4),
    opacities [N], gabs [N,3F] = (f2d_x, f2d_y, w) per Gabor kernel (F in 1,2; f2d in cycles per
    pixel), tile_offsets [th,tw], flatten_ids [M] (gsplat isect lists).

    alpha = min(0.999, o exp(-sigma) [(1 - sum w) + sum w cos(2 pi f2d . delta)]) (paper
    Eq. 12-13). Returns colours [H,W,C] (accumulated, + T*bg if backgrounds given), alpha [H,W,1].
    """
    renders, alphas, _ = _GaborRasterize.apply(
        means2d, conics, colors, opacities, gabs, backgrounds, tile_offsets, flatten_ids,
        int(width), int(height), int(tile_size))
    return renders, alphas
