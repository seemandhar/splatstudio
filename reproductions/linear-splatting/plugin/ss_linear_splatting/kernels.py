"""JIT-built 3DLS rasterizer (csrc/linear_raster.cu, adapted from gsplat 1.5.3
RasterizeToPixels3DGS{Fwd,Bwd}.cu) and its autograd wrapper."""

import functools
from pathlib import Path

import torch

CSRC = Path(__file__).resolve().parent / "csrc"


@functools.cache
def ext():
    """Build on first use (cached by torch in ~/.cache/torch_extensions) and import."""
    from torch.utils.cpp_extension import load

    return load(
        name="ss_linear_splatting_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "linear_raster.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


class _LinearRasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means2d, conics, colors, opacities, backgrounds, tile_offsets, flatten_ids,
                width, height, tile_size, lam, ags):
        args = (means2d.contiguous(), conics.contiguous(), colors.contiguous(),
                opacities.contiguous())
        bg = None if backgrounds is None else backgrounds.contiguous()
        lists = (tile_offsets.int().contiguous(), flatten_ids.int().contiguous())
        renders, alphas, last_ids = ext().linear_forward(
            *args, bg, width, height, tile_size, *lists, float(lam))
        ctx.save_for_backward(*args, *lists, alphas, last_ids)
        ctx.bg = bg
        ctx.meta = (width, height, tile_size, float(lam), float(ags))
        ctx.mark_non_differentiable(last_ids)
        return renders, alphas, last_ids

    @staticmethod
    def backward(ctx, v_renders, v_alphas, _v_last):
        means2d, conics, colors, opacities, offsets, ids, alphas, last_ids = ctx.saved_tensors
        width, height, tile_size, lam, ags = ctx.meta
        if v_renders is None:
            v_renders = torch.zeros(height, width, colors.shape[-1], device=means2d.device)
        if v_alphas is None:
            v_alphas = torch.zeros(height, width, 1, device=means2d.device)
        vm, vc, vcol, vo = ext().linear_backward(
            means2d, conics, colors, opacities, ctx.bg, width, height, tile_size, offsets, ids,
            lam, ags, alphas, last_ids, v_renders.contiguous(), v_alphas.contiguous())
        return vm, vc, vcol, vo, None, None, None, None, None, None, None, None


def rasterize_linear(means2d, conics, colors, opacities, tile_offsets, flatten_ids, width,
                     height, tile_size=16, lam=2.5, ags=1.0, backgrounds=None):
    """One camera, non-packed: means2d [N,2], conics [N,3], colors [N,C] (C in 1,3,4),
    opacities [N], tile_offsets [th,tw], flatten_ids [M] (gsplat isect lists).

    Returns colours [H,W,C] (accumulated, + T*bg if backgrounds given) and alpha [H,W,1].
    Linear falloff alpha = min(0.999, o * max(0, 1 - D/lam)) (paper Eq. 12). `ags` is the AGS
    coefficient c: the backward scales the kernel derivative by exp(-c D^2) (Eq. 13-14: c = 1;
    True -> 1, False/0 -> no AGS).
    """
    renders, alphas, _ = _LinearRasterize.apply(
        means2d, conics, colors, opacities, backgrounds, tile_offsets, flatten_ids, int(width),
        int(height), int(tile_size), float(lam), float(ags))
    return renders, alphas
