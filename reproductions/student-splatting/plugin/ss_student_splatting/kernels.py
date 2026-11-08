"""JIT-built Student's t rasterizer (csrc/t_raster.cu, adapted from the official
diff-t-rasterization renderCUDA) and its autograd wrapper."""

import functools
from pathlib import Path

import torch

CSRC = Path(__file__).resolve().parent / "csrc"


@functools.cache
def ext():
    """Build on first use (cached by torch in ~/.cache/torch_extensions) and import.
    No --use_fast_math: the official extension is compiled without it (powf accuracy)."""
    from torch.utils.cpp_extension import load

    return load(
        name="ss_student_splatting_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "t_raster.cu")],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        verbose=False,
    )


class _TRasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means2d, conics, colors, opacities, nus, backgrounds, tile_offsets,
                flatten_ids, width, height, tile_size, alpha_clip, exact_nu_grad):
        args = tuple(t.contiguous() for t in (means2d, conics, colors, opacities, nus))
        bg = None if backgrounds is None else backgrounds.contiguous()
        lists = (tile_offsets.int().contiguous(), flatten_ids.int().contiguous())
        renders, alphas, last_ids = ext().t_forward(*args, bg, width, height, tile_size, *lists,
                                                    alpha_clip)
        ctx.save_for_backward(*args, *lists, alphas, last_ids)
        ctx.bg = bg
        ctx.meta = (width, height, tile_size, alpha_clip, exact_nu_grad)
        ctx.mark_non_differentiable(last_ids)
        return renders, alphas, last_ids

    @staticmethod
    def backward(ctx, v_renders, v_alphas, _v_last):
        means2d, conics, colors, opacities, nus, offsets, ids, alphas, last_ids = ctx.saved_tensors
        width, height, tile_size, alpha_clip, exact_nu_grad = ctx.meta
        if v_renders is None:
            v_renders = torch.zeros(height, width, colors.shape[-1], device=means2d.device)
        if v_alphas is None:
            v_alphas = torch.zeros(height, width, 1, device=means2d.device)
        vm, vc, vcol, vo, vn = ext().t_backward(
            means2d, conics, colors, opacities, nus, ctx.bg, width, height, tile_size, offsets,
            ids, alphas, last_ids, v_renders.contiguous(), v_alphas.contiguous(), alpha_clip,
            exact_nu_grad)
        return vm, vc, vcol, vo, vn, None, None, None, None, None, None, None, None


def rasterize_t(means2d, conics, colors, opacities, nus, tile_offsets, flatten_ids, width,
                height, tile_size=16, backgrounds=None, alpha_clip=0.99, exact_nu_grad=False):
    """One camera, non-packed: means2d [N,2] (gsplat pixel coords, centres at +0.5), conics [N,3]
    (a, b, c of the inverse 2D covariance), colors [N,C] (C in 1,3,4), signed opacities [N],
    activated nus [N] >= 1, gsplat tile lists.

    alpha = clamp(o (1 + base/nu)^(-(nu+2)/2), -0.99, 0.99), base = d^T conic d (Eq. t_2D).
    Returns colours [H,W,C] (+ T*bg if backgrounds given) and 1 - T [H,W,1] (T can exceed 1
    behind negative components)."""
    renders, alphas, _ = _TRasterize.apply(
        means2d, conics, colors, opacities, nus, backgrounds, tile_offsets, flatten_ids,
        int(width), int(height), int(tile_size), float(alpha_clip), bool(exact_nu_grad))
    return renders, alphas
