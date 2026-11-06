"""JIT-built CUDA jet rasterizer (csrc/jet_raster.cu) and its autograd wrapper.

`rasterize_jets` renders, on gsplat's own projected Gaussians and tile lists, the image I and its
analytic pixel-space derivatives I_x, I_y, I_xy (paper Sec. 5, supp. "3DGS Gradient
Computation") over a constant background, and back-propagates gradients of all four outputs to
means2d / conics / colours / opacities (supp. "Backward Pass").
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
        name="ss_gradient_upscaling_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "jet_raster.cu")],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        verbose=False,
    )


class _JetRasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means2d, conics, colors, opacities, bg, tile_offsets, flatten_ids,
                width, height, tile_size, offx, offy):
        args = (means2d.contiguous(), conics.contiguous(), colors.contiguous(),
                opacities.contiguous(), bg.contiguous())
        lists = (tile_offsets.int().contiguous(), flatten_ids.int().contiguous())
        out, tfin, last = ext().jet_forward(*args, width, height, tile_size, *lists, offx, offy)
        ctx.save_for_backward(*args, *lists, tfin, last)
        ctx.meta = (width, height, tile_size, offx, offy)
        ctx.mark_non_differentiable(tfin, last)
        return out, 1.0 - tfin[..., 0], tfin, last

    @staticmethod
    def backward(ctx, v_out, v_alpha, _v_tfin, _v_last):
        means2d, conics, colors, opacities, bg, offsets, ids, tfin, last = ctx.saved_tensors
        width, height, tile_size, offx, offy = ctx.meta
        if v_out is None:
            v_out = torch.zeros(4, height, width, 3, device=means2d.device)
        if v_alpha is None:
            v_alpha = torch.zeros(height, width, device=means2d.device)
        vm, vc, vcol, vo = ext().jet_backward(
            means2d, conics, colors, opacities, bg, width, height, tile_size, offsets, ids,
            offx, offy, tfin, last, v_out.contiguous(), v_alpha.contiguous())
        return vm, vc, vcol, vo, None, None, None, None, None, None, None, None


def rasterize_jets(means2d, conics, colors, opacities, bg, tile_offsets, flatten_ids, width,
                   height, tile_size=16, offset=(0.0, 0.0)):
    """One camera, non-packed. means2d [N,2], conics [N,3], colors [N,3], opacities [N],
    bg [3], tile_offsets [th,tw], flatten_ids [M] (gsplat isect lists).

    Returns jets [4,H,W,3] = (I, dI/dx, dI/dy, d2I/dxdy) in pixel units including the
    background term, alpha [H,W], final transmittance jet [H,W,4] (not differentiable) and
    last_ids. `offset` shifts every pixel centre (used by the finite-difference probe only).
    """
    out, alpha, tfin, last = _JetRasterize.apply(
        means2d, conics, colors, opacities, bg, tile_offsets, flatten_ids, int(width),
        int(height), int(tile_size), float(offset[0]), float(offset[1]))
    return out, alpha, tfin.detach(), last
