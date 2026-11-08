"""JIT-built unit-sphere rasterizer (csrc/sphere_raster.cu) and its autograd wrapper.

The CUDA file ports the per-pixel tangent-plane evaluation of the official op43dgs pinhole
rasterizer (forward.cu / backward.cu `renderCUDA`) onto gsplat 1.5.3 tile lists, adapting the
tile traversal of reproductions/linear-splatting (itself from gsplat RasterizeToPixels3DGS).
"""

import functools
from pathlib import Path

import torch

CSRC = Path(__file__).resolve().parent / "csrc"
MODES = {"optimal": 0, "ewa": 1}


@functools.cache
def ext():
    """Build on first use (cached by torch in ~/.cache/torch_extensions) and import."""
    from torch.utils.cpp_extension import load

    return load(
        name="ss_optimal_gs_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "sphere_raster.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


class _SphereRasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means2d, conics, colors, opacities, tile_offsets, flatten_ids, width,
                height, tile_size, cam, mode, alpha_max):
        args = (means2d.contiguous(), conics.contiguous(), colors.contiguous(),
                opacities.contiguous())
        lists = (tile_offsets.int().contiguous(), flatten_ids.int().contiguous())
        renders, alphas, last_ids = ext().sphere_forward(
            *args, width, height, tile_size, *lists, cam, mode, alpha_max)
        ctx.save_for_backward(*args, *lists, alphas, last_ids)
        ctx.meta = (width, height, tile_size, cam, mode, alpha_max)
        ctx.mark_non_differentiable(last_ids)
        return renders, alphas, last_ids

    @staticmethod
    def backward(ctx, v_renders, v_alphas, _v_last):
        means2d, conics, colors, opacities, offsets, ids, alphas, last_ids = ctx.saved_tensors
        width, height, tile_size, cam, mode, alpha_max = ctx.meta
        if v_renders is None:
            v_renders = torch.zeros(height, width, colors.shape[-1], device=means2d.device)
        if v_alphas is None:
            v_alphas = torch.zeros(height, width, 1, device=means2d.device)
        vm, vc, vcol, vo = ext().sphere_backward(
            means2d, conics, colors, opacities, width, height, tile_size, offsets, ids, cam,
            mode, alpha_max, alphas, last_ids, v_renders.contiguous(), v_alphas.contiguous())
        return vm, vc, vcol, vo, None, None, None, None, None, None, None, None


def rasterize_sphere(means2d, conics, colors, opacities, tile_offsets, flatten_ids, width,
                     height, K, focal_floor=512.0, mode="optimal", tile_size=16,
                     alpha_max=0.99):
    """One camera: means2d [N,2] (pixels), conics [N,3] (tangent-plane units for 'optimal',
    pixels for 'ewa'), colors [N,C] (C in 3, 4), opacities [N], gsplat tile lists.

    Returns accumulated colours [H,W,C] (no background) and alpha [H,W,1]."""
    fx, fy, cx, cy = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    cam = [fx, fy, cx, cy, max(focal_floor, fx), max(focal_floor, fy)]
    renders, alphas, _ = _SphereRasterize.apply(
        means2d, conics, colors, opacities, tile_offsets, flatten_ids, int(width), int(height),
        int(tile_size), cam, MODES[mode], float(alpha_max))
    return renders, alphas
