"""JIT-built Vol3DGS rasterizer (csrc/vol_raster.cu, adapted from gsplat 1.5.3
RasterizeToPixels3DGS{Fwd,Bwd}.cu with the official slang `evaluate_splat_volr` alpha) and its
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
        name="ss_vol3dgs_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "vol_raster.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


class _VolRasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means_c, precis6, kappa, colors, intrinsics, tile_offsets, flatten_ids,
                width, height, tile_size):
        mk = torch.cat((means_c, kappa[:, None]), dim=1).contiguous()
        pa = precis6[:, :3].contiguous()
        pb = precis6[:, 3:].contiguous()
        colors = colors.contiguous()
        lists = (tile_offsets.int().contiguous(), flatten_ids.int().contiguous())
        renders, alphas, last_ids = ext().vol_forward(
            mk, pa, pb, colors, *intrinsics, width, height, tile_size, *lists)
        ctx.save_for_backward(mk, pa, pb, colors, *lists, alphas, last_ids)
        ctx.meta = (tuple(intrinsics), width, height, tile_size)
        ctx.mark_non_differentiable(last_ids)
        return renders, alphas, last_ids

    @staticmethod
    def backward(ctx, v_renders, v_alphas, _v_last):
        mk, pa, pb, colors, offsets, ids, alphas, last_ids = ctx.saved_tensors
        intrinsics, width, height, tile_size = ctx.meta
        if v_renders is None:
            v_renders = torch.zeros(height, width, colors.shape[-1], device=mk.device)
        if v_alphas is None:
            v_alphas = torch.zeros(height, width, 1, device=mk.device)
        v_mk, v_pa, v_pb, v_col = ext().vol_backward(
            mk, pa, pb, colors, *intrinsics, width, height, tile_size, offsets, ids, alphas,
            last_ids, v_renders.contiguous(), v_alphas.contiguous())
        return (v_mk[:, :3], torch.cat((v_pa, v_pb), dim=1), v_mk[:, 3], v_col,
                None, None, None, None, None, None)


def rasterize_vol(means_c, precis6, kappa, colors, K, tile_offsets, flatten_ids, width, height,
                  tile_size=16):
    """One camera, non-packed. means_c [N,3] camera-space means, precis6 [N,6] camera-space
    inverse covariance (a00 a01 a02 a11 a12 a22), kappa [N] densities, colors [N,C] (C in 1,3,4),
    tile_offsets [th,tw] and flatten_ids [M] from gsplat's isect_tiles (depth-sorted).

    Returns accumulated colours [H,W,C] (no background) and alpha = 1 - T [H,W,1]."""
    intrinsics = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    renders, alphas, _ = _VolRasterize.apply(
        means_c.float(), precis6.float(), kappa.float(), colors.float(), intrinsics,
        tile_offsets, flatten_ids, int(width), int(height), int(tile_size))
    return renders, alphas
