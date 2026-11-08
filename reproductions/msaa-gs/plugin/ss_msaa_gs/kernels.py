"""JIT-built MSAA rasterizer (csrc/msaa_raster.cu, adapted from gsplat 1.5.3
RasterizeToPixels3DGS{Fwd,Bwd}.cu via reproductions/linear-splatting) and its autograd wrapper."""

import functools
from pathlib import Path

import torch

CSRC = Path(__file__).resolve().parent / "csrc"

# Standard 4x MSAA rotated-grid sample positions inside the pixel square [0,1]^2
# (D3D10 / OpenGL standard pattern; centroid = pixel centre). The paper gives no positions.
ROTATED_GRID_4X = ((0.375, 0.125), (0.875, 0.375), (0.125, 0.625), (0.625, 0.875))
PIXEL_CENTRE = ((0.5, 0.5),)


@functools.cache
def ext():
    """Build on first use (cached by torch in ~/.cache/torch_extensions) and import."""
    from torch.utils.cpp_extension import load

    return load(
        name="ss_msaa_gs_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "msaa_raster.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


class _MSAARasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means2d, conics, colors, opacities, tile_offsets, flatten_ids, offsets,
                width, height, tile_size):
        args = (means2d.contiguous(), conics.contiguous(), colors.contiguous(),
                opacities.contiguous())
        lists = (tile_offsets.int().contiguous(), flatten_ids.int().contiguous())
        renders, alphas, sample_T, last_ids = ext().msaa_forward(
            *args, offsets, width, height, tile_size, *lists)
        ctx.save_for_backward(*args, *lists, sample_T, last_ids)
        ctx.offsets = offsets
        ctx.meta = (width, height, tile_size)
        return renders, alphas

    @staticmethod
    def backward(ctx, v_renders, v_alphas):
        means2d, conics, colors, opacities, toffs, ids, sample_T, last_ids = ctx.saved_tensors
        width, height, tile_size = ctx.meta
        if v_renders is None:
            v_renders = torch.zeros(height, width, colors.shape[-1], device=means2d.device)
        if v_alphas is None:
            v_alphas = torch.zeros(height, width, 1, device=means2d.device)
        vm, vc, vcol, vo = ext().msaa_backward(
            means2d, conics, colors, opacities, ctx.offsets, width, height, tile_size, toffs, ids,
            sample_T, last_ids, v_renders.contiguous(), v_alphas.contiguous())
        return vm, vc, vcol, vo, None, None, None, None, None, None


def rasterize_msaa(means2d, conics, colors, opacities, tile_offsets, flatten_ids, width, height,
                   offsets=ROTATED_GRID_4X, tile_size=16):
    """One camera, non-packed: means2d [N,2], conics [N,3], colors [N,C] (C in 1,3,4),
    opacities [N], gsplat tile lists. Each pixel (j,i) is alpha-composited independently at the
    sample positions (j + dx, i + dy) for (dx, dy) in `offsets` (paper Eq. 4) and the results are
    averaged (Eq. 5). Returns colours [H,W,C] (no background) and alpha [H,W,1]."""
    off = torch.tensor(offsets, dtype=torch.float32).reshape(-1, 2).contiguous()
    return _MSAARasterize.apply(means2d, conics, colors, opacities, tile_offsets, flatten_ids,
                                off, int(width), int(height), int(tile_size))
