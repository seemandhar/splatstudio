"""JIT-built WSR rasterizer (csrc/wsr_raster.cu, adapted from gsplat 1.5.3
RasterizeToPixels3DGS{Fwd,Bwd}.cu) and its autograd wrapper."""

import functools
from pathlib import Path

import torch

CSRC = Path(__file__).resolve().parent / "csrc"
ALPHA_THRESHOLD = 1.0 / 255.0
MAX_MAHALANOBIS = 3.33


@functools.cache
def ext():
    """Build on first use (cached by torch in ~/.cache/torch_extensions) and import."""
    from torch.utils.cpp_extension import load

    return load(
        name="ss_sort_free_gs_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "wsr_raster.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


class _WSRRasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means2d, conics, feats, opac, weights, tile_offsets, flatten_ids, width,
                height, tile_size):
        args = (means2d.contiguous(), conics.contiguous(), feats.contiguous(),
                opac.contiguous(), weights.contiguous())
        lists = (tile_offsets.int().contiguous(), flatten_ids.int().contiguous())
        feat_sum, w_sum = ext().wsr_forward(*args, width, height, tile_size, *lists)
        ctx.save_for_backward(*args, *lists)
        ctx.meta = (width, height, tile_size)
        return feat_sum, w_sum

    @staticmethod
    def backward(ctx, v_feat, v_w):
        means2d, conics, feats, opac, weights, offsets, ids = ctx.saved_tensors
        width, height, tile_size = ctx.meta
        if v_feat is None:
            v_feat = torch.zeros(height, width, feats.shape[-1], device=means2d.device)
        if v_w is None:
            v_w = torch.zeros(height, width, 1, device=means2d.device)
        grads = ext().wsr_backward(means2d, conics, feats, opac, weights, width, height,
                                   tile_size, offsets, ids, v_feat.contiguous(), v_w.contiguous())
        return (*grads, None, None, None, None, None)


def rasterize_wsr(means2d, conics, feats, opac, weights, tile_offsets, flatten_ids, width,
                  height, tile_size=16):
    """One camera, non-packed. means2d [N,2], conics [N,3], feats [N,C] (C in 1,3,4),
    opac u [N], weights w [N]; gsplat isect lists. Returns (sum_i f_i a_i w_i [H,W,C],
    sum_i a_i w_i [H,W,1]) with a_i = u_i exp(-1/2 d^T conic d), skipping |a| < 1/255 and
    Mahalanobis distance > 3.33. Order independent."""
    return _WSRRasterize.apply(means2d, conics, feats, opac, weights, tile_offsets, flatten_ids,
                               int(width), int(height), int(tile_size))
