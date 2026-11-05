"""Softmax / alpha / mode depth rasterization (paper §3.1, Eq. 1-3; supp. §A) on top of gsplat.

The CUDA kernels in ``csrc/sgs_depth.cu`` are JIT-compiled on first use (cached by torch in
~/.cache/torch_extensions). They consume gsplat's projection outputs (``info`` of
``gsplat.rasterization``), so gradients reach the Gaussians through gsplat's projection
backward. See the .cu header for the exact numerics (official forward.cu / backward.cu).
"""

from pathlib import Path

import torch

_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        src = Path(__file__).with_name("csrc") / "sgs_depth.cu"
        _EXT = load(name="ss_sparsegs_depth", sources=[str(src)],
                    extra_cuda_cflags=["-O3"], verbose=False)
    return _EXT


def _flat(info):
    if info["means2d"].dim() != 3 or info["means2d"].shape[0] != 1:
        raise ValueError("SparseGS depth needs gsplat packed=False single-camera outputs")
    if int(info["tile_size"]) != 16:
        raise ValueError("SparseGS depth kernel assumes gsplat tile_size 16")
    return (info["means2d"][0], info["conics"][0], info["opacities"][0], info["depths"][0],
            info["isect_offsets"][0].contiguous().int(), info["flatten_ids"].contiguous().int())


class _SoftmaxDepth(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means2d, conics, opac, depths, offsets, flatten_ids, W, H, beta):
        args = [t.contiguous().float() for t in (means2d, conics, opac, depths)]
        out = ext().depth_forward(*args, offsets, flatten_ids, W, H, beta)
        soft, alpha, mode, D, Z, SN, SD, ids, ids_off = out
        ctx.save_for_backward(*args, offsets, flatten_ids, D, Z, SN, SD)
        ctx.dims = (W, H, beta)
        ctx.mark_non_differentiable(alpha, mode, ids, ids_off)
        return soft, alpha, mode, ids, ids_off

    @staticmethod
    def backward(ctx, g_soft, *_):
        m, c, o, d, offsets, fids, D, Z, SN, SD = ctx.saved_tensors
        W, H, beta = ctx.dims
        gm, gc, go, gd = ext().depth_backward(m, c, o, d, offsets, fids, W, H, beta, D, Z, SN,
                                              SD, g_soft.contiguous())
        return gm, gc, go, gd, None, None, None, None, None


def render_depths(info, beta):
    """Returns dict: softmax [H,W] (log depth, differentiable), alpha [H,W], mode [H,W],
    mode_ids [2,H,W] (first contributor, mode; true list index), mode_ids_official
    [2,H,W] (official forward.cu index = range.x + round*256 + thread rank)."""
    W, H = int(info["width"]), int(info["height"])
    soft, alpha, mode, ids, ids_off = _SoftmaxDepth.apply(*_flat(info), W, H, float(beta))
    return {"softmax": soft, "alpha": alpha, "mode": mode, "mode_ids": ids,
            "mode_ids_official": ids_off}


@torch.no_grad()
def prune_marks(info, px, py, lo, hi, power_thresh, n_gauss):
    """Mark every flatten-list entry in [lo, hi] whose power at pixel (px, py) > thresh."""
    means2d, conics, _, _, _, fids = _flat(info)
    i32 = lambda t: t.to(device=means2d.device, dtype=torch.int32).contiguous()
    return ext().prune_mark(means2d.contiguous().float(), conics.contiguous().float(), fids,
                            i32(px), i32(py), i32(lo), i32(hi), float(power_thresh),
                            int(n_gauss))
