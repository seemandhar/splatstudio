"""C3 quantisation primitives and C4 Morton order, following the official c3dgs code.

Official `scene/gaussian_model.py`: `torch.ao.quantization.FakeQuantize(dtype=torch.qint8)` per
attribute (MovingAverageMinMaxObserver, averaging constant 0.01, per-tensor affine, range
[-128, 127]; observers stay enabled, so every forward updates the moving min/max), and
`FakeQuantizationHalf` (float16 positions, identity backward). Morton code: `splitBy3` /
`mortonEncode` / `_sort_morton` (21 bits per axis).
"""

import torch

QUANTIZED = ("features_dc", "features_rest", "opacity", "scaling", "scaling_factor", "rotation")


def fake_quantizers(device):
    """One official 8-bit min-max fake quantiser per attribute (all but positions)."""
    return {
        name: torch.ao.quantization.FakeQuantize(dtype=torch.qint8).to(device)
        for name in QUANTIZED
    }


class HalfSTE(torch.autograd.Function):
    """Official FakeQuantizationHalf: forward x.half().float(), backward identity."""

    @staticmethod
    def forward(ctx, x):
        return x.half().float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def half_ste(x):
    return HalfSTE.apply(x)


def qparams(fq):
    """(scale float32 [1], zero_point int32 [1]) the quantiser used in its last forward."""
    return fq.scale.detach().clone(), fq.zero_point.detach().clone()


def quantize_int8(x, fq):
    """Official save_npz: torch.quantize_per_tensor(x, fq.scale, fq.zero_point, qint8).int_repr()."""
    return torch.quantize_per_tensor(
        x.detach().float().contiguous(), fq.scale, fq.zero_point, fq.dtype
    ).int_repr()


def dequantize(q, scale, zero_point):
    """Official load_npz: (q - zero_point) * scale."""
    return (q.int() - zero_point) * scale


def _split_by_3(a):
    x = a & 0x1FFFFF  # only the first 21 bits
    x = (x | x << 32) & 0x1F00000000FFFF
    x = (x | x << 16) & 0x1F0000FF0000FF
    x = (x | x << 8) & 0x100F00F00F00F00F
    x = (x | x << 4) & 0x10C30C30C30C30C3
    x = (x | x << 2) & 0x1249249249249249
    return x


def morton_encode(pos):
    """Official mortonEncode on integer grid coordinates [N, 3] (int64)."""
    x, y, z = pos.unbind(-1)
    answer = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
    answer |= _split_by_3(x) | _split_by_3(y) << 1 | _split_by_3(z) << 2
    return answer


def morton_order(xyz):
    """Official _sort_morton permutation of the float32 positions."""
    lo, hi = xyz.min(0).values, xyz.max(0).values
    xyz_q = ((2**21 - 1) * (xyz - lo) / (hi - lo)).long()
    return morton_encode(xyz_q).sort().indices
