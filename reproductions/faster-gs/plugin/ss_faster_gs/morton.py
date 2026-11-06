"""Morton (z-order) codes, a torch port of NeRFICG `CudaUtils/MortonEncoding/morton_encoding.cu`
used by Faster-GS `Gaussians.apply_morton_ordering` (paper §3.3, locality-preserving densification).
"""

import torch

_MASKS = (
    (32, 0x1F00000000FFFF),
    (16, 0x1F0000FF0000FF),
    (8, 0x100F00F00F00F00F),
    (4, 0x10C30C30C30C30C3),
    (2, 0x1249249249249249),
)


def _split_by_3(a):
    x = a & 0x1FFFFF
    for shift, mask in _MASKS:
        x = (x | (x << shift)) & mask
    return x


@torch.no_grad()
def morton_codes(positions):
    """int64 codes: normalise by bbox min and the largest bbox side, saturate, 21 bits per axis
    (truncation of x * (2^21 - 1)), interleave x | y << 1 | z << 2."""
    positions = positions.float()
    lo = positions.amin(dim=0)
    size = (positions.amax(dim=0) - lo).amax()
    normalized = ((positions - lo) * (1.0 / size)).clamp(0.0, 1.0)
    q = (normalized * 2097151.0).to(torch.int64)
    return _split_by_3(q[:, 0]) | (_split_by_3(q[:, 1]) << 1) | (_split_by_3(q[:, 2]) << 2)


def morton_order(positions):
    return torch.argsort(morton_codes(positions))
