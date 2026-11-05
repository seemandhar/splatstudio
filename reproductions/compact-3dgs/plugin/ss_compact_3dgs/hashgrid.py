"""Multiresolution hash grid with tiny-cuda-nn "HashGrid" conventions (official `self.recolor`).

Official config (scene/gaussian_model.py): n_levels 16, n_features_per_level 2,
log2_hashmap_size 19 (max_hashmap), base_resolution 16, per_level_scale 1.447, linear
interpolation, coherent prime hash. tcnn conventions reproduced here
(include/tiny-cuda-nn/common_device.h, encodings/grid.h):
  scale_l = 2^(l * log2(per_level_scale)) * base_resolution - 1      (grid_scale)
  res_l   = ceil(scale_l) + 1                                         (grid_resolution)
  size_l  = min(next_multiple(res_l^3, 8), 2^log2_hashmap_size)       (offset table)
  pos     = x * scale_l + 0.5, trilinear over the 8 corners of floor(pos)
  index   = dense x + y res + z res^2 while res^3 <= size_l (and res <= 0x659), else
            (x * 1) ^ (y * 2654435761) ^ (z * 805459861) (uint32); index mod size_l
  init    = U(-1e-4, 1e-4)
Parameters are fp32 here (tcnn trains them in fp16; fidelity gap G3).
"""

import math

import numpy as np
import torch

from . import kernels

PRIMES = (1, 2654435761, 805459861)
MAX_DENSE_BASE = 0x659


def level_table(n_levels=16, log2_hashmap_size=19, base_resolution=16, per_level_scale=1.447):
    """(scales float32 [L], resolutions int32 [L], offsets int64 [L+1]) in grid entries."""
    log2_scale = np.float32(math.log2(per_level_scale))
    scales, res, offsets = [], [], [0]
    for level in range(n_levels):
        # float32 arithmetic as in the CUDA host code (exp2f, float multiply, -1.0f)
        s = np.float32(np.exp2(np.float32(level) * log2_scale)) * np.float32(base_resolution)
        s = np.float32(s - np.float32(1.0))
        r = int(math.ceil(float(s))) + 1
        dense = float(r) ** 3
        max_params = (2**32 - 1) // 2
        n = max_params if dense > max_params else r**3
        n = (n + 7) // 8 * 8
        n = min(n, 1 << log2_hashmap_size)
        scales.append(float(s))
        res.append(r)
        offsets.append(offsets[-1] + n)
    return (torch.tensor(scales, dtype=torch.float32), torch.tensor(res, dtype=torch.int32),
            torch.tensor(offsets, dtype=torch.int64))


def reference_encode(x, params, scales, resolutions, offsets, features=2):
    """Independent torch implementation (int64 arithmetic) of the tcnn grid lookup. x in [0,1]^3."""
    out = []
    table = params.reshape(-1, features)
    mask32 = (1 << 32) - 1
    for level in range(len(scales)):
        size = int(offsets[level + 1] - offsets[level])
        r = int(resolutions[level])
        # fmaf(scale, x, 0.5): exact in float64, one rounding to float32
        pos = (x.double() * float(scales[level]) + 0.5).float()
        g = torch.floor(pos)
        frac = pos - g
        g = g.long()
        acc = torch.zeros(x.shape[0], features, dtype=params.dtype, device=x.device)
        for c in range(8):
            w = torch.ones(x.shape[0], dtype=params.dtype, device=x.device)
            local = []
            for d in range(3):
                if (c >> d) & 1:
                    w = w * frac[:, d]
                    local.append(g[:, d] + 1)
                else:
                    w = w * (1 - frac[:, d])
                    local.append(g[:, d])
            if r <= MAX_DENSE_BASE and r**3 <= size:
                idx = (local[0] + local[1] * r + local[2] * r * r) % size
            else:
                h = ((local[0] * PRIMES[0]) & mask32) ^ ((local[1] * PRIMES[1]) & mask32) \
                    ^ ((local[2] * PRIMES[2]) & mask32)
                idx = h % size
            acc = acc + w[:, None] * table[int(offsets[level]) + idx]
        out.append(acc)
    return torch.cat(out, dim=1)


class _HashFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, params, offsets, scales, resolutions):
        x = x.detach().float().contiguous()
        out = kernels.ext().hash_forward(x, params.contiguous(), offsets, scales, resolutions)
        ctx.save_for_backward(x, offsets, scales, resolutions)
        ctx.n_params = params.numel()
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, offsets, scales, resolutions = ctx.saved_tensors
        grad = kernels.ext().hash_backward(x, grad_out.float().contiguous(), ctx.n_params,
                                           offsets, scales, resolutions)
        return None, grad, None, None, None  # positions are detached (official means.detach())


class HashGrid(torch.nn.Module):
    def __init__(self, n_levels=16, n_features=2, log2_hashmap_size=19, base_resolution=16,
                 per_level_scale=1.447, seed=1337):
        super().__init__()
        if n_features != 2:
            raise ValueError("the CUDA kernel implements 2 features per level (official config)")
        scales, res, offsets = level_table(n_levels, log2_hashmap_size, base_resolution,
                                           per_level_scale)
        self.register_buffer("scales", scales, persistent=False)
        self.register_buffer("resolutions", res, persistent=False)
        self.register_buffer("offsets", offsets, persistent=False)
        self.n_levels, self.n_features = n_levels, n_features
        g = torch.Generator().manual_seed(int(seed))
        n = int(offsets[-1]) * n_features
        self.params = torch.nn.Parameter((torch.rand(n, generator=g) * 2 - 1) * 1e-4)

    @property
    def n_output_dims(self):
        return self.n_levels * self.n_features

    def forward(self, x, params=None):
        p = self.params if params is None else params
        return _HashFn.apply(x, p, self.offsets, self.scales, self.resolutions)

    def reference(self, x, params=None):
        p = self.params if params is None else params
        return reference_encode(x, p, self.scales, self.resolutions, self.offsets, self.n_features)
