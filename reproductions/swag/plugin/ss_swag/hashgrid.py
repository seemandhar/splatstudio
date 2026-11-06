"""Multi-resolution hash encoding (Instant-NGP / tiny-cuda-nn "HashGrid") in pure PyTorch.

SWAG §5.1 / supp. Tab. 7: table size T = 2^19, L = 12 levels, N_min = 16, N_max = 2048; 2 features
per level (input dim 51 = 3 + 24 + 24). Per-level resolution floor(N_min * b^l) with
b = exp((ln N_max - ln N_min) / (L - 1)); a level whose (res+1)^3 grid fits in T is indexed
densely, otherwise with the spatial hash XOR(x*1, y*2654435761, z*805459861) mod T. Trilinear
interpolation of the 8 corners. Table init U(-1e-4, 1e-4) (tcnn). Inputs are in [0, 1]^3.
"""

import math

import torch
from torch import nn

PRIMES = (1, 2654435761, 805459861)


def level_resolutions(levels, n_min, n_max):
    b = math.exp((math.log(n_max) - math.log(n_min)) / max(levels - 1, 1))
    return [math.floor(n_min * b**lvl) for lvl in range(levels)]


def corner_index(corner, res, table_size):
    """corner: int64 [..., 3] grid coordinates in [0, res]; returns int64 table rows."""
    if (res + 1) ** 3 <= table_size:
        return corner[..., 0] + (res + 1) * (corner[..., 1] + (res + 1) * corner[..., 2])
    h = corner[..., 0] * PRIMES[0]
    h = torch.bitwise_xor(h, corner[..., 1] * PRIMES[1])
    h = torch.bitwise_xor(h, corner[..., 2] * PRIMES[2])
    return torch.remainder(h, table_size)


class HashGrid(nn.Module):
    def __init__(self, levels=12, features=2, log2_table=19, n_min=16, n_max=2048, seed=0):
        super().__init__()
        self.levels, self.features = int(levels), int(features)
        self.table_size = 2 ** int(log2_table)
        self.resolutions = level_resolutions(self.levels, n_min, n_max)
        self.sizes = [min((r + 1) ** 3, self.table_size) for r in self.resolutions]
        g = torch.Generator().manual_seed(int(seed))
        # One concatenated table [sum(sizes), F]; level l owns rows offsets[l]:offsets[l+1].
        self.table = nn.Parameter((torch.rand(sum(self.sizes), self.features, generator=g) * 2
                                   - 1) * 1e-4)
        offsets = [0]
        for s in self.sizes:
            offsets.append(offsets[-1] + s)
        self.register_buffer("level_offsets", torch.tensor(offsets, dtype=torch.long),
                             persistent=False)
        self.register_buffer("level_res", torch.tensor(self.resolutions, dtype=torch.int32),
                             persistent=False)
        self.use_cuda = True
        offs = torch.tensor([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)])
        self.register_buffer("offsets", offs, persistent=False)

    @property
    def out_dim(self):
        return self.levels * self.features

    @property
    def tables(self):
        """Per-level views of the concatenated table."""
        o = self.level_offsets.tolist()
        return [self.table[o[i]:o[i + 1]] for i in range(self.levels)]

    def forward(self, x):
        """x: [N, 3] in [0, 1] -> [N, L * F] (CUDA kernel on GPU, PyTorch otherwise)."""
        if self.use_cuda and x.is_cuda and self.features == 2:
            from .hash_cuda import HashEncode

            return HashEncode.apply(x, self.table, self.level_offsets, self.level_res,
                                    self.table_size, self.features)
        return self.forward_torch(x)

    def forward_torch(self, x):
        x = x.clamp(0.0, 1.0)
        outs = []
        for res, table in zip(self.resolutions, self.tables):
            pos = x * res
            base = pos.floor().clamp(max=res - 1)
            frac = pos - base  # [N, 3]
            corners = base.long()[:, None, :] + self.offsets[None]  # [N, 8, 3]
            idx = corner_index(corners, res, self.table_size)  # [N, 8]
            o = self.offsets[None].to(x.dtype)
            w = torch.where(o > 0, frac[:, None, :], 1 - frac[:, None, :]).prod(-1)  # [N, 8]
            outs.append((table[idx] * w[..., None]).sum(1))
        return torch.cat(outs, dim=-1)
