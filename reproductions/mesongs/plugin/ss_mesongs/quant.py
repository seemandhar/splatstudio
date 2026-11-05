"""C5 block quantisation (official `utils/quant_utils.py`: `split_length`, `VanillaQuan`,
`calcScaleZeroPoint`, `round_pass`; `gaussian_renderer.seg_quant_ave`; `scene/gaussian_model.py`:
`quantize_tensor`, `dequantize_tensor`, `torch_vanilla_quant_ave`, `init_qas`).

`BlockQuantizer` holds the running ranges of all `channels x n_block` VanillaQuan modules of one
attribute group (7 RAHT AC channels or 3 log-scale channels) and applies them vectorised:
per block, `max <- max(max, x.max()).clamp(min=0)`, `min <- min(min, x.min()).clamp(max=0)`
(updated in every forward, as the official modules are), `S = (max - min) / (2^b - 1)`,
`Z = round(clamp(2^b - 1 - max / S, 0, 2^b - 1))`, forward
`S * (round_ste(clamp(Z + x / S, 0, 2^b - 1)) - Z)` (gradient 1 inside the clamp range, 0 outside).
"""

import torch


def split_length(length, n):
    base = length / n
    floor = int(base)
    remainder = length - (floor * n)
    return [floor + 1] * remainder + [floor] * (n - remainder)


def round_pass(x):
    y = x.round()
    return (y - x).detach() + x


def calc_scale_zero_point(min_val, max_val, num_bits=8):
    qmin, qmax = 0.0, 2.0**num_bits - 1.0
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = (qmax - max_val / scale).clamp(qmin, qmax).round()
    return scale, zero_point


def quantize_tensor(x, scale, zero_point, num_bits=8):
    """Official quantize_tensor (unsigned): clamp(Z + x / S, 0, 2^b - 1).round()."""
    return (zero_point + x / scale).clamp(0.0, 2.0**num_bits - 1.0).round()


def dequantize_tensor(q, scale, zero_point):
    return scale * (q - zero_point)


class BlockQuantizer:
    """Running-range 8-bit block quantisers for a [L, channels] tensor split along L."""

    def __init__(self, length, channels, n_block, bits, device):
        self.length, self.channels, self.n_block, self.bits = length, channels, n_block, bits
        sizes = torch.tensor(split_length(length, n_block), device=device)
        self.block_of = torch.repeat_interleave(torch.arange(n_block, device=device), sizes)
        self.max_val = None  # [channels, n_block]; official buffers start empty
        self.min_val = None

    def _block_minmax(self, x):
        xt = x.detach().t()  # [channels, L]
        idx = self.block_of.expand(self.channels, -1)
        shape = (self.channels, self.n_block)
        bmax = torch.full(shape, -torch.inf, device=x.device).scatter_reduce(
            1, idx, xt, "amax", include_self=True)
        bmin = torch.full(shape, torch.inf, device=x.device).scatter_reduce(
            1, idx, xt, "amin", include_self=True)
        return bmin, bmax

    @torch.no_grad()
    def update(self, x):
        bmin, bmax = self._block_minmax(x)
        if self.max_val is None:
            self.max_val, self.min_val = bmax, bmin
        else:
            self.max_val = torch.maximum(self.max_val, bmax)
            self.min_val = torch.minimum(self.min_val, bmin)
        self.max_val = self.max_val.clamp(min=0)
        self.min_val = self.min_val.clamp(max=0)

    def qparams(self):
        """(scale, zero_point), each [channels, n_block]."""
        return calc_scale_zero_point(self.min_val, self.max_val, self.bits)

    def expand(self, t):
        """[channels, n_block] -> [L, channels] per-element values."""
        return t[:, self.block_of].t()

    def __call__(self, x, update=True):
        if x.shape != (self.length, self.channels):
            raise ValueError(f"block quantiser built for {(self.length, self.channels)}, "
                             f"got {tuple(x.shape)}")
        if update:
            self.update(x)
        scale, zp = (self.expand(t) for t in self.qparams())
        q = round_pass(torch.clamp(zp + x / scale, 0.0, 2.0**self.bits - 1.0))
        return scale * (q - zp)

    @torch.no_grad()
    def encode(self, x):
        """Integer codes [L, channels] with the current ranges (official torch_vanilla_quant_ave)."""
        scale, zp = (self.expand(t) for t in self.qparams())
        return quantize_tensor(x, scale, zp, self.bits)

    def trans(self):
        """Official trans_array entries: per channel, per block (scale, zero_point)."""
        scale, zp = self.qparams()
        return torch.stack((scale, zp), -1).reshape(-1)

    def state(self):
        return {"min": self.min_val, "max": self.max_val}
