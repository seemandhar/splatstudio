"""Appearance modeling of WildGaussians (paper Sec. 3.2, Eq. 4-5; official method.py
`EmbeddingModel`, `_get_fourier_features`)."""

import math

import torch
from torch import nn

C0 = 0.28209479177387814  # SH band-0 constant (official eval_sh C0)


def fourier_features(xyz, num_freqs=4):
    """_get_fourier_features: centre, divide by the per-axis 0.97 quantile of |x| (-> [0, 1] after
    *0.5 + 0.5), then sin(2*pi*2^k*p + {0, pi/2}), k = 0..num_freqs-1, laid out per axis
    [x: k0 sin, k0 cos, k1 sin, ...]. Equals the paper's sin/cos(pi p 2^m), m = 1..num_freqs."""
    xyz = xyz.float()
    xyz = xyz - xyz.mean(dim=0, keepdim=True)
    xyz = xyz / torch.quantile(xyz.abs(), 0.97, dim=0) * 0.5 + 0.5
    freqs = torch.repeat_interleave(
        2 ** torch.linspace(0, num_freqs - 1, num_freqs, dtype=xyz.dtype, device=xyz.device), 2)
    offsets = torch.tensor([0, 0.5 * math.pi] * num_freqs, dtype=xyz.dtype, device=xyz.device)
    feat = xyz[..., None] * freqs[None, None] * 2 * math.pi + offsets[None, None]
    return torch.sin(feat).reshape(len(xyz), -1)


class AppearanceMLP(nn.Module):
    """EmbeddingModel (appearance_model_sh False): input [SH DC coeffs (3), g_i, e_j]."""

    def __init__(self, image_dim=32, gaussian_dim=24, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(image_dim + 3 + gaussian_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 6))

    def forward(self, gembedding, aembedding, features):
        """features [N, 16*3] flattened SH (coefficient-major, channel-minor, already clamped at 1).
        Returns toned SH features: features * mul (all coefficients) + offset / C0 (DC only),
        (offset, mul) = 0.01 * mlp(...) (official: no +1 on mul, see paper.md Sec. 7.1)."""
        color = features[..., :3]
        inp = torch.cat((color, gembedding, aembedding.expand(len(color), -1)), dim=-1)
        offset, mul = torch.split(self.mlp(inp) * 0.01, [3, 3], dim=-1)
        offset = torch.cat((offset / C0, torch.zeros_like(features[..., 3:])), dim=-1)
        mul = mul.repeat(1, features.shape[-1] // 3)
        return features * mul + offset
