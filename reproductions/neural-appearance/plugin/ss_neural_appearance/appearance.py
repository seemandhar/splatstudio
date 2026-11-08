"""Neural view-dependent appearance (paper §4, Eq. 5-7; official ``Appearance/Neural.py`` and
``Appearance/utils.py``) as a PyTorch module = the paper's unfused reference path ("Default w/o
fusion", supp. Table "neural architecture": same quality as the fused tcnn kernel).

c(d) = phi(c0 + A(d)),  phi = ReLU(. + 0.5)                       (Eq. 2)
x    = (gamma_feat(f), gamma_dir(d))  in R^32   (code order: features first)   (Eq. 5)
gamma_feat^k(f) = (sin(2^j f), cos(2^j f))_{j<k},  F = 8, k = 1               (Eq. 6)
gamma_dir(d) = real SH basis of degrees {0,1,2,3} (tcnn sh_enc polynomials)
A(d) = tanh(MLP_theta(x)),  MLP: 32 -> 16 -> 16 -> 3, ReLU, no biases          (Eq. 7)
"""

import math

import torch
from torch import nn

SH_C0 = 0.28209479177387814

DEFAULTS = {
    "feature_dim": 8,  # NEURAL.FEATURE_DIM
    "feature_frequencies": 1,  # NEURAL.FEATURE_N_FREQUENCIES
    "sh_degrees": [0, 1, 2, 3],  # NEURAL.SH_DEGREES
    "n_neurons": 16,  # NEURAL.N_NEURONS
    "n_hidden_layers": 2,  # NEURAL.N_HIDDEN_LAYERS
}


def frequency_encode(features, n_frequencies):
    """Official ``frequency_encode``: per feature sin(2^j x), cos(2^j x) for j < k (tcnn order)."""
    scaled = features[..., None] * torch.exp2(
        torch.arange(n_frequencies, dtype=torch.float32, device=features.device))
    return torch.stack([scaled.sin(), scaled.cos()], dim=-1).flatten(start_dim=-3)


def sh_basis(directions, active_degree, sh_degrees=(0, 1, 2, 3)):
    """Official ``compute_sh_degrees`` restricted to degrees <= 3: the i-th configured degree >= 1
    is zeroed unless i < active_degree; degree 0 (constant SH_C0, the network's bias input) is
    never gated."""
    if any(d > 3 for d in sh_degrees) or list(sh_degrees) != sorted(set(sh_degrees)):
        raise ValueError("sh_degrees must be strictly ascending degrees in [0, 3]")
    x, y, z = directions.unbind(dim=-1)
    xy, xz, yz, x2, y2, z2 = x * y, x * z, y * z, x * x, y * y, z * z
    values = {
        0: lambda: [torch.full_like(x, SH_C0)],
        1: lambda: [-0.48860251190291987 * y, 0.48860251190291987 * z, -0.48860251190291987 * x],
        2: lambda: [
            1.0925484305920792 * xy,
            -1.0925484305920792 * yz,
            0.94617469575755997 * z2 - 0.31539156525251999,
            -1.0925484305920792 * xz,
            0.54627421529603959 * x2 - 0.54627421529603959 * y2,
        ],
        3: lambda: [
            0.59004358992664352 * y * (-3.0 * x2 + y2),
            2.8906114426405538 * xy * z,
            0.45704579946446572 * y * (1.0 - 5.0 * z2),
            0.3731763325901154 * z * (5.0 * z2 - 3.0),
            0.45704579946446572 * x * (1.0 - 5.0 * z2),
            1.4453057213202769 * z * (x2 - y2),
            0.59004358992664352 * x * (-x2 + 3.0 * y2),
        ],
    }
    n_active = active_degree + (sh_degrees[0] == 0)
    columns = []
    for i, degree in enumerate(sh_degrees):
        if i < n_active:
            columns += values[degree]()
        else:
            columns += [torch.zeros_like(x)] * (2 * degree + 1)
    return torch.stack(columns, dim=-1)


def n_inputs(cfg):
    enc = cfg["feature_dim"] * (2 * cfg["feature_frequencies"] if cfg["feature_frequencies"] > 0 else 1)
    return enc + sum(2 * d + 1 for d in cfg["sh_degrees"])


class ResidualMLP(nn.Module):
    """tcnn FullyFusedMLP (no biases, ReLU hidden, linear output) in fp32.

    Init = tcnn default Xavier-uniform on the (16-padded) layer sizes; output layer zeroed so the
    residual is exactly 0 when the first degree activates (official ``Neural.__init__``)."""

    def __init__(self, cfg, generator=None):
        super().__init__()
        d_in, width, depth = n_inputs(cfg), cfg["n_neurons"], cfg["n_hidden_layers"]
        sizes = [d_in] + [width] * depth
        self.weights = nn.ParameterList()
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            pad_in = (fan_in + 15) // 16 * 16
            bound = math.sqrt(6.0 / (pad_in + fan_out))
            w = (torch.rand(fan_out, fan_in, generator=generator) * 2 - 1) * bound
            self.weights.append(nn.Parameter(w))
        self.weights.append(nn.Parameter(torch.zeros(3, width)))

    def forward(self, x):
        for w in self.weights[:-1]:
            x = torch.relu(x @ w.t())
        return x @ self.weights[-1].t()

    def n_params(self):
        return sum(w.numel() for w in self.weights)


def mlp_input(features, directions, active_degree, cfg):
    enc = frequency_encode(features, cfg["feature_frequencies"]) if cfg["feature_frequencies"] > 0 \
        else features
    return torch.cat([enc, sh_basis(directions, active_degree, cfg["sh_degrees"])], dim=-1)


def neural_residual(means, features, campos, mlp, active_degree, cfg):
    """A(d) = tanh(MLP(x)) for every Gaussian; d = normalize(mu - campos) keeps the gradient to mu
    (DIRECTION_GRADIENT: true). Returns None while active_degree == 0 (residual disabled)."""
    if active_degree <= 0:
        return None
    directions = torch.nn.functional.normalize(means - campos, dim=-1)
    return torch.tanh(mlp(mlp_input(features, directions, active_degree, cfg)))
