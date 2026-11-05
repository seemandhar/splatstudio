"""EAGLES latent decoders (global learned state) and the process-local slot the renderer reads.

Numerics follow the official `compress/decoders.py` (StraightThrough, get_dft_matrix,
DecoderLayer, LatentDecoder with `num_layers_dec = 0`, `use_shift = 1`, `latent_norm = 'none'`
so `div = 1`) and `scene/gaussian_model_sq.py` (latent init by decoder inversion, latent LR
divided by the decoder-scale norm, per-decoder Adam LRs, eps 1e-15).

The SplatStudio runtime has no parameter type for global learned modules, so the density owns a
`DecoderBank` (decoders + their Adam), registers it in `ACTIVE`, and the renderer decodes the
per-Gaussian latents through it while training. At the final step the decoded model is baked
into the base Gaussian fields (`DecoderBank.baked`), after which renderers render those fields.
"""

import math

import torch
from torch import nn

EPSILON = 1e-6  # official decoders.epsilon (1x1 inversion)

# Process-local registry: set by EaglesDensity.initialize, read by EaglesRenderer.render.
ACTIVE = {"bank": None}

# attribute -> base field it decodes into, the latent field, and the decoded tensor shape
TARGETS = {"shN": ("shN", "q_shN"), "rot": ("quats", "q_rot"), "opa": ("opacities", "q_opa")}
OUT_DIM = {"shN": 45, "rot": 4, "opa": 1}

# Effective official values. configs/efficient-3dgs.yaml lists `*_ldecs_lr` (0.005/0.01/0.3) and
# `*_lr_scaling` (0.1) under `quantize_params`, but train_eval.py hands only `opt_params` to
# OptimizationParams (which owns those keys) and QuantizeParams has no such attributes, so the
# official run uses the OptimizationParams defaults: decoder LR 1e-4, latent LR scale 1.0, which
# is exactly paper supp. Table 1. Latent dim, decoder type/std and scale norm do reach the code.
ATTRIBUTES = {
    "shN": {"latent_dim": 16, "matrix": "dft", "std": 0.0005, "decoder_lr": 1e-4,
            "base_lr": 0.0025 / 20.0, "lr_scaling": 1.0, "scale_norm": "div"},
    "rot": {"latent_dim": 8, "matrix": "sq", "std": 0.01, "decoder_lr": 1e-4,
            "base_lr": 0.001, "lr_scaling": 1.0, "scale_norm": "div"},
    "opa": {"latent_dim": 1, "matrix": "sq", "std": 0.5, "decoder_lr": 1e-4,
            "base_lr": 0.05, "lr_scaling": 1.0, "scale_norm": "none"},
}


class StraightThrough(torch.autograd.Function):
    """round() forward, identity backward (official compress/decoders.py:StraightThrough)."""

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def dft_matrix(conv_dim, channels):
    """Official get_dft_matrix: DCT-II rows, F[i,j] = cos(pi/k (i+1/2) j)/sqrt(k) * (sqrt2, j>0)."""
    dft = torch.zeros(conv_dim, channels)
    for i in range(conv_dim):
        for j in range(channels):
            dft[i, j] = math.cos(torch.pi / channels * (i + 0.5) * j) / math.sqrt(channels)
            dft[i, j] = dft[i, j] * (math.sqrt(2) if j > 0 else 1)
    return dft


class DecoderLayer(nn.Module):
    """Linear decoder a = q S + b ('sq') or a = (q F) * s + b ('dft', F fixed)."""

    def __init__(self, in_features, out_features, matrix):
        super().__init__()
        if matrix not in ("sq", "dft"):
            raise ValueError(f"unknown decoder matrix {matrix}")
        self.in_features, self.out_features, self.matrix = in_features, out_features, matrix
        if matrix == "dft":
            self.register_buffer("dft", dft_matrix(in_features, out_features))
            self.scale = nn.Parameter(torch.empty(1, out_features))
        else:
            self.scale = nn.Parameter(torch.empty(in_features, out_features))
        self.shift = nn.Parameter(torch.empty(1, out_features))

    def reset_parameters(self, std, generator=None):
        with torch.no_grad():
            self.scale.normal_(0.0, std, generator=generator)
            self.shift.zero_()

    def forward(self, x):
        if self.matrix == "dft":
            return torch.matmul(x, self.dft) * self.scale + self.shift
        return torch.matmul(x, self.scale) + self.shift

    @torch.no_grad()
    def invert(self, output):
        """Official DecoderLayer.invert (least squares; 1x1 division with epsilon).

        Solved on the CPU with the SVD driver: the rotation system (4 x 8) is underdetermined and
        gets the minimum-norm solution, as LAPACK least squares does.
        """
        if self.in_features == 1 and self.out_features == 1:
            return (output - self.shift) / (self.scale + EPSILON)
        if self.matrix == "dft":
            a, rhs = self.dft.T, ((output - self.shift) / self.scale).T
        else:
            a, rhs = self.scale.T, (output - self.shift).T
        sol = torch.linalg.lstsq(a.detach().cpu().double(), rhs.detach().cpu().double(),
                                 driver="gelsd").solution
        return sol.T.to(device=output.device, dtype=output.dtype).contiguous()

    def size_bits(self):
        """Official LatentDecoder.size(): 32 bits per parameter incl. the fixed DFT matrix and the
        `div` vector (latent_dim ones)."""
        n = self.scale.numel() + self.shift.numel() + self.in_features
        if self.matrix == "dft":
            n += self.dft.numel()
        return 32 * n


def decode_latent(decoder, q):
    """a = D(STE(q)) (paper Eq. 1); div = 1 (latent_norm 'none')."""
    return decoder(StraightThrough.apply(q))


def shape_decoded(name, a):
    if name == "shN":
        return a.reshape(a.shape[0], 15, 3)  # official reshape (N, 15, 3): coefficient-major
    if name == "opa":
        return a.reshape(a.shape[0])
    return a


class DecoderBank(nn.Module):
    def __init__(self, attributes=None, seed=0):
        super().__init__()
        self.attributes = {k: dict(v) for k, v in (attributes or ATTRIBUTES).items()}
        if set(self.attributes) != set(TARGETS):
            raise ValueError(f"attributes must be exactly {sorted(TARGETS)}")
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        self.decoders = nn.ModuleDict()
        for name, a in self.attributes.items():
            layer = DecoderLayer(int(a["latent_dim"]), OUT_DIM[name], a["matrix"])
            layer.reset_parameters(float(a["std"]), g)
            self.decoders[name] = layer
        self.optimizer = None
        self.baked = False
        self.scale = 1.0  # progressive render scale of the next training render

    def build_optimizer(self):
        groups = [{"params": list(self.decoders[n].parameters()), "lr": float(a["decoder_lr"]),
                   "name": f"ldec_{n}"} for n, a in self.attributes.items()]
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)

    @property
    def fields(self):
        return [TARGETS[n][1] for n in self.attributes]

    def decode(self, params, name):
        return shape_decoded(name, decode_latent(self.decoders[name], params[TARGETS[name][1]]))

    def decode_all(self, params):
        return {TARGETS[n][0]: self.decode(params, n) for n in self.attributes}

    def latent_lr(self, name):
        """Official update_learning_rate: lr_3dgs * lr_scaling (/ ||scale||_F if 'div')."""
        a = self.attributes[name]
        lr = float(a["base_lr"]) * float(a["lr_scaling"])
        if a["scale_norm"] == "div":
            lr /= float(self.decoders[name].scale.detach().norm())
        return lr

    @torch.no_grad()
    def initial_latents(self, n, quats, opacity_logits):
        """Supp. Eq. 1, q = D^{-1}(a): SH rest zeros (official), rotation / opacity by inversion."""
        device = quats.device
        return {
            "q_shN": torch.zeros(n, int(self.attributes["shN"]["latent_dim"]), device=device),
            "q_rot": self.decoders["rot"].invert(quats),
            "q_opa": self.decoders["opa"].invert(opacity_logits.reshape(n, 1)),
        }

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def state_tensors(self):
        return {k: v.detach().cpu().clone() for k, v in self.decoders.state_dict().items()}
