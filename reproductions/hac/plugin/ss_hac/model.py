"""HAC global learned state: Scaffold-GS MLPs, the binary hash grid, the context MLP and the
entropy model, owned by one nn.Module with its own Adam (the native runtime holds only
Gaussian-owned fields; this follows the EAGLES DecoderBank pattern).

Numerics follow the official code (github.com/YihangChen-ee/HAC): scene/gaussian_model.py,
utils/encodings.py, utils/entropy_models.py, utils/general_utils.py:get_expon_lr_func.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.autograd import Function

from . import kernels

ANCHOR_ROUND_DIGITS = 16
Q_ANCHOR = 1 / (2**ANCHOR_ROUND_DIGITS - 1)
CLAMP_RANGE = 15_000  # use_clamp in utils/encodings.py

# The live bank during training (one per process); None in a fresh process such as ss-eval.
ACTIVE = {"bank": None}


# ----------------------------------------------------------------------------- STE functions
class STEBinary(Function):
    """utils/encodings.py:STE_binary: sign with a clipped straight-through gradient."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = x.clamp(-1, 1)
        return (x >= 0) * 1.0 + (x < 0) * -1.0

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * ((x.clamp(-1, 1) == x) + 0.0)


class STEMultistep(Function):
    """utils/encodings.py:STE_multistep: clamp to mean +- 15000 Q, round to the step Q."""

    @staticmethod
    def forward(ctx, x, q, x_mean=None):
        if x_mean is None:
            x_mean = x.mean()
        x = torch.clamp(x, min=(x_mean - CLAMP_RANGE * q).detach(),
                        max=(x_mean + CLAMP_RANGE * q).detach())
        return torch.round(x / q) * q

    @staticmethod
    def backward(ctx, g):
        return g, None, None


class QuantizeAnchor(Function):
    """utils/encodings.py:Quantize_anchor: 16-bit uniform anchor quantization in the bounds."""

    @staticmethod
    def forward(ctx, anchors, min_v, max_v):
        interval = (max_v - min_v) * Q_ANCHOR + 1e-6
        v = torch.div(anchors - min_v, interval, rounding_mode="floor")
        v = torch.clamp(v, 0, 2**ANCHOR_ROUND_DIGITS - 1)
        return v * interval + min_v, v

    @staticmethod
    def backward(ctx, g, _):
        return g, None, None


class LowBound(Function):
    """utils/entropy_models.py:Low_bound (likelihood floor 1e-6, gradient pass-through rule)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.clamp(x, min=1e-6)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        g1 = g.clone()
        g1[x < 1e-6] = 0
        return g1 * ((x >= 1e-6) | (g < 0)).to(g.dtype)


def gaussian_bits(x, mean, scale, q, x_mean=None):
    """Entropy_gaussian.forward (Eq. 5-6): -log2 of the Gaussian mass of [x-Q/2, x+Q/2]."""
    if x_mean is None:
        x_mean = x.mean()
    x = torch.clamp(x, min=(x_mean - CLAMP_RANGE * q).detach(),
                    max=(x_mean + CLAMP_RANGE * q).detach())
    scale = torch.clamp(scale, min=1e-9)
    normal = torch.distributions.normal.Normal(mean, scale)
    likelihood = torch.abs(normal.cdf(x + 0.5 * q) - normal.cdf(x - 0.5 * q))
    return -torch.log2(LowBound.apply(likelihood))


def binary_bits(binary):
    """utils/encodings.py:get_binary_vxl_size: Bernoulli entropy of a {0,1} tensor + 32 bits."""
    n = binary.numel()
    pos = binary.sum()
    p = torch.clamp(pos / n, 1e-6, 1 - 1e-6)
    return pos * -torch.log2(p) + (n - pos) * -torch.log2(1 - p) + 32


def expon_lr(lr_init, lr_final, max_steps, step_sub=0):
    """utils/general_utils.py:get_expon_lr_func with lr_delay_steps = 0 (official defaults)."""

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        t = np.clip((step - step_sub) / (max_steps - step_sub), 0, 1)
        return float(np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t))

    return helper


# --------------------------------------------------------------------------------- hash grid
class _GridEncode(Function):
    """utils/encodings.py:_grid_encode (all levels, no binary_vxl, PV = 0)."""

    @staticmethod
    def forward(ctx, inputs, embeddings, offsets, resolutions):
        inputs = inputs.contiguous()
        n, dim = inputs.shape
        levels = offsets.shape[0] - 1
        feats = embeddings.shape[1]
        out = torch.empty(levels, n, feats, device=inputs.device, dtype=embeddings.dtype)
        kernels.grid().grid_encode_forward(inputs, embeddings, offsets, resolutions, out, n, dim,
                                           feats, levels, 0, 128, 0.0, None, None, None)
        ctx.save_for_backward(inputs, embeddings, offsets, resolutions)
        ctx.dims = (n, dim, feats, levels)
        return out.permute(1, 0, 2).reshape(n, levels * feats)

    @staticmethod
    def backward(ctx, grad):
        inputs, embeddings, offsets, resolutions = ctx.saved_tensors
        n, dim, feats, levels = ctx.dims
        grad = grad.view(n, levels, feats).permute(1, 0, 2).contiguous()
        grad_emb = torch.zeros_like(embeddings)
        kernels.grid().grid_encode_backward(grad, inputs, embeddings, offsets, resolutions,
                                            grad_emb, n, dim, feats, levels, 0, 128, None, None,
                                            None, None)
        return None, grad_emb, None, None


class GridEncoder(nn.Module):
    """utils/encodings.py:GridEncoder with ste_binary=True (binary {-1,+1} embeddings, §3.2)."""

    def __init__(self, num_dim, n_features, resolutions, log2_hashmap_size):
        super().__init__()
        res = torch.tensor(resolutions, dtype=torch.int32)
        offsets, offset = [], 0
        max_params = 2**log2_hashmap_size
        for r in resolutions:
            level = int(math.ceil(min(max_params, r**num_dim) / 8) * 8)
            offsets.append(offset)
            offset += level
        offsets.append(offset)
        self.num_dim = num_dim
        self.n_features = n_features
        self.output_dim = len(resolutions) * n_features
        self.register_buffer("offsets_list", torch.tensor(offsets, dtype=torch.int32))
        self.register_buffer("resolutions_list", res)
        self.params = nn.Parameter(torch.empty(offset, n_features))
        self.params.data.uniform_(-1e-4, 1e-4)

    def forward(self, x):
        return _GridEncode.apply(x.reshape(-1, self.num_dim), STEBinary.apply(self.params),
                                 self.offsets_list, self.resolutions_list)


class Mix3D2DEncoding(nn.Module):
    """gaussian_model.py:mix_3D2D_encoding: one 3D grid + xy/xz/yz 2D grids, concatenated."""

    def __init__(self, n_features, res3d, log2, res2d, log2_2d):
        super().__init__()
        self.encoding_xyz = GridEncoder(3, n_features, res3d, log2)
        self.encoding_xy = GridEncoder(2, n_features, res2d, log2_2d)
        self.encoding_xz = GridEncoder(2, n_features, res2d, log2_2d)
        self.encoding_yz = GridEncoder(2, n_features, res2d, log2_2d)
        self.output_dim = (self.encoding_xyz.output_dim + 3 * self.encoding_xy.output_dim)

    def forward(self, x):
        a, b, c = torch.chunk(x, 3, dim=-1)
        return torch.cat([self.encoding_xyz(x), self.encoding_xy(torch.cat([a, b], -1)),
                          self.encoding_xz(torch.cat([a, c], -1)),
                          self.encoding_yz(torch.cat([b, c], -1))], dim=-1)

    def grids(self):
        return [self.encoding_xyz, self.encoding_xy, self.encoding_xz, self.encoding_yz]


# ------------------------------------------------------------------------- Scaffold-GS MLPs
def scaffold_mlps(feat_dim, n_offsets):
    """mlp_opacity / mlp_cov / mlp_color of gaussian_model.py (use_feat_bank = False)."""
    d = feat_dim + 3 + 1
    return nn.ModuleDict({
        "mlp_opacity": nn.Sequential(nn.Linear(d, feat_dim), nn.ReLU(True),
                                     nn.Linear(feat_dim, n_offsets), nn.Tanh()),
        "mlp_cov": nn.Sequential(nn.Linear(d, feat_dim), nn.ReLU(True),
                                 nn.Linear(feat_dim, 7 * n_offsets)),
        "mlp_color": nn.Sequential(nn.Linear(d, feat_dim), nn.ReLU(True),
                                   nn.Linear(feat_dim, 3 * n_offsets), nn.Sigmoid()),
    })


RENDER_MLPS = ("mlp_opacity", "mlp_cov", "mlp_color")


class HACBank(nn.Module):
    """All HAC global modules plus the anchor bounds; `step` is the step of the next render."""

    def __init__(self, c, device):
        super().__init__()
        self.c = c
        self.feat_dim, self.n_offsets = c["feat_dim"], c["n_offsets"]
        self.mlps = scaffold_mlps(self.feat_dim, self.n_offsets)
        self.encoding = Mix3D2DEncoding(c["n_features"], tuple(c["resolutions"]), c["log2"],
                                        tuple(c["resolutions_2d"]), c["log2_2d"])
        ctx_out = (self.feat_dim + 6 + 3 * self.n_offsets) * 2 + 1 + 1 + 1
        self.mlp_grid = nn.Sequential(nn.Linear(self.encoding.output_dim, self.feat_dim * 2),
                                      nn.ReLU(True), nn.Linear(self.feat_dim * 2, ctx_out))
        self.to(device)
        self.register_buffer("x_bound_min", torch.zeros(1, 3, device=device))
        self.register_buffer("x_bound_max", torch.ones(1, 3, device=device))
        self.step = 1
        self.baked = False
        t = c["total_steps"]
        lr = c["lr"]
        self.schedules = {
            "mlp_opacity": expon_lr(lr["mlp_opacity"][0], lr["mlp_opacity"][1], t),
            "mlp_cov": expon_lr(lr["mlp_cov"][0], lr["mlp_cov"][1], t),
            "mlp_color": expon_lr(lr["mlp_color"][0], lr["mlp_color"][1], t),
            "encoding_xyz": expon_lr(lr["encoding_xyz"][0], lr["encoding_xyz"][1], t),
            "mlp_grid": expon_lr(lr["mlp_grid"][0], lr["mlp_grid"][1], t),
        }
        groups = [{"params": list(self.mlps[k].parameters()), "name": k, "lr": 0.0}
                  for k in RENDER_MLPS]
        groups.append({"params": list(self.encoding.parameters()), "name": "encoding_xyz",
                       "lr": 0.0})
        groups.append({"params": list(self.mlp_grid.parameters()), "name": "mlp_grid",
                       "lr": 0.0})
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.update_learning_rate(1)

    # --- optimisation ---------------------------------------------------------------------
    def update_learning_rate(self, step):
        for g in self.optimizer.param_groups:
            g["lr"] = self.schedules[g["name"]](step)

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    # --- anchors / context ----------------------------------------------------------------
    @torch.no_grad()
    def update_anchor_bound(self, anchors):
        """gaussian_model.py:update_anchor_bound (min*1.2 / *0.8 depending on the sign)."""
        lo = anchors.min(dim=0, keepdim=True)[0].detach().clone()
        hi = anchors.max(dim=0, keepdim=True)[0].detach().clone()
        lo = torch.where(lo < 0, lo * 1.2, lo * 0.8)
        hi = torch.where(hi > 0, hi * 1.2, hi * 0.8)
        self.x_bound_min.copy_(lo)
        self.x_bound_max.copy_(hi)

    def quantize_anchor(self, anchors):
        return QuantizeAnchor.apply(anchors, self.x_bound_min, self.x_bound_max)

    def context(self, anchors):
        """Hash-grid context (Eq. 2) -> per-anchor (mu, sigma) for feat/scaling/offsets and the
        adaptive quantization steps Q = Q0 (1 + tanh(r)) (Eq. 4)."""
        x = (anchors - self.x_bound_min) / (self.x_bound_max - self.x_bound_min)
        out = self.mlp_grid(self.encoding(x))
        f, k = self.feat_dim, 3 * self.n_offsets
        (mean, scale, mean_s, scale_s, mean_o, scale_o, qf, qs, qo) = torch.split(
            out, [f, f, 6, 6, k, k, 1, 1, 1], dim=-1)
        q0 = self.c["q0"]
        return {
            "feat": (mean, scale, q0["feat"] * (1 + torch.tanh(qf))),
            "scaling": (mean_s, scale_s, q0["scaling"] * (1 + torch.tanh(qs))),
            "offsets": (mean_o, scale_o, q0["offsets"] * (1 + torch.tanh(qo))),
        }

    def hash_embeddings(self):
        """Binary hash-grid embeddings in {-1, +1} (get_encoding_params)."""
        return torch.cat([STEBinary.apply(g.params) for g in self.encoding.grids()], dim=0)

    def hash_bits(self):
        return binary_bits((self.hash_embeddings() + 1) / 2)

    def mlp_bits(self):
        """get_mlp_size: 32-bit floats of the stored MLPs (render MLPs + context MLP)."""
        n = sum(p.numel() for p in self.mlps.parameters())
        n += sum(p.numel() for p in self.mlp_grid.parameters())
        return 32 * n
