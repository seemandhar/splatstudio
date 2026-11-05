"""ContextGS global learned state and entropy-model primitives.

One nn.Module (`ContextBank`, own Adam) owns everything that is not per-anchor: the Scaffold-GS render MLPs, the
factorized hyperprior density (compressai EntropyBottleneck port), the three per-level context MLPs (`mlp_grid`),
the anchor bounds and the level scales. The renderer reads the live bank through `ACTIVE` (EAGLES/HAC DecoderBank
pattern; copied from the HAC native port, reproductions/hac/plugin/ss_hac/model.py).

Numerics follow the official code (github.com/wyf0912/ContextGS @ eff8d19): scene/gaussian_model.py,
utils/encodings.py, utils/entropy_models.py, utils/general_utils.py:get_expon_lr_func, and compressai 1.2.x
entropy_models.py:EntropyBottleneck (BSD-3-Clause-Clear, InterDigital), which ContextGS uses for the hyperprior.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

ANCHOR_ROUND_DIGITS = 16
Q_ANCHOR = 1 / (2**ANCHOR_ROUND_DIGITS - 1)
CLAMP_RANGE = 15_000  # use_clamp in utils/encodings.py

# The live bank during training (one per process); None in a fresh process such as ss-eval.
ACTIVE = {"bank": None}


# ----------------------------------------------------------------------------- STE functions
class STEMultistep(Function):
    """utils/encodings.py:STE_multistep (ContextGS form): clamp to +-15000 Q, round to Q, identity grad."""

    @staticmethod
    def forward(ctx, x, q):
        x = torch.clamp(x, min=(-CLAMP_RANGE * q).detach(), max=(CLAMP_RANGE * q).detach())
        return torch.round(x / q) * q

    @staticmethod
    def backward(ctx, g):
        return g, None


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
    """utils/entropy_models.py:Low_bound: floor 1e-6, gradient only where x >= 1e-6."""

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


class LowerBound(Function):
    """compressai.ops.LowerBound: max(x, bound); gradient passes if x >= bound or grad < 0."""

    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x)
        ctx.bound = bound
        return torch.clamp(x, min=bound)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * ((x >= ctx.bound) | (g < 0)).to(g.dtype), None


def gaussian_bits(x, mean, scale, q, x_mean):
    """Entropy_gaussian.forward (Eq. 7): -log2 of the Gaussian mass of [x-Q/2, x+Q/2]."""
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


# ------------------------------------------------------------------- factorized hyperprior (Eq. 9)
class FactorizedPrior(nn.Module):
    """compressai EntropyBottleneck(channels) (v1.2.x defaults: filters (3,3,3,3), init_scale 10, tail_mass 1e-9,
    likelihood bound 1e-9). Input [N, C] (ContextGS passes the per-anchor hyper latent with anchors as batch).

    ContextGS never calls the auxiliary quantile loss, so `quantiles` keep their initial value [-10, 0, 10] and the
    medians are 0 for the whole run; they are a buffer here (an Adam parameter without gradient in the official code).
    """

    def __init__(self, channels, filters=(3, 3, 3, 3), init_scale=10.0, likelihood_bound=1e-9):
        super().__init__()
        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.likelihood_bound = float(likelihood_bound)
        full = (1,) + self.filters + (1,)
        scale = init_scale ** (1 / (len(self.filters) + 1))
        self.matrices = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.factors = nn.ParameterList()
        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / full[i + 1]))
            self.matrices.append(nn.Parameter(torch.full((channels, full[i + 1], full[i]), init)))
            self.biases.append(nn.Parameter(torch.empty(channels, full[i + 1], 1).uniform_(-0.5, 0.5)))
            if i < len(self.filters):
                self.factors.append(nn.Parameter(torch.zeros(channels, full[i + 1], 1)))
        init_q = torch.tensor([-init_scale, 0.0, init_scale])
        self.register_buffer("quantiles", init_q.repeat(channels, 1, 1))

    def medians(self):
        return self.quantiles[:, :, 1:2]  # [C, 1, 1]

    def logits_cumulative(self, inputs):
        """_logits_cumulative(stop_gradient=False); inputs [C, 1, M]."""
        logits = inputs
        for i in range(len(self.filters) + 1):
            logits = torch.matmul(F.softplus(self.matrices[i]), logits)
            logits = logits + self.biases[i]
            if i < len(self.filters):
                logits = logits + torch.tanh(self.factors[i]) * torch.tanh(logits)
        return logits

    def likelihood_cm(self, values):
        """_likelihood + likelihood_lower_bound; values [C, 1, M]."""
        lower = self.logits_cumulative(values - 0.5)
        upper = self.logits_cumulative(values + 0.5)
        lik = torch.sigmoid(upper) - torch.sigmoid(lower)
        return LowerBound.apply(lik, self.likelihood_bound)

    def forward(self, x, training):
        """EntropyBottleneck.forward: (noisy or dequantized values, likelihood), both [N, C]."""
        values = x.t().contiguous().unsqueeze(1)  # [C, 1, N]
        if training:
            outputs = values + torch.empty_like(values).uniform_(-0.5, 0.5)
        else:
            med = self.medians()
            outputs = torch.round(values - med) + med
        lik = self.likelihood_cm(outputs)
        return outputs[:, 0].t().contiguous(), lik[:, 0].t().contiguous()

    def symbols(self, x):
        """quantize(mode='symbols', means=medians): integer round(x - median), [N, C]."""
        return torch.round(x - self.medians()[:, 0, 0]).to(torch.int32)

    def dequantize(self, sym):
        return sym.float() + self.medians()[:, 0, 0]

    @torch.no_grad()
    def cdf_table(self, lo, hi):
        """Per-channel CDF at symbol boundaries s-0.5 for s = lo..hi+1 (relative to the median): [C, L+1]."""
        s = torch.arange(lo, hi + 2, device=self.quantiles.device, dtype=torch.float32)
        v = (s - 0.5)[None, None, :] + self.medians()
        return torch.sigmoid(self.logits_cumulative(v))[:, 0]


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


def level_mlps(feat_dim, n_offsets, level_num, hyper_dim):
    """gaussian_model.py:177-188: per-level context MLPs F^k (Eq. 7); level K-1 sees [x, z], the others
    [parent anchor x, parent f, parent l, z]."""
    out_dim = (feat_dim + 6 + 3 * n_offsets) * 2 + 1 + 1 + 1
    context_dim = feat_dim + 6 + 3
    mlps = nn.ModuleList()
    for i in range(level_num):
        in_dim = hyper_dim + 3 if i == level_num - 1 else context_dim + hyper_dim
        mlps.append(nn.Sequential(nn.Linear(in_dim, feat_dim * 2), nn.ReLU(True),
                                  nn.Linear(feat_dim * 2, out_dim)))
    return mlps


class ContextBank(nn.Module):
    """All ContextGS global modules plus the anchor bounds and level scales; `step` = step of the next render."""

    def __init__(self, c, device):
        super().__init__()
        self.c = c
        self.feat_dim, self.n_offsets = c["feat_dim"], c["n_offsets"]
        self.level_num = c["level_num"]
        self.hyper_dim = c["feat_dim"] // c["hyper_divisor"]
        self.mlps = scaffold_mlps(self.feat_dim, self.n_offsets)
        self.prior = FactorizedPrior(self.hyper_dim)
        self.mlp_grid = level_mlps(self.feat_dim, self.n_offsets, self.level_num, self.hyper_dim)
        self.to(device)
        self.register_buffer("x_bound_min", torch.zeros(1, 3, device=device))
        self.register_buffer("x_bound_max", torch.ones(1, 3, device=device))
        self.level_scale = None  # list of K-1 floats, searched once (find_divide_scale)
        self.step = 1
        self.baked = False
        t = c["total_steps"]
        lr = c["lr"]
        self.schedules = {k: expon_lr(lr[k][0], lr[k][1], t)
                          for k in ("mlp_opacity", "mlp_cov", "mlp_color", "latent_codec", "mlp_grid")}
        groups = [{"params": list(self.mlps[k].parameters()), "name": k, "lr": 0.0}
                  for k in RENDER_MLPS]
        groups.append({"params": list(self.prior.parameters()), "name": "latent_codec", "lr": 0.0})
        groups.append({"params": list(self.mlp_grid.parameters()), "name": "mlp_grid", "lr": 0.0})
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.update_learning_rate(1)

    def update_learning_rate(self, step):
        for g in self.optimizer.param_groups:
            g["lr"] = self.schedules[g["name"]](step)

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

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

    def predict(self, level, feat_in):
        """mlp_grid[level] -> (mean, scale, Q) for feat / scaling / offsets, Q = Q0 (1 + tanh r).clamp(1e-9)."""
        out = self.mlp_grid[level](feat_in)
        f, k = self.feat_dim, 3 * self.n_offsets
        mf, sf, ms, ss, mo, so, qf, qs, qo = torch.split(out, [f, f, 6, 6, k, k, 1, 1, 1], dim=-1)
        q0 = self.c["q0"]
        return {
            "feat": (mf, sf, (q0["feat"] * (1 + torch.tanh(qf))).clamp(1e-9)),
            "scaling": (ms, ss, (q0["scaling"] * (1 + torch.tanh(qs))).clamp(1e-9)),
            "offsets": (mo, so, (q0["offsets"] * (1 + torch.tanh(qo))).clamp(1e-9)),
        }

    def mlp_bits(self):
        """get_mlp_size: 32-bit floats of every parameter whose name contains 'mlp' (render MLPs + mlp_grid;
        the factorized prior is not counted, as in the official code)."""
        n = sum(p.numel() for p in self.mlps.parameters())
        n += sum(p.numel() for p in self.mlp_grid.parameters())
        return 32 * n
