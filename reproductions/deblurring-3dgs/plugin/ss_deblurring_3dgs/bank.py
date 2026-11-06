"""Blur MLP F_theta of Deblurring 3DGS and its optimizer, held in a process-local bank.

Port of scene/blur_kernel.py (Embedder, GTnet, init_linear_weights) and the delta
post-processing of gaussian_renderer/__init__.py of
github.com/benhenryL/Deblurring-3D-Gaussian-Splatting. The density component creates the bank
in ``initialize``, publishes it as ``ACTIVE`` for the renderer, and steps its Adam after every
runtime step (official: one Adam with a "GTnet" group, lr 1e-3, eps 1e-15). ``ss-eval`` has no
bank (``ACTIVE is None``), so every camera renders the sharp Gaussians (paper Sec. 3.2).
"""

import torch
from torch import nn

ACTIVE = None  # live bank of the current training process (None in ss-eval)

BANK_DEFAULTS = {
    "mode": "defocus",     # "defocus" (use_pos 0) or "motion" (use_pos 1)
    "lr": 1e-3,            # paper Sec. 4.1; arguments gtnet_lr
    "width": 64,           # paper Sec. 4.1; arguments width
    "hidden": 3,           # paper "first 3 layers shared"; arguments hidden
    "res_pos": 3,          # GTnet res_pos default (code only)
    "res_view": 10,        # GTnet res_view default (code only)
    "num_moments": 4,      # arguments num_moments: M = num_moments + 1 = 5 renders
    "lambda_s": 0.01,      # paper Sec. 4.1; configs lambda_s
    "lambda_p": 0.01,      # paper Sec. 4.1; configs lambda_p
    "max_clamp": 1.1,      # arguments max_clamp (code only)
    "seed": 0,
}


def embed(x, multires):
    """NeRF Embedder (include_input, log_sampling, sin/cos, no pi): [x, sin(2^k x), cos(2^k x)]."""
    freqs = 2.0 ** torch.linspace(0.0, multires - 1, multires)
    out = [x]
    for f in freqs.tolist():
        out += [torch.sin(x * f), torch.cos(x * f)]
    return torch.cat(out, dim=-1)


def init_linear_weights(m):
    """blur_kernel.init_linear_weights: xavier_normal (gain 0.1 if out_features in {2, 3})."""
    if isinstance(m, nn.Linear):
        if m.weight.shape[0] in (2, 3):
            nn.init.xavier_normal_(m.weight, 0.1)
        else:
            nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class GTnet(nn.Module):
    """F_theta (Eq. 7 / 11): shared trunk of ``hidden`` Linear+ReLU layers, one linear head per
    delta. Defocus heads: s 3, r 4. Motion heads: s 3(M+1), r 4(M+1), p 3M (M = num_moments)."""

    def __init__(self, res_pos=3, res_view=10, num_hidden=3, width=64, pos_delta=False,
                 num_moments=4):
        super().__init__()
        self.res_pos, self.res_view = res_pos, res_view
        self.pos_delta, self.num_moments = bool(pos_delta), int(num_moments)
        in_cnl = 3 * (1 + 2 * res_pos) + 3 * (1 + 2 * res_view) + 7
        self.in_channels = in_cnl
        layers = [nn.Linear(in_cnl, width), nn.ReLU()]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        self.linears = nn.Sequential(*layers)
        m = self.num_moments
        if not self.pos_delta:
            self.s, self.r = nn.Linear(width, 3), nn.Linear(width, 4)
        else:
            self.s, self.r = nn.Linear(width, 3 * (m + 1)), nn.Linear(width, 4 * (m + 1))
            self.p = nn.Linear(width, 3 * m)
        self.apply(init_linear_weights)

    def forward(self, pos, scales, rotations, viewdirs):
        x = torch.cat([embed(pos, self.res_pos), embed(viewdirs, self.res_view), scales,
                       rotations], dim=-1)
        h = self.linears(x)
        return self.s(h), self.r(h), (self.p(h) if self.pos_delta else None)


def clamp_delta(delta, lambda_s, max_clamp):
    """Eq. 9-10 as coded: clamp(lambda_s * delta + (1 - lambda_s), min=1, max=max_clamp)."""
    return torch.clamp(lambda_s * delta + (1 - lambda_s), min=1.0, max=max_clamp)


class BlurBank:
    def __init__(self, cfg, device):
        self.cfg = {**BANK_DEFAULTS, **cfg}
        c = self.cfg
        if c["mode"] not in ("defocus", "motion"):
            raise ValueError("blur mode must be 'defocus' or 'motion'")
        self.motion = c["mode"] == "motion"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(c["seed"]))
            self.net = GTnet(int(c["res_pos"]), int(c["res_view"]), int(c["hidden"]),
                             int(c["width"]), self.motion, int(c["num_moments"]))
        self.net.to(device)
        self.reset_optimizer()
        self.counts = {"mlp_steps": 0, "optimizer_resets": 0}

    def reset_optimizer(self):
        """A fresh Adam (official add_points recreates the single optimizer: moments reset)."""
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=float(self.cfg["lr"]),
                                          eps=1e-15)

    def deltas(self, means, scales, quats_normalized, camera_center):
        """Detached inputs (renderer: _pos/_scales/_rotations = .detach(), viewdirs = camera
        centre repeated). Returns clamped (ds, dr, dx) where dx is lambda_p * raw position delta."""
        c = self.cfg
        n = means.shape[0]
        view = camera_center.reshape(1, 3).expand(n, 3)
        ds, dr, dx = self.net(means.detach(), scales.detach(), quats_normalized.detach(), view)
        ds = clamp_delta(ds, c["lambda_s"], c["max_clamp"])
        dr = clamp_delta(dr, c["lambda_s"], c["max_clamp"])
        if dx is not None:
            dx = c["lambda_p"] * dx
        return ds, dr, dx

    @torch.no_grad()
    def step(self):
        if any(p.grad is not None for p in self.net.parameters()):
            self.optimizer.step()
            self.counts["mlp_steps"] += 1
        self.optimizer.zero_grad(set_to_none=True)

    def export(self):
        """Live references (in-place Adam updates keep them current) for the checkpoint."""
        return {"mlp": self.net.state_dict(), "optimizer": self.optimizer.state_dict(),
                "config": dict(self.cfg), "counts": self.counts}
