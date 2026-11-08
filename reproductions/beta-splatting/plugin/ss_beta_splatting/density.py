"""Kernel-agnostic MCMC (paper Sec. 4.4), ported from the official `train.py` densification block
and `BetaModel.{relocate_gs, add_new_gs, _update_params, _sample_alives}`.

Every `interval` steps with from < step < until:
  1. dead = o <= dead_opacity; each dead row copies a multinomial(o_alive) sample. The sample and
     its copies get o' = clamp(1 - (1 - o)^(1/(k+1)), dead_opacity, 1 - eps), k = times sampled
     (paper Eq. o' = 1 - (1-o)^{1/N}). **Only opacity changes** (no 3DGS-MCMC scale adjustment).
     Adam moments of the sampled rows are zeroed.
  2. add min(cap_max, int(1.05 N)) - N copies of multinomial(o) samples, same opacity rule,
     zero moments for the sampled rows and the new rows.
  3. noise: mu += Sigma (eps * (1 - o)^100 * noise_lr * lr_means), Sigma = R S S^T R^T
     (paper: B(o; ln 25) = (1-o)^{4*25}).
Population updates go through gsplat.strategy.ops._update_param_with_optimizer, which keeps the
Parameters trainable leaves (requires_grad copied) and the Adam state aligned.
"""

import math

import torch
from gsplat.strategy.ops import _multinomial_sample, _update_param_with_optimizer
from gsplat.utils import normalized_quat_to_rotmat

OPTIONS = {
    "cap_max": 1_000_000,
    "densify_from": 500,
    "densify_until": 25_000,
    "interval": 100,
    "dead_opacity": 0.005,
    "growth": 1.05,
    "noise_lr": 5e4,
    "noise_power": 100.0,
    "init_opacity": 0.5,
    "init_lobe_angles": True,
}
EPS = torch.finfo(torch.float32).eps


def split_opacity(opacities, counts, min_opacity):
    """o' = 1 - (1 - o)^(1 / (k + 1)), clamped (official `_update_params`); counts = k."""
    new = 1.0 - torch.pow(1.0 - opacities, 1.0 / (counts + 1))
    return new.clamp(min=min_opacity, max=1.0 - EPS)


def sample_with_counts(probs, n):
    """Official `_sample_alives`: multinomial with replacement + per-sample multiplicity."""
    idx = _multinomial_sample(probs / (probs.sum() + EPS), n, replacement=True)
    return idx, torch.bincount(idx)[idx]


@torch.no_grad()
def relocate(params, optimizers, dead_opacity):
    opac = torch.sigmoid(params["opacities"])
    dead = opac <= dead_opacity
    n_dead = int(dead.sum())
    if n_dead == 0:
        return 0
    dead_idx = dead.nonzero(as_tuple=True)[0]
    alive_idx = (~dead).nonzero(as_tuple=True)[0]
    if len(alive_idx) == 0:
        return 0
    s, counts = sample_with_counts(opac[alive_idx], n_dead)
    sampled = alive_idx[s]  # multiplicities are unchanged by the index map
    new_logit = torch.logit(split_opacity(opac[sampled], counts, dead_opacity))

    def param_fn(name, p):
        if name == "opacities":
            p[sampled] = new_logit
        p[dead_idx] = p[sampled]
        return torch.nn.Parameter(p, requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        v[sampled] = 0
        return v

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    return n_dead


@torch.no_grad()
def add_new(params, optimizers, cap_max, growth, dead_opacity):
    n = len(params["means"])
    n_add = max(0, min(int(cap_max), int(growth * n)) - n)
    if n_add == 0:
        return 0
    opac = torch.sigmoid(params["opacities"])
    sampled, counts = sample_with_counts(opac, n_add)
    new_logit = torch.logit(split_opacity(opac[sampled], counts, dead_opacity))

    def param_fn(name, p):
        if name == "opacities":
            p[sampled] = new_logit
        return torch.nn.Parameter(torch.cat([p, p[sampled]]), requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        v[sampled] = 0
        return torch.cat([v, torch.zeros((n_add, *v.shape[1:]), device=v.device, dtype=v.dtype)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    return n_add


@torch.no_grad()
def noise_delta(params, lr, noise_lr, power):
    """Official train.py: randn * (1-o)^100 * noise_lr * xyz_lr, then bmm with the covariance."""
    opac = torch.sigmoid(params["opacities"])[:, None]
    R = normalized_quat_to_rotmat(torch.nn.functional.normalize(params["quats"], dim=-1))
    L = R * torch.exp(params["scales"])[:, None, :]  # build_scaling_rotation: R S
    cov = L @ L.transpose(1, 2)
    eps = torch.randn_like(params["means"]) * torch.pow(1.0 - opac, power) * noise_lr * lr
    return torch.bmm(cov, eps[..., None])[..., 0]


class KernelAgnosticMCMC:
    def __init__(self, spec):
        self.requires = set()
        cfg = {**OPTIONS, **spec.config}
        unknown = set(cfg) - set(OPTIONS)
        if unknown:
            raise ValueError(f"Unknown kernel-agnostic MCMC options: {sorted(unknown)}")
        self.cfg = cfg
        self.state = {"relocated": 0, "added": 0, "events": 0}

    def initialize(self, params, optimizers, scene_scale=1.0):
        """Official create_from_pcd: opacity 0.5, b = 0, lobe theta ~ U[0,pi], phi ~ U[0,2pi]."""
        with torch.no_grad():
            params["opacities"].fill_(math.log(self.cfg["init_opacity"] /
                                               (1.0 - self.cfg["init_opacity"])))
            params["betas"].zero_()
            sb = params["sb_params"]
            sb.zero_()
            if self.cfg["init_lobe_angles"]:
                n, m = sb.shape[:2]
                sb[:, :, 3] = torch.pi * torch.rand(n, m, device=sb.device)
                sb[:, :, 4] = 2 * torch.pi * torch.rand(n, m, device=sb.device)
        # official train.py raises cap_max to the initial point count if it is smaller
        self.cfg["cap_max"] = max(self.cfg["cap_max"], len(params["means"]))

    def active(self, step):
        c = self.cfg
        return c["densify_from"] < step < c["densify_until"] and step % c["interval"] == 0

    def before_backward(self, params, optimizers, step, outputs):
        pass

    def after_backward(self, params, optimizers, step, outputs):
        if not self.active(step):
            return
        c = self.cfg
        self.state["relocated"] += relocate(params, optimizers, c["dead_opacity"])
        self.state["added"] += add_new(params, optimizers, c["cap_max"], c["growth"],
                                       c["dead_opacity"])
        self.state["events"] += 1
        lr = optimizers["means"].param_groups[0]["lr"]
        params["means"].data.add_(noise_delta(params, lr, c["noise_lr"], c["noise_power"]))


def create_density(spec):
    return KernelAgnosticMCMC(spec)
