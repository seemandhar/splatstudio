"""AtomGS Atomized Proliferation (supp. Alg. 1) — port of the official code @5b9ce904
(train.py densification block, scene/gaussian_model.py: create_from_pcd, densify_and_prune,
split_points, clone_points, atomize, reset_opacity, add_densification_stats).

  init     d = sqrt(mean 3-NN sq. dist) (runtime --recipe 3dgs), S = quantile(d, 0.01),
           initial scales max(d, S) (isotropic)
  stats    step < densify_until, radii > 0: accum += ||g[:, :2]||, denom += 1
  densify  step > densify_from and step % interval == 0:
           grads = accum / denom (nan -> 0)
           split: grads >= min(tau_s * step / t_w, tau_s) and max(s) > S; 2 children
                  ~ N(mu, R diag(s)^2 R^T), s / 1.6, rest copied; parents removed
                  (layout: survivors in order, then the 2 child blocks)
           clone: padded = zeros(len after split); padded[:n0] = grads  (official quirk: the
                  pre-split gradient vector is applied to the post-split population)
                  clone padded >= tau_c, no size condition, exact copies appended
           prune: sigmoid(o) < 0.005 or max(s) > 0.1 * extent (every densify)
           statistics restart at zero (densification_postfix)
  atomize  step % interval == 0 and step < t_a: rows with min(s) <= S -> s = (S, S, S);
           Adam moments of all scales zeroed (replace_tensor_to_optimizer); S *= 0.99
  reset    step % 300 == 0 (step < densify_until): opacity = min(opacity, 0.1), moments zeroed
  xyz lr   get_expon_lr_func(1.6e-4 * extent -> 1.6e-6 * extent, max_steps 7000), overriding the
           runtime's 0.01 ** (step / steps) decay.
Difference to the official loop: the runtime applies the optimizer step before this controller
runs (official densifies first; replaced tensors then skip that iteration's step).
"""

import math

import torch
from gsplat.strategy.ops import _update_param_with_optimizer, remove
from gsplat.utils import normalized_quat_to_rotmat

DEFAULTS = {
    "densify_from_iter": 100,
    "densify_until_iter": 15000,
    "densification_interval": 100,
    "clone_threshold": 0.0002,
    "split_threshold": 0.0002,
    "warm_up_until": 7000,
    "atom_proliferation_until": 7000,
    "atom_init_quantile": 0.01,
    "atom_decay": 0.99,
    "atom_init_clamp": True,
    "prune_threshold": 0.005,
    "big_world_factor": 0.1,
    "opacity_reset_interval": 300,
    "opacity_reset_value": 0.1,
    "split_n": 2,
    "split_scale_div": 1.6,
    "official_clone_index_shift": True,
    "position_lr_init": 1.6e-4,
    "position_lr_final": 1.6e-6,
    "position_lr_delay_mult": 0.01,
    "position_lr_delay_steps": 0,
    "position_lr_max_steps": 7000,
}
REQUIRES = {"gradient.screen.inria"}


def expon_lr(step, lr_init, lr_final, delay_steps=0, delay_mult=1.0, max_steps=30000):
    """utils/general_utils.get_expon_lr_func."""
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
        return 0.0
    if delay_steps > 0:
        delay_rate = delay_mult + (1 - delay_mult) * math.sin(
            0.5 * math.pi * min(max(step / delay_steps, 0), 1))
    else:
        delay_rate = 1.0
    t = min(max(step / max_steps, 0), 1)
    return delay_rate * math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)


def quantile_linear(x, q):
    """torch.quantile(x, q) (linear interpolation) without its 2^24-element limit."""
    v = torch.sort(x.flatten()).values
    pos = q * (len(v) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return float(v[lo] + (v[hi] - v[lo]) * (pos - lo))


def _append(params, optimizers, keep, new):
    """params[name] <- cat(params[name][keep], new[name]); Adam moments of new rows are zero."""
    def param_fn(name, p):
        return torch.nn.Parameter(torch.cat([p[keep], new[name].to(p.dtype)]),
                                  requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        extra = len(next(iter(new.values())))
        return torch.cat([v[keep], torch.zeros((extra, *v.shape[1:]), device=v.device,
                                               dtype=v.dtype)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)


@torch.no_grad()
def split_points(params, optimizers, mask, n=2, div=1.6):
    """Official split_points: append n samples per selected Gaussian, then remove parents."""
    sel = torch.where(mask)[0]
    stds = params["scales"][sel].exp().repeat(n, 1)
    samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
    rots = normalized_quat_to_rotmat(
        torch.nn.functional.normalize(params["quats"][sel], dim=-1)).repeat(n, 1, 1)
    new = {}
    for name, p in params.items():
        rep = (n,) + (1,) * (p.dim() - 1)
        new[name] = p[sel].repeat(*rep)
    new["means"] = torch.bmm(rots, samples[..., None])[..., 0] + params["means"][sel].repeat(n, 1)
    new["scales"] = torch.log(params["scales"][sel].exp().repeat(n, 1) / div)
    _append(params, optimizers, ~mask, new)


@torch.no_grad()
def clone_points(params, optimizers, mask):
    """Official clone_points: exact copies appended."""
    keep = torch.ones_like(mask)
    _append(params, optimizers, keep, {name: p[mask] for name, p in params.items()})


@torch.no_grad()
def replace_field(params, optimizers, name, value):
    """Official replace_tensor_to_optimizer: new values, Adam moments zeroed (step kept)."""
    def param_fn(_name, p):
        return torch.nn.Parameter(value.to(p.dtype), requires_grad=p.requires_grad)

    _update_param_with_optimizer(param_fn, lambda key, v: torch.zeros_like(v), params,
                                 optimizers, [name])


class Density:
    requires = REQUIRES

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown AtomGS density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.requires = set(REQUIRES)
        self.state = None
        self.stats = None
        self.extent = None

    # -- runtime hooks -------------------------------------------------------------------
    @torch.no_grad()
    def initialize(self, params, optimizers, scene_scale=1.0):
        self.extent = float(scene_scale)  # 3dgs recipe: getNerfppNorm radius (spatial_lr_scale)
        s = params["scales"]
        if len(s) > 1 and float(s.max() - s.min()) == 0.0:
            raise ValueError("AtomGS needs per-point 3-NN initial scales: train with --recipe 3dgs")
        dist = s[:, 0].exp()
        atom = quantile_linear(dist, float(self.cfg["atom_init_quantile"]))
        if self.cfg["atom_init_clamp"]:
            params["scales"].copy_(torch.log(dist.clamp_min(atom))[:, None].expand(-1, 3))
        self._reset_stats(len(params["means"]), params["means"].device)
        self.state = {"extent": self.extent, "atom_scale": atom, "atom_scale_init": atom,
                      "counts": {"densify": 0, "atomize": 0, "resets": 0}}

    def before_backward(self, params, optimizers, step, outputs):
        if step == 1:
            optimizers["means"].param_groups[0]["lr"] = self.xyz_lr(1)

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        if step < c["densify_until_iter"]:
            self.accumulate(outputs["info"])
            if step > c["densify_from_iter"] and step % c["densification_interval"] == 0:
                split_thr = min(c["split_threshold"] * step / c["warm_up_until"],
                                c["split_threshold"])
                self.densify_and_prune(params, optimizers, c["clone_threshold"], split_thr)
            if step % c["densification_interval"] == 0 and step < c["atom_proliferation_until"]:
                self.atomize(params, optimizers)
            if step % c["opacity_reset_interval"] == 0:
                self.reset_opacity(params, optimizers)
        optimizers["means"].param_groups[0]["lr"] = self.xyz_lr(step + 1)

    # -- pieces ----------------------------------------------------------------------------
    def xyz_lr(self, step):
        c = self.cfg
        return expon_lr(step, c["position_lr_init"] * self.extent,
                        c["position_lr_final"] * self.extent, c["position_lr_delay_steps"],
                        c["position_lr_delay_mult"], c["position_lr_max_steps"])

    def _reset_stats(self, n, dev):
        self.stats = {"grad_accum": torch.zeros(n, device=dev), "denom": torch.zeros(n, device=dev)}

    def accumulate(self, info):
        grad = info["means2d"].grad
        if grad is None:
            raise RuntimeError("AtomGS density: means2d has no gradient (renderer misconfigured)")
        vis = info["radii"] > 0
        self.stats["grad_accum"][vis] += torch.linalg.vector_norm(grad[vis, :2], dim=-1)
        self.stats["denom"][vis] += 1

    @torch.no_grad()
    def densify_and_prune(self, params, optimizers, clone_thr, split_thr):
        c, st = self.cfg, self.stats
        atom = self.state["atom_scale"]
        grads = st["grad_accum"] / st["denom"]
        grads[grads.isnan()] = 0.0
        n0 = len(grads)
        smax = params["scales"].exp().max(dim=1).values
        split_mask = (grads >= split_thr) & (smax > atom)
        n_split = int(split_mask.sum())
        split_points(params, optimizers, split_mask, int(c["split_n"]), float(c["split_scale_div"]))
        n1 = len(params["means"])
        padded = torch.zeros(n1, device=grads.device)
        if c["official_clone_index_shift"]:
            padded[:n0] = grads  # official: pre-split indexing on the post-split population
        else:
            padded[: n0 - n_split] = grads[~split_mask]  # aligned (ablation)
        clone_mask = padded >= clone_thr
        clone_points(params, optimizers, clone_mask)
        prune = (params["opacities"].sigmoid() < c["prune_threshold"]) | (
            params["scales"].exp().max(dim=1).values > c["big_world_factor"] * self.extent)
        remove(params, optimizers, {}, prune)
        self._reset_stats(len(params["means"]), params["means"].device)
        self.state["counts"]["densify"] += 1
        self.state["last"] = {"before": n0, "split": n_split, "clone": int(clone_mask.sum()),
                              "prune": int(prune.sum()), "after": len(params["means"])}

    @torch.no_grad()
    def atomize(self, params, optimizers):
        atom = self.state["atom_scale"]
        dist = params["scales"].exp()
        mask = dist.min(dim=1).values <= atom
        dist[mask] = atom
        replace_field(params, optimizers, "scales", torch.log(dist))
        self.state["atom_scale"] = atom * float(self.cfg["atom_decay"])
        self.state["counts"]["atomize"] += 1

    @torch.no_grad()
    def reset_opacity(self, params, optimizers):
        o = params["opacities"].sigmoid()
        v = torch.min(o, torch.ones_like(o) * float(self.cfg["opacity_reset_value"]))
        replace_field(params, optimizers, "opacities", torch.log(v / (1 - v)))
        self.state["counts"]["resets"] += 1


def create_density(spec):
    return Density(spec)
