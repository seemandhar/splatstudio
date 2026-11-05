"""Official 3DGS density control as used unchanged by StopThePop (train.py +
scene/gaussian_model.py of github.com/r4dl/StopThePop @ 859b11b).

Per training step s (the runtime calls `after_backward` after the optimizer step; the official
loop densifies before it), while s < densify_until:
    accumulate ||dL/d means2D[:, :2]|| (Inria NDC-space screen gradient) and denom for radii > 0;
    s > densify_from and s % 100 == 0: densify_and_clone + densify_and_split + prune;
    opacity_decay == 0 and s % 3000 == 0: reset opacity to min(o, 0.01) (zeroed Adam moments);
    opacity_decay != 0 and s % 50 == 0 and s > densify_from: o <- o * opacity_decay.
Prune: o < 0.005, and once s > 3000 (size_threshold set) max scale > 0.1 * extent. The 20 px
screen-size test is inert because densification_postfix zeroes max_radii2D right before it.
"""

import torch
from gsplat.strategy.ops import _update_param_with_optimizer, remove, reset_opa
from torch import nn

DEFAULTS = {
    "densify_from": 500,  # OptimizationParams.densify_from_iter
    "densify_until": 15000,  # densify_until_iter
    "densify_every": 100,  # densification_interval
    "grad_threshold": 0.0002,  # densify_grad_threshold
    "percent_dense": 0.01,
    "min_opacity": 0.005,  # train.py densify_and_prune(..., 0.005, ...)
    "big_world_after": 3000,  # size_threshold = 20 if iteration > opacity_reset_interval
    "big_world": 0.1,  # big_points_ws: max scale > 0.1 * extent
    "reset_every": 3000,  # opacity_reset_interval
    "reset_value": 0.01,
    "opacity_decay": 0.0,  # --opacity_decay (paper "Opacity Decay" rows: 0.9995)
    "decay_every": 50,
}
REQUIRES = {"gradient.screen.signed"}
BASE_FIELDS = ("means", "scales", "quats", "opacities", "sh0", "shN")


def quat_to_rotmat(quats):
    """utils/general_utils.build_rotation: wxyz (any norm) -> rotation matrices [N,3,3]."""
    q = torch.nn.functional.normalize(quats, dim=-1)
    r, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def append_gaussians(params, optimizers, new):
    """cat_tensors_to_optimizer: append rows, new Adam moments are zero."""
    n_new = len(new["means"])

    def param_fn(name, p):
        return nn.Parameter(torch.cat([p.detach(), new[name].to(p.dtype)]),
                            requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v, torch.zeros((n_new, *v.shape[1:]), dtype=v.dtype,
                                         device=v.device)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)


def decay_opacity(params, optimizers, factor):
    """gaussian_model.decay_opacity: replace_tensor_to_optimizer(inverse_sigmoid(o * f)) —
    a replaced tensor gets zeroed Adam moments."""
    def param_fn(name, p):
        return nn.Parameter(torch.logit(p.detach().sigmoid() * factor), requires_grad=True)

    def optimizer_fn(key, v):
        return torch.zeros_like(v)

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers, names=["opacities"])


class InriaDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown StopThePop density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        if not 0.0 <= float(self.cfg["opacity_decay"]) < 1.0:
            raise ValueError("opacity_decay must be 0 (off) or in (0, 1)")
        self.requires = set(REQUIRES)
        self.state = None
        self.generator = None

    def initialize(self, params, optimizers, scene_scale=1.0):
        extra = set(params) - set(BASE_FIELDS)
        if extra:
            raise ValueError(f"StopThePop density does not know fields {sorted(extra)}")
        self.extent = float(scene_scale)
        self._reset_stats(len(params["means"]), params["means"].device)
        self.state = {"events": []}

    def before_backward(self, params, optimizers, step, outputs):
        if step < self.cfg["densify_until"]:
            outputs["info"]["means2d"].retain_grad()

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.cfg
        if step >= cfg["densify_until"]:
            return
        self._accumulate(outputs["info"])
        if step > cfg["densify_from"] and step % cfg["densify_every"] == 0:
            self._densify(params, optimizers, step)
        decay = float(cfg["opacity_decay"])
        if decay == 0 and step % cfg["reset_every"] == 0:
            reset_opa(params, optimizers, {}, cfg["reset_value"])
        if decay != 0 and step % cfg["decay_every"] == 0 and step > cfg["densify_from"]:
            decay_opacity(params, optimizers, decay)

    def _reset_stats(self, n, device):
        self.vgn = torch.zeros(n, device=device)
        self.denom = torch.zeros(n, device=device)

    def _accumulate(self, info):
        g2 = info["means2d"].grad
        if g2 is None:
            raise RuntimeError("screen gradient missing: render with the StopThePop renderer")
        vis = info["radii"] > 0
        self.vgn[vis] += torch.norm(g2[vis, :2], dim=-1)  # add_densification_stats
        self.denom[vis] += 1

    def _densify(self, params, optimizers, step):
        cfg = self.cfg
        n0 = len(params["means"])
        grads = self.vgn / self.denom
        grads[grads.isnan()] = 0.0
        # densify_and_clone
        small = params["scales"].exp().max(dim=1).values <= cfg["percent_dense"] * self.extent
        clone = (grads >= cfg["grad_threshold"]) & small
        append_gaussians(params, optimizers, {k: params[k][clone] for k in BASE_FIELDS})
        n_clone = int(clone.sum())
        # densify_and_split (padded grads: clones get 0)
        grads = torch.cat([grads, torch.zeros(n_clone, device=grads.device)])
        split = (grads >= cfg["grad_threshold"]) & (
            params["scales"].exp().max(dim=1).values > cfg["percent_dense"] * self.extent)
        K = int(split.sum())
        stds = params["scales"][split].exp().repeat(2, 1)
        samples = torch.normal(torch.zeros_like(stds), stds, generator=self.generator)
        rots = quat_to_rotmat(params["quats"][split]).repeat(2, 1, 1)
        new = {"means": torch.bmm(rots, samples[..., None])[..., 0]
               + params["means"][split].repeat(2, 1),
               "scales": torch.log(params["scales"][split].exp().repeat(2, 1) / (0.8 * 2)),
               "quats": params["quats"][split].repeat(2, 1),
               "opacities": params["opacities"][split].repeat(2),
               "sh0": params["sh0"][split].repeat(2, 1, 1),
               "shN": params["shN"][split].repeat(2, 1, 1)}
        append_gaussians(params, optimizers, new)
        remove(params, optimizers, {}, torch.cat([split, torch.zeros(
            2 * K, dtype=torch.bool, device=split.device)]))
        # prune (screen-size test inert: max_radii2D was zeroed by densification_postfix)
        prune = params["opacities"].sigmoid() < cfg["min_opacity"]
        if step > cfg["big_world_after"]:
            prune |= params["scales"].exp().max(dim=1).values > cfg["big_world"] * self.extent
        remove(params, optimizers, {}, prune)
        self._reset_stats(len(params["means"]), params["means"].device)
        row = {"step": int(step), "n_before": n0, "clone": n_clone, "split": K,
               "pruned": int(prune.sum()), "n_after": len(params["means"])}
        self.state["events"].append(row)
        if step % 1000 == 0:
            print({"stopthepop_density": row}, flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_density(spec):
    return InriaDensity(spec)
