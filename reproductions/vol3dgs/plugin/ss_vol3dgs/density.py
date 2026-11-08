"""Vol3DGS adaptive density control (official scene/gaussian_model_volr.py + train_volr.py).

Per step while step < densify_until and N < max_points: accumulate ||dL/dmu|| (world space,
train_volr.py sets viewspace_point_tensor = gaussians.get_xyz) and a visibility count.
At step > densify_from with step % interval == 0 (densify_and_prune):
  "halved" theta follows the official density2alpha as written (1 - exp(-x)/0.99, i.e.
  1 - sqrt(1 - 0.99 theta)/0.99, ~0.0102 below a true halving), so the theta >= 0.99 split test
  (round trip theta - 0.0101) never fires; config density2alpha: "inverse" gives the intended math.
  clone  grad >= clone_grad and max s <= percent_dense * ext      (new copy: theta halved,
                                                                   original unchanged)
  split  (grad >= split_grad and max s > percent_dense * ext) or theta >= 0.99
         (2 children ~ N(mu, R diag(s)), s / 1.6, theta halved; parent removed)
  prune  theta < min_theta; after step > 3000 also max s > prune_scale * ext
  ext = scene_extent_mult * camera extent; statistics reset (densification_postfix).
The official screen-size prune (max_radii2D > 20) never fires because densification_postfix
zeroes max_radii2D before the check, so it is omitted. Opacity reset is disabled.

The controller also applies the paper's optimizer recipe: constant LRs for theta / SH set in
initialize(), and the Inria exponential means LR 1.6e-4 -> 1e-5 (x extent) every step
(overriding the runtime's built-in decay, which runs before after_backward).

Order: the native runtime steps the optimizer before after_backward, whereas train_volr.py
densifies before optimizer.step (so Inria's freshly replaced params skip that step's update).
"""

import math

import torch
from gsplat.strategy.ops import duplicate, remove, split

from .reference import (
    DENSITY2ALPHA_MODES,
    alpha2density,
    density2alpha,
    halve_theta,
    inverse_sigmoid,
)

OPTIONS = {
    "clone_grad": 5e-5,
    "split_grad": 1e-7,
    "densify_from": 500,
    "densify_until": 15000,
    "interval": 200,
    "percent_dense": 0.01,
    "extent_mult": 2.0,
    "min_theta": 0.05,
    "split_theta": 0.99,
    "prune_scale": 0.01,
    "prune_after": 3000,
    "max_points": 7_000_000,
    "density2alpha": "official",
    "lr": {"opacities": 0.03, "sh0": 0.003, "shN": 0.003 / 20, "scales": 0.005, "quats": 0.001},
    "means_lr_init": 1.6e-4,
    "means_lr_final": 1e-5,
    "means_lr_steps": 30000,
}


def expon_lr(step, init, final, max_steps):
    """Inria get_expon_lr_func with lr_delay_steps = 0."""
    t = min(max(step / max_steps, 0.0), 1.0)
    return math.exp(math.log(init) * (1 - t) + math.log(final) * t)


class VolDensity:
    requires = frozenset({"gradient.world.means"})

    def __init__(self, spec):
        cfg = {**OPTIONS, **spec.config}
        unknown = set(cfg) - set(OPTIONS)
        if unknown:
            raise ValueError(f"Unknown Vol3DGS density options: {sorted(unknown)}")
        if cfg["density2alpha"] not in DENSITY2ALPHA_MODES:
            raise ValueError(f"density2alpha must be one of {DENSITY2ALPHA_MODES}")
        self.cfg = cfg
        self.state = None
        self.extent = None
        self.scene_scale = None
        self.last_counts = {}

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.scene_scale = float(scene_scale)
        self.extent = self.cfg["extent_mult"] * self.scene_scale
        for name, lr in self.cfg["lr"].items():
            optimizers[name].param_groups[0]["lr"] = float(lr)
        optimizers["means"].param_groups[0]["lr"] = self.means_lr(1)
        n = len(params["means"])
        dev = params["means"].device
        self.state = {"grad_accum": torch.zeros(n, device=dev), "count": torch.zeros(n, device=dev)}

    def means_lr(self, step):
        c = self.cfg
        return expon_lr(step, c["means_lr_init"] * self.scene_scale,
                        c["means_lr_final"] * self.scene_scale, c["means_lr_steps"])

    def before_backward(self, params, optimizers, step, outputs):
        means_in = outputs["info"]["means_in"]
        if means_in.requires_grad:
            means_in.retain_grad()

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        info = outputs["info"]
        if "set_next_step" in info:
            info["set_next_step"](step + 1)
        optimizers["means"].param_groups[0]["lr"] = self.means_lr(step + 1)
        c = self.cfg
        if not (step < c["densify_until"] and len(params["means"]) < c["max_points"]):
            return
        grad = info["means_in"].grad
        if grad is None:
            raise RuntimeError("Vol3DGS density: world-space means gradient was not retained")
        vis = info["visible"]
        self.state["grad_accum"][vis] += grad[vis].norm(dim=-1)
        self.state["count"][vis] += 1
        if step > c["densify_from"] and step % c["interval"] == 0:
            self.densify_and_prune(params, optimizers, step)

    @torch.no_grad()
    def densify_and_prune(self, params, optimizers, step):
        c = self.cfg
        ext = self.extent
        grads = self.state["grad_accum"] / self.state["count"]
        grads[grads.isnan()] = 0.0

        # clone (densify_and_clone): copies keep everything except a halved density
        smax = params["scales"].exp().max(dim=-1).values
        clone = (grads >= c["clone_grad"]) & (smax <= c["percent_dense"] * ext)
        n_clone = int(clone.sum())
        if n_clone:
            theta_new = halve_theta(torch.sigmoid(params["opacities"][clone]),
                                    params["scales"][clone].exp(), c["density2alpha"])
            duplicate(params, optimizers, self.state, clone)
            params["opacities"].data[-n_clone:] = inverse_sigmoid(theta_new)

        # split (densify_and_split) on the post-clone population; clones have zero gradient
        grads = torch.cat((grads, torch.zeros(n_clone, device=grads.device)))
        scales = params["scales"].exp()
        smax = scales.max(dim=-1).values
        theta = torch.sigmoid(params["opacities"])
        theta_rt = density2alpha(alpha2density(theta, scales), scales, c["density2alpha"])  # code's round trip
        sel = ((grads >= c["split_grad"]) & (smax > c["percent_dense"] * ext)) | (
            theta_rt >= c["split_theta"])
        n_split = int(sel.sum())
        if n_split:
            theta_child = halve_theta(theta[sel], scales[sel], c["density2alpha"])
            split(params, optimizers, self.state, sel)
            params["opacities"].data[-2 * n_split:] = inverse_sigmoid(theta_child).repeat(2)

        # prune
        theta = torch.sigmoid(params["opacities"])
        prune = theta < c["min_theta"]
        if step > c["prune_after"]:
            prune |= params["scales"].exp().max(dim=-1).values > c["prune_scale"] * ext
        n_prune = int(prune.sum())
        if n_prune:
            remove(params, optimizers, self.state, prune)

        # densification_postfix: statistics reset
        n = len(params["means"])
        dev = params["means"].device
        self.state["grad_accum"] = torch.zeros(n, device=dev)
        self.state["count"] = torch.zeros(n, device=dev)
        for name, p in params.items():
            if not (p.is_leaf and p.requires_grad):
                raise RuntimeError(f"Vol3DGS density froze field {name}")
        self.last_counts = {"step": step, "clone": n_clone, "split": n_split, "prune": n_prune,
                            "gaussians": n}


def create_density(spec):
    return VolDensity(spec)
