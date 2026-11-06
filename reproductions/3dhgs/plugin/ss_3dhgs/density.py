"""3D-HGS density control: official train.py + scene/gaussian_model.py of
github.com/lihaolin88/3DHGS @ dde1bd5 (vanilla Inria ADC adapted to two opacities).

Per training step s (the runtime calls `after_backward` after the optimizer step; the official
loop densifies before it):
  s % 5000 == 0: lr of both opacity halves /= 1.4               (train.py:128-132)
  s < densify_until (20000):
      vgn += ||dL/d means2D[:, :2]||, denom += 1 for radii > 0  (add_densification_stats)
      s > 500 and s % 100 == 0: densify_and_prune                (clone, split, prune)
      s % 3000 == 0: both halves -> min(alpha, 0.02), Adam moments zeroed (reset_opacity)
"""

import torch
from gsplat.strategy.ops import _update_param_with_optimizer, remove
from torch import nn

from . import core

DEFAULTS = {
    "densify_from": 500,  # densify_from_iter
    "densify_until": 20000,  # densify_until_iter (3D-HGS default; 3DGS uses 15000)
    "densify_every": 100,  # densification_interval
    "grad_threshold": 0.0002,  # densify_grad_threshold
    "percent_dense": 0.01,
    "min_opacity": 0.01,  # train.py:136 ("change 0.005 to 0.01"), supp. B.2; on max(a1, a2)
    "big_world_after": 3000,  # size_threshold set when iteration > opacity_reset_interval
    "big_world": 0.1,  # big_points_ws: max scale > 0.1 * extent
    "reset_every": 3000,  # opacity_reset_interval
    "reset_value": 0.02,  # reset_opacity (supp. B.2)
    "clone_opacity_rescale": True,  # densify_and_clone: clone gets 1 - sqrt(1 - alpha)
    "opacity_lr_decay": 1.4,  # train.py:132 (supp. B.2)
    "opacity_lr_decay_every": 5000,
}
REQUIRES = {"gradient.screen.signed"}
FIELDS = ("means", "scales", "quats", "opacities", "opacity2", "hgs_normal", "sh0", "shN")
OPACITY_FIELDS = ("opacities", "opacity2")


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


def reset_opacity(params, optimizers, value):
    """reset_opacity + replace_tensor_to_optimizer: both halves clamped, moments zeroed."""
    cap = torch.logit(torch.tensor(value)).item()

    def param_fn(name, p):
        return nn.Parameter(torch.clamp(p, max=cap), requires_grad=p.requires_grad)

    _update_param_with_optimizer(param_fn, lambda k, v: torch.zeros_like(v), params, optimizers,
                                 names=list(OPACITY_FIELDS))


class HGSDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown 3D-HGS density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.requires = set(REQUIRES)
        self.state = None
        self.generator = None

    # ------------------------------------------------------------------ runtime hooks
    def initialize(self, params, optimizers, scene_scale=1.0):
        if set(params) != set(FIELDS):
            raise ValueError(f"3D-HGS density expects fields {sorted(FIELDS)}, got {sorted(params)}")
        self.extent = float(scene_scale)
        self._reset_stats(len(params["means"]), params["means"].device)
        self.state = {"events": [], "opacity_lr": []}

    def before_backward(self, params, optimizers, step, outputs):
        if step < self.cfg["densify_until"]:
            outputs["info"]["means2d"].retain_grad()

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.cfg
        if step % cfg["opacity_lr_decay_every"] == 0:
            for k in OPACITY_FIELDS:
                for group in optimizers[k].param_groups:
                    group["lr"] /= cfg["opacity_lr_decay"]
            self.state["opacity_lr"].append(
                {"step": int(step), "lr": optimizers["opacities"].param_groups[0]["lr"]})
        if step >= cfg["densify_until"]:
            return
        self._accumulate(outputs["info"])
        if step > cfg["densify_from"] and step % cfg["densify_every"] == 0:
            self._densify(params, optimizers, step)
        if step % cfg["reset_every"] == 0:
            reset_opacity(params, optimizers, cfg["reset_value"])

    # ------------------------------------------------------------------ statistics
    def _reset_stats(self, n, device):
        self.vgn = torch.zeros(n, device=device)
        self.denom = torch.zeros(n, device=device)

    def _accumulate(self, info):
        g2 = info["means2d"].grad
        if g2 is None:
            raise RuntimeError("3D-HGS statistics missing: render with the 3D-HGS renderer")
        vis = info["radii"] > 0
        self.vgn[vis] += torch.norm(g2[vis, :2], dim=-1)
        self.denom[vis] += 1

    # ------------------------------------------------------------------ densification
    def _densify(self, params, optimizers, step):
        cfg = self.cfg
        n0 = len(params["means"])
        grads = core.averaged(self.vgn, self.denom)
        limit = cfg["percent_dense"] * self.extent

        # ---- densify_and_clone: copies; the clone's opacities become 1 - sqrt(1 - alpha).
        # The official in-place update of the originals (`self._opacity[mask].copy_`) writes
        # into an advanced-indexing temporary, so the originals keep their opacity.
        clone = (grads >= cfg["grad_threshold"]) & (params["scales"].exp().max(dim=1).values
                                                     <= limit)
        new = {k: params[k][clone] for k in FIELDS}
        if cfg["clone_opacity_rescale"]:
            for k in OPACITY_FIELDS:
                new[k] = core.clone_opacity_logit(new[k])
        append_gaussians(params, optimizers, new)
        n_clone = int(clone.sum())

        # ---- densify_and_split: N=2 samples R N(0, s) + mu, scale s / 1.6, rest copied
        grads = torch.cat([grads, torch.zeros(n_clone, device=grads.device)])
        split = (grads >= cfg["grad_threshold"]) & (params["scales"].exp().max(dim=1).values
                                                     > limit)
        K = int(split.sum())
        off = core.split_offsets(params["scales"][split].exp(), params["quats"][split],
                                 self.generator)
        new = {k: params[k][split].repeat(2, *([1] * (params[k].dim() - 1))) for k in FIELDS}
        new["means"] = off + params["means"][split].repeat(2, 1)
        new["scales"] = torch.log(params["scales"][split].exp().repeat(2, 1) / (0.8 * 2))
        append_gaussians(params, optimizers, new)
        remove(params, optimizers, {}, torch.cat([split, torch.zeros(
            2 * K, dtype=torch.bool, device=split.device)]))

        # ---- prune: max over the two halves < min_opacity; world-size after step 3000.
        # max_radii2D was zeroed by densification_postfix, so the official screen-size test
        # (> 20 px) never fires (TODO-runtime.md); only the world-size test acts.
        alpha_max = torch.stack([params[k] for k in OPACITY_FIELDS], -1).sigmoid().max(-1).values
        prune = alpha_max < cfg["min_opacity"]
        if step > cfg["big_world_after"]:
            prune |= params["scales"].exp().max(dim=1).values > cfg["big_world"] * self.extent
        remove(params, optimizers, {}, prune)
        self._reset_stats(len(params["means"]), params["means"].device)
        row = {"step": int(step), "n_before": n0, "clone": n_clone, "split": K,
               "pruned": int(prune.sum()), "n_after": len(params["means"])}
        self.state["events"].append(row)
        if step % 1000 == 0:
            print({"3dhgs": row}, flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_density(spec):
    return HGSDensity(spec)
