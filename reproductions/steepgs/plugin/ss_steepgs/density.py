"""SteepGS density control (official train.py + scene/gaussian_model.py, `--densify_strategy
steepest --S_estimator inv_cov`, github.com/facebookresearch/SteepGS commit 63eee45).

Per training step s (the runtime calls `after_backward` after the optimizer step; the official
loop densifies before it):
  s < densify_until: accumulate for visible Gaussians (radii > 0)
      vgn   += ||dL/d means2D[:, :2]||          (Inria NDC-space screen gradient)
      S     += (sum_pix dL/dG) * Sigma^-1        (inv_cov estimator, Sigma of the pre-step params)
      denom += 1
    s > densify_from and s % 100 == 0: densify_and_split_steepest + prune, reset statistics;
    s % 3000 == 0: opacity reset to min(o, 0.01) with zeroed Adam moments.
"""

import torch
from gsplat.strategy.ops import _update_param_with_optimizer, remove, reset_opa
from torch import nn

from . import core
from .renderer import SPLITTING_CAPABILITY

DEFAULTS = {
    "densify_from": 500,  # OptimizationParams.densify_from_iter (paper §4.1)
    "densify_until": 15000,  # densify_until_iter
    "densify_every": 100,  # densification_interval (paper §4.1)
    "grad_threshold": 0.0002,  # densify_grad_threshold ("uncertain" filter)
    "S_threshold": -1e-6,  # densify_S_threshold (paper §4.1)
    "min_grad": 1e-4,  # stationary filter (gaussian_model.py:486), only with 'stationary'
    "percent_dense": 0.01,
    "min_opacity": 0.005,  # train.py:190
    "big_world_after": 3000,  # size_threshold set when iteration > opacity_reset_interval
    "big_world": 0.1,  # big_points_ws: max scale > 0.1 * extent
    "reset_every": 3000,  # opacity_reset_interval
    "reset_value": 0.01,
    "options": [],  # densify_strategy attributes after 'steepest' (core.OPTIONS)
    "strategy": "steepest",  # 'adc' = official 3DGS branch of densify_and_prune (control)
}
REQUIRES = {"gradient.screen.signed", SPLITTING_CAPABILITY}
BASE_FIELDS = ("means", "scales", "quats", "opacities", "sh0", "shN")


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


class SteepDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown SteepGS density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.options = tuple(self.cfg["options"])
        bad = set(self.options) - set(core.OPTIONS)
        if bad:
            raise ValueError(f"Unknown SteepGS densify_strategy attributes: {sorted(bad)}")
        if self.cfg["strategy"] not in ("steepest", "adc"):
            raise ValueError("strategy must be 'steepest' or 'adc'")
        if self.cfg["S_threshold"] >= 0:
            raise ValueError("S_threshold must be a negative number.")
        self.requires = set(REQUIRES)
        self.state = None
        self.generator = None
        self._cov_inv = None
        self._xyz_grad = None
        self._hook = None

    # ------------------------------------------------------------------ runtime hooks
    def initialize(self, params, optimizers, scene_scale=1.0):
        extra = set(params) - set(BASE_FIELDS)
        if extra:
            raise ValueError(f"SteepGS density does not know fields {sorted(extra)}")
        self.extent = float(scene_scale)
        self._reset_stats(len(params["means"]), params["means"].device)
        self.state = {"events": []}

    def before_backward(self, params, optimizers, step, outputs):
        if step >= self.cfg["densify_until"]:
            return
        info = outputs["info"]
        info["means2d"].retain_grad()
        with torch.no_grad():  # Sigma^-1 of the parameters that produced this render
            self._cov_inv = core.inverse_covariance(params["scales"].detach(),
                                                    params["quats"].detach())
        if "stationary" in self.options:
            def grab(g):
                self._xyz_grad = g.detach().clone()
            self._hook = params["means"].register_hook(grab)

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.cfg
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        if step >= cfg["densify_until"]:
            return
        self._accumulate(outputs["info"])
        if step > cfg["densify_from"] and step % cfg["densify_every"] == 0:
            self._densify(params, optimizers, step)
        if step % cfg["reset_every"] == 0:
            reset_opa(params, optimizers, {}, cfg["reset_value"])

    # ------------------------------------------------------------------ statistics
    def _reset_stats(self, n, device):
        self.vgn = torch.zeros(n, device=device)
        self.xyz_grad = torch.zeros(n, 3, device=device)
        self.S = torch.zeros(n, 3, 3, device=device)
        self.denom = torch.zeros(n, device=device)

    def _accumulate(self, info):
        g2 = info["means2d"].grad
        gs = info["splitting_mats"].grad
        if g2 is None or gs is None:
            raise RuntimeError("SteepGS statistics missing: render with the SteepGS renderer")
        vis = info["radii"] > 0
        self.vgn[vis] += torch.norm(g2[vis, :2], dim=-1)  # add_densification_stats
        dLdG = gs[vis][:, 0, 0]
        self.S[vis] += dLdG[:, None, None] * self._cov_inv[vis]  # inv_cov estimator
        if "stationary" in self.options:
            self.xyz_grad[vis] += self._xyz_grad[vis]
        self.denom[vis] += 1

    # ------------------------------------------------------------------ densification
    def _densify(self, params, optimizers, step):
        if self.cfg["strategy"] == "adc":
            return self._densify_adc(params, optimizers, step)
        cfg, opts = self.cfg, self.options
        n0 = len(params["means"])
        vgn = core.averaged(self.vgn, self.denom)
        S = core.averaged(self.S, self.denom)
        xyz_norm = torch.norm(core.averaged(self.xyz_grad, self.denom), dim=-1)
        lam, vec = core.smallest_eig(S)
        selected = core.select(opts, vgn, lam, cfg["grad_threshold"], cfg["S_threshold"],
                               xyz_norm, self.denom, cfg["min_grad"])
        max_scale = params["scales"].exp().max(dim=1).values
        big = max_scale > cfg["percent_dense"] * self.extent
        split_mask, clone_mask = selected & big, selected & ~big

        # offsets of split offspring (computed on the pre-clone population, as official)
        scales_split = params["scales"][split_mask].exp()
        if "no_eig_upd" in opts:
            off_split = core.random_split_offsets(scales_split, params["quats"][split_mask],
                                                  opts, self.generator)
        else:
            off_split = core.split_offsets(vec[split_mask], scales_split, opts, self.generator)

        # ---- clone: copy and original both at p + lr * v_min (gaussian_model.py:590-612)
        if "no_eig_upd" in opts:
            off_clone = torch.zeros_like(vec[clone_mask])
        else:
            lr = optimizers["means"].param_groups[0]["lr"]
            off_clone = vec[clone_mask] * lr
        means = params["means"]
        new = {"means": means[clone_mask] + off_clone}
        means.data[clone_mask] = means.data[clone_mask] + off_clone
        for k in ("scales", "quats", "sh0", "shN"):
            new[k] = params[k][clone_mask]
        o = params["opacities"][clone_mask]
        new["opacities"] = torch.logit(o.sigmoid() * 0.5) if "no_div_opacity" in opts else o
        append_gaussians(params, optimizers, new)
        n_clone = int(clone_mask.sum())

        # ---- split: 2 offspring at p +- |n| v_min, scale s / 1.6, originals removed
        pad = torch.zeros(n_clone, dtype=torch.bool, device=split_mask.device)
        split_full = torch.cat([split_mask, pad])
        K = int(split_full.sum())
        new = {"means": off_split + params["means"][split_full].repeat(2, 1),
               "scales": torch.log(params["scales"][split_full].exp().repeat(2, 1) / (0.8 * 2)),
               "quats": params["quats"][split_full].repeat(2, 1),
               "sh0": params["sh0"][split_full].repeat(2, 1, 1),
               "shN": params["shN"][split_full].repeat(2, 1, 1)}
        o = params["opacities"][split_full].repeat(2)
        new["opacities"] = torch.logit(o.sigmoid() * 0.5) if "no_div_opacity" in opts else o
        append_gaussians(params, optimizers, new)
        remove(params, optimizers, {}, torch.cat([split_full, torch.zeros(
            2 * K, dtype=torch.bool, device=split_full.device)]))

        self._prune_and_log(params, optimizers, step, n0, selected=int(selected.sum()),
                            clone=n_clone, split=K,
                            saddle_eig=int((lam < cfg["S_threshold"]).sum()),
                            uncertain=int((vgn >= cfg["grad_threshold"]).sum()))

    def _densify_adc(self, params, optimizers, step):
        """Official 'adc' branch: densify_and_clone + densify_and_split (3DGS)."""
        cfg = self.cfg
        n0 = len(params["means"])
        grads = core.averaged(self.vgn, self.denom)
        small = params["scales"].exp().max(dim=1).values <= cfg["percent_dense"] * self.extent
        clone = (grads >= cfg["grad_threshold"]) & small
        append_gaussians(params, optimizers, {k: params[k][clone] for k in BASE_FIELDS})
        n_clone = int(clone.sum())
        grads = torch.cat([grads, torch.zeros(n_clone, device=grads.device)])
        split = (grads >= cfg["grad_threshold"]) & (
            params["scales"].exp().max(dim=1).values > cfg["percent_dense"] * self.extent)
        K = int(split.sum())
        off = core.random_split_offsets(params["scales"][split].exp(), params["quats"][split],
                                        (), self.generator)
        new = {"means": off + params["means"][split].repeat(2, 1),
               "scales": torch.log(params["scales"][split].exp().repeat(2, 1) / (0.8 * 2)),
               "quats": params["quats"][split].repeat(2, 1),
               "opacities": params["opacities"][split].repeat(2),
               "sh0": params["sh0"][split].repeat(2, 1, 1),
               "shN": params["shN"][split].repeat(2, 1, 1)}
        append_gaussians(params, optimizers, new)
        remove(params, optimizers, {}, torch.cat([split, torch.zeros(
            2 * K, dtype=torch.bool, device=split.device)]))
        self._prune_and_log(params, optimizers, step, n0, selected=n_clone + K, clone=n_clone,
                            split=K)

    def _prune_and_log(self, params, optimizers, step, n0, **counts):
        cfg = self.cfg
        # ---- prune (densify_and_prune): max_radii2D was zeroed by densification_postfix, so
        # the screen-size test (> 20 px) never fires; only the world-size test acts.
        prune = params["opacities"].sigmoid() < cfg["min_opacity"]
        if step > cfg["big_world_after"]:
            prune |= params["scales"].exp().max(dim=1).values > cfg["big_world"] * self.extent
        remove(params, optimizers, {}, prune)
        self._reset_stats(len(params["means"]), params["means"].device)
        row = {"step": int(step), "n_before": n0, **counts, "pruned": int(prune.sum()),
               "n_after": len(params["means"])}
        self.state["events"].append(row)
        if step % 1000 == 0:
            print({"steepgs": row}, flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_density(spec):
    return SteepDensity(spec)
