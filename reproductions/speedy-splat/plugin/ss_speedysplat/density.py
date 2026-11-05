"""Speedy-Splat density control: official 3D-GS ADC (scene/gaussian_model.py) plus Soft and
Hard Pruning with the efficient sensitivity score (train.py `prune`, speedy-splat @ 34c45c6).

Per training step s (the runtime calls `after_backward` after the optimizer step; the official
loop densifies/soft-prunes before it and hard-prunes after it):
  s < densify_until:
    accumulate ||dL/d means2D[:, :2]|| and the visit count for Gaussians with radii > 0;
    s > densify_from and s % 100 == 0: densify_and_clone + densify_and_split + prune;
    SOFT prune (s >= prune_from, s < prune_until, s % prune_every == 0): soft_ratio;
    s % 3000 == 0: opacity reset to min(o, 0.01) with zeroed Adam moments.
  HARD prune (s >= densify_until, s >= prune_from, s < prune_until, s % prune_every == 0):
    hard_ratio.
"""

import torch
from gsplat.strategy.ops import _update_param_with_optimizer, remove, reset_opa
from torch import nn

DEFAULTS = {
    "densify_from": 500,  # densify_from_iter
    "densify_until": 15000,  # densify_until_iter
    "densify_every": 100,  # densification_interval
    "grad_threshold": 0.0002,  # densify_grad_threshold
    "percent_dense": 0.01,
    "min_opacity": 0.005,  # train.py densify_and_prune(..., 0.005, ...)
    "big_world_after": 3000,  # size_threshold set when iteration > opacity_reset_interval
    "big_world": 0.1,  # big_points_ws: max scale > 0.1 * extent
    "reset_every": 3000,  # opacity_reset_interval
    "reset_value": 0.01,
    "prune": True,  # Speedy-Splat soft + hard pruning (False = paper rows +SnugBox/+AccuTile)
    "prune_from": 6000,  # prune_from_iter
    "prune_until": 30000,  # prune_until_iter (exclusive)
    "prune_every": 3000,  # prune_interval
    "soft_ratio": 0.80,  # densify_prune_ratio (paper §4.2.2)
    "hard_ratio": 0.30,  # after_densify_prune_ratio (paper §4.2.3)
    "score_background": [0.0, 0.0, 0.0],  # background of the score renders (official bg)
}
REQUIRES = {"gradient.screen.signed"}
BASE_FIELDS = ("means", "scales", "quats", "opacities", "sh0", "shN")


def quat_to_rotmat(quats):
    """utils/general_utils.build_rotation (wxyz, normalised)."""
    r, x, y, z = torch.nn.functional.normalize(quats, dim=-1).unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def prune_mask(scores, ratio):
    """GaussianModel.prune_gaussians: prune scores <= sorted[int(ratio * (N - 1))]."""
    if len(scores) == 0:
        return torch.zeros(0, dtype=torch.bool, device=scores.device)
    sorted_scores, _ = torch.sort(scores, dim=0)
    value = sorted_scores[int(ratio * (sorted_scores.shape[0] - 1))]
    return scores <= value


def prune_steps(cfg, steps=30000):
    """(soft, hard) prune iterations implied by the official train.py conditions."""
    soft, hard = [], []
    for s in range(1, steps + 1):
        on = s >= cfg["prune_from"] and s < cfg["prune_until"] and s % cfg["prune_every"] == 0
        if on and s < cfg["densify_until"]:
            soft.append(s)
        elif on and s >= cfg["densify_until"]:
            hard.append(s)
    return soft, hard


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


class SpeedyDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown Speedy-Splat density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        for k in ("soft_ratio", "hard_ratio"):
            if not 0.0 <= float(self.cfg[k]) < 1.0:
                raise ValueError(f"{k} must be in [0, 1)")
        self.requires = set(REQUIRES)
        self.state = None
        self.generator = None
        self.cameras = None
        # score_fn(params, cameras, sh_degree, background) -> [N]; default = vendored kernel.
        self.score_fn = None

    # ------------------------------------------------------------------ runtime hooks
    def initialize(self, params, optimizers, scene_scale=1.0):
        extra = set(params) - set(BASE_FIELDS)
        if extra:
            raise ValueError(f"Speedy-Splat density does not know fields {sorted(extra)}")
        self.extent = float(scene_scale)
        self._reset_stats(len(params["means"]), params["means"].device)
        self.state = {"events": [], "prunes": []}

    def set_training_cameras(self, cameras):
        self.cameras = list(cameras)

    def before_backward(self, params, optimizers, step, outputs):
        if step < self.cfg["densify_until"]:
            outputs["info"]["means2d"].retain_grad()

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.cfg
        prune_on = (cfg["prune"] and step >= cfg["prune_from"] and step < cfg["prune_until"]
                    and step % cfg["prune_every"] == 0)
        if step < cfg["densify_until"]:
            self._accumulate(outputs["info"])
            if step > cfg["densify_from"] and step % cfg["densify_every"] == 0:
                self._densify(params, optimizers, step)
            if prune_on:  # --- Soft Pruning (before the opacity reset, train.py order)
                self._prune(params, optimizers, step, cfg["soft_ratio"], "soft")
            if step % cfg["reset_every"] == 0:
                reset_opa(params, optimizers, {}, cfg["reset_value"])
        elif prune_on:  # --- Hard Pruning
            self._prune(params, optimizers, step, cfg["hard_ratio"], "hard")

    # ------------------------------------------------------------------ pruning
    def scores(self, params, step):
        if self.cameras is None:
            raise RuntimeError("Speedy-Splat pruning needs set_training_cameras()")
        degree = min((step - 1) // 1000, 3)  # SH degree the runtime uses at this step
        bg = torch.tensor(self.cfg["score_background"], dtype=torch.float32,
                          device=params["means"].device)
        fn = self.score_fn
        if fn is None:
            from .raster import speedy_scores as fn
        return fn(params, self.cameras, degree, bg)

    def _prune(self, params, optimizers, step, ratio, kind):
        n0 = len(params["means"])
        scores = self.scores(params, step)
        if scores.shape != (n0,):
            raise RuntimeError(f"score shape {tuple(scores.shape)} != ({n0},)")
        mask = prune_mask(scores, ratio)
        remove(params, optimizers, {}, mask)
        keep = ~mask
        self.vgn, self.denom = self.vgn[keep], self.denom[keep]
        row = {"step": int(step), "kind": kind, "ratio": float(ratio), "n_before": n0,
               "n_after": len(params["means"]), "zero_scores": int((scores == 0).sum())}
        self.state["prunes"].append(row)
        print({"speedy-splat-prune": row}, flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ ADC (3D-GS)
    def _reset_stats(self, n, device):
        self.vgn = torch.zeros(n, device=device)
        self.denom = torch.zeros(n, device=device)

    def _accumulate(self, info):
        g2 = info["means2d"].grad
        if g2 is None:
            raise RuntimeError("screen-space gradient missing: render with the Speedy renderer")
        vis = info["radii"] > 0
        self.vgn[vis] += torch.norm(g2[vis, :2], dim=-1)  # add_densification_stats
        self.denom[vis] += 1

    def _densify(self, params, optimizers, step):
        cfg = self.cfg
        n0 = len(params["means"])
        grads = self.vgn / self.denom
        grads[grads.isnan()] = 0.0
        thr = cfg["percent_dense"] * self.extent
        # densify_and_clone
        clone = (grads >= cfg["grad_threshold"]) & (params["scales"].exp().max(dim=1).values
                                                   <= thr)
        append_gaussians(params, optimizers, {k: params[k][clone] for k in BASE_FIELDS})
        n_clone = int(clone.sum())
        # densify_and_split (N = 2) on the padded gradient
        grads = torch.cat([grads, torch.zeros(n_clone, device=grads.device)])
        split = (grads >= cfg["grad_threshold"]) & (params["scales"].exp().max(dim=1).values
                                                   > thr)
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
        # prune (densify_and_prune): max_radii2D was zeroed by densification_postfix, so the
        # screen-size test (> 20 px) never fires; only the world-size test acts.
        pr = params["opacities"].sigmoid() < cfg["min_opacity"]
        if step > cfg["big_world_after"]:
            pr |= params["scales"].exp().max(dim=1).values > cfg["big_world"] * self.extent
        remove(params, optimizers, {}, pr)
        self._reset_stats(len(params["means"]), params["means"].device)
        row = {"step": int(step), "n_before": n0, "clone": n_clone, "split": K,
               "pruned": int(pr.sum()), "n_after": len(params["means"])}
        self.state["events"].append(row)
        if step % 1000 == 0:
            print({"speedy-splat-densify": row}, flush=True)
        if torch.cuda.is_available() and step % 1000 == 0:
            torch.cuda.empty_cache()


def create_density(spec):
    return SpeedyDensity(spec)

