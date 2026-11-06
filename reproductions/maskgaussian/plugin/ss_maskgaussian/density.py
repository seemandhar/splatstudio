"""MaskGaussian density control: official 3D-GS ADC (MaskGaussian scene/gaussian_model.py, the
Inria code base) with mask-sampling pruning.

Per training step s (the runtime calls `after_backward` after the optimizer step; the official
loop densifies before it):
  s < densify_until:
    accumulate ||dL/d means2D[:, :2]|| and the visit count for Gaussians with radii > 0;
    s > densify_from and s % 100 == 0: densify_and_clone + densify_and_split, then prune
      (opacity < 0.005) | (never present in `prune_samples` Gumbel draws) | (big in world space
      after 3000);   [densify_and_prune]
    s % 3000 == 0: opacity reset to min(o, 0.01) with zeroed Adam moments.
  s >= densify_until and s % mask_prune_every == 0: remove Gaussians never present in
    `prune_samples` draws  [mask_prune].
  s == total_steps and final_sample: keep exactly one Gumbel sample of the mask
    [save_ply_default writes only Gaussians with M = 1; test renders use that model].
The mask field (`mask_scores`, [N, 2]) is copied on clone/split like every other attribute.
"""

import torch
from gsplat.strategy.ops import _update_param_with_optimizer, remove, reset_opa
from torch import nn

from .renderer import DEFAULTS as RENDERER_DEFAULTS
from .renderer import FIELD, sample_mask

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
    "prune_samples": 10,  # sample_times in densify_and_prune / mask_prune
    "mask_prune_every": 1000,  # mask_prune_iter
    "mask_prune_in_densify": True,  # never-sampled test inside densify_and_prune
    "total_steps": 30000,  # iterations (final single-sample save)
    "final_sample": True,  # save_ply_default: keep one mask sample at the last step
    "score_offset": list(RENDERER_DEFAULTS["score_offset"]),  # must match the renderer
    "tau": RENDERER_DEFAULTS["tau"],
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


def never_sampled(scores, offset, tau=1.0, samples=10):
    """Official: gumbel_softmax(log_softmax(score repeated `samples` times))[..., 0].sum(0)==0."""
    hits = torch.zeros(len(scores), dtype=scores.dtype, device=scores.device)
    for _ in range(int(samples)):
        hits += sample_mask(scores, offset, tau)[:, 0]
    return hits == 0


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


def mask_prune_steps(cfg, steps):
    """Steps at which train.py calls gaussians.mask_prune() (elif branch after densification)."""
    return [s for s in range(1, steps + 1)
            if s >= cfg["densify_until"] and s % cfg["mask_prune_every"] == 0]


class MaskDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown MaskGaussian density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        if int(self.cfg["prune_samples"]) < 1:
            raise ValueError("prune_samples must be >= 1")
        self.requires = set(REQUIRES)
        self.state = None
        self.generator = None

    # ------------------------------------------------------------------ runtime hooks
    def initialize(self, params, optimizers, scene_scale=1.0):
        extra = set(params) - set(BASE_FIELDS) - {FIELD}
        if extra or FIELD not in params:
            raise ValueError(f"MaskGaussian density needs exactly the base fields + {FIELD}")
        self.fields = tuple(BASE_FIELDS) + (FIELD,)
        self.extent = float(scene_scale)
        self._reset_stats(len(params["means"]), params["means"].device)
        self.state = {"events": [], "mask_prunes": []}

    def before_backward(self, params, optimizers, step, outputs):
        if step < self.cfg["densify_until"]:
            outputs["info"]["means2d"].retain_grad()

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.cfg
        if step < cfg["densify_until"]:
            self._accumulate(outputs["info"])
            if step > cfg["densify_from"] and step % cfg["densify_every"] == 0:
                self._densify(params, optimizers, step)
            if step % cfg["reset_every"] == 0:
                reset_opa(params, optimizers, {}, cfg["reset_value"])
        elif step % cfg["mask_prune_every"] == 0 and step < cfg["total_steps"]:
            self._remove(params, optimizers, self._never_sampled(params), step, "mask_prune")
        if cfg["final_sample"] and step == cfg["total_steps"]:
            # Official order at 30000: scene.save (one sample) precedes that step's mask_prune,
            # whose result is never saved; so only the single-sample keep acts here.
            keep = sample_mask(params[FIELD], cfg["score_offset"], float(cfg["tau"]))[:, 0] > 0.5
            self._remove(params, optimizers, ~keep, step, "final_sample")

    # ------------------------------------------------------------------ mask pruning
    def _never_sampled(self, params):
        cfg = self.cfg
        return never_sampled(params[FIELD], cfg["score_offset"], float(cfg["tau"]),
                             cfg["prune_samples"])

    def _remove(self, params, optimizers, mask, step, kind):
        n0 = len(params["means"])
        if bool(mask.all()):
            raise RuntimeError(f"{kind} would remove every Gaussian")
        remove(params, optimizers, {}, mask)
        keep = ~mask
        self.vgn, self.denom = self.vgn[keep], self.denom[keep]
        row = {"step": int(step), "kind": kind, "n_before": n0, "n_after": len(params["means"])}
        self.state["mask_prunes"].append(row)
        print({"maskgaussian-prune": row}, flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ ADC (3D-GS)
    def _reset_stats(self, n, device):
        self.vgn = torch.zeros(n, device=device)
        self.denom = torch.zeros(n, device=device)

    def _accumulate(self, info):
        g2 = info["means2d"].grad
        if g2 is None:
            raise RuntimeError("screen-space gradient missing: render with the MaskGaussian "
                               "renderer")
        vis = info["radii"] > 0
        self.vgn[vis] += torch.norm(g2[vis, :2], dim=-1)  # add_densification_stats
        self.denom[vis] += 1

    def _densify(self, params, optimizers, step):
        cfg = self.cfg
        n0 = len(params["means"])
        grads = self.vgn / self.denom
        grads[grads.isnan()] = 0.0
        thr = cfg["percent_dense"] * self.extent
        # densify_and_clone (all attributes incl. the mask scores are copied)
        clone = (grads >= cfg["grad_threshold"]) & (params["scales"].exp().max(dim=1).values
                                                   <= thr)
        append_gaussians(params, optimizers, {k: params[k][clone] for k in self.fields})
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
               "shN": params["shN"][split].repeat(2, 1, 1),
               FIELD: params[FIELD][split].repeat(2, 1)}
        append_gaussians(params, optimizers, new)
        remove(params, optimizers, {}, torch.cat([split, torch.zeros(
            2 * K, dtype=torch.bool, device=split.device)]))
        # densify_and_prune: opacity | never sampled in 10 draws | big world-space. max_radii2D
        # was zeroed by densification_postfix, so the screen-size test (> 20 px) never fires.
        low = params["opacities"].sigmoid() < cfg["min_opacity"]
        unsampled = (self._never_sampled(params) if cfg["mask_prune_in_densify"]
                     else torch.zeros_like(low))
        pr = low | unsampled
        if step > cfg["big_world_after"]:
            pr |= params["scales"].exp().max(dim=1).values > cfg["big_world"] * self.extent
        remove(params, optimizers, {}, pr)
        self._reset_stats(len(params["means"]), params["means"].device)
        row = {"step": int(step), "n_before": n0, "clone": n_clone, "split": K,
               "pruned": int(pr.sum()), "pruned_unsampled": int(unsampled.sum()),
               "n_after": len(params["means"])}
        self.state["events"].append(row)
        if step % 1000 == 0:
            print({"maskgaussian-densify": row}, flush=True)
        if torch.cuda.is_available() and step % 1000 == 0:
            torch.cuda.empty_cache()


def create_density(spec):
    return MaskDensity(spec)
