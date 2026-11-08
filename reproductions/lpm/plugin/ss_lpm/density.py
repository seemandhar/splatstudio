"""LPM density controller: 3DGS ADC with Localized Points Management.

Follows the official training loop (GS-LPM `train.py` + `lpm/lpm.py`, @7c06026):
- statistics as 3DGS (mean view-space positional gradient) for steps < densify_until;
- every `densify_every` steps in (densify_from, densify_until): error-source zones from this
  step's render, then `lpm_densify_and_prune` (C4) - this *replaces* 3DGS densify_and_prune;
  when no zone is found nothing is densified or pruned and statistics keep accumulating;
- global 3DGS opacity reset every `opacity_reset_every` steps (< densify_until);
- every `reset_every` steps in (reset_from, reset_until): localized opacity reset (C5).

The runtime exposes training images only through renders, so every rendered training target is
cached and the offline SuperPoint/LightGlue pairing (official `get_paired_views`, 3 random
neighbours within `angle`) is built lazily from the neighbours already seen (fidelity gap 1).
"""

import random

import torch
from gsplat import DefaultStrategy
from gsplat.strategy.ops import duplicate, remove, reset_opa, split

from . import core

DEFAULTS = {
    "densify_from": 500,
    "densify_until": 15000,
    "densify_every": 100,
    "grad_threshold": 0.0002,
    "percent_dense": 0.01,
    "zone_grad_ratio": 0.5,
    "min_opacity": 0.005,
    "world_prune_after": 3000,  # official size_threshold is None until opacity_reset_interval
    "world_prune_ratio": 0.1,
    "opacity_reset_every": 3000,
    "reset_value": 0.01,
    "reset_from": 500,
    "reset_until": 15000,
    "reset_every": 100,
    "reset_top_ratio": 0.7,
    "angle": 45.0,
    "neighbours": 3,
    "error_quantile": 0.4,
    "error_patch": 16,
    "error_fill": 0.85,
    "enable_addition": True,
    "enable_reset": True,
    "seed": 0,
}
REQUIRES = ("gradient.screen.signed", "training.target_image")
FIELDS = ("means", "scales", "quats", "opacities", "sh0", "shN")


def check_trainable(params):
    """Guard against the frozen-field pitfall (TODO-runtime.md): every field stays a trainable leaf."""
    for k, p in params.items():
        if not (isinstance(p, torch.nn.Parameter) and p.is_leaf and p.requires_grad):
            raise RuntimeError(f"LPM density left field {k!r} non-trainable")


class LPMDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown lpm-density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.requires = set(REQUIRES)
        # Used only for its gradient bookkeeping (_update_state / step_pre_backward).
        self.strategy = DefaultStrategy(refine_stop_iter=self.cfg["densify_until"])
        self.state = None
        self.rng = random.Random(self.cfg["seed"])
        self.cameras = None
        self.matcher = None
        self.matcher_factory = None  # probes may inject a stub

    # ------------------------------------------------------------------ setup
    def initialize(self, params, optimizers, scene_scale=1.0):
        self.strategy.check_sanity(params, optimizers)
        self.state = self.strategy.initialize_state(scene_scale=scene_scale)
        self.state.update(
            lpm_steps=0, lpm_zones=0, lpm_extra=0, lpm_reset=0, lpm_no_zone=0, lpm_no_pair=0,
            lpm_matched_pairs=0,
        )

    def set_training_cameras(self, cameras):
        self.cameras = list(cameras)
        self.viewmats = torch.stack([c["viewmat"] for c in self.cameras]).flatten(1)
        dirs = core.center_ray_dirs(self.cameras)
        mask = core.neighbour_mask(dirs, self.cfg["angle"]).cpu()
        self.neighbours = [torch.nonzero(row).flatten().tolist() for row in mask]
        self.images = [None] * len(self.cameras)  # uint8 HWC on CPU
        self.feats = [None] * len(self.cameras)
        self.pairs = [[] for _ in self.cameras]  # (pts_cur, pts_ref, ref_index)

    def view_index(self, viewmat):
        d = (self.viewmats - viewmat.reshape(1, -1).to(self.viewmats)).abs().amax(dim=1)
        k = int(torch.argmin(d))
        return k if float(d[k]) < 1e-5 else None

    # ------------------------------------------------------------------ pairing (C2)
    def _matcher(self, device):
        if self.matcher is None:
            if self.matcher_factory is not None:
                self.matcher = self.matcher_factory(device)
            else:
                from .matching import Matcher

                self.matcher = Matcher(device)
        return self.matcher

    def _features(self, k, device):
        if self.feats[k] is None:
            img = self.images[k].to(device).float() / 255
            self.feats[k] = self._matcher(device).features(img)
        return self.feats[k]

    def _ensure_pairs(self, i, device):
        """Official: up to `neighbours` random angle-neighbours (the view itself if none),
        matched once. Here restricted to neighbours whose image has been seen; topped up later."""
        valid = self.neighbours[i] or [i]
        want = min(self.cfg["neighbours"], len(valid))
        if len(self.pairs[i]) >= want:
            return
        have = {p[2] for p in self.pairs[i]}
        cand = [j for j in valid if j not in have and self.images[j] is not None]
        for j in self.rng.sample(cand, min(want - len(self.pairs[i]), len(cand))):
            m = self._matcher(device)
            p0, p1 = m.match(self._features(i, device), self._features(j, device))
            self.pairs[i].append((p0.to(device), p1.to(device), j))
            self.state["lpm_matched_pairs"] += 1

    def zones_for(self, i, rgb, device):
        """Error-source zones for training view i rendered as `rgb` (HWC). None = no pair yet."""
        self._ensure_pairs(i, device)
        if not self.pairs[i]:
            return None
        pts_cur, pts_ref, j = self.pairs[i][self.rng.randrange(len(self.pairs[i]))]
        target = self.images[i].to(device).float() / 255
        emap = core.error_map(
            rgb.detach().float(), target,
            self.cfg["error_quantile"], self.cfg["error_patch"], self.cfg["error_fill"],
        )
        r_cur, r_ref = core.paired_regions(emap, pts_cur, pts_ref)
        return core.zones3d(self.cameras[i], self.cameras[j], r_cur, r_ref)

    # ------------------------------------------------------------------ hooks
    def before_backward(self, params, optimizers, step, outputs):
        if step < self.cfg["densify_until"]:
            self.strategy.step_pre_backward(params, optimizers, self.state, step, outputs["info"])

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.cfg
        if self.cameras is None:
            # contract probes call the hook at step 1 without cameras; training always sets them
            if step > min(cfg["densify_from"], cfg["reset_from"]):
                raise RuntimeError("LPM density needs set_training_cameras() before training")
            if step < cfg["densify_until"]:
                self.strategy._update_state(params, self.state, outputs["info"], packed=False)
            return
        i = self.view_index(outputs["viewmat"])
        if i is not None and self.images[i] is None and step < max(cfg["densify_until"], cfg["reset_until"]):
            self.images[i] = (outputs["target_image"].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
        if step < cfg["densify_until"]:
            self.strategy._update_state(params, self.state, outputs["info"], packed=False)

        do_add = (cfg["enable_addition"] and cfg["densify_from"] < step < cfg["densify_until"]
                  and step % cfg["densify_every"] == 0)
        do_reset = (cfg["enable_reset"] and cfg["reset_from"] < step < cfg["reset_until"]
                    and step % cfg["reset_every"] == 0)
        zones = None
        if (do_add or do_reset) and i is not None:
            zones = self.zones_for(i, outputs["rgb"], params["means"].device)
            if zones is None:
                self.state["lpm_no_pair"] += 1
            else:
                self.state["lpm_steps"] += 1
                self.state["lpm_zones"] += len(zones)
        # official: points_addition only acts when zones were found (else no densification)
        if not cfg["enable_addition"]:
            if cfg["densify_from"] < step < cfg["densify_until"] and step % cfg["densify_every"] == 0:
                self.densify_and_prune(params, optimizers, step, [])  # plain 3DGS (ablation)
        elif do_add:
            if zones:
                self.densify_and_prune(params, optimizers, step, zones)
            else:
                self.state["lpm_no_zone"] += 1
        if step < cfg["densify_until"] and step % cfg["opacity_reset_every"] == 0:
            reset_opa(params, optimizers, self.state, value=cfg["reset_value"])
            check_trainable(params)
        if do_reset and zones:
            self.localized_reset(params, i, zones)
        if step == max(cfg["densify_until"], cfg["reset_until"]):
            self.matcher, self.feats, self.images = None, None, [None] * len(self.cameras)
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ C4
    def densify_and_prune(self, params, optimizers, step, zones):
        cfg, st = self.cfg, self.state
        grads = st["grad2d"] / st["count"].clamp_min(1)
        max_scale = params["scales"].exp().max(dim=1).values
        in_zone = core.in_zones(params["means"], zones)
        clone, split_mask, n_extra = core.densify_masks(
            grads, max_scale, in_zone, cfg["grad_threshold"],
            cfg["percent_dense"] * st["scene_scale"], cfg["zone_grad_ratio"],
        )
        n_clone = int(clone.sum())
        if n_clone:
            duplicate(params, optimizers, st, clone)
        split_mask = torch.cat([split_mask, torch.zeros(n_clone, dtype=torch.bool, device=split_mask.device)])
        if int(split_mask.sum()):
            split(params, optimizers, st, split_mask)
        world = (cfg["world_prune_ratio"] * st["scene_scale"]
                 if step > cfg["world_prune_after"] else None)
        prune = core.prune_mask(
            params["opacities"].detach(), params["scales"].exp().max(dim=1).values, n_extra,
            cfg["min_opacity"], world,
        )
        if int(prune.sum()):
            remove(params, optimizers, st, prune)
        # official densification_postfix zeroes all statistics
        st["grad2d"].zero_()
        st["count"].zero_()
        st["lpm_extra"] += n_extra
        check_trainable(params)

    # ------------------------------------------------------------------ C5
    def localized_reset(self, params, i, zones):
        cam_pos = core.camera_center(self.cameras[i]).to(params["means"].device)
        mask = core.reset_mask(
            cam_pos, zones, params["means"].detach(), params["opacities"].detach(),
            self.cfg["reset_top_ratio"],
        )
        n = int(mask.sum())
        if n:
            # official writes _opacity in place; Adam moments are left untouched
            # same float32 value as the global reset (official: inverse_sigmoid(0.01) both times)
            value = torch.logit(torch.tensor(self.cfg["reset_value"])).item()
            params["opacities"].data[mask] = value
            self.state["lpm_reset"] += n
        check_trainable(params)


def create_density(spec):
    return LPMDensity(spec)
