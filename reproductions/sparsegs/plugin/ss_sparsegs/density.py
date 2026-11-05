"""SparseGS density: Inria 3DGS adaptive density as configured by SparseGS, the warp-target
generation and the dip-test floater pruning (C5). Official train.py / scene/gaussian_model.py.

after_backward(step) (the runtime already stepped the optimizers) performs, in order:
  1. xyz lr of iteration step+1: expon(1.6e-4 -> 1.6e-6 over position_lr_max_steps=30000) x
     extent (Inria update_learning_rate; the runtime's own schedule is overridden);
  2. at step == warp_start (4999): renders the alpha depth of every training view and builds the
     forward-warped targets of the warp cameras (official, once);
  3. step < densify_until (18000): statistics |dL/d ndc xy| and max screen radius of visible
     Gaussians, from the warp render on warp iterations, else from the training render;
     step > 500 and step % 100 == 0: clone / split (grad 2e-4, percent_dense 0.01) and prune
     opacity < 0.05 (SparseGS passes 0.05, Inria 0.005), plus radius > 20 px and scale >
     0.1 extent once step > 3000; step % 3000 == 0: opacity reset to min(o, 0.01);
  4. step == prune_step: floater pruning (paper: once, after training completes).
"""

import math

import numpy as np
import torch
from gsplat.strategy.ops import duplicate, remove, reset_opa, split

from .context import CTX, forward_warp, warp_viewmats
from .depth_raster import prune_marks

DEFAULTS = {
    "total_steps": 10000, "seed": 0,
    "position_lr_init": 0.00016, "position_lr_final": 0.0000016, "position_lr_max_steps": 30000,
    "densify_from_iter": 500, "densify_until_iter": 18000, "densification_interval": 100,
    "grad_threshold": 0.0002, "percent_dense": 0.01, "min_opacity": 0.05,
    "max_screen_size": 20, "opacity_reset_interval": 3000, "reset_value": 0.01,
    "warp": True, "warp_start": 4999, "warp_angles": [-2.5, 2.5],
    "prune": True, "prune_step": 10000, "prune_perc": 0.97, "prune_exp": 7.5,
    "prune_min_perc": 80.0, "power_thresh": -4.0, "mode_range": "official",
    "near_plane": 0.2, "beta": 5.0, "diptest_pylib": "~/models/bmd/pylib",
}


def expon_lr(step, lr_init, lr_final, max_steps):
    """Inria get_expon_lr_func with lr_delay_steps = 0."""
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
        return 0.0
    t = min(max(step / max_steps, 0.0), 1.0)
    return math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)


def calc_diff(mode, depth):
    """utils/prune_utils.calc_diff (torch.std is unbiased, as in the official code)."""
    mean, std = depth.mean(), depth.std()
    return (depth - mode) / (depth + torch.maximum(depth.min(), mean - 3 * std))


def prune_percentile(avg_dip, perc, exp, min_perc):
    """perc = prune_perc * 100 * exp(-prune_exp * avg_dip), at least 80 (official)."""
    return max(float(min_perc), float(perc) * 100 * float(np.exp(-float(exp) * avg_dip)))


def dipstat(values, pylib):
    try:
        import diptest
    except ImportError:
        import os
        import sys

        sys.path.append(os.path.expanduser(pylib))
        import diptest
    return float(diptest.dipstat(np.asarray(values, dtype=np.float64)))


class SparseGSDensity:
    requires = frozenset({"gradient.screen.signed"})

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown SparseGS density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        if self.cfg["mode_range"] not in ("official", "paper"):
            raise ValueError("mode_range must be 'official' or 'paper'")
        self.state = {"grad2d": None, "count": None, "radii": None, "log": []}
        self.extent = 1.0
        self.cameras = None
        self.last_row = {}

    # ------------------------------------------------------------------ setup
    def initialize(self, params, optimizers, scene_scale=1.0):
        self.extent = float(scene_scale)
        # mix the config seed with the run seed (runtime calls torch.manual_seed(run seed) first)
        CTX.seed((int(self.cfg["seed"]) + int(torch.initial_seed())) % (2**31))
        CTX.warp_cams, CTX.warp_stack, CTX.warp_info, CTX.ellipse = None, [], None, None
        CTX._draw = (None, (False, False))
        optimizers["means"].param_groups[0]["lr"] = self.means_lr(1)
        self._reset_stats(len(params["means"]), params["means"].device)

    def set_training_cameras(self, cameras):
        self.cameras = cameras
        CTX.cameras = cameras

    def means_lr(self, iteration):
        c = self.cfg
        return expon_lr(iteration, float(c["position_lr_init"]) * self.extent,
                        float(c["position_lr_final"]) * self.extent,
                        int(c["position_lr_max_steps"]))

    def _reset_stats(self, n, device):
        self.state["grad2d"] = torch.zeros(n, device=device)
        self.state["count"] = torch.zeros(n, device=device)
        self.state["radii"] = torch.zeros(n, device=device)

    # ------------------------------------------------------------------ hooks
    def before_backward(self, params, optimizers, step, outputs):
        outputs["info"]["means2d"].retain_grad()

    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        row = {"step": step}
        optimizers["means"].param_groups[0]["lr"] = self.means_lr(step + 1)
        if c["warp"] and step == int(c["warp_start"]):
            row["warp_targets"] = self.build_warp_targets(params)
        if step < int(c["densify_until_iter"]):
            info = outputs["info"]
            if CTX.warp_info is not None and CTX.warp_info[0] == step:
                info = CTX.warp_info[1]  # official "Densification for warp Cam"
            self.accumulate(info)
            if step > int(c["densify_from_iter"]) and step % int(c["densification_interval"]) == 0:
                size = int(c["max_screen_size"]) if step > int(c["opacity_reset_interval"]) else None
                row.update(self.densify_and_prune(params, optimizers, size))
            if step % int(c["opacity_reset_interval"]) == 0:
                reset_opa(params, optimizers, self.state, value=float(c["reset_value"]))
                row["opacity_reset"] = True
        CTX.warp_info = None
        if c["prune"] and step == int(c["prune_step"]):
            row["floater_prune"] = self.prune_floaters(params, optimizers)
        self.last_row = row
        if len(row) > 1 and ("floater_prune" in row or "warp_targets" in row or step % 1000 == 0):
            row["gaussians"] = len(params["means"])
            self.state["log"].append(row)
            print({"sparsegs": row}, flush=True)

    # ------------------------------------------------------------------ densification
    def accumulate(self, info):
        means2d = info["means2d"]
        if means2d.grad is None:
            return
        n = means2d.shape[-2]
        if self.state["grad2d"] is None or len(self.state["grad2d"]) != n:
            self._reset_stats(n, means2d.device)
        g = means2d.grad[0].clone()
        g[:, 0] *= info["width"] / 2.0  # Inria gradients are w.r.t. NDC
        g[:, 1] *= info["height"] / 2.0
        radii = info["visibility_radii"]
        vis = radii > 0
        self.state["grad2d"][vis] += g[vis].norm(dim=-1)
        self.state["count"][vis] += 1
        self.state["radii"][vis] = torch.maximum(self.state["radii"][vis], radii[vis].float())

    @torch.no_grad()
    def densify_and_prune(self, params, optimizers, max_screen_size):
        c, st = self.cfg, self.state
        thr, pd = float(c["grad_threshold"]), float(c["percent_dense"])
        grads = st["grad2d"] / st["count"]
        grads[grads.isnan()] = 0.0
        big = params["scales"].exp().max(dim=1).values > pd * self.extent
        is_dup = (grads >= thr) & ~big
        n_dup = int(is_dup.sum())
        if n_dup:
            duplicate(params, optimizers, st, is_dup)
        padded = torch.cat([grads, torch.zeros(n_dup, device=grads.device)])
        big = params["scales"].exp().max(dim=1).values > pd * self.extent
        is_split = (padded >= thr) & big
        n_split = int(is_split.sum())
        if n_split:
            split(params, optimizers, st, is_split)
        prune = params["opacities"].sigmoid() < float(c["min_opacity"])
        if max_screen_size:
            # Official quirk: densification_postfix (called by clone and split) zeroes
            # max_radii2D before densify_and_prune checks it (gaussian_model.py:358 vs :409),
            # so the screen-size prune never fires; only the world-size prune is active.
            # Applying a real screen-size prune collapsed test PSNR after step 3000
            # (garden control 18.3 -> 12.2; iteration 1).
            prune |= params["scales"].exp().max(dim=1).values > 0.1 * self.extent
        n_prune = int(prune.sum())
        if n_prune:
            if n_prune == len(prune):
                raise RuntimeError("SparseGS densification would remove every Gaussian")
            remove(params, optimizers, st, prune)
        self._reset_stats(len(params["means"]), params["means"].device)
        return {"cloned": n_dup, "split": n_split, "pruned": n_prune}

    # ------------------------------------------------------------------ warp targets
    @torch.no_grad()
    def build_warp_targets(self, params, renderer=None):
        if not self.cameras:
            raise RuntimeError("SparseGS warp needs the training cameras")
        rnd = renderer or _depth_renderer(self.cfg)
        table = CTX.frames()
        dev = params["means"].device
        ranges = []
        for cam in self.cameras:
            d = rnd.render(params, cam, 0, with_depth=True)["depth_alpha"]
            ranges.append((float(d.min()), float(d.max())))
        cams = []
        for vm, src in warp_viewmats([cam["viewmat"] for cam in self.cameras],
                                     self.cfg["warp_angles"]):
            s = self.cameras[src]
            k = table.index(s["viewmat"])
            target, mask = forward_warp(table.image(k, dev), table.lookup(s["viewmat"], dev),
                                        s["K"], s["viewmat"], vm.to(dev), *ranges[src])
            cams.append({"camera": {"viewmat": vm.to(dev), "K": s["K"], "width": s["width"],
                                    "height": s["height"]},
                         "target": target.float(), "mask": mask.float(), "src": src})
        CTX.warp_cams, CTX.warp_stack = cams, []
        return {"n": len(cams), "mean_mask": float(np.mean([float(x["mask"].mean()) for x in cams])),
                "depth_ranges": ranges}

    # ------------------------------------------------------------------ C5 floater pruning
    @torch.no_grad()
    def prune_floaters(self, params, optimizers, renderer=None):
        c = self.cfg
        if not self.cameras:
            raise RuntimeError("SparseGS floater pruning needs the training cameras")
        rnd = renderer or _depth_renderer(c)
        key = "mode_ids_official" if c["mode_range"] == "official" else "mode_ids"
        views, dips = [], []
        for cam in self.cameras:
            out = rnd.render(params, cam, 0, with_depth=True)
            diff = calc_diff(out["depth_mode"], out["depth_alpha"])
            pos = diff[diff > 0]
            dips.append(dipstat(pos.cpu().numpy(), c["diptest_pylib"]) if len(pos) > 3 else 0.0)
            views.append((out["info"], diff, out[key]))
        avg_dip = float(np.mean(dips))
        perc = prune_percentile(avg_dip, c["prune_perc"], c["prune_exp"], c["prune_min_perc"])
        n = len(params["means"])
        mask = torch.zeros(n, dtype=torch.bool, device=params["means"].device)
        per_view = []
        for info, diff, ids in views:
            pos = diff[diff > 0]
            if len(pos) == 0:
                per_view.append(0)
                continue
            thr = float(np.percentile(pos.cpu().numpy(), perc))
            flagged = diff > thr
            ys, xs = torch.nonzero(flagged, as_tuple=True)
            lo, hi = ids[0][flagged], ids[1][flagged]
            keep = (lo != -1) & (hi != -1)  # official neg_mask
            sub = prune_marks(info, xs[keep], ys[keep], lo[keep], hi[keep],
                              float(c["power_thresh"]), n)
            per_view.append(int(sub.sum()))
            mask |= sub
        n_pruned = int(mask.sum())
        if n_pruned:
            if n_pruned == n:
                raise RuntimeError("SparseGS floater pruning would remove every Gaussian")
            remove(params, optimizers, self.state, mask)
            self._reset_stats(len(params["means"]), params["means"].device)
        return {"avg_dip": avg_dip, "percentile": perc, "per_view": per_view,
                "pruned": n_pruned}


def _depth_renderer(cfg):
    from types import SimpleNamespace

    from .renderer import Renderer

    return Renderer(SimpleNamespace(config={"beta": cfg["beta"], "near_plane": cfg["near_plane"]}))


def create_density(spec):
    return SparseGSDensity(spec)
