"""PGSR adaptive density control (official train.py densification block and
scene/gaussian_model.py @de24f1a: add_densification_stats, densify_and_clone,
densify_and_split, densify_and_prune, reset_opacity, multi-view observe trim).

  stats   step < densify_until: max_radii2D = max(., radii) where out_observe > 0 & radii > 0;
          radii > 0: accum += ||g[:, :2]||, accum_abs += ||g_abs[:, :2]||, denom += 1
  densify step > densify_from and step % 100 == 0 (grads = accum/denom, nan -> 0):
          clone  grads >= thr & max(s) <= percent_dense * extent (cap max_all_points by
                 gradient quantile); new means sampled from N(mu, Sigma), rest copied
          split  (post-clone population, clones padded with zeros) grads >= thr &
                 max(s) > percent_dense * extent; if no cap: plus AbsGS split of the remaining
                 Gaussians with grads_abs >= abs_thr, max_radii2D > 20, max(s) > pd * extent,
                 at most min(max_all - n - n_split, 50000) by grads_abs quantile;
                 2 samples, s / 1.6, opacity copied
          prune  sigmoid(o) < opacity_cull; after step 3000 also max(s) > 0.1 * extent.
                 The official screen-size test (max_radii2D > 20) never fires because
                 densification_postfix zeroes max_radii2D before it; replicated (not applied).
  trim    step % 1000 == 0 and step < densify_until: render every training view, prune
          Gaussians with out_observe > 0 in fewer than 2 views
  reset   step % 3000 == 0 and step < densify_until: opacity = min(o, 0.01), moments zeroed
  xyz lr  get_expon_lr_func(1.6e-4 * extent -> 1.6e-6 * extent, 30000 steps)
The controller also reloads the training split (cameras + grey images) from ``--data`` and
builds the official neighbour graph for the multi-view losses (``link.LINK``). The runtime
applies the optimizer step before this controller runs (official: densify first).
"""

import math
import sys
import time
from pathlib import Path

import torch
from gsplat.strategy.ops import _update_param_with_optimizer, remove, reset_opa, split
from gsplat.utils import normalized_quat_to_rotmat

from . import geometry as G
from .link import LINK

DEFAULTS = {
    "densify_from_iter": 500,
    "densify_until_iter": 15000,
    "densification_interval": 100,
    "densify_grad_threshold": 0.0002,
    "densify_abs_grad_threshold": 0.0002,  # run_mip360.py (code default 0.0008)
    "abs_split_radii2D_threshold": 20,
    "max_abs_split_points": 50000,
    "max_all_points": 6000000,
    "percent_dense": 0.001,
    "opacity_cull_threshold": 0.005,
    "opacity_reset_interval": 3000,
    "opacity_reset_value": 0.01,
    "big_world_factor": 0.1,
    "use_multi_view_trim": True,
    "multi_view_trim_interval": 1000,
    "observe_threshold": 2,
    "geometry_from_iter": 7000,  # min(single_view_weight_from_iter, multi_view_weight_from_iter)
    "multi_view_num": 8,
    "multi_view_max_angle": 30.0,
    "multi_view_min_dis": 0.01,
    "multi_view_max_dis": 1.5,
    "position_lr_init": 1.6e-4,
    "position_lr_final": 1.6e-6,
    "position_lr_delay_mult": 0.01,
    "position_lr_delay_steps": 0,
    "position_lr_max_steps": 30000,
    "data": None,        # training-image source; default ss-train --data
    "downscale": None,   # default ss-train --downscale (1)
}
REQUIRES = {"gradient.screen.signed", "gradient.screen.absolute", "render.observe_count"}


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


def _argv_value(flag):
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


@torch.no_grad()
def clone_resampled(params, optimizers, mask):
    """densify_and_clone: append copies; means drawn from N(mu, R diag(s)^2 R^T)."""
    sel = torch.where(mask)[0]
    stds = params["scales"][sel].exp()
    rots = normalized_quat_to_rotmat(torch.nn.functional.normalize(params["quats"][sel], dim=-1))
    samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
    new_means = torch.bmm(rots, samples[..., None])[..., 0] + params["means"][sel]

    def param_fn(name, p):
        new = new_means if name == "means" else p[sel]
        return torch.nn.Parameter(torch.cat([p, new]), requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=v.device)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)


def _quantile_mask(values, limited, n):
    """Official cap: threshold = quantile(values, 1 - limited / n); keep values > threshold."""
    q = min(max(1.0 - float(limited) / float(n), 0.0), 1.0)
    return values > torch.quantile(values, q)


def select_clone(grads, smax, thr, dense, max_all):
    n0 = grads.shape[0]
    sel = (grads >= thr) & (smax <= dense)
    if int(sel.sum()) + n0 > max_all:
        g = grads.clone()
        g[~sel] = 0
        sel = _quantile_mask(g, max_all - n0, n0)
    return sel


def select_split(grads, grads_abs, radii, smax, c, dense):
    """densify_and_split selection on the post-clone population (inputs already padded)."""
    n1 = grads.shape[0]
    sel = (grads >= c["densify_grad_threshold"]) & (smax > dense)
    n_abs = 0
    if int(sel.sum()) + n1 > c["max_all_points"]:
        g = grads.clone()
        g[~sel] = 0
        sel = _quantile_mask(g, c["max_all_points"] - n1, n1)
    else:
        ga = grads_abs.clone()
        ga[sel] = 0
        ga[~((smax > dense) & (radii > c["abs_split_radii2D_threshold"]))] = 0
        sel_abs = ga >= c["densify_abs_grad_threshold"]
        limited = min(c["max_all_points"] - n1 - int(sel.sum()), c["max_abs_split_points"])
        if int(sel_abs.sum()) > limited:
            sel_abs = _quantile_mask(ga, limited, n1)
        n_abs = int((sel_abs & ~sel).sum())
        sel = sel | sel_abs
    return sel, n_abs


class Density:
    requires = REQUIRES

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown PGSR density options: {sorted(unknown)}")
        self.c = {**DEFAULTS, **spec.config}
        self.requires = set(REQUIRES)
        self.extent = None
        self.stats = None
        self.counts = {"densify": 0, "trim": 0, "resets": 0, "trimmed": 0}
        self.state = {"counts": self.counts}
        LINK.cameras = LINK.gray = LINK.neighbours = None
        LINK.extra_optimizers = []

    # -- runtime hooks -------------------------------------------------------------------
    def initialize(self, params, optimizers, scene_scale=1.0):
        for name in ("means", "scales", "quats", "opacities"):
            if name not in params or name not in optimizers:
                raise ValueError(f"pgsr-density requires the {name} field")
        self.extent = float(scene_scale)  # recipe 3dgs: getNerfppNorm radius
        self.state["extent"] = self.extent
        self._reset_stats(len(params["means"]), params["means"].device)
        optimizers["means"].param_groups[0]["lr"] = self.xyz_lr(1)
        LINK.render_geo = 1 > self.c["geometry_from_iter"]

    def set_training_cameras(self, cameras):
        """Runtime hook: reload the training split's images (the runtime exposes cameras only)."""
        root = self.c["data"] or _argv_value("--data")
        if root is None:
            raise RuntimeError("pgsr-density needs the training images for the multi-view "
                               "losses: set density config 'data' or run through ss-train --data")
        downscale = int(self.c["downscale"] or _argv_value("--downscale") or 1)
        from splatstudio.data import load_dataset

        t0 = time.perf_counter()
        dataset = load_dataset(Path(root), downscale)
        names = dataset["splits"]["train"]  # training split only; test images are never kept
        if len(names) != len(cameras):
            raise RuntimeError("Reloaded training split does not match the runtime's cameras")
        device = cameras[0]["K"].device
        images = []
        for name, cam in zip(names, cameras):
            frame = dataset["frames"][name]
            same = (frame["width"] == cam["width"] and frame["height"] == cam["height"]
                    and torch.allclose(frame["K"].to(device), cam["K"])
                    and torch.allclose(frame["viewmat"].to(device), cam["viewmat"]))
            if not same:
                raise RuntimeError(f"Reloaded training view {name} differs from the runtime camera")
            images.append(frame["image"])
        del dataset
        self.set_training_frames(cameras, images)
        self.state["load_seconds"] = time.perf_counter() - t0

    def set_training_frames(self, cameras, images):
        """Register training cameras + HWC images in [0,1]; builds grey images and the graph."""
        if len(cameras) != len(images) or not cameras:
            raise ValueError("cameras and images must be non-empty and aligned")
        device = cameras[0]["K"].device
        cams = [dict(c) for c in cameras]
        gray = [G.gray(im.to(device).float()).contiguous() for im in images]
        c = self.c
        neighbours = G.neighbour_graph(torch.stack([cm["viewmat"] for cm in cams]),
                                       c["multi_view_num"], c["multi_view_max_angle"],
                                       c["multi_view_min_dis"], c["multi_view_max_dis"])
        LINK.set_views(cams, gray, neighbours)
        self.state["neighbours"] = {"views": len(cams),
                                    "with_neighbours": sum(bool(n) for n in neighbours),
                                    "mean_count": sum(map(len, neighbours)) / len(cams)}

    def before_backward(self, params, optimizers, step, outputs):
        if not LINK.active:
            raise RuntimeError("pgsr-density: training views not registered")

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c = self.c
        LINK.step_extra()  # exposure coefficients (official app_model.optimizer.step())
        if step < c["densify_until_iter"]:
            self.accumulate(outputs["info"])
            if step > c["densify_from_iter"] and step % c["densification_interval"] == 0:
                self.densify_and_prune(params, optimizers, step > c["opacity_reset_interval"])
        if (c["use_multi_view_trim"] and step % c["multi_view_trim_interval"] == 0
                and step < c["densify_until_iter"]):
            self.trim(params, optimizers)
        if step < c["densify_until_iter"] and step % c["opacity_reset_interval"] == 0:
            reset_opa(params, optimizers, {}, c["opacity_reset_value"])
            self.counts["resets"] += 1
        optimizers["means"].param_groups[0]["lr"] = self.xyz_lr(step + 1)
        LINK.render_geo = (step + 1) > c["geometry_from_iter"]

    # -- pieces ----------------------------------------------------------------------------
    def xyz_lr(self, step):
        c = self.c
        return expon_lr(step, c["position_lr_init"] * self.extent,
                        c["position_lr_final"] * self.extent, c["position_lr_delay_steps"],
                        c["position_lr_delay_mult"], c["position_lr_max_steps"])

    def _reset_stats(self, n, dev):
        self.stats = {k: torch.zeros(n, device=dev)
                      for k in ("accum", "accum_abs", "denom", "denom_abs", "max_radii")}

    def accumulate(self, info):
        g, ga = info["means2d"].grad, info["means2d_abs"].grad
        if g is None or ga is None:
            raise RuntimeError("pgsr-density: screen-space gradients missing")
        radii = info["radii"]
        vis = radii > 0
        st = self.stats
        m = (info["observe"] > 0) & vis
        st["max_radii"][m] = torch.max(st["max_radii"][m], radii[m].float())
        st["accum"][vis] += torch.linalg.vector_norm(g[vis, :2], dim=-1)
        st["accum_abs"][vis] += torch.linalg.vector_norm(ga[vis, :2], dim=-1)
        st["denom"][vis] += 1
        st["denom_abs"][vis] += 1

    def densify_and_prune(self, params, optimizers, world_prune):
        c, st = self.c, self.stats
        grads = st["accum"] / st["denom"]
        grads[grads.isnan()] = 0.0
        grads_abs = st["accum_abs"] / st["denom_abs"]
        grads_abs[grads_abs.isnan()] = 0.0
        radii = st["max_radii"].clone()
        dense = c["percent_dense"] * self.extent
        n0 = len(grads)
        smax = params["scales"].exp().max(dim=1).values
        clone = select_clone(grads, smax, c["densify_grad_threshold"], dense, c["max_all_points"])
        n_clone = int(clone.sum())
        if n_clone:
            clone_resampled(params, optimizers, clone)
        pad = torch.zeros(n_clone, device=grads.device)
        smax = params["scales"].exp().max(dim=1).values
        sel, n_abs = select_split(torch.cat([grads, pad]), torch.cat([grads_abs, pad]),
                                  torch.cat([radii, pad]), smax, c, dense)
        split(params, optimizers, {}, sel)
        prune = params["opacities"].sigmoid() < c["opacity_cull_threshold"]
        if world_prune:
            # big_points_vs (max_radii2D > 20) is identically False upstream (see docstring)
            prune = prune | (params["scales"].exp().max(dim=1).values
                             > c["big_world_factor"] * self.extent)
        remove(params, optimizers, {}, prune)
        n = len(params["means"])
        self._reset_stats(n, params["means"].device)
        self.counts["densify"] += 1
        self.state["last"] = {"before": n0, "clone": n_clone, "split": int(sel.sum()),
                              "abs_split": n_abs, "prune": int(prune.sum()), "after": n}

    def observe_counts(self, params):
        cnt = torch.zeros(len(params["means"]), device=params["means"].device)
        for cam in LINK.cameras:
            out = LINK.renderer.render(params, cam, 0, render_geo=False)
            cnt[out["info"]["observe"] > 0] += 1
        return cnt

    def trim(self, params, optimizers):
        prune = self.observe_counts(params) < self.c["observe_threshold"]
        if int(prune.sum()) > 0:
            remove(params, optimizers, {}, prune)
            keep = ~prune
            self.stats = {k: v[keep] for k, v in self.stats.items()}
        self.counts["trim"] += 1
        self.counts["trimmed"] += int(prune.sum())


def create_density(spec):
    return Density(spec)
