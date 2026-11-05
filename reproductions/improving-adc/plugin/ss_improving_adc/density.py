"""Improving-ADC density control (arXiv 2503.14274, Sec. 4, Alg. 2).

Port of the official code https://github.com/fraunhoferhhi/Improving-ADC-3DGS (65901d8):
`train.py` (threshold schedule, sigma accumulation, densify/reset schedule),
`scene/__init__.py` (corrected extent, spatial_lr_scale), `scene/gaussian_model.py`
(densify_and_clone / densify_and_split with revised opacity, significance-aware pruning, no size
pruning, PixelGS add_densification_stats) and the PixelGS rasterizer (pixel counts, squared depth
scaling of the 2D gradient). Population changes go through gsplat.strategy.ops so params and Adam
moments stay aligned.
"""

import json

import torch
from gsplat.strategy.ops import duplicate, remove, reset_opa, split

from . import core
from .kernels import pixel_stats

DEFAULTS = {
    "densify_from": 500,
    "densify_until": 15000,
    "interval": 100,
    # C2 exponentially ascending threshold (train.py:62-67)
    "threshold": "exponential",  # "constant" = ablation (3DGS fixed threshold)
    "grad_start": 1e-4,
    "grad_end": 4e-4,
    "grad_constant": 2e-4,
    # C1 corrected extent (scene/__init__.py:73-79,94)
    "extent": "sfm_mean",  # "camera" = ablation (3DGS 1.1 * max camera distance)
    "scale_means_lr": True,  # spatial_lr_scale = e' (create_from_pcd(pcd, cameras_extent))
    # C3 significance-aware pruning (gaussian_model.py:443-471)
    "pruning": "significance",  # "opacity" = ablation (3DGS opacity pruning)
    "min_opacity": 0.005,
    # base: PixelGS gradient and Bulo revised opacity
    "pixel_gradient": True,
    "depth_gamma": 0.37,
    "depth_power": 2.0,
    "revised_opacity": True,
    "percent_dense": 0.01,
    # 3DGS opacity reset (train.py:155)
    "reset_every": 3000,
    "reset_value": 0.01,
    "log": True,  # one JSON line per densification call on stdout (train log)
}


class ImprovedADC:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown Improving-ADC density options: {sorted(unknown)}")
        c = {**DEFAULTS, **spec.config}
        if c["threshold"] not in ("exponential", "constant"):
            raise ValueError("threshold must be 'exponential' or 'constant'")
        if c["extent"] not in ("sfm_mean", "camera"):
            raise ValueError("extent must be 'sfm_mean' or 'camera'")
        if c["pruning"] not in ("significance", "opacity"):
            raise ValueError("pruning must be 'significance' or 'opacity'")
        self.cfg = c
        self.requires = {"gradient.screen.signed"}
        self.state = None
        self.intervals = (c["densify_until"] - c["densify_from"]) // c["interval"]
        self.history = []  # per densification call, for logs and probes

    # ------------------------------------------------------------------ lifecycle
    def initialize(self, params, optimizers, scene_scale=1.0):
        n = len(params["means"])
        dev = params["means"].device
        self.sfm = params["means"].detach().clone()  # initial = SfM points (recipe 3dgs)
        self.scene_scale = float(scene_scale)
        self.state = {
            "grad_accum": torch.zeros(n, device=dev),
            "pix_count": torch.zeros(n, device=dev),
            "sigma": torch.zeros(n, device=dev),
            # until set_training_cameras runs both extents default to the runtime scale
            "camera_extent": float(scene_scale),
            "extent": float(scene_scale),
            "calls": 0,
        }
        self.optimizers = optimizers

    def set_training_cameras(self, cameras):
        centers = torch.stack([torch.linalg.inv(c["viewmat"].float())[:3, 3] for c in cameras])
        e_cam = core.camera_extent(centers)
        e_new = core.corrected_extent(self.sfm.to(centers.device), centers)
        self.state["camera_extent"] = e_cam
        self.state["extent"] = e_new if self.cfg["extent"] == "sfm_mean" else e_cam
        if self.cfg["scale_means_lr"] and self.cfg["extent"] == "sfm_mean":
            # official spatial_lr_scale = e'; the runtime multiplied the means LR by scene_scale
            # and captures initial_lr after this hook, so its exponential decay follows.
            for g in self.optimizers["means"].param_groups:
                g["lr"] *= e_new / self.scene_scale
        self.sfm = None
        if self.cfg["log"]:
            print(json.dumps({"improving_adc": "extent", "camera_extent": e_cam,
                              "corrected_extent": e_new, "used": self.state["extent"],
                              "means_lr": self.optimizers["means"].param_groups[0]["lr"]}),
                  flush=True)

    def before_backward(self, params, optimizers, step, outputs):
        if step < self.cfg["densify_until"]:
            outputs["info"]["means2d"].retain_grad()

    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        if step >= c["densify_until"]:
            return
        self.accumulate(outputs["info"])
        if step > c["densify_from"] and step % c["interval"] == 0:
            self.densify_and_prune(params, optimizers, step)
        if step % c["reset_every"] == 0:
            reset_opa(params, optimizers, {}, value=c["reset_value"])

    # ------------------------------------------------------------------ statistics
    @torch.no_grad()
    def accumulate(self, info):
        """PixelGS add_densification_stats + per-step sigma (train.py:127-147, Eq. 15)."""
        count, blend = pixel_stats(info)
        self.state["sigma"] += blend
        g = info["means2d"].grad
        if g is None:
            raise RuntimeError("means2d gradient missing; before_backward must retain it")
        g = g[0].clone()
        g[:, 0] *= info["width"] / 2.0  # Inria NDC gradient units (as gsplat DefaultStrategy)
        g[:, 1] *= info["height"] / 2.0
        vis = (info["radii"][0] > 0).all(dim=-1)
        norm = g.norm(dim=-1)
        if self.cfg["pixel_gradient"]:
            f = core.depth_factor(info["depths"][0], self.cfg["depth_gamma"]
                                  * self.state["camera_extent"], self.cfg["depth_power"])
            self.state["grad_accum"][vis] += (norm * f * count)[vis]
            self.state["pix_count"][vis] += count[vis]
        else:  # 3DGS per-view mean
            self.state["grad_accum"][vis] += norm[vis]
            self.state["pix_count"][vis] += 1.0

    def grads(self):
        return torch.nan_to_num(self.state["grad_accum"] / self.state["pix_count"], nan=0.0)

    def current_threshold(self, call):
        c = self.cfg
        if c["threshold"] == "constant":
            return c["grad_constant"]
        return core.threshold(call, c["grad_start"], c["grad_end"], self.intervals)

    # ------------------------------------------------------------------ densify + prune
    @torch.no_grad()
    def densify_and_prune(self, params, optimizers, step):
        c = self.cfg
        self.state["calls"] += 1
        thr = self.current_threshold(self.state["calls"])
        grads = self.grads()
        sigma = self.state["sigma"].clone()
        n_old = len(params["means"])
        extent = self.state["extent"]
        stats = {}  # no running state needs to follow the ops: all statistics reset below

        # clone (densify_and_clone): parent + clone opacity 1 - sqrt(1 - o) (Eq. 8)
        max_scale = params["scales"].exp().max(dim=-1).values
        clone = (grads >= thr) & (max_scale <= c["percent_dense"] * extent)
        if c["revised_opacity"] and clone.any():
            o = params["opacities"][clone].sigmoid()
            params["opacities"].data[clone] = torch.logit(core.revised_opacity(o))
        duplicate(params, optimizers, stats, clone)
        n_clone = int(clone.sum())

        # split (densify_and_split): padded grads -> new clones never split
        padded = torch.zeros(len(params["means"]), device=grads.device)
        padded[:n_old] = grads
        max_scale = params["scales"].exp().max(dim=-1).values
        split_mask = (padded >= thr) & (max_scale > c["percent_dense"] * extent)
        split(params, optimizers, stats, split_mask, revised_opacity=c["revised_opacity"])
        n_split = int(split_mask.sum())

        # prune
        opacity = params["opacities"].sigmoid()
        low = opacity < c["min_opacity"]
        if c["pruning"] == "significance":
            prune = core.significance_prune_mask(sigma, split_mask[:n_old], opacity,
                                                 c["min_opacity"])
        else:
            prune = low
        n_low = int(low.sum())
        remove(params, optimizers, stats, prune)

        n = len(params["means"])
        dev = params["means"].device
        for k in ("grad_accum", "pix_count", "sigma"):
            self.state[k] = torch.zeros(n, device=dev)
        row = {"improving_adc": "densify", "step": step, "call": self.state["calls"],
               "threshold": thr, "before": n_old, "clone": n_clone, "split": n_split,
               "low_opacity": n_low, "pruned": int(prune.sum()), "after": n}
        self.history.append(row)
        if c["log"]:
            print(json.dumps(row), flush=True)
        return row


def create_density(spec):
    return ImprovedADC(spec)
