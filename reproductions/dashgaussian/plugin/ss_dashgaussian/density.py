"""DashGaussian density controller for the 3DGS backbone.

Owns the resolution schedule (C1), the resolution-guided primitive schedule with top-k
densification (C2), the momentum budget (C3) and the position-LR schedule (C4). Follows the
official `train_dash.py`, `utils/schedule_utils.py` and `scene/gaussian_model.py:prune_and_densify`.
"""

import sys
import time
from pathlib import Path

import torch
from gsplat.strategy import ops

from . import schedule as S
from .link import LINK
from .renderer import scaled_size

DEFAULTS = {
    # 3DGS densification values kept by DashGaussian (arguments/__init__.py).
    "grow_grad2d": 0.0002,
    "grow_scale3d": 0.01,  # percent_dense
    "prune_opa": 0.005,
    "refine_start": 500,
    "refine_every": 100,
    "densify_until": 27000,  # full_eval.py --dash: --densify_until_iter 27000
    "reset_every": 3000,
    "reset_value": 0.01,
    "total_steps": 30000,  # position_lr_max_steps / iterations
    "lr_final_factor": 0.01,  # 1.6e-6 / 1.6e-4
    # DashGaussian scheduler (utils/schedule_utils.py).
    "start_significance_factor": 4.0,  # a
    "max_reso_scale": 8.0,
    "reso_sample_num": 32,
    "max_densify_rate": 0.2,
    "momentum_init_factor": 5,  # momentum = 5 * P_init, P_fin^(0) = 6 * P_init
    "integrate_factor": 0.98,  # gamma (Eq. 5)
    "momentum_step_cap": 1000000,
    "max_n_gaussian": -1,  # >0: preset upper bound, momentum disabled (--preset_upperbound)
    "resolution_mode": "freq",  # "freq" | "const"
    "densify_mode": "freq",  # "freq" | "free"
    # Where to read the training images for the FFT schedule; None -> ss-train command line.
    "data": None,
    "downscale": None,
}


def _argv_value(flag):
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


class DashDensity:
    requires = {"gradient.screen.signed"}

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown dash-density options: {sorted(unknown)}")
        c = {**DEFAULTS, **spec.config}
        if c["resolution_mode"] not in ("freq", "const"):
            raise ValueError("resolution_mode must be 'freq' or 'const'")
        if c["densify_mode"] not in ("freq", "free"):
            raise ValueError("densify_mode must be 'freq' or 'free'")
        self.c = c
        self.state = None
        self.stats = None
        self.schedule = None
        self.schedule_active = False
        self.current_scale = 1
        self.scene_scale = 1.0
        self.history = []  # (step, scale for the following steps, gaussians, P_fin) per refine
        LINK.attach(self)

    # ---- set-up -------------------------------------------------------------------------
    def initialize(self, params, optimizers, scene_scale=1.0):
        c = self.c
        for name in ("means", "scales", "quats", "opacities"):
            if name not in params or name not in optimizers:
                raise ValueError(f"dash-density requires the {name} field")
        self.scene_scale = float(scene_scale)
        self.init_n = len(params["means"])
        if c["max_n_gaussian"] > 0:
            self.momentum = -1
            self.max_n = int(c["max_n_gaussian"])
        else:
            self.momentum = int(c["momentum_init_factor"] * self.init_n)
            self.max_n = self.init_n + self.momentum
        # The runtime has already multiplied the means LR by the scene scale.
        self.lr_init = optimizers["means"].param_groups[0]["lr"]
        self.lr_final = self.lr_init * c["lr_final_factor"]
        self._reset_stats(params)
        self._sync_state()

    def set_training_cameras(self, cameras):
        """Runtime hook: locate and load the training images, then build the FFT schedule."""
        if self.c["resolution_mode"] == "const":
            return
        root = self.c["data"] or _argv_value("--data")
        if root is None:
            raise RuntimeError(
                "dash-density needs the training images for its frequency schedule: set density "
                "config 'data' or run through ss-train --data"
            )
        downscale = self.c["downscale"] or _argv_value("--downscale") or 1
        from splatstudio.data import load_dataset

        t0 = time.perf_counter()
        dataset = load_dataset(Path(root), int(downscale))
        names = dataset["splits"]["train"]
        if len(names) != len(cameras):
            raise RuntimeError("Reloaded training split does not match the runtime's cameras")
        device = cameras[0]["K"].device
        images = []
        for name, cam in zip(names, cameras):
            frame = dataset["frames"][name]
            same = (
                frame["width"] == cam["width"]
                and frame["height"] == cam["height"]
                and torch.allclose(frame["K"].to(device), cam["K"])
                and torch.allclose(frame["viewmat"].to(device), cam["viewmat"])
            )
            if not same:
                raise RuntimeError(f"Reloaded training view {name} differs from the runtime camera")
            images.append(frame["image"].to(device).permute(2, 0, 1))
        self.set_training_images(images)
        self.schedule_seconds = time.perf_counter() - t0

    def set_training_images(self, images):
        """Build the resolution schedule from [3,H,W] training images (official scheduler init)."""
        c = self.c
        with torch.no_grad():
            self.schedule = S.build_schedule(
                images,
                increase_until=c["densify_until"],
                start_significance_factor=c["start_significance_factor"],
                max_reso_scale=c["max_reso_scale"],
                sample_num=c["reso_sample_num"],
            )
        self.decay_from = S.lr_decay_from(self.schedule)
        self.schedule_active = True
        self.current_scale = self.scale_at(1)
        self._sync_state()

    # ---- schedules ----------------------------------------------------------------------
    def scale_at(self, iteration):
        if self.c["resolution_mode"] == "const":
            return 1
        if self.schedule is None:
            raise RuntimeError("dash-density: resolution schedule not built (no training images)")
        return S.res_scale(self.schedule, iteration, self.c["densify_until"])

    def is_refine_step(self, step):
        c = self.c
        return step < c["densify_until"] and step > c["refine_start"] and step % c["refine_every"] == 0

    def densify_rate(self, step, cur_n):
        c = self.c
        if c["densify_mode"] == "free":
            return 1.0
        return S.densify_rate(step, cur_n, self.current_scale, self.init_n, self.max_n,
                              c["refine_every"], c["densify_until"], c["densify_until"],
                              c["max_densify_rate"])

    # ---- runtime callbacks --------------------------------------------------------------
    def before_backward(self, params, optimizers, step, outputs):
        if self.c["resolution_mode"] == "freq" and not self.schedule_active:
            raise RuntimeError("dash-density: resolution schedule not built before training")
        if self.schedule_active:
            # Guard against renderer/schedule desync: the rendered size must be the scheduled one.
            full_h, full_w = outputs["full_size"]
            h, w = outputs["rgb"].shape[:2]
            if outputs["render_scale"] != self.current_scale or (h, w) != scaled_size(
                full_h, full_w, self.current_scale
            ):
                raise RuntimeError("Rendered size disagrees with the DashGaussian schedule")
        outputs["info"]["means2d"].retain_grad()

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c = self.c
        self.update_lr(optimizers, step)
        if step < c["densify_until"]:
            self._accumulate(params, outputs["info"])
            if self.is_refine_step(step):
                rate = self.densify_rate(step, len(params["means"]))
                added = self.prune_and_densify(params, optimizers, rate)
                if self.momentum != -1:
                    self.momentum = S.update_momentum(
                        self.momentum, added, c["integrate_factor"], c["momentum_step_cap"])
                    self.max_n = self.init_n + self.momentum
                if self.schedule_active:
                    self.current_scale = self.scale_at(step)
                self.history.append((step, self.current_scale, len(params["means"]), self.max_n))
            if step % c["reset_every"] == 0:
                ops.reset_opa(params, optimizers, self.stats, value=c["reset_value"])
        self._sync_state()

    def update_lr(self, optimizers, step):
        """supp. Sec. 7.2: position LR held until decay_from, then log-linear (set for step+1).

        Only with the frequency schedule (official train_dash.py replaces the xyz scheduler only
        for resolution_mode='freq'); otherwise the runtime's 3DGS schedule stays in effect.
        """
        if self.schedule_active:
            optimizers["means"].param_groups[0]["lr"] = S.position_lr(
                step + 1, self.lr_init, self.lr_final, self.c["total_steps"], self.decay_from)

    # ---- densification ------------------------------------------------------------------
    def _reset_stats(self, params):
        n, device = len(params["means"]), params["means"].device
        self.stats = {"grad2d": torch.zeros(n, device=device),
                      "count": torch.zeros(n, device=device)}

    def _accumulate(self, params, info):
        grads = info["means2d"].grad
        if grads is None:
            raise RuntimeError("means2d has no gradient; before_backward must retain it")
        grads = grads.clone()
        # 3DGS NDC gradient at the render size (gsplat DefaultStrategy._update_state).
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        sel = (info["radii"] > 0.0).all(dim=-1)
        ids = torch.where(sel)[1]
        self.stats["grad2d"].index_add_(0, ids, grads[sel].norm(dim=-1))
        self.stats["count"].index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

    def prune_and_densify(self, params, optimizers, rate):
        """Official prune_and_densify: prune first, then clone/split the top-k qualified."""
        c = self.c
        cur_n = len(params["means"])
        prune = torch.sigmoid(params["opacities"].flatten()) < c["prune_opa"]
        if prune.any():
            ops.remove(params, optimizers, self.stats, prune)
        n_left = len(params["means"])
        n_densify = min(int(cur_n * (1 + rate) - n_left), n_left)
        grads = self.stats["grad2d"] / self.stats["count"]
        grads[grads.isnan()] = 0.0
        qualified = grads >= c["grow_grad2d"]
        top = torch.zeros_like(qualified)
        if n_densify > 0:
            top[torch.topk(grads, n_densify).indices] = True
        small = torch.exp(params["scales"]).max(dim=-1).values <= c["grow_scale3d"] * self.scene_scale
        clone = qualified & top & small
        split = qualified & top & ~small
        n_clone = int(clone.sum())
        if n_clone:
            ops.duplicate(params, optimizers, self.stats, clone)
        if split.any():
            split = torch.cat([split, torch.zeros(n_clone, dtype=torch.bool, device=split.device)])
            ops.split(params, optimizers, self.stats, split)
        self._reset_stats(params)
        self.last_densify = {"n_densify": n_densify, "clone": n_clone, "split": int(split.sum()),
                             "pruned": int(prune.sum()), "qualified": int(qualified.sum())}
        # Number of primitives naturally passing the threshold -> momentum force P_add (Eq. 5).
        return int(qualified.sum())

    def _sync_state(self):
        # Checkpointed with weights_only=True: plain numbers and lists only.
        self.state = {
            "init_n": int(getattr(self, "init_n", 0)),
            "max_n": int(getattr(self, "max_n", 0)),
            "momentum": int(getattr(self, "momentum", 0)),
            "current_scale": float(self.current_scale),
            "schedule_scales": list(self.schedule["scales"]) if self.schedule else [],
            "schedule_begins": list(self.schedule["begins"]) if self.schedule else [],
            "lr_decay_from": int(getattr(self, "decay_from", 0)),
        }


def create_density(spec):
    return DashDensity(spec)
