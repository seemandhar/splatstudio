"""DecompGS density controller (arXiv 2507.20239), 3DGS densification with:

- C1 global-to-local densification (G2L): before the boundary T_2 only the split operation
  runs; from T_2 on, 3DGS clone + split.
- C2 energy-guided coarse-to-fine resolution (C2F, Eq. 1-3): owns the resolution schedule the
  renderer follows (via LINK); position LR held constant at reduced resolution, decayed after.
- C3 adaptive opacity pruning: tau = min(alpha_sorted[floor(N p)], tau_u) replaces 0.005.

Everything else follows the 3DGS(-accel) `densify_and_prune` / training loop: densify every
100 steps in (500, T_densify), clone if max scale <= 0.01 extent else split in 2 (scale / 1.6),
prune after growth, world-size prune (> 0.1 extent) after the first reset, opacity reset to
0.01 every 3000 steps while densifying. No official DecompGS code exists.
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
    # 3DGS densification defaults (arguments/__init__.py), kept by the paper.
    "grow_grad2d": 0.0002,
    "grow_scale3d": 0.01,  # percent_dense
    "prune_scale3d": 0.1,  # world-size prune, after step > reset_every
    "refine_start": 500,
    "refine_every": 100,
    "densify_until": 25000,  # T_densify (Impl. details: extended to 25K)
    "reset_every": 3000,
    "reset_value": 0.01,
    "prune_opa": 0.005,  # fixed 3DGS threshold, used only when adaptive_prune is off
    "total_steps": 30000,
    "lr_final_factor": 0.01,  # 1.6e-6 / 1.6e-4
    # C1 G2L
    "g2l": True,
    "global_mode": "skip_clone",  # "skip_clone" (Fig. 2) | "split_all" (split clone candidates too)
    "g2l_boundary": None,  # None -> T_2 from the C2F schedule
    # C2 C2F
    "c2f": True,
    "max_scale_factor": 8,  # K
    "lr_mode": "hold_then_decay",  # "hold_then_decay" | "3dgs"
    # C3 adaptive pruning
    "adaptive_prune": True,
    "prune_ratio": 0.03,  # p
    "prune_upper": 0.05,  # tau_u
    # Training images for the energy schedule; None -> ss-train command line.
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


class DecompDensity:
    requires = {"gradient.screen.signed"}

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown decomp-density options: {sorted(unknown)}")
        c = {**DEFAULTS, **spec.config}
        if c["global_mode"] not in ("skip_clone", "split_all"):
            raise ValueError("global_mode must be 'skip_clone' or 'split_all'")
        if c["lr_mode"] not in ("hold_then_decay", "3dgs"):
            raise ValueError("lr_mode must be 'hold_then_decay' or '3dgs'")
        if not 0 < c["prune_ratio"] < 1 or c["max_scale_factor"] < 1:
            raise ValueError("prune_ratio must be in (0, 1) and max_scale_factor >= 1")
        self.c = c
        self.state = None
        self.stats = None
        self.ends = None  # {r: T_r}
        self.schedule_active = False
        self.current_scale = 1
        self.boundary = c["g2l_boundary"]
        self.scene_scale = 1.0
        self.history = []  # per refine step: dict(step, scale, clone, split, pruned, tau, n)
        LINK.attach(self)

    @property
    def needs_images(self):
        c = self.c
        return c["c2f"] or (c["g2l"] and c["g2l_boundary"] is None)

    # ---- set-up -------------------------------------------------------------------------
    def initialize(self, params, optimizers, scene_scale=1.0):
        for name in ("means", "scales", "quats", "opacities"):
            if name not in params or name not in optimizers:
                raise ValueError(f"decomp-density requires the {name} field")
        self.scene_scale = float(scene_scale)
        # The runtime has already multiplied the means LR by the scene scale.
        self.lr_init = optimizers["means"].param_groups[0]["lr"]
        self.lr_final = self.lr_init * self.c["lr_final_factor"]
        self._reset_stats(params)
        self._sync_state()

    def set_training_cameras(self, cameras):
        """Runtime hook: reload the training images (same split/order) and build the schedule."""
        if not self.needs_images:
            return
        root = self.c["data"] or _argv_value("--data")
        if root is None:
            raise RuntimeError("decomp-density needs the training images for its energy "
                               "schedule: set density config 'data' or run through ss-train --data")
        downscale = self.c["downscale"] or _argv_value("--downscale") or 1
        from splatstudio.data import load_dataset

        t0 = time.perf_counter()
        dataset = load_dataset(Path(root), int(downscale))
        names = dataset["splits"]["train"]
        if len(names) != len(cameras):
            raise RuntimeError("Reloaded training split does not match the runtime's cameras")
        device = cameras[0]["K"].device
        for name, cam in zip(names, cameras):
            frame = dataset["frames"][name]
            same = (frame["width"] == cam["width"] and frame["height"] == cam["height"]
                    and torch.allclose(frame["K"].to(device), cam["K"])
                    and torch.allclose(frame["viewmat"].to(device), cam["viewmat"]))
            if not same:
                raise RuntimeError(f"Reloaded training view {name} differs from the runtime camera")
        images = (dataset["frames"][n]["image"].to(device).permute(2, 0, 1) for n in names)
        self.set_training_images(images)
        self.schedule_seconds = time.perf_counter() - t0

    def set_training_images(self, images):
        """Eq. 2-3 from [3,H,W] training images: energies E_r and end iterations T_r."""
        with torch.no_grad():
            self.energies = S.scale_energies(images, self.c["max_scale_factor"])
        self.ends = S.end_iterations(self.energies, self.c["densify_until"])
        if self.c["g2l_boundary"] is None:
            self.boundary = S.full_resolution_step(self.ends)  # T_2 (Ablation, G2L paragraph)
        self.schedule_active = bool(self.c["c2f"])
        self.current_scale = self.scale_at(1)
        self._sync_state()

    # ---- schedules ----------------------------------------------------------------------
    def scale_at(self, iteration):
        if not self.schedule_active:
            return 1
        return S.scale_at(self.ends, iteration)

    def clone_allowed(self, step):
        if not self.c["g2l"]:
            return True
        if self.boundary is None:
            raise RuntimeError("decomp-density: G2L boundary unknown (schedule not built)")
        return step >= self.boundary

    def update_lr(self, optimizers, step):
        """Position LR for step + 1: held during reduced resolution, then log-linear decay."""
        if self.schedule_active and self.c["lr_mode"] == "hold_then_decay":
            optimizers["means"].param_groups[0]["lr"] = S.position_lr(
                step + 1, self.lr_init, self.lr_final, self.c["total_steps"],
                S.full_resolution_step(self.ends))

    # ---- runtime callbacks --------------------------------------------------------------
    def before_backward(self, params, optimizers, step, outputs):
        if self.needs_images and self.ends is None and self.c["c2f"]:
            raise RuntimeError("decomp-density: C2F schedule not built before training")
        full_h, full_w = outputs["full_size"]
        h, w = outputs["rgb"].shape[:2]
        if outputs["render_scale"] != self.current_scale or (h, w) != scaled_size(
                full_h, full_w, self.current_scale):
            raise RuntimeError("Rendered size disagrees with the C2F schedule")
        if self.current_scale != self.scale_at(step):
            raise RuntimeError(f"C2F scale out of sync at step {step}")
        outputs["info"]["means2d"].retain_grad()

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c = self.c
        self.update_lr(optimizers, step)
        if step < c["densify_until"]:
            self._accumulate(outputs["info"])
            if step > c["refine_start"] and step % c["refine_every"] == 0:
                self.densify_and_prune(params, optimizers, step)
            if step % c["reset_every"] == 0:
                ops.reset_opa(params, optimizers, self.stats, value=c["reset_value"])
        self.current_scale = self.scale_at(step + 1)
        self._sync_state()

    # ---- densification ------------------------------------------------------------------
    def _reset_stats(self, params):
        n, device = len(params["means"]), params["means"].device
        self.stats = {"grad2d": torch.zeros(n, device=device),
                      "count": torch.zeros(n, device=device)}

    def _accumulate(self, info):
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

    def prune_threshold(self, params):
        if not self.c["adaptive_prune"]:
            return float(self.c["prune_opa"])
        return S.adaptive_threshold(torch.sigmoid(params["opacities"].flatten()),
                                    self.c["prune_ratio"], self.c["prune_upper"])

    def densify_and_prune(self, params, optimizers, step):
        """3DGS densify_and_prune with the G2L phase rule and the adaptive prune threshold."""
        c = self.c
        grads = self.stats["grad2d"] / self.stats["count"]
        grads[grads.isnan()] = 0.0
        qualified = grads >= c["grow_grad2d"]
        small = torch.exp(params["scales"]).max(dim=-1).values <= c["grow_scale3d"] * self.scene_scale
        if self.clone_allowed(step):  # local refinement: split + clone
            clone, split = qualified & small, qualified & ~small
        elif c["global_mode"] == "skip_clone":  # global spread: split only
            clone, split = torch.zeros_like(qualified), qualified & ~small
        else:  # "split_all": every qualified Gaussian is split
            clone, split = torch.zeros_like(qualified), qualified
        n_clone, n_split = int(clone.sum()), int(split.sum())
        if n_clone:
            ops.duplicate(params, optimizers, self.stats, clone)
        if n_split:
            split = torch.cat([split, torch.zeros(n_clone, dtype=torch.bool, device=split.device)])
            ops.split(params, optimizers, self.stats, split)
        # Prune after growth on the full set (3DGS order).
        tau = self.prune_threshold(params)
        prune = torch.sigmoid(params["opacities"].flatten()) < tau
        if step > c["reset_every"]:
            big = torch.exp(params["scales"]).max(dim=-1).values > c["prune_scale3d"] * self.scene_scale
            prune |= big
        n_prune = int(prune.sum())
        if n_prune:
            ops.remove(params, optimizers, self.stats, prune)
        self._reset_stats(params)
        self.last_densify = {"step": step, "scale": self.current_scale, "clone": n_clone,
                             "split": n_split, "pruned": n_prune, "tau": tau,
                             "n": len(params["means"])}
        self.history.append(self.last_densify)

    def _sync_state(self):
        # Checkpointed with weights_only=True: plain numbers and lists only.
        self.state = {
            "current_scale": int(self.current_scale),
            "g2l_boundary": int(self.boundary) if self.boundary is not None else -1,
            "c2f_end_iterations": [int(self.ends[r]) for r in sorted(self.ends)] if self.ends
            else [],
        }


def create_density(spec):
    return DecompDensity(spec)
