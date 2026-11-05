"""EDGS densification-free density control (CompVis/EDGS `source/trainer.py`, no_densify=True).

- initialize: apply the correspondence initialization's per-point extras (`edgs_init.npz`, found by
  the SHA-256 of the initial means) - isotropic log scales, opacity logits, SH rest; opacity LR 0.025
  (`configs/gs/base.yaml`); means LR floored at lr(8000) (`train.max_lr: True`).
- after_backward: prune opacity < 0.005 every step before 15000 (`EDGSTrainer.prune`); opacity
  `exp(logit) * 0.99` every 10 steps before 15000 (`train.reduce_opacity: True`). No clone/split,
  no opacity reset.
"""

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from gsplat.strategy.ops import remove

from .core import lr_floored

DEFAULTS = {
    "init_root": "~/datasets/edgs-init",
    "sh_init": True,
    "prune_opacity": 0.005,
    "prune_until": 15000,
    "opacity_decay": 0.99,
    "decay_every": 10,
    "decay_until": 15000,
    "means_lr_floor_step": 8000,
    "total_steps": 30000,
    "opacity_lr": 0.025,
    # The runtime's --recipe 3dgs computes 3-NN scales with a [4096, N] cdist (61 GB for EDGS's
    # ~4M points), so EDGS trains with --recipe splatstudio and restores 3DGS's means-LR scale
    # (getNerfppNorm camera extent of the training views) from set_training_cameras.
    "camera_extent_lr": True,
}


def means_sha256(means):
    return hashlib.sha256(
        np.ascontiguousarray(means.detach().float().cpu().numpy()).tobytes()
    ).hexdigest()


def find_sidecar(root, sha):
    root = Path(root).expanduser()
    for path in sorted(root.rglob("edgs_init.npz")):
        with np.load(path, allow_pickle=False) as data:
            if str(data["points_sha256"]) == sha:
                return path
    raise FileNotFoundError(
        f"No edgs_init.npz under {root} matches the initial points (sha256 {sha[:12]}...). "
        "Train on a scene prepared by ss-edgs-init with --num-gaussians >= its point count."
    )


class EDGSDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown edgs-density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.requires = set()
        self.state = {}
        self.init_override = None  # dict of arrays (probes); otherwise looked up by hash
        self.base_means_lr = None

    @torch.no_grad()
    def initialize(self, params, optimizers, scene_scale=1.0):
        cfg = self.cfg
        extras = self.init_override
        if extras is None:
            sha = means_sha256(params["means"])
            path = find_sidecar(cfg["init_root"], sha)
            with np.load(path, allow_pickle=False) as data:
                extras = {k: data[k] for k in ("log_scales", "opacity_logits", "shN")}
            self.state["init_file"] = str(path)
        dev = params["means"].device
        n = len(params["means"])
        log_scales = torch.as_tensor(extras["log_scales"], dtype=torch.float32, device=dev)
        opac = torch.as_tensor(extras["opacity_logits"], dtype=torch.float32, device=dev)
        if log_scales.shape != (n,) or opac.shape != (n,):
            raise ValueError("EDGS init extras do not match the number of Gaussians")
        params["scales"].copy_(log_scales[:, None].expand(-1, 3))
        params["opacities"].copy_(opac)
        params["quats"].zero_()
        params["quats"][:, 0] = 1
        if cfg["sh_init"]:
            shn = torch.as_tensor(extras["shN"], dtype=torch.float32, device=dev)
            if shn.shape != params["shN"].shape:
                raise ValueError("EDGS SH init does not match shN")
            params["shN"].copy_(shn)
        else:
            params["shN"].zero_()  # released code: features_rest = 0
        optimizers["opacities"].param_groups[0]["lr"] = cfg["opacity_lr"]
        self.optimizers = optimizers
        self.runtime_scene_scale = float(scene_scale)
        self.base_means_lr = optimizers["means"].param_groups[0]["lr"]
        optimizers["means"].param_groups[0]["lr"] = self._means_lr(1)
        self.extent_applied = not cfg["camera_extent_lr"]
        self.state.update(initial_gaussians=n, pruned=0)

    def set_training_cameras(self, cameras):
        if not self.cfg["camera_extent_lr"]:
            return
        centres = torch.stack([torch.linalg.inv(c["viewmat"])[:3, 3] for c in cameras])
        extent = 1.1 * float(torch.linalg.vector_norm(centres - centres.mean(0), dim=1).max())
        extent = max(extent, 0.01)
        self.base_means_lr = self.base_means_lr / self.runtime_scene_scale * extent
        self.optimizers["means"].param_groups[0]["lr"] = self._means_lr(1)
        self.state["camera_extent"] = extent
        self.extent_applied = True

    def _means_lr(self, step):
        return lr_floored(
            self.base_means_lr, step, self.cfg["means_lr_floor_step"], self.cfg["total_steps"]
        )

    def before_backward(self, params, optimizers, step, outputs):
        pass

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.cfg
        if not self.extent_applied:
            raise RuntimeError("edgs-density needs set_training_cameras() for the camera-extent LR")
        if step < cfg["prune_until"]:
            mask = params["opacities"].sigmoid() < cfg["prune_opacity"]
            if mask.any() and not mask.all():
                remove(params, optimizers, {}, mask)
                self.state["pruned"] += int(mask.sum())
        if step < cfg["decay_until"] and step % cfg["decay_every"] == 0:
            # torch.log(torch.exp(_opacity) * 0.99) on the logit, exactly as the official code.
            params["opacities"].add_(math.log(cfg["opacity_decay"]))
        # LR for the next step (the runtime's own decay is applied just before this call).
        optimizers["means"].param_groups[0]["lr"] = self._means_lr(step + 1)


def create_density(spec):
    return EDGSDensity(spec)
