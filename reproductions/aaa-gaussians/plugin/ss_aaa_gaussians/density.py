"""3DGS-MCMC density control as used by AAA-Gaussians (official train.py +
scene/gaussian_model.py of DerThomy/AAA-Gaussians @ dc5b3f2, from ubc-vision/3dgs-mcmc), plus
the final-step training-view 3D filter (``compute_3D_filter`` at the last save).

Per step s (the runtime calls ``after_backward`` after the optimizer step; the official loop
relocates/adds before it and injects noise after it):
    500 < s < 25000 and s % 100 == 0: relocate dead (sigmoid(o) <= 0.005) Gaussians onto
        opacity-sampled live ones, then add min(cap, 1.05 N) - N opacity-sampled copies
        (compute_relocation opacity/scale update, zeroed Adam moments);
    every step: means += noise_lr * lr_xyz * Sigma @ (eps * op_sigmoid(1 - o)).
gsplat.strategy.MCMCStrategy implements exactly these operations (gsplat ops keep every
Parameter a trainable leaf); it is wrapped, not re-derived.

Initialisation (official ``create_from_pcd``): opacity 0.5 and scale 0.1 * sqrt(mean 3-NN
distance^2); the runtime's ``--recipe 3dgs`` provides log sqrt(mean 3-NN d^2), we add log 0.1.
cap_max: per-scene values of the official full_eval.py, looked up from the ``--data`` basename
unless given in the config.
"""

import math
import sys
from pathlib import Path

import torch
from gsplat.strategy import MCMCStrategy

from .filters import train_filter_3d

# full_eval.py:17-25 (DerThomy/AAA-Gaussians @ dc5b3f2)
CAPS = {
    "bicycle": 6131954, "flowers": 3636448, "garden": 5834784, "stump": 4961797,
    "treehill": 3783761, "room": 1593376, "counter": 1222956, "kitchen": 1852335,
    "bonsai": 1244819, "truck": 2541226, "train": 1026508, "drjohnson": 3405153,
    "playroom": 2546116,
}
DEFAULTS = {
    "cap_max": None,  # None -> CAPS[basename of ss-train --data]
    "noise_lr": 5e5,  # OptimizationParams.noise_lr
    "refine_start_iter": 500,  # densify_from_iter
    "refine_stop_iter": 25000,  # densify_until_iter
    "refine_every": 100,  # densification_interval
    "min_opacity": 0.005,  # dead_mask = opacity <= 0.005
    "init_opacity": 0.5,  # create_from_pcd
    "init_scale_mult": 0.1,  # log(sqrt(dist2) * 0.1)
    "filter_kernel": 0.3,  # mip_filter.py 0.3 ** 0.5 / forward.cu kernel_size
    "filter_near": 0.2,
    "filter_margin": 0.15,
    "total_steps": None,  # None -> ss-train --steps (default 30000): final step = save
}
REQUIRES = set()


def _argv_value(flag):
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def resolve_cap(cap):
    if cap is not None:
        return int(cap)
    root = _argv_value("--data")
    scene = Path(root).name if root else None
    if scene not in CAPS:
        raise RuntimeError(f"aaa-mcmc-density: no official cap for scene {scene!r}; set "
                           "config 'cap_max'")
    return CAPS[scene]


def resolve_total_steps(total):
    if total is not None:
        return int(total)
    v = _argv_value("--steps")
    return int(v) if v is not None else 30000


class AAADensity:
    requires = REQUIRES

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown aaa-mcmc-density options: {sorted(unknown)}")
        self.c = {**DEFAULTS, **spec.config}
        self.strategy = MCMCStrategy(
            cap_max=1, noise_lr=float(self.c["noise_lr"]),
            refine_start_iter=int(self.c["refine_start_iter"]),
            refine_stop_iter=int(self.c["refine_stop_iter"]),
            refine_every=int(self.c["refine_every"]), min_opacity=float(self.c["min_opacity"]))
        self.state = None
        self.cameras = None
        self.total_steps = None
        self.filter_steps = []

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.strategy.cap_max = resolve_cap(self.c["cap_max"])
        self.total_steps = resolve_total_steps(self.c["total_steps"])
        with torch.no_grad():
            params["opacities"].fill_(math.log(self.c["init_opacity"]
                                               / (1 - self.c["init_opacity"])))
            params["scales"].add_(math.log(self.c["init_scale_mult"]))
            if "filter_3d" in params:
                params["filter_3d"].zero_()  # official filter_3D = None during training
        self.strategy.check_sanity(params, optimizers)
        self.state = self.strategy.initialize_state()
        self.state.update(cap_max=self.strategy.cap_max, total_steps=self.total_steps)

    def set_training_cameras(self, cameras):
        self.cameras = list(cameras)

    def before_backward(self, params, optimizers, step, outputs):
        pass

    def update_filter(self, params):
        if self.cameras is None:
            raise RuntimeError("aaa-mcmc-density needs the training cameras "
                               "(density.set_training_cameras) for the 3D filter")
        f = train_filter_3d(params["means"], self.cameras, self.c["filter_kernel"],
                            self.c["filter_near"], self.c["filter_margin"])
        params["filter_3d"].data.copy_(f.to(params["filter_3d"].dtype))

    def after_backward(self, params, optimizers, step, outputs):
        lr = optimizers["means"].param_groups[0]["lr"]
        n_before = len(params["means"])
        self.strategy.step_post_backward(params, optimizers, self.state, step, outputs["info"],
                                         lr=lr)
        if len(params["means"]) != n_before or step % self.strategy.refine_every == 0:
            bad = [k for k, p in params.items() if not (p.is_leaf and p.requires_grad)]
            if bad:
                raise RuntimeError(f"aaa-mcmc-density froze or detached fields {bad}")
        if self.total_steps is not None and step == self.total_steps:
            self.update_filter(params)  # official compute_3D_filter(trainCameras) at save
            self.filter_steps.append(step)


def create_density(spec):
    return AAADensity(spec)
