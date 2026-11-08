"""CRiM-GS density control: Mip-Splatting 3DGS densification + 3D filter, and the motion kernel.

Densification follows the CoMoGaussian train.py (the authors' released successor code, used
because CRiM-GS has no code): densify_and_prune(2e-4, 0.005, extent, size_threshold=None) when
500 < step < 20000 and step % 100 == 0, **no opacity reset** and **no screen/world-size prune**;
compute_3D_filter(train cameras) at start, after each densification and every 100 steps after
densification (until total - 100). gsplat's DefaultStrategy with reset_every = 10**9 reproduces
exactly that (its too-big prune only runs when step > reset_every; gsplat 1.5.x's own reset is
a no-op anyway, see splatstudio.backends.gsplat.GSPLAT_RESET_BROKEN).

The CRiM kernel (``ss_crim_gs.kernel``) is created in ``set_training_cameras`` and stepped in
``after_backward`` (CoMo: kernel optimizer step in the same iteration as the Gaussians).
"""

import torch
from gsplat import DefaultStrategy

from . import kernel as kernel_mod
from .filters import compute_filter_3d, is_filter_step
from .kernel import KERNEL_DEFAULTS, CRiMKernel

REQUIRES = ("gradient.screen.signed",)
NO_RESET = 10**9
DEFAULTS = {
    "grow_grad2d": 0.0002,       # densify_grad_threshold
    "prune_opa": 0.005,          # densify_and_prune min_opacity
    "grow_scale3d": 0.01,        # percent_dense
    "refine_start_iter": 500,    # densify_from_iter
    "refine_stop_iter": 20000,   # densify_until_iter (CoMo arguments)
    "refine_every": 100,         # densification_interval
    "opacity_reset_every": 0,    # 0 = none (CoMo train.py never calls reset_opacity)
    "filter_coef": 0.2,          # Mip-Splatting 3D filter
    "near": 0.2,
    "frustum_margin": 0.15,
    "filter_interval": 100,
    "total_steps": 40000,
    "kernel_export_every": 5000,
    **{f"kernel_{k}": v for k, v in KERNEL_DEFAULTS.items()},
}


class CRiMDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown crim-gs density options: {sorted(unknown)}")
        c = {**DEFAULTS, **spec.config}
        self.cfg = c
        if int(c["opacity_reset_every"]) != 0:
            raise ValueError("opacity reset is not part of the CRiM-GS/CoMo recipe; "
                             "opacity_reset_every must be 0")
        self.strategy = DefaultStrategy(
            prune_opa=c["prune_opa"], grow_grad2d=c["grow_grad2d"], grow_scale3d=c["grow_scale3d"],
            refine_start_iter=int(c["refine_start_iter"]),
            refine_stop_iter=int(c["refine_stop_iter"]), refine_every=int(c["refine_every"]),
            reset_every=NO_RESET, absgrad=False)
        self.kernel_cfg = {k[len("kernel_"):]: v for k, v in c.items() if k.startswith("kernel_")
                           and k != "kernel_export_every"}
        self.kernel_cfg["total_steps"] = int(c["total_steps"])
        self.requires = set(REQUIRES)
        self.state = None
        self.params = None
        self.cameras = None
        self.kernel = None
        self.filter_updates = 0

    # -- runtime hooks -------------------------------------------------------------------
    def set_training_cameras(self, cameras):
        cameras = list(cameras)
        if not cameras:
            raise ValueError("CRiM-GS needs the training cameras")
        self.cameras = cameras
        device = cameras[0]["viewmat"].device
        self.kernel = CRiMKernel(cameras, self.kernel_cfg, device)
        kernel_mod.ACTIVE = self.kernel
        if self.params is not None:
            self.update_filter(self.params)

    def _require_cameras(self):
        if not self.cameras or self.kernel is None:
            raise RuntimeError("CRiM-GS density needs set_training_cameras(...) before training")

    def update_filter(self, params):
        self._require_cameras()
        device = params["means"].device
        if self.cameras[0]["viewmat"].device != device:
            self.cameras = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in c.items()}
                            for c in self.cameras]
        f = compute_filter_3d(params["means"], self.cameras, self.cfg["filter_coef"],
                              self.cfg["near"], self.cfg["frustum_margin"])
        params["filter_3d"].data.copy_(f.to(params["filter_3d"].dtype))
        self.filter_updates += 1

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.strategy.check_sanity(params, optimizers)
        self.state = self.strategy.initialize_state(scene_scale=scene_scale)
        self.params = params
        if self.cameras:
            self.update_filter(params)

    def before_backward(self, params, optimizers, step, outputs):
        self._require_cameras()
        self.strategy.step_pre_backward(params, optimizers, self.state, step, outputs["info"])

    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        self.kernel.after_step(step)
        self.strategy.step_post_backward(params, optimizers, self.state, step, outputs["info"],
                                         packed=False)
        if is_filter_step(step, int(c["filter_interval"]), int(c["refine_start_iter"]),
                          int(c["refine_stop_iter"]), int(c["total_steps"])):
            self.update_filter(params)
        every = int(c["kernel_export_every"])
        if step == int(c["total_steps"]) or (every > 0 and step % every == 0):
            self.state["kernel"] = self.kernel.export()
        for name, p in params.items():  # requires_grad pitfall guard (TODO-runtime.md)
            if not (p.is_leaf and p.requires_grad):
                raise RuntimeError(f"field {name} is no longer a trainable leaf at step {step}")


def create_density(spec):
    return CRiMDensity(spec)
