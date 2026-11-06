"""Spec-Gaussian density controller: 3DGS adaptive density control on absolute (L1) screen
gradients (paper Sec. 3.3, tau_g = 0.0005 for the performance version), plus ownership of the
global ASG decoder bank.

Wraps SplatStudio's gsplat ``Density`` (DefaultStrategy with absgrad, 3DGS opacity reset every
3000 steps). Additional official behaviour (scene/gaussian_model.py, train.py of
github.com/ingra14m/Specular-Gaussians @0b40ce3):
  * ``spatial_lr_scale = 5`` is hard-coded for the xyz learning rate (not the camera extent);
    the runtime's means LR (1.6e-4 * scene_scale) is rescaled once in ``initialize``.
  * the decoder (Theta + Psi) is optimised by its own Adam with a linear LR decay, stepped after
    every runtime step, and packed into ``spec_blob`` so the checkpoint carries it.
"""

from dataclasses import replace

import torch
from splatstudio.backends.gsplat import Density

from . import bank as bank_mod
from .asg import DEFAULTS as ASG_DEFAULTS
from .asg import pack
from .renderer import BLOB

REQUIRES = {"gradient.screen.absolute"}
STRATEGY_DEFAULTS = {
    "absgrad": True,
    "grow_grad2d": 0.0005,  # densify_grad_threshold (arguments/__init__.py, performance version)
    "prune_opa": 0.005,  # densify_and_prune(.., 0.005, ..)
    "grow_scale3d": 0.01,  # percent_dense
    "prune_scale3d": 0.1,  # big_points_ws: 0.1 * extent
    "refine_start_iter": 500,
    "refine_stop_iter": 15000,
    "refine_every": 100,
    "reset_every": 3000,
}
SPEC_DEFAULTS = {
    "spatial_lr_scale": 5.0,  # gaussian_model.py: self.spatial_lr_scale = 5
    "mlp_lr_init": 0.0025,  # SpecularModel: get_linear_noise_func(feature_lr, ...)
    "mlp_lr_final": 0.0025 / 20,
    "mlp_lr_max_steps": 30000,  # specular_lr_max_steps
}
DEFAULTS = {**STRATEGY_DEFAULTS, **SPEC_DEFAULTS, **ASG_DEFAULTS}


class SpecDensity:
    requires = REQUIRES

    def __init__(self, spec):
        cfg = {**DEFAULTS, **spec.config}
        unknown = set(cfg) - set(DEFAULTS) - {"opacity_reset"}
        if unknown:
            raise ValueError(f"Unknown density options: {sorted(unknown)}")
        self.cfg = cfg
        strategy = {k: cfg[k] for k in STRATEGY_DEFAULTS}
        if "opacity_reset" in cfg:
            strategy["opacity_reset"] = cfg["opacity_reset"]
        self.base = Density(replace(spec, config=strategy))
        if self.base.requires != REQUIRES:
            raise ValueError("Spec-Gaussian densification needs absolute screen gradients")
        self.bank = None

    @property
    def state(self):
        return self.base.state

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.base.initialize(params, optimizers, scene_scale)
        group = optimizers["means"].param_groups[0]
        group["lr"] = group["lr"] * self.cfg["spatial_lr_scale"] / scene_scale
        self.bank = bank_mod.SpecBank(self.cfg, params["means"].device)
        bank_mod.ACTIVE = self.bank
        self.base.state["spec"] = {"spatial_lr_scale": float(self.cfg["spatial_lr_scale"]),
                                   "camera_scene_scale": float(scene_scale),
                                   "means_lr0": float(group["lr"]), "mlp_updates": 0}
        self.bake(params)

    def before_backward(self, params, optimizers, step, outputs):
        self.base.before_backward(params, optimizers, step, outputs)

    def after_backward(self, params, optimizers, step, outputs):
        lr = self.bank.after_step(step)
        self.base.after_backward(params, optimizers, step, outputs)
        self.bake(params)
        self.base.state["spec"]["mlp_updates"] = self.bank.updates
        self.base.state["spec"]["mlp_lr"] = float(lr)

    @torch.no_grad()
    def bake(self, params):
        flat = pack(self.bank.network, self.bank.step)
        blob = params[BLOB].view(-1)
        if flat.numel() > blob.numel():
            raise ValueError(f"{blob.numel()} Gaussians cannot hold {flat.numel()} decoder floats")
        blob.zero_()
        blob[: flat.numel()] = flat


def create_density(spec):
    return SpecDensity(spec)
