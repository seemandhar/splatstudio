"""GS-IR density / stage controller.

Stage 1: 3DGS adaptive density control (gsplat DefaultStrategy wrapped by SplatStudio's
``Density``, including the 3DGS opacity-reset fix), the means LR schedule of the official code
(exponential 1.6e-4 -> 1.6e-6 x extent over ``stage1_steps`` = position_lr_max_steps 30000,
constant afterwards; the runtime's own schedule spans all steps and is overridden here), and
the official normal initialisation (0, 0, 1).
At the end of step ``pbr_start`` (after its optimizer step) the Gaussians are frozen in
practice (no stage-2 loss reaches them), occlusion volumes are baked (baking.py) and the
decomposition bank (cube light, irradiance volumes, light Adam) is created.
Stage 2: no densification; the light optimizer is stepped after every runtime step.
"""

import torch
from splatstudio.backends.gsplat import Density
from splatstudio.specs import ComponentSpec

from . import bank as bank_mod
from .baking import bake_occlusion

STRATEGY_DEFAULTS = {"absgrad": False, "grow_grad2d": 0.0002}
DEFAULTS = {
    "strategy": dict(STRATEGY_DEFAULTS),
    "pbr_start": 30000,
    "stage1_steps": 30000,
    "lr_final_ratio": 0.01,
    # baking (README Mip-NeRF 360: --bound 16.0 --occlu_res 256 --occlusion 0.4; baking.py)
    "occ_res": 256,
    "occ_bound": 16.0,
    "occ_threshold": 0.4,
    "occ_valid": 1.5,
    "occ_bands": 4,
    "occ_cube_res": 256,
    "occ_near": 0.2,
    "occ_batch": 32,
    "indirect": True,
    # decomposition state (train.py)
    "cube_res": 256,
    "irr_res": 64,
    "irr_bands": 3,
    "irr_bound": 1.5,
    "light_lr": 0.05,
    "seed": 0,
}


def merge(defaults, config):
    out = dict(defaults)
    for k, v in config.items():
        if k not in defaults:
            raise ValueError(f"Unknown gsir-density option: {k}")
        out[k] = {**defaults[k], **v} if isinstance(defaults[k], dict) else v
    return out


class GSIRDensity:
    requires = {"gradient.screen.signed"}

    def __init__(self, spec):
        self.cfg = merge(DEFAULTS, spec.config)
        self.base = Density(ComponentSpec("adaptive-density", "density",
                                          "splatstudio.backends.gsplat:create_density",
                                          ("gradient.screen.signed",), (),
                                          dict(self.cfg["strategy"])))
        self.bank = None
        self._params = None
        self.initial_lr = None
        self.events = []

    # The runtime saves ``density.state`` in checkpoint.pt; the GS-IR bank travels with it.
    @property
    def state(self):
        out = {"strategy": self.base.state, "events": list(self.events)}
        if self.bank is not None:
            out["gsir"] = self.bank.export(self._params)
        return out

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.base.initialize(params, optimizers, scene_scale)
        with torch.no_grad():
            if "gsir_normal" in params:
                n = torch.zeros_like(params["gsir_normal"])
                n[:, 2] = 1.0
                params["gsir_normal"].copy_(n)
        self.initial_lr = optimizers["means"].param_groups[0]["lr"]
        self.bank = bank_mod.GSIRBank(self.cfg, params["means"].device)
        bank_mod.ACTIVE["bank"] = self.bank
        self._params = params

    def before_backward(self, params, optimizers, step, outputs):
        if self.bank is None or self.bank.stage == 1:
            self.base.before_backward(params, optimizers, step, outputs)

    def after_backward(self, params, optimizers, step, outputs):
        self._params = params
        c = self.cfg
        if self.bank is None or self.bank.stage == 1:
            self.base.after_backward(params, optimizers, step, outputs)
        else:
            self.bank.step_light()
        if self.initial_lr is not None:
            s1 = int(c["stage1_steps"])
            optimizers["means"].param_groups[0]["lr"] = self.initial_lr * (
                float(c["lr_final_ratio"]) ** (min(step, s1) / s1))
        if self.bank is not None and self.bank.stage == 1 and step == int(c["pbr_start"]):
            self.start_decomposition(params, step)

    def start_decomposition(self, params, step):
        c = self.cfg
        baked = None
        if c["indirect"]:
            baked = bake_occlusion(
                params, res=int(c["occ_res"]), bound=float(c["occ_bound"]),
                valid=float(c["occ_valid"]), threshold=float(c["occ_threshold"]),
                bands=int(c["occ_bands"]), cube_res=int(c["occ_cube_res"]),
                near=float(c["occ_near"]), batch=int(c["occ_batch"]),
                log=lambda m: print(m, flush=True))
            print(f"gs-ir baking done: {baked['stats']}", flush=True)
        self.bank.start_decomposition(baked, seed=int(c["seed"]))
        self.events.append({"step": int(step), "event": "decomposition_start",
                            "gaussians": int(params["means"].shape[0]),
                            **({k: v for k, v in baked["stats"].items()} if baked else {})})
        torch.cuda.empty_cache()


def create_density(spec):
    return GSIRDensity(spec)
