"""WE-GS density: 3DGS adaptive density control (splatstudio gsplat ``Density`` = gsplat
DefaultStrategy + the 3DGS opacity reset; paper §5.1 "following similar settings as in 3DGS"),
the learned bank (encoder + G_phi, stepped by its own Adam) and the 3DGS position-LR schedule
(decay over position_lr_max_steps = 30000, then constant; authors' OptimizationParams default).
Pattern from reproductions/swag ``ss_swag.density``.
"""

import json

from splatstudio import ComponentSpec
from splatstudio.backends.gsplat import Density as GsplatDensity

from . import bank as bank_mod
from .bank import BANK_DEFAULTS, WEGSBank

STRATEGY_DEFAULTS = {
    "refine_every": 100,
    "refine_start_iter": 500,
    "refine_stop_iter": 15000,
    "reset_every": 3000,
    "grow_grad2d": 0.0002,
}
DEFAULTS = {**STRATEGY_DEFAULTS, "position_lr_max_steps": 30000, "position_lr_ratio": 0.01,
            "diagnostics_every": 100,
            **{f"bank_{k}": v for k, v in BANK_DEFAULTS.items()}}
REQUIRES = ("gradient.screen.signed",)


def position_lr(lr0, step, max_steps, ratio):
    """3DGS get_expon_lr_func(lr0, ratio*lr0, max_steps): log-linear, clamped at max_steps."""
    t = min(max(step / max_steps, 0.0), 1.0)
    return lr0 * ratio**t


class Density:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown we-gs density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        strat = {k: self.cfg[k] for k in STRATEGY_DEFAULTS}
        self.inner = GsplatDensity(ComponentSpec("inner", "density", "", REQUIRES, (), strat))
        self.requires = set(self.inner.requires)
        self.bank = None
        self.lr0 = None
        self.counters = {"transient_gaussians_last": 0, "history": []}

    @property
    def bank_cfg(self):
        return {k[5:]: v for k, v in self.cfg.items() if k.startswith("bank_")}

    @property
    def state(self):
        s = dict(self.inner.state or {})
        if self.bank is not None:
            s["bank"] = self.bank.export()
        s["counters"] = dict(self.counters)
        return s

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.inner.initialize(params, optimizers, scene_scale)
        self.lr0 = optimizers["means"].param_groups[0]["lr"]  # already x scene_scale
        self.bank = WEGSBank(self.bank_cfg, params["means"].device)
        self.bank.train()
        bank_mod.ACTIVE = self.bank

    def before_backward(self, params, optimizers, step, outputs):
        if self.bank is None:
            raise RuntimeError("we-gs density was not initialised")
        self.inner.before_backward(params, optimizers, step, outputs)

    def after_backward(self, params, optimizers, step, outputs):
        self.bank.after_step(step)
        if "transient_gaussians" in outputs:
            self.counters["transient_gaussians_last"] = int(outputs["transient_gaussians"])
            if step == 1 or step % int(self.cfg["diagnostics_every"]) == 0:
                row = {"step": int(step), "transient_gaussians": int(outputs["transient_gaussians"]),
                       "gaussians": len(params["means"]),
                       "static_mask_mean": float(outputs["static_mask_mean"])}
                self.counters["history"].append(row)
                print(json.dumps({"wegs_diagnostics": row}), flush=True)
        optimizers["means"].param_groups[0]["lr"] = position_lr(
            self.lr0, step, int(self.cfg["position_lr_max_steps"]),
            float(self.cfg["position_lr_ratio"]))
        self.inner.after_backward(params, optimizers, step, outputs)


def create_density(spec):
    return Density(spec)
