"""SWAG density: 3DGS adaptive density control (splatstudio gsplat ``Density`` = gsplat
DefaultStrategy + the 3DGS opacity reset) with the paper's schedule (supp. Tab. 7: densify every
2000 iterations until 30000), plus the learned bank (embeddings / hash grid / MLP) and the 3DGS
position-LR schedule (decay over ``position_lr_max_steps`` = 30000, then constant; A6).
"""

import torch
from splatstudio import ComponentSpec
from splatstudio.backends.gsplat import Density as GsplatDensity

from . import bank as bank_mod
from .bank import BANK_DEFAULTS, SwagBank

STRATEGY_DEFAULTS = {
    "refine_every": 2000,        # supp. Tab. 7 densification interval
    "refine_start_iter": 500,    # 3DGS default (A6)
    "refine_stop_iter": 30000,   # supp. Tab. 7 densification end
    "reset_every": 3000,         # 3DGS default (A6)
    "grow_grad2d": 0.0002,
}
DEFAULTS = {**STRATEGY_DEFAULTS, "position_lr_max_steps": 30000, "position_lr_ratio": 0.01,
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
            raise ValueError(f"Unknown swag density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        strat = {k: self.cfg[k] for k in STRATEGY_DEFAULTS}
        self.inner = GsplatDensity(ComponentSpec("inner", "density", "", REQUIRES, (), strat))
        self.requires = set(self.inner.requires)
        self.bank = None
        self.points = None
        self.lr0 = None

    @property
    def bank_cfg(self):
        return {k[5:]: v for k, v in self.cfg.items() if k.startswith("bank_")}

    @property
    def state(self):
        s = dict(self.inner.state or {})
        if self.bank is not None:
            s["bank"] = self.bank.export()
        return s

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.inner.initialize(params, optimizers, scene_scale)
        self.points = params["means"].detach().clone()
        self.lr0 = optimizers["means"].param_groups[0]["lr"]  # already x scene_scale

    def set_training_cameras(self, cameras):
        dev = self.points.device
        self.bank = SwagBank([c["viewmat"] for c in cameras], self.points, self.bank_cfg, dev)
        self.bank.train()
        bank_mod.ACTIVE = self.bank

    def before_backward(self, params, optimizers, step, outputs):
        if self.bank is None:
            raise RuntimeError("swag density needs set_training_cameras (native ss-train)")
        self.inner.before_backward(params, optimizers, step, outputs)

    def after_backward(self, params, optimizers, step, outputs):
        self.bank.after_step(step)
        # Runtime already set its own decay; override with the 3DGS schedule for the next step.
        optimizers["means"].param_groups[0]["lr"] = position_lr(
            self.lr0, step, int(self.cfg["position_lr_max_steps"]),
            float(self.cfg["position_lr_ratio"]))
        self.inner.after_backward(params, optimizers, step, outputs)


def create_density(spec):
    return Density(spec)


@torch.no_grad()
def transient_variance(renderer, params, bank, cameras, degree=3):
    """C3: per-Gaussian Var_I[delta_alpha_tilde^I] over training images (eval U = 0.5)."""
    s = s2 = None
    for i, cam in enumerate(cameras):
        bank.set_override(bank.image_embeddings[i].detach())
        try:
            _, _, ex = renderer.conditioned(params, cam, degree, bank)
        finally:
            bank.set_override(None)
        d = ex["delta_tilde"].double()
        s = d if s is None else s + d
        s2 = d * d if s2 is None else s2 + d * d
    n = len(cameras)
    return (s2 / n - (s / n) ** 2).clamp_min(0).float()
