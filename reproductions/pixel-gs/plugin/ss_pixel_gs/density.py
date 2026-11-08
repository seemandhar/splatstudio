"""Pixel-GS density control (arXiv 2403.15530, Sec. 3.2-3.3).

gsplat DefaultStrategy (3DGS clone/split/prune with 3DGS defaults) whose densification statistic
is replaced by the official Pixel-GS `add_densification_stats` (github.com/zhengzhang01/Pixel-GS
05c9d55, scene/gaussian_model.py:405-407):
    accum[vis] += ||grad_ndc[vis]|| * f * pixels[vis];  denom[vis] += pixels[vis]
where f is the depth scaling the official rasterizer applies to grad_means2D in backward
(diff_gaussian_rasterization/__init__.py:144). gsplat's means2d is part of the autograd graph, so
the scaling is applied to the statistic only — exactly what the official code's screen-space dummy
gradient affects. Population changes go through gsplat.strategy.ops (DefaultStrategy).
"""

from dataclasses import dataclass

import torch
from gsplat import DefaultStrategy
from gsplat.strategy.ops import reset_opa
from splatstudio.backends.gsplat import GSPLAT_RESET_BROKEN, resets_opacity

from .core import view_contribution
from .kernels import pixel_counts

PIXEL_KEYS = ("pixel_aware", "depth_scaling", "gamma_depth")


@dataclass
class PixelGSStrategy(DefaultStrategy):
    pixel_aware: bool = True  # C1, Eq. 7
    depth_scaling: bool = True  # C2, Eq. 10
    gamma_depth: float = 0.37  # paper Sec. 4.1, code arguments/__init__.py:99

    def _update_state(self, params, state, info, packed=False):
        if packed:
            raise ValueError("Pixel-GS statistics require packed=False")
        if self.absgrad or self.key_for_gradient != "means2d":
            raise ValueError("Pixel-GS uses the signed means2d gradient")
        if info["n_cameras"] != 1:
            raise ValueError("Pixel-GS statistics expect one camera per step")
        grads = info["means2d"].grad
        if grads is None:
            raise RuntimeError("means2d gradient missing; step_pre_backward must retain it")
        grads = grads[0].clone()  # [N,2]
        grads[:, 0] *= info["width"] / 2.0  # Inria NDC units (as gsplat DefaultStrategy)
        grads[:, 1] *= info["height"] / 2.0
        n = len(params["means"])
        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n, device=grads.device)
        if self.refine_scale2d_stop_iter > 0:
            raise ValueError("screen-size pruning is a no-op in official Pixel-GS/3DGS; keep 0")
        vis = (info["radii"][0] > 0).all(dim=-1)  # radii > 0 (official visibility_filter)
        ids = torch.where(vis)[0]
        pixels = pixel_counts(info) if self.pixel_aware else None
        num, den = view_contribution(
            grads.norm(dim=-1),
            pixels,
            info["depths"][0],
            pixel_aware=self.pixel_aware,
            depth_scaling=self.depth_scaling,
            gamma_depth=self.gamma_depth,
            radius=float(state["scene_scale"]),  # Eq. 9 radius = 3DGS cameras_extent
        )
        state["grad2d"].index_add_(0, ids, num[vis])
        state["count"].index_add_(0, ids, den[vis])
        # _grow_gs: grad2d / count.clamp_min(1) == official accum/denom with NaN -> 0
        # (count == 0 implies grad2d == 0).


class PixelGSDensity:
    """splatstudio.backends.gsplat.Density with the Pixel-GS strategy."""

    def __init__(self, spec):
        config = dict(spec.config)
        self.opacity_reset = bool(config.pop("opacity_reset", True))
        self.strategy = PixelGSStrategy(**config)
        self.requires = {"gradient.screen.signed"}
        self.state = None

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.strategy.check_sanity(params, optimizers)
        self.state = self.strategy.initialize_state(scene_scale=scene_scale)

    def before_backward(self, params, optimizers, step, outputs):
        self.strategy.step_pre_backward(params, optimizers, self.state, step, outputs["info"])

    def after_backward(self, params, optimizers, step, outputs):
        self.strategy.step_post_backward(
            params, optimizers, self.state, step, outputs["info"], packed=False
        )
        if self.opacity_reset and GSPLAT_RESET_BROKEN and resets_opacity(self.strategy, step):
            reset_opa(params, optimizers, self.state, value=self.strategy.prune_opa * 2.0)


def create_density(spec):
    return PixelGSDensity(spec)
