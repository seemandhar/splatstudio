"""gsplat renderer with DropGaussian (paper §3.2, Eq. 5-6; official gaussian_renderer/__init__.py).

Training renders (autograd enabled) multiply the activated opacity by a fresh dropout mask
M = F.dropout(1, p=r_t): kept Gaussians get 1/(1-r_t) (opacity compensation, Eq. 5), dropped
ones 0. r_t = gamma * t / total_steps (Eq. 6), t = 1-based training-render count (the native
runtime renders once per step; the density component checks the counter against `step`).
Renders without autograd (evaluation, ss-eval) use all Gaussians with unscaled opacity.

Official conventions kept: black background, rendered image clamped to [0, 1] (training too).
"""

import torch
import torch.nn.functional as F
from gsplat import fully_fused_projection, rasterization

CAPABILITIES = {"render.rgb", "gradient.screen.signed"}
BACKGROUNDS = {"black": 0.0, "white": 1.0}
DEFAULTS = {"gamma": 0.2, "total_steps": 10000, "background": "black", "clamp_rgb": True,
            "st_alpha_clamp": True}
ALPHA_CAP = 0.999


def drop_rate(step, gamma, total_steps):
    """Eq. 6: r_t = gamma * t / t_total (official: 0.2 * (iteration / 10000))."""
    return float(gamma) * float(step) / float(total_steps)


def compensation(n, rate, device):
    """Eq. 5 factor M(i): 1/(1-r) for kept Gaussians, 0 for dropped (torch.nn.Dropout on ones)."""
    ones = torch.ones(n, dtype=torch.float32, device=device)
    if rate <= 0.0:
        return ones
    if rate >= 1.0:
        raise ValueError(f"drop rate must be < 1, got {rate}")
    return F.dropout(ones, p=rate, training=True)


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown DropGaussian renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        self.gamma = float(cfg["gamma"])
        self.total_steps = int(cfg["total_steps"])
        if not 0.0 <= self.gamma < 1.0 or self.total_steps < 1:
            raise ValueError("gamma must be in [0, 1) and total_steps >= 1")
        if cfg["background"] not in BACKGROUNDS:
            raise ValueError("background must be black or white")
        self.background = BACKGROUNDS[cfg["background"]]
        self.clamp_rgb = bool(cfg["clamp_rgb"])
        self.st_alpha_clamp = bool(cfg["st_alpha_clamp"])
        self.capabilities = set(CAPABILITIES)
        self.step = 0  # number of training (autograd) renders so far

    def render(self, params, camera, sh_degree=3):
        opacities = params["opacities"].sigmoid()
        scales = params["scales"].exp()
        viewmats, Ks = camera["viewmat"][None], camera["K"][None]
        width, height = int(camera["width"]), int(camera["height"])
        drop = None
        visibility_radii = None
        if torch.is_grad_enabled():
            self.step += 1
            rate = drop_rate(min(self.step, self.total_steps), self.gamma, self.total_steps)
            mask = compensation(len(opacities), rate, opacities.device)
            if rate > 0.0:
                # gsplat culls Gaussians with opacity < 1/255 (i.e. every dropped one); the
                # Inria rasterizer's radius ignores opacity, so dropped in-frustum Gaussians still
                # count as visible for densification statistics. Reproduce that visibility.
                with torch.no_grad():
                    visibility_radii = fully_fused_projection(
                        params["means"], None, params["quats"], scales, viewmats, Ks,
                        width, height, packed=False, opacities=opacities,
                    )[0]
                opacities = opacities * mask
            drop = {"step": self.step, "rate": rate,
                    "kept_fraction": float((mask > 0).float().mean()) if len(mask) else 1.0,
                    "mask": mask}
        if self.st_alpha_clamp:
            # gsplat's backward drops the opacity/geometry gradient wherever opac * G > 0.999;
            # the official (Inria) rasterizer clamps alpha in the forward only and always
            # back-propagates. Compensated opacities o/(1-r) exceed 1 often, so cap the opacity
            # given to gsplat at 0.999 with a straight-through gradient (forward unchanged:
            # gsplat clamps alpha at 0.999 anyway; only pixels with G in (0.999/o, 1) differ).
            capped = opacities.clamp(max=ALPHA_CAP)
            opacities = opacities + (capped - opacities).detach()
        colors = torch.cat((params["sh0"], params["shN"]), dim=1)
        render, alpha, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree,
            packed=False,
            render_mode="RGB",
            rasterize_mode="classic",
        )
        rgb = render[0] + (1 - alpha[0]) * self.background
        if self.clamp_rgb:
            rgb = rgb.clamp(0, 1)  # official renderer clamps before the loss
        info["visibility_radii"] = info["radii"] if visibility_radii is None else visibility_radii
        return {"rgb": rgb, "alpha": alpha[0], "info": info, "drop": drop}


def create_renderer(spec):
    return Renderer(spec)
