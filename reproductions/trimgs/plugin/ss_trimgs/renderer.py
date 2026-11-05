"""Flat-Gaussian gsplat renderer (Trim3DGS FlattenGaussianModel) + extra passes for the loss.

`render` is the stock SplatStudio gsplat renderer except that axis 0 of every Gaussian's scale
is the constant `flat_eps` (official `get_scaling`). It forwards the camera so that the normal
loss can render the depth/normal pass of `tune.py:normal_regularization` with the same
rasterizer and camera.
"""

import torch
from gsplat import rasterization

from . import core

CAPS = (
    "render.rgb",
    "render.depth.expected_z",
    "gradient.screen.signed",
    "render.flat_normal",
    "training.camera",
)


def render_features(params, camera, eps, features):
    """Alpha-blend per-Gaussian features [N,D] over a black background (override_color)."""
    out, alpha, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=core.flat_scales(params["scales"], eps),
        opacities=params["opacities"].sigmoid(),
        colors=features,
        viewmats=camera["viewmat"][None],
        Ks=camera["K"][None],
        width=camera["width"],
        height=camera["height"],
        sh_degree=None,
        packed=False,
        render_mode="RGB",
        rasterize_mode="classic",
    )
    return out[0], alpha[0], info


class FlatRenderer:
    def __init__(self, spec):
        unknown = set(spec.config) - {"flat_eps"}
        if unknown:
            raise ValueError(f"Unknown renderer options: {sorted(unknown)}")
        self.eps = float(spec.config.get("flat_eps", 1e-8))
        self.capabilities = set(CAPS)

    def render(self, params, camera, sh_degree=3):
        colors = torch.cat((params["sh0"], params["shN"]), dim=1)
        out, alpha, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=core.flat_scales(params["scales"], self.eps),
            opacities=params["opacities"].sigmoid(),
            colors=colors,
            viewmats=camera["viewmat"][None],
            Ks=camera["K"][None],
            width=camera["width"],
            height=camera["height"],
            sh_degree=sh_degree,
            packed=False,
            render_mode="RGB+ED",
            rasterize_mode="classic",
        )
        return {
            "rgb": out[0, ..., :3] + (1 - alpha[0]),  # runtime convention: white background
            "depth": out[0, ..., 3:4],
            "alpha": alpha[0],
            "info": info,
            "camera": {k: camera[k] for k in ("viewmat", "K", "width", "height")},
            "flat_eps": self.eps,
        }


def create_renderer(spec):
    return FlatRenderer(spec)
