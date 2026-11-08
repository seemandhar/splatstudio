"""gsplat renderer whose opacity is masked by the LP-3DGS Gumbel-sigmoid mask (Eq. 8)."""

import torch
from gsplat import rasterization

from .core import effective_opacity

CAPABILITIES = {"render.rgb", "render.depth.expected_z", "gradient.screen.signed"}
DEFAULTS = {"tau": 0.5}


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown LP-3DGS renderer options: {sorted(unknown)}")
        self.tau = float({**DEFAULTS, **spec.config}["tau"])
        if self.tau <= 0:
            raise ValueError("tau must be > 0")
        self.capabilities = set(CAPABILITIES)

    def render(self, params, camera, sh_degree=3, noise=None):
        colors = torch.cat((params["sh0"], params["shN"]), dim=1)
        render, alpha, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"].exp(),
            opacities=effective_opacity(params, self.tau, noise),
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
            "rgb": render[0, ..., :3] + (1 - alpha[0]),
            "depth": render[0, ..., 3:4],
            "alpha": alpha[0],
            "info": info,
        }


def create_renderer(spec):
    return Renderer(spec)
