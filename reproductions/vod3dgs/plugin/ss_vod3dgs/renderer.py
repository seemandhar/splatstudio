"""gsplat rasterization with VoD view-dependent opacity (Eq. 9). Same call as the built-in
SplatStudio renderer; only the opacity input changes."""

import torch
from gsplat import rasterization

from .vod import camera_center, vod_logits

CAPS = {"render.rgb", "render.depth.expected_z", "gradient.screen.signed"}


class Renderer:
    capabilities = CAPS

    def __init__(self, spec):
        if spec.config:
            raise ValueError(f"Unknown renderer options: {sorted(spec.config)}")

    def opacities(self, params, camera):
        c = camera_center(camera["viewmat"])
        return torch.sigmoid(vod_logits(params["opacities"], params["vod_S"], params["means"], c)), c

    def render(self, params, camera, sh_degree=3):
        opac, center = self.opacities(params, camera)
        colors = torch.cat((params["sh0"], params["shN"]), dim=1)
        render, alpha, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"].exp(),
            opacities=opac,
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
            "camera_center": center.detach(),
        }


def create_renderer(spec):
    return Renderer(spec)
