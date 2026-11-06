"""gsplat renderer (absolute gradients, AbsGS) that renders training views at the ResGS
image-pyramid resolution (official: per-stage cameras built from cv2.pyrDown images with the
original FoV, utils/camera_utils.py:loadCam). Evaluation (no_grad) is always full resolution.
"""

import torch
from gsplat import rasterization

from .link import LINK
from .pyramid import scaled_camera

PYRAMID_CAPABILITY = "render.image_pyramid"
CAPABILITIES = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed",
                "gradient.screen.absolute", PYRAMID_CAPABILITY)


class Renderer:
    capabilities = set(CAPABILITIES)

    def __init__(self, spec):
        if spec.config:
            raise ValueError("resgs-renderer accepts no configuration")

    def render(self, params, camera, sh_degree=3):
        level, owner = LINK.training_level() if torch.is_grad_enabled() else (0, None)
        cam = scaled_camera(camera, level)
        colors = torch.cat((params["sh0"], params["shN"]), dim=1)
        render, alpha, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"].exp(),
            opacities=params["opacities"].sigmoid(),
            colors=colors,
            viewmats=cam["viewmat"][None],
            Ks=cam["K"][None],
            width=cam["width"],
            height=cam["height"],
            sh_degree=sh_degree,
            packed=False,
            render_mode="RGB+ED",
            absgrad=True,
            rasterize_mode="classic",
        )
        return {
            "rgb": render[0, ..., :3] + (1 - alpha[0]),
            "depth": render[0, ..., 3:4],
            "alpha": alpha[0],
            "info": info,
            "pyramid_level": level,
            "pyramid_owner": owner,
        }


def create_renderer(spec):
    return Renderer(spec)
