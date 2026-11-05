"""gsplat renderer that renders training views at the DashGaussian resolution.

Reduced resolution = (floor(H / r), floor(W / r)) with the same field of view, i.e. pixel
intrinsics scaled by (w/W, h/H). This is what the official code does by passing `render_size`
to the 3DGS rasterizer with unchanged tan(fov) (gaussian_renderer/__init__.py:37). The gsplat
`info` then carries the reduced width/height, so densification gradients are normalised at the
render size exactly like 3DGS's NDC gradients.
"""

import math

import torch
from gsplat import rasterization

from .link import LINK

SCHEDULE_CAPABILITY = "render.resolution_schedule"
CAPABILITIES = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed",
                SCHEDULE_CAPABILITY)


def scaled_size(height, width, scale):
    """Output size of F.interpolate(scale_factor=1/scale, recompute_scale_factor=True)."""
    if scale == 1:
        return height, width
    factor = 1.0 / scale
    return int(math.floor(height * factor)), int(math.floor(width * factor))


def scaled_camera(camera, scale):
    h, w = scaled_size(int(camera["height"]), int(camera["width"]), scale)
    if (h, w) == (camera["height"], camera["width"]):
        return camera
    K = camera["K"].clone()
    K[0] *= w / camera["width"]
    K[1] *= h / camera["height"]
    return {**camera, "K": K, "width": w, "height": h}


class Renderer:
    capabilities = set(CAPABILITIES)

    def __init__(self, spec):
        if spec.config:
            raise ValueError("dash-renderer accepts no configuration")

    def render(self, params, camera, sh_degree=3):
        # Evaluation (no_grad) is always full resolution; training follows the schedule.
        scale = LINK.training_scale() if torch.is_grad_enabled() else 1
        cam = scaled_camera(camera, scale)
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
            rasterize_mode="classic",
        )
        return {
            "rgb": render[0, ..., :3] + (1 - alpha[0]),
            "depth": render[0, ..., 3:4],
            "alpha": alpha[0],
            "info": info,
            "render_scale": scale,
            "full_size": (int(camera["height"]), int(camera["width"])),
        }


def create_renderer(spec):
    return Renderer(spec)
