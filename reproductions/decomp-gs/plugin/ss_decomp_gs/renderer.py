"""gsplat renderer that renders training views at the DecompGS C2F resolution.

Adapted from the DashGaussian port (reproductions/dashgaussian/plugin/ss_dashgaussian/renderer.py).

Reduced resolution = (floor(H / r), floor(W / r)) with the same field of view, i.e. pixel
intrinsics scaled by (w/W, h/H). This is what DashGaussian's code (which DecompGS cites for its C2F scheme) does by passing `render_size`
to the 3DGS rasterizer with unchanged tan(fov). The gsplat
`info` then carries the reduced width/height, so densification gradients are normalised at the
render size exactly like 3DGS's NDC gradients.
"""

import torch
from gsplat import rasterization

from .link import LINK
from .schedule import downscaled_size

SCHEDULE_CAPABILITY = "render.c2f_resolution"
CAPABILITIES = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed",
                SCHEDULE_CAPABILITY)


def scaled_size(height, width, scale):
    """floor(H/r) x floor(W/r), identical to schedule.downscaled_size."""
    return downscaled_size(height, width, scale)


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
            raise ValueError("decomp-renderer accepts no configuration")

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
