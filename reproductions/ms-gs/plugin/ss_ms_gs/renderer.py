"""MS-GS renderer: pixel-coverage selection (C1, C3) in front of the gsplat rasterizer.

Per render: S_k for every Gaussian (coverage.pixel_coverage, the official kernel's formula),
the selection rule (coverage.select), then gsplat `rasterization` on the selected subset only,
like the official kernel that gives culled Gaussians radius 0 (not drawn, no gradient, not
visible). Training renders (grad enabled) are drawn at the resolution index chosen by the MS
sampler (density controller via LINK): size (int(W/2^r), int(H/2^r)), pixel intrinsics scaled by
(w/W, h/H) with the centred principal point -- the official FoV camera (utils/camera_utils.py).
"""

import torch
from gsplat import rasterization

from .coverage import pixel_coverage, select
from .link import LINK


@torch.no_grad()
def coverage_and_mask(params, camera, cfg):
    """(S_k, keep) for every Gaussian: fused CUDA kernel for CUDA tensors, torch reference on
    CPU (probe msgs-cuda-kernel checks they agree)."""
    if params["means"].is_cuda:
        from .kernel import coverage_select_cuda

        return coverage_select_cuda(params, camera, cfg)
    px = pixel_coverage(params, camera, cfg["opacity_level"])
    keep = select(px, params["ms_max_px"].detach(), params["ms_min_px"].detach(),
                  filter_small=cfg["filter_small"], filter_large=cfg["filter_large"],
                  min_rel=cfg["min_rel"], max_rel=cfg["max_rel"],
                  abs_threshold=cfg["abs_threshold"])
    return px, keep

SELECTION_CAPABILITY = "render.multiscale_selection"
CAPABILITIES = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed",
                SELECTION_CAPABILITY)
DEFAULTS = {
    "filter_small": True,  # --filter_small
    "filter_large": True,  # filter_large = insert_large (train.py:125)
    "min_rel": 0.5,  # S_rel^min (Eq. 7; forward.cu:271)
    "max_rel": 2.0,  # forward.cu:279 (paper text 1.5)
    "abs_threshold": 2.0,  # S_T = 2 px
    "opacity_level": 1.0 / 255.0,  # sigma_T
}


def scaled_camera(camera, level):
    if level == 0:
        return camera
    s = 2**level
    W, H = int(camera["width"]), int(camera["height"])
    w, h = int(W / s), int(H / s)
    K = camera["K"].clone()
    K[0] *= w / W
    K[1] *= h / H
    return {**camera, "K": K, "width": w, "height": h}


class Renderer:
    capabilities = set(CAPABILITIES)

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown ms-gs renderer options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}

    def selection(self, params, camera):
        return coverage_and_mask(params, camera, self.cfg)

    def render(self, params, camera, sh_degree=3):
        level = LINK.training_level() if torch.is_grad_enabled() else 0
        cam = scaled_camera(camera, level)
        px, keep = self.selection(params, cam)
        # Culled Gaussians get opacity 0: gsplat's projection then gives them radius 0
        # (opacity < 1/255 cull), i.e. not drawn, no gradient, not visible -- as the official
        # kernel's early return.
        opac = params["opacities"].sigmoid() * keep.to(params["opacities"].dtype)
        colors = torch.cat((params["sh0"], params["shN"]), dim=1)
        render, alpha, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"].exp(),
            opacities=opac,
            colors=colors,
            viewmats=cam["viewmat"][None],
            Ks=cam["K"][None],
            width=int(cam["width"]),
            height=int(cam["height"]),
            sh_degree=sh_degree,
            packed=False,
            render_mode="RGB+ED",
            rasterize_mode="classic",
        )
        info["ms_keep"] = keep
        info["ms_px"] = px
        info["ms_level"] = level
        # Runtime convention: RGB over a white background.
        return {
            "rgb": render[0, ..., :3] + (1 - alpha[0]),
            "depth": render[0, ..., 3:4],
            "alpha": alpha[0],
            "info": info,
            "ms_level": level,
        }


def create_renderer(spec):
    return Renderer(spec)
