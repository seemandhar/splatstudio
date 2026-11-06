"""BAGS renderer: Mip-Splatting gsplat render of the sharp image C and depth D.

* training camera (grad enabled and viewmat matches a live-bank training view): rendered at the
  coarse-to-fine resolution of the current step (1/4 before 3000, 1/2 before 6000, full after;
  official train.py upsample_iter and getTrainCameras(scale)); K, width and height are scaled
  like the official FoV cameras (fx * W_s / W, cx = W_s / 2 here because cx = W / 2).
* any other camera (test / validation views, ss-eval, no bank): full-resolution sharp render.
  The BPN is never applied here; the losses apply it to training views only, so every
  evaluation render is the sharp, multi-view-consistent C (paper: "we render only the clear C").

Mip-Splatting: 3D filter on scales / opacities, 2D Mip filter = gsplat antialiased mode with
eps2d = kernel_size 0.1. Background black (official white_background False).
D = alpha-blended Euclidean distance |mu - campos| (official rasterizer ``depths = dist``,
backward through the distance), rendered as a 4th colour channel next to the SH colour
(evaluated with gsplat's spherical_harmonics, +0.5, clamp >= 0, like gsplat / Inria).
"""

import torch
from gsplat import rasterization
from gsplat.cuda._wrapper import spherical_harmonics

from . import bank as bank_mod
from .filters import filtered_scales_opacities
from .schedule import resolution_divisor

CAPABILITIES = {
    "render.rgb", "render.depth.ray_distance", "render.multiscale", "render.blur_view",
    "render.antialiased_2d", "render.filter_3d", "gradient.screen.signed",
}
DEFAULTS = {"kernel_size": 0.1, "background": [0.0, 0.0, 0.0], "upsample_steps": [3000, 6000]}


def scaled_camera(camera, divisor):
    """Camera of the training image downscaled by ``divisor`` (official loadCam: round(W/d))."""
    if divisor == 1:
        return camera
    W, H = int(camera["width"]), int(camera["height"])
    Ws, Hs = int(round(W / divisor)), int(round(H / divisor))
    K = camera["K"].clone()
    K[0, :] = K[0, :] * (Ws / W)
    K[1, :] = K[1, :] * (Hs / H)
    out = dict(camera)
    out.update(K=K, width=Ws, height=Hs)
    return out


def render_sharp(params, camera, sh_degree, background, kernel_size=0.1):
    scales, opac = filtered_scales_opacities(params["scales"], params["opacities"],
                                             params["filter_3d"])
    means = params["means"]
    campos = torch.linalg.inv(camera["viewmat"])[:3, 3]
    delta = means - campos
    sh = torch.cat((params["sh0"], params["shN"]), dim=1)
    k = (sh_degree + 1) ** 2
    rgb = torch.clamp_min(spherical_harmonics(sh_degree, delta, sh[:, :k]) + 0.5, 0.0)
    dist = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    colors = torch.cat([rgb, dist], dim=-1)
    bg = torch.cat([background, background.new_zeros(1)])[None]
    render, alpha, info = rasterization(
        means=means, quats=params["quats"], scales=scales, opacities=opac, colors=colors,
        viewmats=camera["viewmat"][None], Ks=camera["K"][None], width=int(camera["width"]),
        height=int(camera["height"]), sh_degree=None, packed=False, render_mode="RGB",
        eps2d=kernel_size, rasterize_mode="antialiased", backgrounds=bg)
    return {"rgb": render[0, ..., :3], "depth": render[0, ..., 3:4], "alpha": alpha[0],
            "info": info}


class Renderer:
    capabilities = CAPABILITIES

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown BAGS renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        self.kernel_size = float(cfg["kernel_size"])
        self.background = [float(v) for v in cfg["background"]]
        self.upsample_steps = tuple(int(s) for s in cfg["upsample_steps"])
        self.capabilities = set(CAPABILITIES)
        self._bg = {}

    def _background(self, dev):
        if dev not in self._bg:
            self._bg[dev] = torch.tensor(self.background, dtype=torch.float32, device=dev)
        return self._bg[dev]

    def render(self, params, camera, sh_degree=3):
        dev = params["means"].device
        bank = bank_mod.ACTIVE
        view = None
        if bank is not None and torch.is_grad_enabled():
            view = bank.view_index(camera["viewmat"])
        divisor = 1
        if view is not None:
            if tuple(bank.cfg["upsample_steps"]) != self.upsample_steps:
                raise RuntimeError("renderer and density upsample_steps differ")
            divisor = resolution_divisor(bank.step + 1, self.upsample_steps)
        cam = scaled_camera(camera, divisor)
        out = render_sharp(params, cam, sh_degree, self._background(dev), self.kernel_size)
        out["bags"] = {"view": view, "bank": bank if view is not None else None,
                       "divisor": divisor}
        return out


def create_renderer(spec):
    return Renderer(spec)
