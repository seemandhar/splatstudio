"""Upscaling renderer: render at W//s (with analytic image derivatives) and upscale to W.

Training with upscaling (paper Sec. 5 "Scene Reconstruction"): the rasterizer runs at the low
resolution, the image is upscaled to the ground-truth resolution and the loss is evaluated there;
gradients flow back through the upscaler and the derivative rendering.

upscaler = "spline":  jet rasterizer (csrc/jet_raster.cu) + gradient-aware spline ("Spline").
upscaler = "bicubic": stock gsplat rasterization + PyTorch bicubic ("Bicubic" rows; the paper
                      uses PyTorch's differentiable bicubic).
Projection, SH evaluation and tile intersection are gsplat 1.5.3's (classic mode, eps2d 0.3,
near 0.01, far 1e10, tile 16) exactly as `gsplat.rasterization` calls them.
"""

import math

import torch
from gsplat import rasterization
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    spherical_harmonics,
)

from .kernels import rasterize_jets
from .upscale import bicubic_upscale, bilinear_upscale, spline_upscale

TARGET_CAPABILITY = "training.target_image"  # consumed by the Taming density (plan #30)
CAPABILITIES = ("render.rgb", "gradient.screen.signed", TARGET_CAPABILITY)
UPSCALERS = ("spline", "bicubic")


def low_res_camera(camera, factor):
    """Camera of the low-res rendering: size floor(W/s) (brush render.rs, runtime --downscale),
    intrinsics scaled per axis so low-res pixel centres sit at (X+0.5) w/W."""
    W, H = int(camera["width"]), int(camera["height"])
    w, h = max(2, math.floor(W / factor)), max(2, math.floor(H / factor))
    K = camera["K"].clone()
    K[0] = K[0] * (w / W)
    K[1] = K[1] * (h / H)
    return {"viewmat": camera["viewmat"], "K": K, "width": w, "height": h}


def project(params, camera, sh_degree, tile_size=16):
    """gsplat.rasterization's classic pipeline up to (excluding) rasterize_to_pixels."""
    means = params["means"]
    viewmats, Ks = camera["viewmat"][None], camera["K"][None]
    W, H = int(camera["width"]), int(camera["height"])
    opacities = params["opacities"].sigmoid()
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means, None, params["quats"], params["scales"].exp(), viewmats, Ks, W, H,
        eps2d=0.3, near_plane=0.01, far_plane=1e10, radius_clip=0.0, packed=False,
        calc_compensations=False, opacities=opacities)
    shs = torch.cat((params["sh0"], params["shN"]), dim=1)
    campos = torch.inverse(viewmats)[..., :3, 3]
    dirs = means[None] - campos[:, None]
    colors = spherical_harmonics(sh_degree, dirs, shs[None], masks=(radii > 0).all(dim=-1))
    colors = torch.clamp_min(colors + 0.5, 0.0)
    tw, th = math.ceil(W / tile_size), math.ceil(H / tile_size)
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tw, th, packed=False, n_images=1)
    offsets = isect_offset_encode(isect_ids, 1, tw, th).reshape(1, th, tw)
    meta = {
        "radii": radii, "means2d": means2d, "depths": depths, "conics": conics,
        "opacities": opacities[None], "tile_width": tw, "tile_height": th,
        "tiles_per_gauss": tiles_per_gauss, "isect_ids": isect_ids, "flatten_ids": flatten_ids,
        "isect_offsets": offsets, "width": W, "height": H, "tile_size": tile_size,
        "n_batches": 1, "n_cameras": 1,
    }
    return colors[0], meta


def render_jets(params, camera, sh_degree, bg=None, tile_size=16, offset=(0.0, 0.0)):
    """Image jets [4,h,w,3] (I, I_x, I_y, I_xy) over a white background, alpha [h,w], meta."""
    colors, meta = project(params, camera, sh_degree, tile_size)
    if bg is None:
        bg = torch.ones(3, device=colors.device)
    jets, alpha, _, _ = rasterize_jets(
        meta["means2d"][0], meta["conics"][0], colors, meta["opacities"][0], bg,
        meta["isect_offsets"][0], meta["flatten_ids"], meta["width"], meta["height"], tile_size,
        offset)
    return jets, alpha, meta


def render_gsplat(params, camera, sh_degree):
    """Stock gsplat rendering (as splatstudio.backends.gsplat.Renderer), white background."""
    render, alpha, info = rasterization(
        means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(),
        colors=torch.cat((params["sh0"], params["shN"]), dim=1),
        viewmats=camera["viewmat"][None], Ks=camera["K"][None], width=camera["width"],
        height=camera["height"], sh_degree=sh_degree, packed=False, rasterize_mode="classic")
    return render[0] + (1 - alpha[0]), alpha[0], info


class UpscaleRenderer:
    def __init__(self, spec):
        cfg = {"factor": 4.0, "upscaler": "spline", "tile_size": 16, **spec.config}
        unknown = set(cfg) - {"factor", "upscaler", "tile_size"}
        if unknown:
            raise ValueError(f"Unknown upscale renderer options: {sorted(unknown)}")
        if cfg["upscaler"] not in UPSCALERS:
            raise ValueError(f"upscaler must be one of {UPSCALERS}")
        if float(cfg["factor"]) < 1.0:
            raise ValueError("factor must be >= 1")
        self.factor = float(cfg["factor"])
        self.upscaler = cfg["upscaler"]
        self.tile_size = int(cfg["tile_size"])
        self.capabilities = set(CAPABILITIES)

    def render(self, params, camera, sh_degree=3):
        H, W = int(camera["height"]), int(camera["width"])
        low = low_res_camera(camera, self.factor)
        if self.upscaler == "spline":
            jets, alpha, info = render_jets(params, low, sh_degree, tile_size=self.tile_size)
            rgb = spline_upscale(jets[0], jets[1], jets[2], jets[3], H, W)
        else:
            rgb_low, alpha, info = render_gsplat(params, low, sh_degree)
            rgb = bicubic_upscale(rgb_low, H, W)
            alpha = alpha[..., 0]
        outputs = {
            "rgb": rgb,
            # coverage at target size, for inspection only (no loss uses it)
            "alpha": bilinear_upscale(alpha[..., None], H, W),
            "info": info,
            "low_res": (low["width"], low["height"]),
            # for the Taming density's score cache (full-resolution view and target)
            "target_image": camera.get("image"),
            "camera": {k: camera[k] for k in ("K", "viewmat", "width", "height")},
        }
        return outputs


def create_renderer(spec):
    return UpscaleRenderer(spec)
