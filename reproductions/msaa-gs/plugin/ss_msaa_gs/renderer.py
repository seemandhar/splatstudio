"""MSAA-GS renderer: gsplat 1.5.3 classic pipeline (`gsplat.rasterization`, non-packed, one
camera, RGB+ED) with the `rasterize_to_pixels` stage replaced by the multi-sample CUDA
rasterizer (csrc/msaa_raster.cu). Projection (EWA, eps2d 0.3, near 0.01, far 1e10,
opacity-aware radii), SH evaluation, tile intersection and the expected-depth channel follow
`gsplat.rasterization` exactly. Used for training and evaluation (paper Eq. 14)."""

import math

import torch
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    spherical_harmonics,
)

from .kernels import PIXEL_CENTRE, ROTATED_GRID_4X, rasterize_msaa

CAPABILITIES = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed")
PATTERNS = {"rotated-grid-4x": ROTATED_GRID_4X, "centre-1x": PIXEL_CENTRE}
OPTIONS = {"pattern": "rotated-grid-4x", "tile_size": 16, "eps2d": 0.3}


def project(params, camera, sh_degree, tile_size=16, eps2d=0.3):
    """gsplat.rasterization (classic, non-packed, one camera) up to rasterize_to_pixels."""
    means = params["means"]
    viewmats, Ks = camera["viewmat"][None], camera["K"][None]
    W, H = int(camera["width"]), int(camera["height"])
    opacities = params["opacities"].sigmoid()
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means, None, params["quats"], params["scales"].exp(), viewmats, Ks, W, H,
        eps2d=eps2d, near_plane=0.01, far_plane=1e10, radius_clip=0.0, packed=False,
        calc_compensations=False, opacities=opacities)
    opacities = opacities[None]  # [1,N]
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
        "opacities": opacities, "tile_width": tw, "tile_height": th,
        "tiles_per_gauss": tiles_per_gauss, "isect_ids": isect_ids, "flatten_ids": flatten_ids,
        "isect_offsets": offsets, "width": W, "height": H, "tile_size": tile_size,
        "n_batches": 1, "n_cameras": 1, "gaussian_ids": None, "camera_ids": None,
    }
    return colors, meta


def render_msaa(params, camera, sh_degree=3, offsets=ROTATED_GRID_4X, tile_size=16, eps2d=0.3):
    """Raw (no background) colours [H,W,3], expected depth [H,W,1], alpha [H,W,1], meta."""
    colors, meta = project(params, camera, sh_degree, tile_size, eps2d)
    feats = torch.cat((colors[0], meta["depths"][0][..., None]), dim=-1)
    render, alpha = rasterize_msaa(
        meta["means2d"][0], meta["conics"][0], feats, meta["opacities"][0],
        meta["isect_offsets"][0], meta["flatten_ids"], meta["width"], meta["height"],
        offsets, tile_size)
    depth = render[..., 3:4] / alpha.clamp(min=1e-10)
    return render[..., :3], depth, alpha, meta


class MSAARenderer:
    def __init__(self, spec):
        cfg = {**OPTIONS, **spec.config}
        unknown = set(cfg) - set(OPTIONS)
        if unknown:
            raise ValueError(f"Unknown MSAA renderer options: {sorted(unknown)}")
        if cfg["pattern"] not in PATTERNS:
            raise ValueError(f"pattern must be one of {sorted(PATTERNS)}")
        self.offsets = PATTERNS[cfg["pattern"]]
        self.tile_size = int(cfg["tile_size"])
        self.eps2d = float(cfg["eps2d"])
        self.capabilities = set(CAPABILITIES)

    def render(self, params, camera, sh_degree=3):
        rgb, depth, alpha, info = render_msaa(
            params, camera, sh_degree, self.offsets, self.tile_size, self.eps2d)
        # white background composited with the sample-averaged alpha == mean of the S
        # per-sample composites over white (Eq. 5 is linear)
        return {"rgb": rgb + (1 - alpha), "depth": depth, "alpha": alpha, "info": info}


def create_renderer(spec):
    return MSAARenderer(spec)
