"""3DLS renderer: gsplat 1.5.3 classic pipeline with the rasterize_to_pixels stage replaced by
the linear-kernel CUDA rasterizer (csrc/linear_raster.cu).

Projection (EWA, eps2d 0.3, near 0.01, far 1e10), SH evaluation, tile intersection and the
RGB+ED depth channel follow `gsplat.rasterization` exactly. The only other change is the
bounding radius: the linear kernel's support is D < lambda and alpha >= 1/255 requires
D <= lambda (1 - (1/255)/o), so the per-axis box half-size is ceil(extend * sigma_axis) with that
extend (gsplat uses min(3.33, sqrt(2 ln(255 o))) for the Gaussian).
"""

import math

import torch
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    spherical_harmonics,
)

from .kernels import rasterize_linear

ALPHA_THRESHOLD = 1.0 / 255.0
CAPABILITIES = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed")
OPTIONS = {"lam": 2.5, "ags": True, "ags_coef": 1.0, "tile_size": 16, "eps2d": 0.3}


def linear_radii(means2d, conics, opacities, radii, width, height, lam):
    """Per-axis bounding radii [...,N,2] (int32) of the linear kernel's alpha >= 1/255 support.

    Starts from gsplat's 3.33-sigma visibility (radii > 0), which contains the linear support
    (lam <= 3.33 is checked by the renderer); culls o < 1/255 and boxes outside the image as
    gsplat's projection does.
    """
    with torch.no_grad():
        a, b, c = conics[..., 0], conics[..., 1], conics[..., 2]
        det = (a * c - b * b).clamp_min(1e-12)
        var_x, var_y = c / det, a / det  # diagonal of Sigma_2D = conic^-1
        extend = lam * (1.0 - ALPHA_THRESHOLD / opacities.clamp_min(1e-12))
        rx = torch.ceil(extend * var_x.clamp_min(0).sqrt())
        ry = torch.ceil(extend * var_y.clamp_min(0).sqrt())
        mx, my = means2d[..., 0], means2d[..., 1]
        ok = (radii > 0).all(dim=-1) & (opacities >= ALPHA_THRESHOLD)
        ok &= ~((mx + rx <= 0) | (mx - rx >= width) | (my + ry <= 0) | (my - ry >= height))
        out = torch.stack((rx, ry), dim=-1).to(torch.int32)
        return torch.where(ok[..., None], out, torch.zeros_like(out))


def project(params, camera, sh_degree, lam, tile_size=16, eps2d=0.3):
    """gsplat.rasterization (classic, non-packed, one camera) up to rasterize_to_pixels."""
    means = params["means"]
    viewmats, Ks = camera["viewmat"][None], camera["K"][None]
    W, H = int(camera["width"]), int(camera["height"])
    opacities = params["opacities"].sigmoid()[None]  # [1,N]
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means, None, params["quats"], params["scales"].exp(), viewmats, Ks, W, H,
        eps2d=eps2d, near_plane=0.01, far_plane=1e10, radius_clip=0.0, packed=False,
        calc_compensations=False)
    radii = linear_radii(means2d, conics, opacities, radii, W, H, lam)
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


def render_linear(params, camera, sh_degree=3, lam=2.5, ags=1.0, tile_size=16, eps2d=0.3,
                  with_depth=True):
    """Raw (no background) colours [H,W,3], expected depth [H,W,1] or None, alpha [H,W,1], meta."""
    colors, meta = project(params, camera, sh_degree, lam, tile_size, eps2d)
    feats = colors[0]
    if with_depth:
        feats = torch.cat((feats, meta["depths"][0][..., None]), dim=-1)
    render, alpha = rasterize_linear(
        meta["means2d"][0], meta["conics"][0], feats, meta["opacities"][0],
        meta["isect_offsets"][0], meta["flatten_ids"], meta["width"], meta["height"],
        tile_size, lam, ags)
    depth = render[..., 3:4] / alpha.clamp(min=1e-10) if with_depth else None
    return render[..., :3], depth, alpha, meta


class LinearRenderer:
    def __init__(self, spec):
        cfg = {**OPTIONS, **spec.config}
        unknown = set(cfg) - set(OPTIONS)
        if unknown:
            raise ValueError(f"Unknown linear renderer options: {sorted(unknown)}")
        self.lam = float(cfg["lam"])
        if not 0 < self.lam <= 3.33:
            # gsplat's projection culls beyond 3.33 sigma; a wider support would be clipped.
            raise ValueError("lam must be in (0, 3.33]")
        # AGS coefficient c of omega = exp(-c D^2); paper Eq. 13 has c = 1
        self.ags = float(cfg["ags_coef"]) if bool(cfg["ags"]) else 0.0
        if self.ags < 0:
            raise ValueError("ags_coef must be non-negative")
        self.tile_size = int(cfg["tile_size"])
        self.eps2d = float(cfg["eps2d"])
        self.capabilities = set(CAPABILITIES)

    def render(self, params, camera, sh_degree=3):
        rgb, depth, alpha, info = render_linear(
            params, camera, sh_degree, self.lam, self.ags, self.tile_size, self.eps2d)
        # dataset convention: RGB over a white background
        return {"rgb": rgb + (1 - alpha), "depth": depth, "alpha": alpha, "info": info}


def create_renderer(spec):
    return LinearRenderer(spec)
