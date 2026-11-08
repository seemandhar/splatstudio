"""Optimal Gaussian Splatting renderer: tangent-plane projection (projection.py) + unit-sphere
rasterizer (csrc/sphere_raster.cu). SH evaluation and tile intersection are gsplat 1.5.3's.

`projection="ewa"` is the control: identical pipeline and Inria blending rules with the
standard EWA covariance and planar pixel-space evaluation.
"""

import math

import torch
from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles, spherical_harmonics

from .kernels import rasterize_sphere
from .projection import project_gaussians

CAPABILITIES = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed")
OPTIONS = {
    "projection": "optimal",  # "optimal" (paper) | "ewa" (control)
    "focal_floor": 512.0,     # f' = max(512, f), official forward.cu
    "eps2d": 0.3,             # low-pass, official forward.cu
    "near": 0.2,              # official auxiliary.h in_frustum
    "alpha_max": 0.99,        # Inria alpha clamp
    "official_cov_grad": True,  # official closed-form dJ/dt (incl. j12_x typo)
    "tile_size": 16,
}


def project(params, camera, sh_degree, cfg):
    W, H = int(camera["width"]), int(camera["height"])
    viewmat, K = camera["viewmat"], camera["K"]
    proj = project_gaussians(
        params["means"], params["quats"], params["scales"], viewmat, K, W, H,
        mode=cfg["projection"], focal_floor=float(cfg["focal_floor"]),
        eps2d=float(cfg["eps2d"]), near=float(cfg["near"]),
        official_cov_grad=bool(cfg["official_cov_grad"]))
    tile_size = int(cfg["tile_size"])
    tw, th = math.ceil(W / tile_size), math.ceil(H / tile_size)
    means2d = proj["means2d"][None]  # [1,N,2]: the density controller reads this gradient
    radii = proj["radii"][None]
    depths = proj["depths"][None]
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tw, th, packed=False, n_images=1)
    offsets = isect_offset_encode(isect_ids, 1, tw, th).reshape(1, th, tw)
    # forward.cu: a Gaussian whose rectangle covers no tile is not visible (radius 0)
    radii = torch.where((tiles_per_gauss > 0)[..., None], radii, torch.zeros_like(radii))
    visible = (radii > 0).all(-1)
    shs = torch.cat((params["sh0"], params["shN"]), dim=1)
    campos = torch.inverse(viewmat)[:3, 3]
    dirs = params["means"][None] - campos
    colors = spherical_harmonics(sh_degree, dirs, shs[None], masks=visible)
    colors = torch.clamp_min(colors + 0.5, 0.0)
    opacities = params["opacities"].sigmoid()
    meta = {
        "radii": radii, "means2d": means2d, "depths": depths, "conics": proj["conics"],
        "opacities": opacities, "tile_width": tw, "tile_height": th,
        "tiles_per_gauss": tiles_per_gauss, "isect_ids": isect_ids, "flatten_ids": flatten_ids,
        "isect_offsets": offsets, "width": W, "height": H, "tile_size": tile_size,
        "n_batches": 1, "n_cameras": 1, "gaussian_ids": None, "camera_ids": None,
    }
    return colors[0], meta


def render_optimal(params, camera, sh_degree=3, cfg=None, with_depth=True):
    """Raw colours [H,W,3] (no background), expected depth [H,W,1] or None, alpha, meta."""
    cfg = {**OPTIONS, **(cfg or {})}
    colors, meta = project(params, camera, sh_degree, cfg)
    feats = colors
    if with_depth:
        feats = torch.cat((feats, meta["depths"][0][..., None]), dim=-1)
    render, alpha = rasterize_sphere(
        meta["means2d"][0], meta["conics"], feats, meta["opacities"], meta["isect_offsets"][0],
        meta["flatten_ids"], meta["width"], meta["height"], camera["K"],
        focal_floor=float(cfg["focal_floor"]), mode=cfg["projection"],
        tile_size=meta["tile_size"], alpha_max=float(cfg["alpha_max"]))
    depth = render[..., 3:4] / alpha.clamp(min=1e-10) if with_depth else None
    return render[..., :3], depth, alpha, meta


class OptimalRenderer:
    def __init__(self, spec):
        cfg = {**OPTIONS, **spec.config}
        unknown = set(cfg) - set(OPTIONS)
        if unknown:
            raise ValueError(f"Unknown optimal-gs renderer options: {sorted(unknown)}")
        if cfg["projection"] not in ("optimal", "ewa"):
            raise ValueError("projection must be 'optimal' or 'ewa'")
        if not 0 < float(cfg["alpha_max"]) < 1:
            raise ValueError("alpha_max must be in (0, 1)")
        self.cfg = cfg
        self.capabilities = set(CAPABILITIES)

    def render(self, params, camera, sh_degree=3):
        rgb, depth, alpha, info = render_optimal(params, camera, sh_degree, self.cfg)
        # dataset convention: RGB over a white background
        return {"rgb": rgb + (1 - alpha), "depth": depth, "alpha": alpha, "info": info}


def create_renderer(spec):
    return OptimalRenderer(spec)
