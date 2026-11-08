"""3DGabSplat renderer: gsplat 1.5.3 classic pipeline with the rasterize_to_pixels stage
replaced by the 2D Gabor CUDA rasterizer (csrc/gabor_raster.cu).

Projection (EWA, eps2d 0.3, near 0.01, far 1e10), SH evaluation, tile intersection and the
RGB+ED depth channel follow `gsplat.rasterization`. The Gaussian 3.33-sigma tile boxes are kept:
the Gabor modulation is <= 1, so the Gabor support lies inside the Gaussian one. 3D frequencies
are projected per view to 2D (frequency.py, paper Eq. 9-10); weights are sigmoid-activated
(Sec. 3.4.1).
"""

import math

import torch
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    spherical_harmonics,
)

from .frequency import project_frequencies
from .kernels import rasterize_gabor

CAPABILITIES = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed")
OPTIONS = {"num_freqs": 2, "tile_size": 16, "eps2d": 0.3}
FREQ, WEIGHT = "gabor_freq", "gabor_weight"


def project(params, camera, sh_degree, tile_size=16, eps2d=0.3):
    """gsplat.rasterization (classic, non-packed, one camera) up to rasterize_to_pixels, plus
    the per-view 2D Gabor parameters gabs [N,3F] = (f2d_x, f2d_y, w)."""
    means = params["means"]
    viewmats, Ks = camera["viewmat"][None], camera["K"][None]
    W, H = int(camera["width"]), int(camera["height"])
    opacities = params["opacities"].sigmoid()[None]  # [1,N]
    scales = params["scales"].exp()
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means, None, params["quats"], scales, viewmats, Ks, W, H,
        eps2d=eps2d, near_plane=0.01, far_plane=1e10, radius_clip=0.0, packed=False,
        calc_compensations=False)
    visible = (radii[0] > 0).all(dim=-1)
    f2d = project_frequencies(means, params["quats"], scales, params[FREQ], camera["viewmat"],
                              camera["K"], W, H, conics[0], visible)  # [N,F,2]
    w = params[WEIGHT].sigmoid()  # [N,F]
    gabs = torch.cat((f2d, w[..., None]), dim=-1).reshape(len(means), -1)  # [N,3F]
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
        "opacities": opacities, "gabs": gabs, "tile_width": tw, "tile_height": th,
        "tiles_per_gauss": tiles_per_gauss, "isect_ids": isect_ids, "flatten_ids": flatten_ids,
        "isect_offsets": offsets, "width": W, "height": H, "tile_size": tile_size,
        "n_batches": 1, "n_cameras": 1, "gaussian_ids": None, "camera_ids": None,
    }
    return colors, meta


def render_gabor(params, camera, sh_degree=3, tile_size=16, eps2d=0.3, with_depth=True):
    """Raw (no background) colours [H,W,3], expected depth [H,W,1] or None, alpha [H,W,1], meta."""
    colors, meta = project(params, camera, sh_degree, tile_size, eps2d)
    feats = colors[0]
    if with_depth:
        feats = torch.cat((feats, meta["depths"][0][..., None]), dim=-1)
    render, alpha = rasterize_gabor(
        meta["means2d"][0], meta["conics"][0], feats, meta["opacities"][0], meta["gabs"],
        meta["isect_offsets"][0], meta["flatten_ids"], meta["width"], meta["height"], tile_size)
    depth = render[..., 3:4] / alpha.clamp(min=1e-10) if with_depth else None
    return render[..., :3], depth, alpha, meta


class GaborRenderer:
    def __init__(self, spec):
        cfg = {**OPTIONS, **spec.config}
        unknown = set(cfg) - set(OPTIONS)
        if unknown:
            raise ValueError(f"Unknown Gabor renderer options: {sorted(unknown)}")
        self.num_freqs = int(cfg["num_freqs"])
        if self.num_freqs not in (1, 2):
            raise ValueError("num_freqs must be 1 or 2 (CUDA kernel instantiations)")
        self.tile_size = int(cfg["tile_size"])
        self.eps2d = float(cfg["eps2d"])
        self.capabilities = set(CAPABILITIES)

    def render(self, params, camera, sh_degree=3):
        if params[FREQ].shape[1:] != (self.num_freqs, 3) or \
                params[WEIGHT].shape[1:] != (self.num_freqs,):
            raise ValueError("gabor_freq/gabor_weight shapes do not match num_freqs")
        rgb, depth, alpha, info = render_gabor(params, camera, sh_degree, self.tile_size,
                                               self.eps2d)
        # dataset convention: RGB over a white background
        return {"rgb": rgb + (1 - alpha), "depth": depth, "alpha": alpha, "info": info}


def create_renderer(spec):
    return GaborRenderer(spec)
