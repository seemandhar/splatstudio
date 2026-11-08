"""SSS renderer: Inria-faithful EWA projection (projection.py), gsplat 1.5.3 tiling and SH, and the
Student's t / signed-opacity rasterizer (csrc/t_raster.cu, ported from the official
diff-t-rasterization).

Follows the official `t_renderer.render` (black background, scale modifier 1):
  o     = tanh(opacities)                      (signed opacity, Sec. 3.3; nt_model)
  nu    = hardtanh(degrees, 1, 10000)          (nt_model.nu_degree_activation)
  c     = clamp_min(SH(mu - campos) + 0.5, 0)  (computeColorFromSH)
  alpha = clamp(o (1 + base/nu)^(-(nu+2)/2), -0.99, 0.99)
"""

import math

import torch
import torch.nn.functional as F
from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles, spherical_harmonics

from .kernels import rasterize_t
from .projection import project

CAPABILITIES = ("render.rgb",)
OPTIONS = {"background": 0.0, "tile_size": 16, "near_plane": 0.2, "fov_clamp": 1.3,
           "alpha_clip": 0.99, "exact_nu_grad": False, "nu_min": 1.0, "nu_max": 10000.0}


def signed_opacity(raw):
    return torch.tanh(raw)


def nu_activation(raw, nu_min=1.0, nu_max=10000.0):
    return F.hardtanh(raw, nu_min, nu_max)


def colors_sh(params, camera, sh_degree, visible):
    coeffs = torch.cat((params["sh0"], params["shN"]), dim=1)  # [N,16,3]
    campos = torch.inverse(camera["viewmat"])[:3, 3]
    dirs = params["means"] - campos
    sh = spherical_harmonics(sh_degree, dirs, coeffs, masks=visible)
    return torch.clamp_min(sh + 0.5, 0.0)


def tile_lists(proj, width, height, tile_size):
    tw, th = math.ceil(width / tile_size), math.ceil(height / tile_size)
    _, isect_ids, flatten_ids = isect_tiles(
        proj["means2d"][None].detach(), proj["radii"][None], proj["depths"][None].detach(),
        tile_size, tw, th, packed=False, n_images=1)
    offsets = isect_offset_encode(isect_ids, 1, tw, th).reshape(th, tw)
    return offsets, flatten_ids


def render_t(params, camera, sh_degree, cfg):
    W, H = int(camera["width"]), int(camera["height"])
    ts = int(cfg["tile_size"])
    nus = nu_activation(params["degrees"], cfg["nu_min"], cfg["nu_max"])
    proj = project(params["means"], params["quats"], params["scales"], nus, camera["viewmat"],
                   camera["K"], W, H, ts, cfg["near_plane"], cfg["fov_clamp"])
    offsets, flatten_ids = tile_lists(proj, W, H, ts)
    colors = colors_sh(params, camera, sh_degree, proj["valid"])
    opac = signed_opacity(params["opacities"])
    bg = None
    if cfg["background"] != 0.0:
        bg = torch.full((3,), float(cfg["background"]), device=colors.device)
    rgb, alpha = rasterize_t(proj["means2d"], proj["conics"], colors, opac, nus, offsets,
                             flatten_ids, W, H, ts, bg, cfg["alpha_clip"], cfg["exact_nu_grad"])
    info = {"radii": proj["radii"], "means2d": proj["means2d"], "depths": proj["depths"],
            "visible": proj["valid"], "n_isects": int(flatten_ids.numel())}
    return rgb, alpha, info


class TRenderer:
    def __init__(self, spec):
        cfg = {**OPTIONS, **spec.config}
        unknown = set(cfg) - set(OPTIONS)
        if unknown:
            raise ValueError(f"Unknown t-renderer options: {sorted(unknown)}")
        self.cfg = cfg
        self.capabilities = set(CAPABILITIES)

    def render(self, params, camera, sh_degree=3):
        rgb, alpha, info = render_t(params, camera, sh_degree, self.cfg)
        return {"rgb": rgb, "alpha": alpha, "info": info}


def create_renderer(spec):
    return TRenderer(spec)
