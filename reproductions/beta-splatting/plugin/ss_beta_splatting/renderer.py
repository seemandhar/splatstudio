"""DBS renderer: gsplat 1.5.3 projection + tiling, Spherical Beta colour and the Beta-kernel
rasterizer (csrc/beta_raster.cu, ported from the official fork).

Follows the official `BetaModel.render` -> fork `gsplat.rendering.rasterization` (sh_degree 0,
sb_number 2, packed False, eps2d 0.3, near 0.01, tile 16, black background):
  c0     = clamp_min(SH_0(sh0) + 0.5, 0)
  colour = clamp_min(SB(c0, softplus_{10 ln2}(lobe rgb), theta, phi, b_m; mu - campos), 0)
  alpha  = min(0.999, o (1 - r^2)^{4 e^b}),  r^2 < 1
The bounding box is the exact 1-sigma support: half-sizes ceil(sqrt(Sigma2D_xx)),
ceil(sqrt(Sigma2D_yy)) (the fork uses the enclosing circle ceil(sqrt(lambda_max)); both cover the
same pixels since the kernel is zero for r >= 1).
"""

import math

import torch
import torch.nn.functional as F
from gsplat.cuda._wrapper import fully_fused_projection, isect_offset_encode, isect_tiles

from .kernels import rasterize_beta, spherical_beta

SH_C0 = 0.28209479177387814
SOFTPLUS_BETA = math.log(2) * 10  # official sb_params_activation
CAPABILITIES = ("render.rgb",)
OPTIONS = {"sb_number": 2, "background": 0.0, "tile_size": 16, "eps2d": 0.3, "near_plane": 0.01}


def beta_activation(raw):
    """beta(b) = 4 e^b (paper Eq. beta)."""
    return 4.0 * torch.exp(raw)


def sb_activation(raw):
    """[..., 6] = (r, g, b, theta, phi, b_m): softplus(beta = 10 ln 2) on the lobe colour."""
    return torch.cat((F.softplus(raw[..., :3], beta=SOFTPLUS_BETA), raw[..., 3:]), dim=-1)


def beta_radii(means2d, conics, radii, width, height):
    """Per-axis int32 radii [...,N,2] of the r^2 < 1 support, culled like the fork's projection."""
    with torch.no_grad():
        a, b, c = conics[..., 0], conics[..., 1], conics[..., 2]
        det = (a * c - b * b).clamp_min(1e-20)
        rx = torch.ceil((c / det).clamp_min(0).sqrt())  # sqrt(Sigma_xx)
        ry = torch.ceil((a / det).clamp_min(0).sqrt())  # sqrt(Sigma_yy)
        mx, my = means2d[..., 0], means2d[..., 1]
        ok = (radii > 0).all(dim=-1)  # gsplat: det > 0, near/far, 3-sigma box on screen
        ok &= ~((mx + rx <= 0) | (mx - rx >= width) | (my + ry <= 0) | (my - ry >= height))
        out = torch.stack((rx, ry), dim=-1).to(torch.int32)
        return torch.where(ok[..., None], out, torch.zeros_like(out))


def project(params, camera, tile_size=16, eps2d=0.3, near_plane=0.01):
    means = params["means"]
    viewmats, Ks = camera["viewmat"][None], camera["K"][None]
    W, H = int(camera["width"]), int(camera["height"])
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means, None, params["quats"], params["scales"].exp(), viewmats, Ks, W, H, eps2d=eps2d,
        near_plane=near_plane, far_plane=1e10, radius_clip=0.0, packed=False,
        calc_compensations=False)
    radii = beta_radii(means2d, conics, radii, W, H)
    tw, th = math.ceil(W / tile_size), math.ceil(H / tile_size)
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tw, th, packed=False, n_images=1)
    offsets = isect_offset_encode(isect_ids, 1, tw, th).reshape(1, th, tw)
    return {
        "radii": radii, "means2d": means2d, "depths": depths, "conics": conics,
        "tile_width": tw, "tile_height": th, "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids, "flatten_ids": flatten_ids, "isect_offsets": offsets,
        "width": W, "height": H, "tile_size": tile_size, "n_cameras": 1,
    }


def colors_sb(params, camera, visible):
    """Per-Gaussian RGB [N,3] for one camera (official rendering.py lines 380-414)."""
    c0 = torch.clamp_min(SH_C0 * params["sh0"][:, 0, :] + 0.5, 0.0)
    campos = torch.inverse(camera["viewmat"])[:3, 3]
    dirs = (params["means"] - campos).detach()
    sb = sb_activation(params["sb_params"])
    return torch.clamp_min(spherical_beta(dirs, c0, sb, visible), 0.0)


def render_beta(params, camera, tile_size=16, eps2d=0.3, near_plane=0.01, background=0.0):
    """Returns rgb [H,W,3] (over `background`), alpha [H,W,1], meta."""
    meta = project(params, camera, tile_size, eps2d, near_plane)
    visible = (meta["radii"][0] > 0).all(dim=-1)
    colors = colors_sb(params, camera, visible)
    opacities = params["opacities"].sigmoid()
    betas = beta_activation(params["betas"])
    bg = None
    if background != 0.0:
        bg = torch.full((3,), float(background), device=colors.device)
    rgb, alpha = rasterize_beta(
        meta["means2d"][0], meta["conics"][0], colors, opacities, betas,
        meta["isect_offsets"][0], meta["flatten_ids"], meta["width"], meta["height"], tile_size,
        bg)
    meta["opacities"] = opacities[None]
    return rgb, alpha, meta


class BetaRenderer:
    def __init__(self, spec):
        cfg = {**OPTIONS, **spec.config}
        unknown = set(cfg) - set(OPTIONS)
        if unknown:
            raise ValueError(f"Unknown beta renderer options: {sorted(unknown)}")
        self.sb_number = int(cfg["sb_number"])
        self.background = float(cfg["background"])
        self.tile_size = int(cfg["tile_size"])
        self.eps2d = float(cfg["eps2d"])
        self.near_plane = float(cfg["near_plane"])
        self.capabilities = set(CAPABILITIES)

    def render(self, params, camera, sh_degree=0):
        # sh_degree is ignored: DBS uses SH degree 0 (base colour) + Spherical Beta lobes.
        if params["sb_params"].shape[1] != self.sb_number:
            raise ValueError("sb_params field does not match sb_number")
        rgb, alpha, info = render_beta(params, camera, self.tile_size, self.eps2d,
                                       self.near_plane, self.background)
        return {"rgb": rgb, "alpha": alpha, "info": info}


def create_renderer(spec):
    return BetaRenderer(spec)
