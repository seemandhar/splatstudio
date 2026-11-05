"""SparseGS renderer: gsplat RGB + the official softmax / alpha / mode depth (paper §3.1).

Official gaussian_renderer/__init__.py renders RGB with SH, black background
(white_background False) and returns ``depth`` = softmax depth (log), ``alpha_depth``,
``modes``, ``mode_id``. Here RGB is gsplat ``rasterization`` (classic, packed=False, near plane
0.2 = Inria in_frustum culling) and the depth maps come from ``depth_raster`` on the same
projection. Depth passes run only while autograd is enabled (training) or on request
(``with_depth=True``), so evaluation renders RGB only.
"""

import torch
from gsplat import fully_fused_projection, rasterization

from .depth_raster import render_depths

CAPABILITIES = {"render.rgb", "gradient.screen.signed", "render.depth.softmax",
                "render.depth.alpha", "render.depth.mode"}
DEFAULTS = {"beta": 5.0, "near_plane": 0.2, "background": 0.0}


def rasterize(params, camera, sh_degree, near_plane=0.2, background=0.0):
    colors = torch.cat((params["sh0"], params["shN"]), dim=1)
    bg = torch.full((1, 3), float(background), device=params["means"].device)
    render, alpha, info = rasterization(
        means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(), colors=colors,
        viewmats=camera["viewmat"][None], Ks=camera["K"][None],
        width=int(camera["width"]), height=int(camera["height"]), sh_degree=sh_degree,
        near_plane=near_plane, packed=False, render_mode="RGB", rasterize_mode="classic",
        backgrounds=bg)
    return render[0], alpha[0], info


@torch.no_grad()
def inria_radii(params, camera, near_plane=0.2):
    """Screen radius [N] as the Inria rasterizer defines it (preprocessCUDA: 3 sqrt(lambda_max)
    of the 0.3-dilated 2D covariance, ceil; 0 when culled), from gsplat's projection without
    opacity-aware shrinking."""
    radii, _, _, conics, _ = fully_fused_projection(
        params["means"], None, params["quats"], params["scales"].exp(),
        camera["viewmat"][None], camera["K"][None], int(camera["width"]), int(camera["height"]),
        near_plane=near_plane, packed=False)
    a, b, c = conics[0].unbind(-1)
    det_conic = (a * c - b * b).clamp_min(1e-20)
    cxx, cyy = c / det_conic, a / det_conic  # covariance = inverse conic
    det = cxx * cyy - (b / det_conic) ** 2
    mid = 0.5 * (cxx + cyy)
    lam = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
    r = torch.ceil(3.0 * torch.sqrt(lam))
    return torch.where((radii[0] > 0).all(dim=-1), r, torch.zeros_like(r))


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown SparseGS renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        self.beta = float(cfg["beta"])
        self.near_plane = float(cfg["near_plane"])
        self.background = float(cfg["background"])
        self.capabilities = set(CAPABILITIES)

    def render(self, params, camera, sh_degree=3, with_depth=None):
        rgb, alpha, info = rasterize(params, camera, sh_degree, self.near_plane, self.background)
        out = {"rgb": rgb, "alpha": alpha, "info": info,
               "camera": {k: camera[k] for k in ("viewmat", "K", "width", "height")}}
        if with_depth is None:
            with_depth = torch.is_grad_enabled()
        if with_depth:
            dep = render_depths(info, self.beta)
            out["depth_softmax"] = dep["softmax"]
            out["depth_alpha"] = dep["alpha"]
            out["depth_mode"] = dep["mode"]
            out["mode_ids"] = dep["mode_ids"]
            out["mode_ids_official"] = dep["mode_ids_official"]
        if torch.is_grad_enabled():
            info["visibility_radii"] = inria_radii(params, camera, self.near_plane)
        return out


def create_renderer(spec):
    return Renderer(spec)
