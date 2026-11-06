"""gsplat renderer with GaussianPro's hybrid geometric representation (paper §4.1).

RGB pass: identical to SplatStudio's baseline gsplat renderer (SH colour, RGB+ED, white
background), so the photometric path matches our gsplat baseline.

Geometry pass (official ``gaussian_renderer.render(return_depth, return_normal)``): the same
Gaussians rasterised with precomputed colours [z_i, n_i^cam] over a black background
(official bg for Mip-NeRF 360), giving the accumulated depth Σ z_i α_i T_i (``render_depth``)
and the alpha-blended camera normal normalised per pixel (``render_normal``,
``F.normalize(dim=0)``). It runs
  * with autograd on training renders of a view that already holds a propagated normal
    (the planar loss, Eq. 8), and
  * without autograd when the density controller needs the rendered depth for propagation.

``st_alpha_clamp``: gsplat's backward drops the gradient wherever o·G > 0.999 (TODO-runtime);
the official Inria rasterizer clamps alpha (0.99) in the forward only. Propagated Gaussians
start fully opaque (official opacity inverse_sigmoid(1)), so the opacity given to gsplat is
capped at 0.999 with a straight-through gradient (DropGaussian precedent).
"""

import torch
import torch.nn.functional as F
from gsplat import rasterization

from . import geometry as G
from .link import LINK

CAPABILITIES = {
    "render.rgb",
    "render.depth.expected_z",
    "render.depth.accumulated",
    "render.normal.camera",
    "render.alpha",
    "gradient.screen.signed",
}
DEFAULTS = {"st_alpha_clamp": True}
ALPHA_CAP = 0.999


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown GaussianPro renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        self.st_alpha_clamp = bool(cfg["st_alpha_clamp"])
        self.capabilities = set(CAPABILITIES)
        LINK.renderer = self

    def opacities(self, params):
        o = params["opacities"].sigmoid()
        if self.st_alpha_clamp:
            o = o + (o.clamp(max=ALPHA_CAP) - o).detach()
        return o

    def render(self, params, camera, sh_degree=3):
        opacities = self.opacities(params)
        colors = torch.cat((params["sh0"], params["shN"]), dim=1)
        render, alpha, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"].exp(),
            opacities=opacities,
            colors=colors,
            viewmats=camera["viewmat"][None],
            Ks=camera["K"][None],
            width=camera["width"],
            height=camera["height"],
            sh_degree=sh_degree,
            packed=False,
            render_mode="RGB+ED",
        )
        out = {
            "rgb": render[0, ..., :3] + (1 - alpha[0]),
            "depth": render[0, ..., 3:4],
            "alpha": alpha[0],
            "info": info,
            "gp_view": None,
        }
        if not torch.is_grad_enabled():
            return out
        view = LINK.view_index(camera)
        out["gp_view"] = view
        if view is not None and LINK.normal_targets[view] is not None:
            geo = self.geometry(params, camera, opacities)
            out["normal_cam"] = geo["normal"]
            out["normal_target"] = LINK.normal_targets[view].to(geo["normal"].device)
            out["geo_info"] = geo["info"]
        return out

    def geometry(self, params, camera, opacities=None):
        """Accumulated depth [H,W], normalised camera normal [H,W,3], alpha [H,W]."""
        if opacities is None:
            opacities = self.opacities(params)
        viewmat = camera["viewmat"].float()
        z = G.camera_depths(params["means"], viewmat)
        n = G.camera_normals(params["means"], params["quats"], params["scales"], viewmat)
        feats = torch.cat((z[:, None], n), dim=1)
        render, alpha, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"].exp(),
            opacities=opacities,
            colors=feats,
            viewmats=viewmat[None],
            Ks=camera["K"][None],
            width=camera["width"],
            height=camera["height"],
            sh_degree=None,
            packed=False,
            render_mode="RGB",
        )
        return {
            "depth": render[0, ..., 0],
            "normal": F.normalize(render[0, ..., 1:4], dim=-1),
            "alpha": alpha[0, ..., 0],
            "info": info,
        }


def create_renderer(spec):
    return Renderer(spec)
