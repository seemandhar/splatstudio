"""DNGaussian renderers (official gaussian_renderer/__init__.py: render, render_for_depth,
render_for_opa), on gsplat instead of the ashawkey diff-gaussian-rasterization.

* photometric render (``Renderer.render``): colours from the Neural Color Renderer
  (colors_precomp, no SH), opacity sigmoid(o), black background (white_background False),
  rasterizer near plane 0.2 (Inria in_frustum culling; zfar only enters the projection matrix).
  While training it uses the live network in ``neural.ACTIVE``. Renders without autograd after a
  bake (validation, ss-eval) use the cached geometry feature + packed colour MLP (the paper's
  inference path) and, like the official render.py ``--near 10``, drop Gaussians closer than
  ``eval_near`` to the camera centre (official: to any eval camera, see README gaps).
* ``hard_depth``: Eq. 5 — every opacity = tau (0.95), colours ones, scales / rotations
  detached; gradients reach only the centres.
* ``soft_depth``: Eq. 6 — centres, scales, rotations detached, real opacity; gradients reach
  only the opacity logits.
Depth is gsplat ``render_mode="D"``: sum_i T_i alpha_i z_i (camera z, not alpha-normalised, no
background), which is what the ashawkey rasterizer returns as ``depth``.
"""

import torch
from gsplat import fully_fused_projection, rasterization

from . import neural

CAPABILITIES = {"render.rgb", "gradient.screen.signed"}
DEFAULTS = {"near_plane": 0.2, "background": 0.0, "eval_near": 10.0}


def _raster(means, quats, scales, opacities, colors, camera, mode, near_plane, bg=None):
    render, alpha, info = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
        viewmats=camera["viewmat"][None], Ks=camera["K"][None],
        width=int(camera["width"]), height=int(camera["height"]),
        near_plane=near_plane, sh_degree=None, packed=False, render_mode=mode,
        rasterize_mode="classic", backgrounds=bg,
    )
    return render[0], alpha[0], info


def hard_depth(params, camera, tau=0.95, near_plane=0.2):
    """Eq. 5 hard depth [1, 1, H, W]; differentiable w.r.t. means only."""
    n = len(params["means"])
    dev = params["means"].device
    depth, alpha, info = _raster(
        params["means"], params["quats"].detach(), params["scales"].detach().exp(),
        torch.full((n,), float(tau), device=dev), torch.ones(n, 3, device=dev), camera, "D",
        near_plane)
    return depth.permute(2, 0, 1)[None], alpha, info


def soft_depth(params, camera, near_plane=0.2):
    """Eq. 6 soft depth [1, 1, H, W]; differentiable w.r.t. opacity logits only."""
    n = len(params["means"])
    dev = params["means"].device
    depth, alpha, info = _raster(
        params["means"].detach(), params["quats"].detach(), params["scales"].detach().exp(),
        params["opacities"].sigmoid(), torch.ones(n, 3, device=dev), camera, "D", near_plane)
    return depth.permute(2, 0, 1)[None], alpha, info


@torch.no_grad()
def inria_radii(params, camera, near_plane=0.2):
    """Visibility as the Inria rasterizer defines it (radius from the 3-sigma extent, no
    opacity-aware shrinking or culling): gsplat projection without opacities. [1, N, 2]"""
    return fully_fused_projection(
        params["means"], None, params["quats"], params["scales"].exp(),
        camera["viewmat"][None], camera["K"][None], int(camera["width"]), int(camera["height"]),
        near_plane=near_plane, packed=False)[0]


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown DNGaussian renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        self.near_plane = float(cfg["near_plane"])
        self.background = float(cfg["background"])
        self.eval_near = float(cfg["eval_near"])
        self.capabilities = set(CAPABILITIES)

    def colors_live(self, params, camera):
        net = neural.ACTIVE["net"]
        if net is None:
            raise RuntimeError("DNGaussian renderer: no active neural colour renderer (the density "
                               "component creates it in initialize)")
        return net(params["means"], neural.view_dirs(params["means"], camera["viewmat"]))

    def render(self, params, camera, sh_degree=3):
        live = torch.is_grad_enabled() or (neural.ACTIVE["net"] is not None
                                           and not neural.ACTIVE["baked"])
        means, quats = params["means"], params["quats"]
        scales, opacities = params["scales"].exp(), params["opacities"].sigmoid()
        if live:
            colors = self.colors_live(params, camera)
            keep = None
        else:
            mlp, _deg = neural.unpack_color_net(params)
            d = neural.view_dirs(means, camera["viewmat"])
            colors = neural.color_head(mlp, torch.cat([neural.sh_encode(d, _deg),
                                                       params["dng_geo"]], dim=-1))
            keep = None
            if self.eval_near > 0:
                center = torch.linalg.inv(camera["viewmat"])[:3, 3]
                keep = (means - center[None]).norm(dim=1) >= self.eval_near
                means, quats, scales = means[keep], quats[keep], scales[keep]
                opacities, colors = opacities[keep], colors[keep]
        bg = torch.full((1, 3), self.background, device=means.device)
        rgb, alpha, info = _raster(means, quats, scales, opacities, colors, camera, "RGB",
                                   self.near_plane, bg)
        if live:
            info["visibility_radii"] = inria_radii(params, camera, self.near_plane)
        return {"rgb": rgb, "alpha": alpha, "info": info, "opacity": opacities,
                "camera": {k: camera[k] for k in ("viewmat", "K", "width", "height")},
                "eval_kept": None if keep is None else int(keep.sum())}


def create_renderer(spec):
    return Renderer(spec)
