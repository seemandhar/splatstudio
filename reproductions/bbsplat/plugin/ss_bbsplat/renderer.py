"""BBSplat renderer component: official textured-billboard CUDA rasterizer (gaussian_renderer.render
of the official code).

Outputs (HWC for the runtime): rgb (over a black background, official default), depth (expected
camera z = allmap[0] / alpha), alpha, normal (world), surface_normal (normal from the expected
depth x alpha, detached alpha), distortion; ``info`` holds the per-billboard impact (Eq. 7) and
radii. With gradients disabled (evaluation) the textures go through the official 8-bit
save/load round trip first (render.py renders the quantised npz textures), unless
``eval_quantize`` is false.
"""

import torch
import torch.nn.functional as F

from . import texture as tex
from .raster import depth_to_normal, raster_settings, rasterize

CAPABILITIES = {
    "render.rgb",
    "render.depth.expected_z",
    "render.alpha",
    "render.normal.world",
    "render.normal_from_depth.world",
    "render.distortion.bbsplat",
    "render.impact.bbsplat",
}
DEFAULTS = {
    "background": [0.0, 0.0, 0.0],  # train.py: white_background False
    "depth_ratio": 0.0,  # pipeline default (expected depth for the normal-from-depth map)
    "eval_quantize": True,  # render.py loads quantised textures (texture_preproc=True)
}


def activated(params, quantized=False):
    """Official getters: means, SH [N,16,3], exp of the two plane scales [N,2] (the runtime's
    third scale column is unused), normalised wxyz quaternions, sigmoid textures."""
    alpha, color = tex.activated(params)
    if quantized:
        alpha, color = tex.quantize(alpha, color)
    return (params["means"],
            torch.cat((params["sh0"], params["shN"]), dim=1),
            alpha, color,
            params["scales"][:, :2].exp(),
            F.normalize(params["quats"], dim=-1))


def render(params, camera, sh_degree, background, depth_ratio=0.0, quantized=False):
    means, sh, alpha_t, color_t, scales, rots = activated(params, quantized)
    s = raster_settings(camera, background, sh_degree)
    color, radii, impact, allmap = rasterize(means, sh, alpha_t, color_t, scales, rots, s)
    alpha = allmap[1:2]
    rend_normal = (allmap[2:5].permute(1, 2, 0) @ (s["viewmatrix"][:3, :3].T)).permute(2, 0, 1)
    median = torch.nan_to_num(allmap[5:6], 0, 0)
    expected = torch.nan_to_num(allmap[0:1] / alpha, 0, 0)
    surf_depth = expected * (1 - depth_ratio) + depth_ratio * median
    surf_normal = depth_to_normal(s, surf_depth).permute(2, 0, 1) * alpha.detach()
    return {"color": color, "radii": radii, "impact": impact, "alpha": alpha,
            "rend_normal": rend_normal, "rend_dist": allmap[6:7], "surf_depth": surf_depth,
            "expected_depth": expected, "surf_normal": surf_normal, "allmap": allmap,
            "settings": s}


class Renderer:
    capabilities = CAPABILITIES

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown BBSplat renderer options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.capabilities = set(CAPABILITIES)

    def render(self, params, camera, sh_degree=3):
        dev = params["means"].device
        bg = torch.tensor(self.cfg["background"], dtype=torch.float32, device=dev)
        quantized = bool(self.cfg["eval_quantize"]) and not torch.is_grad_enabled()
        out = render(params, camera, sh_degree, bg, float(self.cfg["depth_ratio"]), quantized)
        return {
            "rgb": out["color"].permute(1, 2, 0),
            "depth": out["expected_depth"].permute(1, 2, 0),
            "alpha": out["alpha"].permute(1, 2, 0),
            "normal": out["rend_normal"].permute(1, 2, 0),
            "surface_normal": out["surf_normal"].permute(1, 2, 0),
            "distortion": out["rend_dist"].permute(1, 2, 0),
            "info": {"impact": out["impact"], "radii": out["radii"]},
            "bbsplat": out,
        }


def create_renderer(spec):
    return Renderer(spec)
