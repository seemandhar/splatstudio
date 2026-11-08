"""HDGS renderer component: official stage-1 rasterizer (k-sort + frustum, SH colours) until the
stage switch, then the official textured rasterizer (params[FLAG] == 1, texture from STORE).

Outputs (HWC for the runtime): rgb, depth (expected camera z), alpha, normal (world, blended),
surface_normal (from the expected depth, x alpha), distortion; ``hdgs`` holds the official CHW
maps used by the 2DGS regularisers and ``info`` the densification statistics.
"""

import torch

from . import texture as tex
from .raster import render

CAPABILITIES = {
    "render.rgb",
    "render.depth.expected_z",
    "render.alpha",
    "render.normal.world",
    "render.normal_from_depth.world",
    "render.distortion.hdgs",
    "gradient.hdgs",
}
DEFAULTS = {
    "sort_window": 16,  # paper §3.3 "16 for real scenes"; README; official eval360S16
    "background": [0.0, 0.0, 0.0],  # run_360.py: no --white_background
    "depth_ratio": 0.0,  # expected depth for the normal-from-depth map (unbounded scenes)
    "tex_res": 200,  # code-texture/run_360.py --tex_res (outdoor 200, indoor 500)
    "max_tex": 500,  # train_texture.py init_texture(max_tex=500)
}


def textured(params):
    return tex.FLAG in params and bool(params[tex.FLAG].detach()[0, 0] > 0.5)


class Renderer:
    capabilities = CAPABILITIES

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown HDGS renderer options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.capabilities = set(CAPABILITIES)

    def render(self, params, camera, sh_degree=3):
        dev = params["means"].device
        bg = torch.tensor(self.cfg["background"], dtype=torch.float32, device=dev)
        texture = index = None
        is_tex = textured(params)
        if is_tex:
            texture, index = tex.ensure_loaded(params, float(self.cfg["tex_res"]),
                                               int(self.cfg["max_tex"]))
        out = render(params, camera, sh_degree, bg, int(self.cfg["sort_window"]),
                     float(self.cfg["depth_ratio"]), texture, index)
        return {
            "rgb": out["color"].permute(1, 2, 0),
            "depth": out["expected_depth"].permute(1, 2, 0),
            "alpha": out["alpha"].permute(1, 2, 0),
            "normal": out["rend_normal"].permute(1, 2, 0),
            "surface_normal": out["surf_normal"].permute(1, 2, 0),
            "distortion": out["rend_dist"].permute(1, 2, 0),
            "info": {"means2d": out["means2d"], "radii": out["radii"]},
            "hdgs": out,
            "textured": is_tex,
        }


def create_renderer(spec):
    return Renderer(spec)
