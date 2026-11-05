"""DOF-GS renderer: official DOF rasterizer with per-view (f_m, Q_m) from the live bank.

* training camera (viewmat matches a bank view): ``rgb`` = defocused render with the view's
  learned focal distance / aperture (training.py: ``render(..., select_view=idx)``), plus
  ``rgb_aif`` (F=[1], K=[0], the official ``valid_F``/``valid_K``) once step > infocus_start;
* any other camera (test/validation views, or no live bank as in ss-eval): All-in-Focus render
  (Q = 0) as ``rgb`` -- the official evaluation of test views.
Background black, 2D kernel 0.1. ``band_limit`` (per-scene official option, variants.py):
true -> Mip-Splatting 3D filter on scales/opacities; false -> raw exp(scales)/sigmoid(opacity)
(renderer/__init__.py:63-82). The density component must use the same value (method() sets
both).
"""

import torch
import torch.nn.functional as F

from . import bank as bank_mod
from .filters import filtered_scales_opacities
from .raster import rasterize

CAPABILITIES = {
    "render.rgb", "render.rgb.all_in_focus", "render.depth.ray_distance", "render.coc",
    "render.dof_view", "gradient.screen.signed",
}
DEFAULTS = {"background": [0.0, 0.0, 0.0], "kernel_size": 0.1, "band_limit": True,
            "infocus_start_iter": 5000}


def activated(params, band_limit=True):
    if band_limit:
        scales, opac = filtered_scales_opacities(params["scales"], params["opacities"],
                                                 params["filter_3d"])
    else:
        scales, opac = params["scales"].exp(), params["opacities"].sigmoid()
    quats = F.normalize(params["quats"], dim=-1)
    sh = torch.cat((params["sh0"], params["shN"]), dim=1)
    return scales, opac, quats, sh


def render_dof(params, camera, sh_degree, background, focal, aperture, view=0,
               kernel_size=0.1, band_limit=True):
    scales, opac, quats, sh = activated(params, band_limit)
    return rasterize(params["means"], scales, opac, quats, sh, camera, focal, aperture, view,
                     sh_degree, background, kernel_size)


def aif_camera(device):
    return torch.ones(1, 1, device=device), torch.zeros(1, 1, device=device)


class Renderer:
    capabilities = CAPABILITIES

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown DOF-GS renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        self.background = [float(v) for v in cfg["background"]]
        self.kernel_size = float(cfg["kernel_size"])
        self.band_limit = bool(cfg["band_limit"])
        self.infocus_start = int(cfg["infocus_start_iter"])
        self.capabilities = set(CAPABILITIES)
        self._bg = {}

    def _background(self, dev):
        if dev not in self._bg:
            self._bg[dev] = torch.tensor(self.background, dtype=torch.float32, device=dev)
        return self._bg[dev]

    def render(self, params, camera, sh_degree=3):
        dev = params["means"].device
        bg = self._background(dev)
        bank = bank_mod.ACTIVE
        view = bank.view_index(camera["viewmat"]) if bank is not None else None
        kw = {"kernel_size": self.kernel_size, "band_limit": self.band_limit}
        if view is None:
            f0, k0 = aif_camera(dev)
            out = render_dof(params, camera, sh_degree, bg, f0, k0, 0, **kw)
            return {"rgb": out["rgb"], "rgb_aif": out["rgb"], "depth": out["depth"],
                    "coc": out["coc"], "dof": {"view": None, "bank": None},
                    "info": {"means2d": out["means2d"], "radii": out["radii"]}}
        out = render_dof(params, camera, sh_degree, bg, bank.focal(), bank.aperture(), view, **kw)
        result = {"rgb": out["rgb"], "depth": out["depth"], "coc": out["coc"],
                  "dof": {"view": view, "bank": bank}, "rgb_aif": None,
                  "info": {"means2d": out["means2d"], "radii": out["radii"]}}
        if torch.is_grad_enabled() and bank.step + 1 > self.infocus_start:
            f0, k0 = aif_camera(dev)
            result["rgb_aif"] = render_dof(params, camera, sh_degree, bg, f0, k0, 0, **kw)["rgb"]
        return result


def create_renderer(spec):
    return Renderer(spec)
