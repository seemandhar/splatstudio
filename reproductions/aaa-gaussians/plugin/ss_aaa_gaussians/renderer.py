"""AAA-Gaussians renderer: the vendored official rasterizer (raster.py) with configs/aaa.json,
used for training and evaluation like the official train.py/render.py; black background
(official ``white_background`` False).

Config: ``splatting`` = full settings dict (default configs/aaa.json) plus optional flat
overrides (``sort_mode``, ``sort_order``, queue sizes, culling flags, ``load_balancing``,
``proper_ewa_scaling``, ``eval_3D``, ``new_aabb``).

The per-Gaussian training-view filter ``params["filter_3d"]`` is always passed: it is zero
during training (official ``filter_3D = None``: per-view adaptive filter only) and is set to
the official ``compute_3D_filter`` value by the density at the final step, so test renders use
max(filter_3d^2, k z^2/f^2) (paper Eq. 12).
"""

import torch

from .raster import AAA, BOOL_KEYS, make_settings, render

CAPABILITIES = {"render.rgb"}
DEFAULTS = {"background": [0.0, 0.0, 0.0], "splatting": AAA}
FLAT_KEYS = {"sort_mode", "sort_order", "tile_4x4", "tile_2x2", "per_pixel", "rect_bounding",
             "tight_opacity_bounding", "tile_based_culling", "hierarchical_4x4_culling",
             *BOOL_KEYS}


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS) - FLAT_KEYS
        if unknown:
            raise ValueError(f"Unknown AAA renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        flat = {k: v for k, v in cfg.items() if k in FLAT_KEYS}
        self.settings = make_settings(cfg["splatting"], **flat)
        self.background = [float(v) for v in cfg["background"]]
        self.capabilities = set(CAPABILITIES)
        self._bg = {}

    def render(self, params, camera, sh_degree=3):
        dev = params["means"].device
        if dev not in self._bg:
            self._bg[dev] = torch.tensor(self.background, dtype=torch.float32, device=dev)
        color, means2d, radii = render(params, camera, sh_degree, self._bg[dev], self.settings)
        return {
            "rgb": color.permute(1, 2, 0),
            "info": {"means2d": means2d, "radii": radii,
                     "width": int(camera["width"]), "height": int(camera["height"])},
        }


def create_renderer(spec):
    return Renderer(spec)
