"""StopThePop renderer: vendored official sorted rasterizer (raster.py) used for training and
evaluation (official train.py renders every iteration with the splatting config), black
background (official `white_background` False).

Config: `splatting` = full settings dict (default configs/hierarchical.json), optional flat
overrides `sort_mode`, `sort_order` and ExtendedSettings keys (e.g. `per_pixel`, `tile_2x2`,
`tile_based_culling`, `load_balancing`).

outputs["info"]: `means2d` [N,3] (retained grad: Inria NDC-space screen gradient in [:, :2]),
`radii` [N].
"""

import torch

from .raster import HIERARCHICAL, make_settings, render

CAPABILITIES = {"render.rgb", "gradient.screen.signed"}
DEFAULTS = {"background": [0.0, 0.0, 0.0], "splatting": HIERARCHICAL}
FLAT_KEYS = {"sort_mode", "sort_order", "tile_4x4", "tile_2x2", "per_pixel", "rect_bounding",
             "tight_opacity_bounding", "tile_based_culling", "hierarchical_4x4_culling",
             "load_balancing", "proper_ewa_scaling"}


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS) - FLAT_KEYS
        if unknown:
            raise ValueError(f"Unknown StopThePop renderer options: {sorted(unknown)}")
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
