"""SteepGS renderer: vendored official rasterizer (raster.py), black background (official
`white_background` False), splitting statistics with the `inv_cov` estimator (README default).

outputs["info"]: `means2d` [N,3] (retained grad: Inria NDC-space screen gradient in [:, :2]),
`splitting_mats` [N,3,3] leaf (grad[:, 0, 0] = sum over pixels of dL/dG), `radii` [N].
"""

import torch

from .raster import SPLITTING_ESTIMATORS, render

SPLITTING_CAPABILITY = "gradient.splitting.inv_cov"
CAPABILITIES = {"render.rgb", "gradient.screen.signed", SPLITTING_CAPABILITY}
DEFAULTS = {"background": [0.0, 0.0, 0.0], "S_estimator": "inv_cov"}


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown SteepGS renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        if cfg["S_estimator"] != "inv_cov":
            # The density component implements only the inv_cov post-processing
            # (gaussian_model.add_densification_stats); `partial`/`approx` are not ported.
            raise ValueError(f"Only S_estimator 'inv_cov' is ported (got {cfg['S_estimator']}; "
                             f"kernel supports {sorted(SPLITTING_ESTIMATORS)})")
        self.s_estimator = cfg["S_estimator"]
        self.background = [float(v) for v in cfg["background"]]
        self.capabilities = set(CAPABILITIES)
        self._bg = {}

    def render(self, params, camera, sh_degree=3):
        dev = params["means"].device
        if dev not in self._bg:
            self._bg[dev] = torch.tensor(self.background, dtype=torch.float32, device=dev)
        color, means2d, splitting, radii = render(params, camera, sh_degree, self._bg[dev],
                                                  self.s_estimator)
        return {
            "rgb": color.permute(1, 2, 0),
            "info": {"means2d": means2d, "splitting_mats": splitting, "radii": radii,
                     "width": int(camera["width"]), "height": int(camera["height"])},
        }


def create_renderer(spec):
    return Renderer(spec)
