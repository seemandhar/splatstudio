"""3D-HGS renderer: vendored official half-Gaussian rasterizer (raster.py), black background
(official `white_background` False).

Training renders (autograd on) use the official training path (`cov3D_precomp_small` empty).
Evaluation renders (autograd off) with `test_time_cull` use the official test path, where
`GaussianModel.load_ply` makes `small_gaussian` non-empty and the kernel switches to the
half-aware ("efficient rasterizer", paper §3.4 / supp. B.1) tile rectangles.

outputs["info"]: `means2d` [N,3] (retained grad: Inria NDC-space screen gradient in [:, :2]),
`radii` [N].
"""

import torch

from .raster import render

CAPABILITIES = {"render.rgb", "gradient.screen.signed"}
DEFAULTS = {"background": [0.0, 0.0, 0.0], "test_time_cull": True}


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown 3D-HGS renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        self.background = [float(v) for v in cfg["background"]]
        self.test_time_cull = bool(cfg["test_time_cull"])
        self.capabilities = set(CAPABILITIES)
        self._bg = {}

    def render(self, params, camera, sh_degree=3):
        dev = params["means"].device
        if dev not in self._bg:
            self._bg[dev] = torch.tensor(self.background, dtype=torch.float32, device=dev)
        cull = self.test_time_cull and not torch.is_grad_enabled()
        color, means2d, radii = render(params, camera, sh_degree, self._bg[dev], cull)
        return {
            "rgb": color.permute(1, 2, 0),
            "info": {"means2d": means2d, "radii": radii, "test_time_cull": cull,
                     "width": int(camera["width"]), "height": int(camera["height"])},
        }


def create_renderer(spec):
    return Renderer(spec)
