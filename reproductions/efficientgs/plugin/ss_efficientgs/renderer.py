"""gsplat renderer with per-Gaussian SH degree (EfficientGS Sec. 3.3) and gradient taps.

Colours follow gsplat's own SH path (`spherical_harmonics(deg, means - campos, coeffs)`,
`clamp_min(+0.5, 0)`) but are computed here and passed post-activation (`sh_degree=None`), so
that (a) each Gaussian uses its own degree from the `sh_order` field — bands above it are
multiplied by zero, which equals the official `computeColorFromSH(idx, sparse_shs_degree[idx])`
in value and gradient — and (b) the per-Gaussian colours are available to the Eq. 6 kernel.

The runtime's global SH schedule (`sh_degree` argument, +1 per 1000 steps) is only used as an
upper bound: per-Gaussian degrees are 0 until the paper's first sparse increment (16000), and
evaluation (sh_degree = 3) renders every Gaussian at its stored degree.

Outputs besides rgb/alpha/info: `raw_rgb` / `raw_alpha` (gsplat's pre-background outputs, whose
gradients the density retains for Eq. 6), `colors` (per-Gaussian RGB, detached), the frame's
`target_image` and `viewmat` (training images are cached by the density for Eq. 9).
"""

import torch
from gsplat import rasterization, spherical_harmonics

from . import core

CAPABILITIES = (
    "render.rgb",
    "gradient.screen.signed",
    "render.pixel_gradients",
    "training.target_image",
)


class EfficientGSRenderer:
    def __init__(self, spec):
        if spec is not None and spec.config:
            raise ValueError("EfficientGS renderer accepts no configuration")
        self.capabilities = set(CAPABILITIES)

    def colors(self, params, camera, sh_degree=3):
        means = params["means"]
        order = params["sh_order"][:, 0].detach().round().clamp(0, min(int(sh_degree), 3))
        degree = int(order.max().item()) if len(order) else 0
        campos = torch.linalg.inv(camera["viewmat"])[:3, 3]
        dirs = means - campos[None, :]
        if degree == 0:
            coeffs = params["sh0"]
        else:
            k = (degree + 1) ** 2 - 1
            mask = core.rest_mask(order, degree)[..., None]
            coeffs = torch.cat((params["sh0"], params["shN"][:, :k] * mask), dim=1)
        rgb = spherical_harmonics(degree, dirs, coeffs)
        return torch.clamp_min(rgb + 0.5, 0.0), degree

    def render(self, params, camera, sh_degree=3):
        colors, degree = self.colors(params, camera, sh_degree)
        render, alpha, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"].exp(),
            opacities=params["opacities"].sigmoid(),
            colors=colors[None],
            viewmats=camera["viewmat"][None],
            Ks=camera["K"][None],
            width=camera["width"],
            height=camera["height"],
            sh_degree=None,
            packed=False,
            render_mode="RGB",
            rasterize_mode="classic",
        )
        # Dataset convention: RGB over a white background.
        return {
            "rgb": render[0] + (1 - alpha[0]),
            "alpha": alpha[0],
            "info": info,
            "raw_rgb": render,
            "raw_alpha": alpha,
            "colors": colors.detach(),
            "sh_degree_used": degree,
            "target_image": camera.get("image"),
            "viewmat": camera["viewmat"],
        }


def create_renderer(spec):
    return EfficientGSRenderer(spec)
