"""Renderer: gsplat RGB rasterization with absolute gradients plus the CoMe confidence map.

The SH colour is evaluated exactly as gsplat's `rasterization(sh_degree=...)` does
(dirs = means - camera centre, `spherical_harmonics`, clamp_min(c + 0.5, 0)), and RGB is rasterised
with absolute gradients. The confidence map follows CoMe's rasteriser
(submodules/diff-gaussian-rasterization/.../hierarchical_render.cuh:1231, :1513-1515, :1643):
C_hat(p) = sum_i b_i(p) exp(gamma_i) / max(1 - T(p), 1e-3), i.e. opacity-normalised (1 everywhere
at init), and dL/dC_hat reaches only gamma -- never opacity, means, conics or the absolute
screen-space gradient. We obtain exactly that with a second rasterisation on detached geometry.
"""

import torch
from gsplat import rasterization, spherical_harmonics

CAPABILITIES = {
    "render.rgb",
    "render.depth.expected_z",
    "render.confidence",
    "gradient.screen.signed",
    "gradient.screen.absolute",
}


def sh_colors(params, viewmat, sh_degree):
    campos = torch.linalg.inv(viewmat)[:3, 3]
    dirs = params["means"] - campos[None]
    coeffs = torch.cat((params["sh0"], params["shN"]), dim=1)
    return torch.clamp_min(spherical_harmonics(sh_degree, dirs, coeffs) + 0.5, 0.0)


class Renderer:
    def __init__(self, spec):
        if spec.config:
            raise ValueError("cad-renderer accepts no configuration")
        self.capabilities = set(CAPABILITIES)

    def render(self, params, camera, sh_degree=3):
        rgb = sh_colors(params, camera["viewmat"], sh_degree)
        render, alpha, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"].exp(),
            opacities=params["opacities"].sigmoid(),
            colors=rgb,
            viewmats=camera["viewmat"][None],
            Ks=camera["K"][None],
            width=camera["width"],
            height=camera["height"],
            sh_degree=None,
            packed=False,
            render_mode="RGB+ED",
            absgrad=True,
        )
        return {
            "rgb": render[0, ..., :3] + (1 - alpha[0]),  # white background (runtime convention)
            "confidence": render_confidence(params, camera),
            "depth": render[0, ..., 3:4],
            "alpha": alpha[0],
            "info": info,
            "camera": {k: camera[k] for k in ("viewmat", "K", "width", "height")},
            "sh_degree": sh_degree,
        }


def render_confidence(params, camera):
    """CoMe confidence map: normalised blend of exp(gamma); gradient only to gamma."""
    conf, calpha, _ = rasterization(
        means=params["means"].detach(),
        quats=params["quats"].detach(),
        scales=params["scales"].detach().exp(),
        opacities=params["opacities"].detach().sigmoid(),
        colors=params["confidence"].exp(),  # gamma_bar = exp(gamma), [N,1]
        viewmats=camera["viewmat"][None],
        Ks=camera["K"][None],
        width=camera["width"],
        height=camera["height"],
        sh_degree=None,
        packed=False,
        render_mode="RGB",
    )
    return conf[0] / calpha[0].clamp_min(1e-3)  # hierarchical_render.cuh:1231


def create_renderer(spec):
    return Renderer(spec)
