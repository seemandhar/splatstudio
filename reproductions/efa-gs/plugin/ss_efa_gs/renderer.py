"""Mip-Splatting renderer (3D smoothing filter + 2D Mip filter) with absolute screen gradients.

Same rendering as reproductions/mip-splatting (`MipRenderer`): 3D filter applied to scales and
opacities before projection (Mip-Splatting Eq. 9), 2D Mip filter = gsplat
`rasterize_mode="antialiased"` with eps2d = kernel_size 0.1 (Eq. 10). Additionally passes
`absgrad=True` so the density controller gets the per-pixel absolute gradient sums that the
official EFA-GS / Mip-Splatting (GOF densification) rasterizer writes into grad[:, 2].
Background: configurable, default black (official `_white_background = False`, train.py:62
bg_color [0, 0, 0]); the runtime's built-in gsplat renderer (and our baselines) use white.
"""

import torch
from gsplat import rasterization

from .mip_filters import filtered_scales_opacities

CAPABILITIES = frozenset(
    {
        "render.rgb",
        "render.depth.expected_z",
        "gradient.screen.signed",
        "gradient.screen.absolute",
        "render.antialiased_2d",
        "render.filter_3d",
    }
)


class EFARenderer:
    def __init__(self, spec):
        unknown = set(spec.config) - {"kernel_size_2d", "background"}
        if unknown:
            raise ValueError(f"Unknown renderer options: {sorted(unknown)}")
        self.kernel_size_2d = float(spec.config["kernel_size_2d"])
        self.background = [float(v) for v in spec.config["background"]]
        if len(self.background) != 3:
            raise ValueError("background must be an RGB triple")
        self.capabilities = set(CAPABILITIES)

    def render(self, params, camera, sh_degree=3):
        scales, opacities = filtered_scales_opacities(
            params["scales"], params["opacities"], params["filter_3d"]
        )
        colors = torch.cat((params["sh0"], params["shN"]), dim=1)
        # RGB background; gsplat appends the zero for the expected-depth channel (RGB+ED).
        bg = torch.tensor(self.background, device=params["means"].device)[None]
        render, alpha, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=camera["viewmat"][None],
            Ks=camera["K"][None],
            width=camera["width"],
            height=camera["height"],
            sh_degree=sh_degree,
            packed=False,
            render_mode="RGB+ED",
            eps2d=self.kernel_size_2d,
            rasterize_mode="antialiased",
            absgrad=True,
            backgrounds=bg,
        )
        return {
            "rgb": render[0, ..., :3],
            "depth": render[0, ..., 3:4],
            "alpha": alpha[0],
            "info": info,
        }


def create_renderer(spec):
    return EFARenderer(spec)
