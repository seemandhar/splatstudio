"""gsplat renderer with Mip-Splatting's 3D smoothing filter and 2D Mip filter.

The 2D Mip filter (paper Eq. 10) is gsplat's `rasterize_mode="antialiased"` with
`eps2d = kernel_size_2d`: gsplat's `add_blur` adds eps2d to the projected covariance and scales
opacity by sqrt(det(Sigma2D) / det(Sigma2D + eps2d I)), differentiably — the official
`computeCov2D` (forward.cu) with kernel_size 0.1. The 3D filter (Eq. 9) is applied to
scales/opacities before projection, as in the official `render()`.
"""

import torch
from gsplat import rasterization

from .filters import filtered_scales_opacities

CAPABILITIES = frozenset(
    {
        "render.rgb",
        "render.depth.expected_z",
        "gradient.screen.signed",
        "render.antialiased_2d",
        "render.filter_3d",
    }
)


class MipRenderer:
    def __init__(self, spec):
        unknown = set(spec.config) - {"kernel_size_2d"}
        if unknown:
            raise ValueError(f"Unknown renderer options: {sorted(unknown)}")
        self.kernel_size_2d = float(spec.config["kernel_size_2d"])
        self.capabilities = set(CAPABILITIES)

    def render(self, params, camera, sh_degree=3):
        scales, opacities = filtered_scales_opacities(
            params["scales"], params["opacities"], params["filter_3d"]
        )
        colors = torch.cat((params["sh0"], params["shN"]), dim=1)
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
        )
        return {
            "rgb": render[0, ..., :3] + (1 - alpha[0]),
            "depth": render[0, ..., 3:4],
            "alpha": alpha[0],
            "info": info,
        }


def create_renderer(spec):
    return MipRenderer(spec)
