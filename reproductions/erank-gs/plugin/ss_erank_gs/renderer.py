"""Vanilla 3DGS splatting with gsplat (classic mode: 0.3 px dilation like the Inria
rasterizer), absolute-gradient statistics for the fixed ADC, and an explicit background.

The official erank-gs DTU runs use white_background = False: RGB is composited on black and
the targets are RGB * alpha-mask (black outside the object), see tools/prepare_dtu_2dgs.py.
"""

import torch
from gsplat import rasterization

DEFAULTS = {"background": [0.0, 0.0, 0.0]}
CAPABILITIES = {"render.rgb", "render.depth.expected_z", "gradient.screen.signed",
                "gradient.screen.absolute"}


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown erank renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        if len(cfg["background"]) != 3:
            raise ValueError("background must be an RGB triple")
        self.background = [float(v) for v in cfg["background"]]
        self.capabilities = set(CAPABILITIES)

    def render(self, params, camera, sh_degree=3):
        colors = torch.cat((params["sh0"], params["shN"]), dim=1)
        render, alpha, info = rasterization(
            means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
            opacities=params["opacities"].sigmoid(), colors=colors,
            viewmats=camera["viewmat"][None], Ks=camera["K"][None],
            width=camera["width"], height=camera["height"], sh_degree=sh_degree,
            packed=False, render_mode="RGB+ED", absgrad=True, rasterize_mode="classic",
        )
        bg = torch.tensor(self.background, device=render.device, dtype=render.dtype)
        return {"rgb": render[0, ..., :3] + (1 - alpha[0]) * bg, "depth": render[0, ..., 3:4],
                "alpha": alpha[0], "info": info}


def create_renderer(spec):
    return Renderer(spec)
