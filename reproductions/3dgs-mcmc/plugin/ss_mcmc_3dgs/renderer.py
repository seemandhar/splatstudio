"""gsplat classic 3DGS rasterization over a configurable background (adapted from
reproductions/mh-3dgs). Official 3dgs-mcmc trains and evaluates over black (white_background is
False in every real-scene config)."""

import torch
from gsplat import rasterization

CAPABILITIES = {"render.rgb", "render.depth.expected_z", "gradient.screen.signed"}


def render(params, camera, sh_degree, background):
    colors = torch.cat((params["sh0"], params["shN"]), dim=1)
    out, alpha, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(),
        colors=colors,
        viewmats=camera["viewmat"][None],
        Ks=camera["K"][None],
        width=camera["width"],
        height=camera["height"],
        sh_degree=sh_degree,
        packed=False,
        render_mode="RGB+ED",
        rasterize_mode="classic",
    )
    if info["means2d"].requires_grad:
        info["means2d"].retain_grad()  # honest gradient.screen.signed capability
    rgb = out[0, ..., :3] + background * (1 - alpha[0])
    return {"rgb": rgb, "depth": out[0, ..., 3:4], "alpha": alpha[0], "info": info}


class Renderer:
    capabilities = CAPABILITIES

    def __init__(self, spec):
        unknown = set(spec.config) - {"background"}
        if unknown:
            raise ValueError(f"Unknown mcmc-raster options: {sorted(unknown)}")
        self.background = list(spec.config.get("background", [0.0, 0.0, 0.0]))
        if len(self.background) != 3:
            raise ValueError("background must be RGB")
        self._bg = {}

    def bg(self, device):
        if device not in self._bg:
            self._bg[device] = torch.tensor(self.background, dtype=torch.float32, device=device)
        return self._bg[device]

    def render(self, params, camera, sh_degree=3):
        return render(params, camera, sh_degree, self.bg(params["means"].device))


def create_renderer(spec):
    return Renderer(spec)
