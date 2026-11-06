"""RobustSplat renderer: gsplat ``rasterization`` over a black background (official
``white_background=False``) with the Inria near-plane cull (z <= 0.2). Exposes the camera and SH
degree so the mask losses can re-render the coarse view and look up per-view data."""

import torch
from gsplat import rasterization

CAPABILITIES = frozenset({"render.rgb", "gradient.screen.signed"})
DEFAULTS = {"background": "black", "near_plane": 0.2, "far_plane": 1e10}


def rasterize(params, camera, sh_degree, near_plane=0.2, far_plane=1e10):
    """[H,W,3] RGB on black, alpha [H,W,1] and gsplat info for one camera."""
    colors = torch.cat((params["sh0"], params["shN"]), dim=1)
    render, alpha, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(),
        colors=colors,
        viewmats=camera["viewmat"][None],
        Ks=camera["K"][None],
        width=int(camera["width"]),
        height=int(camera["height"]),
        near_plane=near_plane,
        far_plane=far_plane,
        sh_degree=int(sh_degree),
        packed=False,
        render_mode="RGB",
        rasterize_mode="classic",
    )
    return render[0], alpha[0], info


class Renderer:
    def __init__(self, spec):
        config = {**DEFAULTS, **spec.config}
        unknown = set(config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown rs-renderer options: {sorted(unknown)}")
        if config["background"] != "black":
            raise ValueError("RobustSplat renders on black (official white_background=False)")
        self.near = float(config["near_plane"])
        self.far = float(config["far_plane"])
        self.capabilities = set(CAPABILITIES)

    def render(self, params, camera, sh_degree=3):
        rgb, alpha, info = rasterize(params, camera, sh_degree, self.near, self.far)
        return {"rgb": rgb, "alpha": alpha, "info": info, "camera": camera,
                "sh_degree": int(sh_degree), "near_plane": self.near, "far_plane": self.far}


def create_renderer(spec):
    return Renderer(spec)
