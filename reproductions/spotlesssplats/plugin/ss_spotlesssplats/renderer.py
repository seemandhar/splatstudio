"""SpotLessSplats renderer: the official gsplat call (RGB only, black background) with the
SH colours evaluated explicitly (same formula gsplat uses internally) so the UBP kernel sees the
exact colours, and a hook that stores dL/dI for the utilisation statistic."""

import torch
from gsplat import rasterization, spherical_harmonics

CAPABILITIES = frozenset(
    {"render.rgb", "gradient.screen.signed", "render.rgb_grad", "render.sh_colors"}
)
DEFAULTS = {"background": "black", "near_plane": 0.01, "far_plane": 1e10}


def sh_colors(params, camera, sh_degree):
    """gsplat.rendering.rasterization colour path: SH(dirs = means - campos) + 0.5, clamp >= 0."""
    coeffs = torch.cat((params["sh0"], params["shN"]), dim=1)  # [N,16,3]
    campos = torch.linalg.inv(camera["viewmat"])[:3, 3]
    dirs = params["means"] - campos[None]
    colors = spherical_harmonics(sh_degree, dirs, coeffs)
    return torch.clamp_min(colors + 0.5, 0.0)


class Renderer:
    def __init__(self, spec):
        config = {**DEFAULTS, **spec.config}
        unknown = set(config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown sls-renderer options: {sorted(unknown)}")
        if config["background"] != "black":
            raise ValueError("SpotLessSplats renders on black (official spotless_trainer.py)")
        self.near = float(config["near_plane"])
        self.far = float(config["far_plane"])
        self.capabilities = set(CAPABILITIES)

    def render(self, params, camera, sh_degree=3):
        colors = sh_colors(params, camera, sh_degree)
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
            near_plane=self.near,
            far_plane=self.far,
            sh_degree=None,
            packed=False,
            render_mode="RGB",
            rasterize_mode="classic",
        )
        rgb = render[0]
        # The hook writes into a holder that does not reference the graph: closing over
        # ``outputs`` would create an rgb -> hook -> outputs -> rgb cycle that keeps every
        # step's autograd graph alive until the cyclic GC runs.
        grads = {}
        outputs = {"rgb": rgb, "alpha": alpha[0], "info": info, "camera": camera,
                   "sh_colors": colors.detach(), "grads": grads}
        if rgb.requires_grad:
            def keep(grad, grads=grads):
                grads["rgb"] = grad.detach()
            rgb.register_hook(keep)
        return outputs


def create_renderer(spec):
    return Renderer(spec)
