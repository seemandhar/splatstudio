"""gsplat renderer for free (stages A/B) and mesh-bound (stage D) Gaussians."""

import torch
from gsplat import rasterization

from .context import new_context
from .geometry import bound_gaussians

CAPS = (
    "render.rgb",
    "render.depth.expected_z",
    "render.depth.accumulated",
    "gradient.screen.signed",
    "training.camera",
    "sugar.mesh_binding",
)


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - {"alpha_cap"}
        if unknown:
            raise ValueError(f"Unknown renderer options: {sorted(unknown)}")
        self.alpha_cap = float(spec.config.get("alpha_cap", 0.999))
        self.capabilities = set(CAPS)
        self.ctx = new_context()
        self.ctx.renderer = self

    def gaussians(self, params):
        """Activated means, quats, scales of the current representation."""
        mesh = self.ctx.mesh
        if mesh is None:
            return params["means"], params["quats"], params["scales"].exp()
        return bound_gaussians(
            mesh.verts, mesh.faces, mesh.bary, params["sugar_rot2d"], params["scales"][:, 1:],
            mesh.log_thickness,
        )

    def render(self, params, camera, sh_degree=3):
        means, quats, scales = self.gaussians(params)
        if self.ctx.mesh is not None:
            sh_degree = min(sh_degree, self.ctx.sh_cap)
        opacities = params["opacities"].sigmoid()
        # gsplat's backward drops opacity/geometry gradients where opac * G > 0.999, the Inria
        # rasterizer used by SuGaR does not; SuGaR drives opacities to 1 (entropy loss, opaque
        # surface Gaussians). Straight-through cap: forward ~unchanged, gradients kept.
        capped = opacities.clamp(max=self.alpha_cap)
        capped = opacities + (capped - opacities).detach()
        render, alpha, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=capped,
            colors=torch.cat((params["sh0"], params["shN"]), dim=1),
            viewmats=camera["viewmat"][None],
            Ks=camera["K"][None],
            width=camera["width"],
            height=camera["height"],
            sh_degree=sh_degree,
            packed=False,
            render_mode="RGB+D",
        )
        accumulated = render[0, ..., 3:4]
        a = alpha[0]
        return {
            "rgb": render[0, ..., :3] + (1 - a),  # runtime convention: white background
            "depth": accumulated / a.clamp_min(1e-10),
            "depth_accumulated": accumulated,
            "alpha": a,
            "opacities": opacities,
            "gaussians": (means, quats, scales),
            "camera": camera,
            "info": info,
        }


def create_renderer(spec):
    return Renderer(spec)
