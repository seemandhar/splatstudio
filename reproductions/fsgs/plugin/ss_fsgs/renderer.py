"""gsplat renderer with the official FSGS rasterizer conventions.

* depth = sum_i z_i alpha_i T_i (paper Eq. 7; official diff-gaussian-rasterization-confidence
  forward.cu:361), i.e. gsplat render_mode "RGB+D" (accumulated, not alpha-normalised);
* black background (official `white_background=False`), rendered RGB not clamped in training
  (official gaussian_renderer/__init__.py returns the raw rasterizer image);
* SH schedule: official train.py raises the degree every 500 iterations before rendering
  iteration t (degree = min(t // 500, 3)); the runtime passes min((step-1)//1000, 3). The renderer
  counts its training (autograd) renders through `render()` — exactly one per native step — and
  uses the official degree. Pseudo-view renders go through `rasterize` directly with the same
  degree (outputs["sh_degree"]) and are not counted. Renders without autograd use degree 3.
* straight-through opacity cap at 0.999: gsplat's backward drops opacity/conic/mean gradients
  where opac * G > 0.999 (RasterizeToPixels3DGSBwd.cu:222), the Inria rasterizer clamps alpha in
  the forward pass only and always back-propagates (TODO-runtime.md, found via DropGaussian).
"""

import torch
from gsplat import rasterization

CAPABILITIES = {
    "render.rgb",
    "render.depth.accumulated_z",
    "render.camera",
    "gradient.screen.signed",
}
BACKGROUNDS = {"black": 0.0, "white": 1.0}
ALPHA_CAP = 0.999
DEFAULTS = {"background": "black", "sh_interval": 500, "max_sh_degree": 3, "eval_sh_degree": 3,
            "st_alpha_clamp": True}


def rasterize(params, camera, sh_degree, background=0.0, st_alpha_clamp=True):
    opacities = params["opacities"].sigmoid()
    if st_alpha_clamp:
        capped = opacities.clamp(max=ALPHA_CAP)
        opacities = opacities + (capped - opacities).detach()
    colors = torch.cat((params["sh0"], params["shN"]), dim=1)
    render, alpha, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=params["scales"].exp(),
        opacities=opacities,
        colors=colors,
        viewmats=camera["viewmat"][None],
        Ks=camera["K"][None],
        width=int(camera["width"]),
        height=int(camera["height"]),
        sh_degree=sh_degree,
        packed=False,
        render_mode="RGB+D",
        rasterize_mode="classic",
    )
    return {
        "rgb": render[0, ..., :3] + (1 - alpha[0]) * background,
        "depth": render[0, ..., 3:4],
        "alpha": alpha[0],
        "info": info,
    }


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown FSGS renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        if cfg["background"] not in BACKGROUNDS:
            raise ValueError("background must be black or white")
        self.background = BACKGROUNDS[cfg["background"]]
        self.sh_interval = int(cfg["sh_interval"])
        self.max_sh_degree = int(cfg["max_sh_degree"])
        self.eval_sh_degree = int(cfg["eval_sh_degree"])
        self.st_alpha_clamp = bool(cfg["st_alpha_clamp"])
        self.training_renders = 0
        self.capabilities = set(CAPABILITIES)

    def degree(self):
        if torch.is_grad_enabled():
            self.training_renders += 1
            return min(self.training_renders // self.sh_interval, self.max_sh_degree)
        return self.eval_sh_degree

    def render(self, params, camera, sh_degree=3):
        degree = self.degree()  # runtime's degree replaced by the official schedule
        out = rasterize(params, camera, degree, self.background, self.st_alpha_clamp)
        out.update(camera=camera, sh_degree=degree, background=self.background,
                   st_alpha_clamp=self.st_alpha_clamp)
        return out


def create_renderer(spec):
    return Renderer(spec)
