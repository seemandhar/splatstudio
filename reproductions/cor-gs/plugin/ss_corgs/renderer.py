"""gsplat renderer with the official CoR-GS (FSGS-fork) rasterizer conventions.

* `--random_background` (scripts/run_llff.sh; train.py:146): every training iteration draws
  bg = torch.rand(3) and uses it for the training-view render of both fields and for both
  pseudo-view renders; evaluation renders use the black background (render.py:61,
  white_background=False). The renderer draws bg once per training (autograd) render and
  publishes it as outputs["background"] for the loss components.
* SH schedule: official train.py:121 raises the degree every 500 iterations before rendering
  iteration t -> degree = min(t // 500, 3). The runtime passes min((step-1)//1000, 3); the
  renderer counts its training renders (exactly one per native step) and uses the official
  degree. Renders without autograd (evaluation) use degree 3 (render.py loads the ply with
  active_sh_degree = max).
* depth = sum_i z_i alpha_i T_i (diff-gaussian-rasterization-confidence), gsplat "RGB+D".
* straight-through opacity cap at 0.999: gsplat's backward drops gradients where
  opac * G > 0.999 (RasterizeToPixels3DGSBwd.cu:222), the Inria rasterizer clamps alpha in the
  forward pass only (TODO-runtime.md, found via DropGaussian).
Adapted from the FSGS plugin renderer (reproductions/fsgs/plugin/ss_fsgs/renderer.py).
"""

import torch
from gsplat import rasterization

CAPABILITIES = {
    "render.rgb",
    "render.depth.accumulated_z",
    "render.camera",
    "gradient.screen.signed",
}
ALPHA_CAP = 0.999
DEFAULTS = {"train_background": "random", "eval_background": "black", "sh_interval": 500,
            "max_sh_degree": 3, "eval_sh_degree": 3, "st_alpha_clamp": True}
FIXED = {"black": 0.0, "white": 1.0}


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
            raise ValueError(f"Unknown CoR-GS renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        if cfg["train_background"] not in ("random", *FIXED):
            raise ValueError("train_background must be random, black or white")
        if cfg["eval_background"] not in FIXED:
            raise ValueError("eval_background must be black or white")
        self.train_background = cfg["train_background"]
        self.eval_background = FIXED[cfg["eval_background"]]
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

    def background(self, device):
        if not torch.is_grad_enabled():
            return self.eval_background
        if self.train_background == "random":
            return torch.rand(3, device=device)  # official: torch.rand((3), device="cuda")
        return FIXED[self.train_background]

    def render(self, params, camera, sh_degree=3):
        degree = self.degree()  # runtime's degree replaced by the official schedule
        bg = self.background(params["means"].device)
        out = rasterize(params, camera, degree, bg, self.st_alpha_clamp)
        out.update(camera=camera, sh_degree=degree, background=bg,
                   st_alpha_clamp=self.st_alpha_clamp)
        return out


def create_renderer(spec):
    return Renderer(spec)
