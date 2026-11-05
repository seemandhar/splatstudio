"""gsplat renderer with AD-GS's background convention and accumulated depth.

* depth = sum_i z_i alpha_i T_i (paper §3.3.1(b); CoR-GS rasterizer forward.cu:361), i.e.
  gsplat render_mode "RGB+D", not alpha-normalised expected depth;
* training renders use a fresh uniform random background each call (official
  `--random_background`: one random colour per iteration shared by all renders of that
  iteration — see `background` in the outputs, reused by the co-training loss);
* renders without autograd (evaluation, `ss-eval`) use the official test background (black).
* iteration 1 — SH schedule: the official loop raises the SH degree every 500 iterations
  (train.py:191, `oneupSHdegree` before rendering iteration t, so degree = min(t // 500, 3)).
  The runtime passes min((step-1)//1000, 3); the renderer counts its autograd renders (exactly
  one per training step in the native trainer) and uses the official degree instead. Renders
  without autograd use the final degree `eval_sh_degree` (3, reached at step 1500).
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


def rasterize(params, camera, sh_degree, background):
    colors = torch.cat((params["sh0"], params["shN"]), dim=1)
    render, alpha, info = rasterization(
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
        config = dict(spec.config)
        self.train_background = config.pop("train_background", "random")
        self.eval_background = config.pop("eval_background", "black")
        self.sh_interval = int(config.pop("sh_interval", 500))
        self.max_sh_degree = int(config.pop("max_sh_degree", 3))
        self.eval_sh_degree = int(config.pop("eval_sh_degree", 3))
        self.training_renders = 0
        if config:
            raise ValueError(f"Unknown renderer options: {sorted(config)}")
        if self.train_background not in {"random", *BACKGROUNDS}:
            raise ValueError("train_background must be random, black or white")
        if self.eval_background not in BACKGROUNDS:
            raise ValueError("eval_background must be black or white")
        self.capabilities = set(CAPABILITIES)

    def background(self, device):
        if torch.is_grad_enabled():
            if self.train_background == "random":
                return torch.rand(3, device=device)
            return torch.full((3,), BACKGROUNDS[self.train_background], device=device)
        return torch.full((3,), BACKGROUNDS[self.eval_background], device=device)

    def degree(self):
        if torch.is_grad_enabled():
            self.training_renders += 1
            return min(self.training_renders // self.sh_interval, self.max_sh_degree)
        return self.eval_sh_degree

    def render(self, params, camera, sh_degree=3):
        sh_degree = self.degree()  # runtime's degree replaced by the official schedule
        bg = self.background(params["means"].device)
        out = rasterize(params, camera, sh_degree, bg)
        out.update(camera=camera, background=bg, sh_degree=sh_degree)
        return out


def create_renderer(spec):
    return Renderer(spec)
