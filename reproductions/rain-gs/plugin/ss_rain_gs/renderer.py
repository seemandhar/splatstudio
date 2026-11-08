"""gsplat renderer with RAIN-GS progressive Gaussian low-pass filter control.

Paper §4.2 / supp. A.1.2, Alg. 2 line 8; official train.py (cvlab-kaist/RAIN-GS, MIT):

    if iteration == 1 or (iteration % c2f_every_step == 0 and iteration < densify_until_iter):
        low_pass = min(max(H * W / N / (9 * pi), 0.3), 300)

H, W are the sampled training view's size and N the current number of Gaussians; between
updates the previous value is kept. The value is the `s` of Eq. 2 (Sigma' + sI), which is
gsplat's `eps2d` in "classic" mode (the Inria rasterizer's `low_pass`). Test renders
(no autograd) use the official default 0.3.

Also the RAIN-GS SH schedule (official train.py:84, `--ours_new`): one SH degree up every 1000
steps starting at iteration 5000 (degree 1 at 5000, 3 at 7000), capped by the runtime degree.

The training step is the 1-based count of autograd renders (the native runtime renders once per
step); the density component checks it against the runtime step every iteration.
"""

import math

import torch
from gsplat import rasterization

CAPABILITIES = {"render.rgb", "render.depth.expected_z", "gradient.screen.signed"}
BACKGROUNDS = {"black": 0.0, "white": 1.0}
DEFAULTS = {
    "progressive_lowpass": True,
    "lowpass_min": 0.3,
    "lowpass_max": 300.0,
    "lowpass_every": 1000,
    "lowpass_until": 25000,  # densify_until_iter (15000) + warmup_iter (10000)
    "eval_lowpass": 0.3,
    "sh_start": 5000,
    "sh_every": 1000,
    "background": "black",  # official default (white_background False)
}


def lowpass_value(height, width, n, lo=0.3, hi=300.0):
    """min(max(HW / (9 pi N), 0.3), 300) (Alg. 2 l.8; train.py `low_pass`)."""
    return min(max(height * width / n / (9 * math.pi), lo), hi)


def lowpass_due(step, every=1000, until=25000):
    """train.py: `iteration == 1 or (iteration % c2f_every_step == 0 and iteration < until)`."""
    return step == 1 or (step % every == 0 and step < until)


def sh_degree_at(step, start=5000, every=1000, max_degree=3):
    """oneupSHdegree() at iterations start, start+every, ... before rendering."""
    if step < start:
        return 0
    return min((step - start) // every + 1, max_degree)


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown rain-gs renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        if cfg["background"] not in BACKGROUNDS:
            raise ValueError("background must be black or white")
        self.cfg = cfg
        self.background = BACKGROUNDS[cfg["background"]]
        self.capabilities = set(CAPABILITIES)
        self.step = 0  # number of training (autograd) renders so far
        self.lowpass = float(cfg["lowpass_min"])
        self.history = []  # (step, eps2d) at each recomputation

    def training_lowpass(self, height, width, n):
        """Advance the training counter and return eps2d for this render."""
        cfg = self.cfg
        self.step += 1
        if not cfg["progressive_lowpass"]:
            self.lowpass = 0.3
        elif lowpass_due(self.step, cfg["lowpass_every"], cfg["lowpass_until"]):
            self.lowpass = lowpass_value(height, width, n, cfg["lowpass_min"], cfg["lowpass_max"])
            self.history.append((self.step, self.lowpass))
        return self.lowpass

    def render(self, params, camera, sh_degree=3):
        width, height = int(camera["width"]), int(camera["height"])
        if torch.is_grad_enabled():
            eps2d = self.training_lowpass(height, width, len(params["means"]))
            degree = min(sh_degree, sh_degree_at(self.step, self.cfg["sh_start"],
                                                 self.cfg["sh_every"]))
        else:
            eps2d = float(self.cfg["eval_lowpass"])
            degree = sh_degree
        colors = torch.cat((params["sh0"], params["shN"]), dim=1)
        render, alpha, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"].exp(),
            opacities=params["opacities"].sigmoid(),
            colors=colors,
            viewmats=camera["viewmat"][None],
            Ks=camera["K"][None],
            width=width,
            height=height,
            sh_degree=degree,
            packed=False,
            render_mode="RGB+ED",
            eps2d=eps2d,
            rasterize_mode="classic",
        )
        info["rain_step"] = self.step
        info["rain_eps2d"] = eps2d
        info["rain_sh_degree"] = degree
        return {
            "rgb": render[0, ..., :3] + (1 - alpha[0]) * self.background,
            "depth": render[0, ..., 3:4],
            "alpha": alpha[0],
            "info": info,
        }


def create_renderer(spec):
    return Renderer(spec)
