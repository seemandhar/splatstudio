"""CoherentGS renderer: pixel/ray-structured Gaussians (Eq. 3) and multisampling (Sec. 4.2).

Training render (grad enabled, bank active): means = g(D_init + DeltaD, p) and opacity =
sigmoid(logit(alpha0) + DeltaAlpha) come from the decoders for all N*H*W Gaussians; DC colour is
the fixed pixel colour (official training_setup never optimises f_dc), so sh0 is detached.
The SH degree follows the official schedule (bank.sh_degree), not the runtime's. For official
iteration > 12000 the view is rendered at s = randint(1, 4) times the resolution (same FoV: K
scaled by s) and image, accumulated depth and alpha are AvgPool2d(s)-ed, as train.py does.
The derived means / opacity logits are copied (detached) into params on every training render,
so checkpoints hold the actual Gaussians and evaluation renders params alone.

Evaluation render (no grad, or no bank): plain gsplat render of params at full resolution.

Depth: gsplat "RGB+D" accumulated camera z; the official loss uses depth / (alpha + 1e-6) after
pooling (train.py), exposed as outputs["coherent"]["disparity"] = 1 / (that + 1).

gsplat's backward drops opacity/geometry gradients where opac * G > 0.999 (TODO-runtime.md);
the Inria rasterizer always back-propagates. Opacities given to gsplat are capped at 0.999 with
a straight-through gradient (same fix as the DropGaussian port).
"""

import torch
import torch.nn.functional as F
from gsplat import rasterization

from . import bank as bank_mod

CAPABILITIES = {"render.rgb", "render.depth.expected_z", "render.coherent.xyz",
                "render.coherent.disparity"}
DEFAULTS = {"alpha_cap": 0.999, "near_plane": 0.01}


def scaled_camera(camera, s):
    if s == 1:
        return camera
    K = camera["K"].clone()
    K[:2] *= s  # fx, cx, fy, cy: same FoV, s x s sub-pixels per original pixel
    return {**camera, "K": K, "width": int(camera["width"]) * s, "height": int(camera["height"]) * s}


def pool(x, s):
    """AvgPool2d(s) of an HWC tensor."""
    if s == 1:
        return x
    return F.avg_pool2d(x.permute(2, 0, 1)[None], s)[0].permute(1, 2, 0)


def raster(means, quats, log_scales, opacities, colors, camera, sh_degree, alpha_cap, near):
    capped = opacities.clamp(max=alpha_cap)
    opacities = opacities + (capped - opacities).detach()
    render, alpha, info = rasterization(
        means=means, quats=quats, scales=log_scales.exp(), opacities=opacities, colors=colors,
        viewmats=camera["viewmat"][None], Ks=camera["K"][None], width=int(camera["width"]),
        height=int(camera["height"]), sh_degree=sh_degree, packed=False, render_mode="RGB+D",
        near_plane=near)
    rgb = render[0, ..., :3] + (1 - alpha[0])  # white background (official white_background)
    return rgb, render[0, ..., 3:4], alpha[0], info


class Renderer:
    capabilities = CAPABILITIES

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        self.alpha_cap = float(cfg["alpha_cap"])
        self.near = float(cfg["near_plane"])

    def render(self, params, camera, sh_degree=3):
        bank = bank_mod.ACTIVE["bank"]
        if bank is None or not torch.is_grad_enabled():
            return self.render_params(params, camera, sh_degree)
        return self.render_train(params, camera, bank)

    def render_params(self, params, camera, sh_degree):
        colors = torch.cat((params["sh0"], params["shN"]), dim=1)
        rgb, d, alpha, info = raster(params["means"], params["quats"], params["scales"],
                                     params["opacities"].sigmoid(), colors, camera, sh_degree,
                                     self.alpha_cap, self.near)
        depth = d / (alpha + 1e-6)
        return {"rgb": rgb, "depth": depth, "alpha": alpha, "info": info}

    def render_train(self, params, camera, bank):
        step = bank.step
        if step is None:
            raise RuntimeError("CoherentGS bank has no step; density.initialize was not run")
        view = bank.view_index(camera["viewmat"])
        means, logits = bank.gaussians()
        if means.shape != params["means"].shape:
            raise RuntimeError("CoherentGS: params population differs from the pixel structure")
        with torch.no_grad():  # persistence: checkpoint holds the derived Gaussians
            params["means"].copy_(means)
            params["opacities"].copy_(logits)
        s = bank.samples(step)
        colors = torch.cat((params["sh0"].detach(), params["shN"]), dim=1)
        rgb, d, alpha, info = raster(means, params["quats"], params["scales"], logits.sigmoid(),
                                     colors, scaled_camera(camera, s), bank.sh_degree(step),
                                     self.alpha_cap, self.near)
        rgb, d, alpha = pool(rgb, s), pool(d, s), pool(alpha, s)
        depth = d / (alpha + 1e-6)  # official ren_depth = depth / (alpha + 1e-6)
        disparity = (1 / (depth + 1)).permute(2, 0, 1)  # [1, H, W]
        return {"rgb": rgb, "depth": depth, "alpha": alpha, "info": info, "camera": camera,
                "coherent": {"bank": bank, "view": view, "xyz": means, "disparity": disparity,
                             "samples": s}}


def create_renderer(spec):
    return Renderer(spec)
