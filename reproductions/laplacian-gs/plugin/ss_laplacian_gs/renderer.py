"""Laplacian-GS renderer: per-level fields, signed residual rendering, image-domain composition.

Official sources: gaussian_renderer/__init__.py (render_size, clamps), submodules/*_pn
(forward.cu: residual colours without `+0.5` and without the max(.,0) clamp),
render_pn.py (evaluation composition), train_lap.py (training composition with the cache).
"""

import torch
from gsplat import rasterization, spherical_harmonics

from . import pyramid as P
from .link import LINK

LEVEL_CAPABILITY = "render.laplacian_level"
CAPABILITIES = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed",
                LEVEL_CAPABILITY)
LEVEL_FIELD = "lap_level"
RENDER_DEFAULTS = {"levels": 2, "iter_plan": [10000, 20000], "blur_up": True,
                   "base_background": "black"}


def check_hierarchy(levels, iter_plan):
    levels = int(levels)
    plan = [int(t) for t in iter_plan]
    if levels < 2 or len(plan) != levels or min(plan) < 1:
        raise ValueError("laplacian-gs needs levels >= 2 and one positive iteration count per level")
    return levels, plan


def stage_level(stage, levels):
    """Stage 0 trains the coarsest level L-1, the last stage trains level 0."""
    return levels - 1 - stage


def final_degree(iterations):
    """SH degree after a stage: oneupSHdegree at every 1000-th stage-local iteration (max 3)."""
    return min(int(iterations) // 1000, 3)


def level_camera(camera, size):
    h, w = size
    if (h, w) == (camera["height"], camera["width"]):
        return camera
    K = camera["K"].clone()
    K[0] *= w / camera["width"]
    K[1] *= h / camera["height"]
    return {**camera, "K": K, "width": w, "height": h}


def _subset(params, mask, detach):
    out = {}
    for k in ("means", "quats", "scales", "opacities", "sh0", "shN"):
        v = params[k].detach() if detach else params[k]
        out[k] = v if mask is None else v[mask]
    return out


def render_field(g, camera, degree, signed, background):
    """Render one field. signed=False: Eq. 5 base colours; signed=True: residual (no shift)."""
    coeffs = torch.cat((g["sh0"], g["shN"]), dim=1)
    common = dict(
        means=g["means"], quats=g["quats"], scales=g["scales"].exp(),
        opacities=g["opacities"].sigmoid(), viewmats=camera["viewmat"][None],
        Ks=camera["K"][None], width=camera["width"], height=camera["height"], packed=False,
        render_mode="RGB+ED", rasterize_mode="classic",
    )
    if signed:
        campos = torch.linalg.inv(camera["viewmat"])[:3, 3]
        n_coeffs = (degree + 1) ** 2
        colors = spherical_harmonics(degree, g["means"] - campos, coeffs[:, :n_coeffs])
        render, alpha, info = rasterization(colors=colors, sh_degree=None, **common)
        rgb = render[0, ..., :3].clamp(-1.0, 1.0)  # gaussian_renderer: clamp(-1, 1)
    else:
        render, alpha, info = rasterization(colors=coeffs, sh_degree=degree, **common)
        rgb = render[0, ..., :3] + background * (1 - alpha[0])
        rgb = rgb.clamp(0.0, 1.0)  # gaussian_renderer: clamp(0, 1)
    return {"rgb": rgb, "depth": render[0, ..., 3:4], "alpha": alpha[0], "info": info}


class Renderer:
    capabilities = set(CAPABILITIES)

    def __init__(self, spec):
        unknown = set(spec.config) - set(RENDER_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown lap-renderer options: {sorted(unknown)}")
        c = {**RENDER_DEFAULTS, **spec.config}
        self.levels, self.plan = check_hierarchy(c["levels"], c["iter_plan"])
        self.blur_up = bool(c["blur_up"])
        if c["base_background"] not in ("white", "black"):
            raise ValueError("base_background must be 'white' or 'black'")
        self.background = 1.0 if c["base_background"] == "white" else 0.0
        # Final SH degree of each level (index = level), used for archived fields and eval.
        self.degrees = [0] * self.levels
        for s, t in enumerate(self.plan):
            self.degrees[stage_level(s, self.levels)] = final_degree(t)

    def _level_mask(self, params, level):
        return params[LEVEL_FIELD].detach() == float(level)

    def _render_level(self, params, camera, sizes, level, degree, detach):
        mask = self._level_mask(params, level)
        if not bool(mask.any()):
            return None
        cam = level_camera(camera, sizes[level])
        out = render_field(_subset(params, mask, detach), cam, degree,
                           signed=level < self.levels - 1, background=self.background)
        out["mask"] = mask
        return out

    @torch.no_grad()
    def compose(self, params, camera, down_to=0, sizes=None):
        """Eq. 7 from the coarsest level down to `down_to` with frozen (archived) fields."""
        sizes = sizes or P.level_sizes(camera["height"], camera["width"], self.levels)
        top = self.levels - 1
        base = self._render_level(params, camera, sizes, top, self.degrees[top], True)
        if base is None:
            raise RuntimeError("laplacian-gs: no base-level Gaussians in the state")
        image = base["rgb"]
        for level in range(top - 1, down_to - 1, -1):
            image = P.up(image, sizes[level], self.blur_up)
            res = self._render_level(params, camera, sizes, level, self.degrees[level], True)
            if res is not None:
                image = image + res["rgb"]
        return image

    def render(self, params, camera, sh_degree=3):
        sizes = P.level_sizes(camera["height"], camera["width"], self.levels)
        if not torch.is_grad_enabled():
            # Evaluation: all fields rendered separately and composed (render_pn.py).
            rgb = self.compose(params, camera, 0, sizes)
            return {"rgb": rgb, "depth": torch.zeros_like(rgb[..., :1]),
                    "alpha": torch.ones_like(rgb[..., :1]), "info": {}, "lap_level": 0}
        level, local_step = LINK.training_stage()
        degree = final_degree(local_step)  # stage-local SH schedule (new field starts at 0)
        out = self._render_level(params, camera, sizes, level, degree, False)
        if out is None:
            raise RuntimeError(f"laplacian-gs: no Gaussians at the active level {level}")
        if level < self.levels - 1:
            # Cached Up(I^{l+1}) of the archived fields (train_lap.py stage cache), Eq. 7.
            cache = P.up(self.compose(params, camera, level + 1, sizes), sizes[level],
                         self.blur_up)
            out["rgb"] = cache + out["rgb"]
        out["active_ids"] = torch.where(out.pop("mask"))[0]
        out["lap_level"] = level
        out["sh_degree"] = degree
        return out


def create_renderer(spec):
    return Renderer(spec)
