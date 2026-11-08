"""gsplat renderer with Trick-GS masks and progressive schedules.

- GM (paper Eq. 3, Compact-3DGS): M = sg(1[sigmoid(m) > eps_m] - sigmoid(m)) + sigmoid(m)
  multiplies the activated scale and opacity.
- SHM (paper Eq. 5, RDO-Gaussian): the same STE per SH degree l=1..3 (eps_sh) multiplies the
  2l+1 coefficients of that degree; active from `sh_mask_start` during training, always at test.
- DS (Eq. 6): training renders at the scheduled size with K scaled per axis (same FoV).
- SC (§3.2.4c, RAIN-GS): eps2d (the s of Sigma' + sI) = max(HW/(9 pi N), 0.3), recomputed at
  step 1 and every 1000 steps until 10000 from the training render size; 0.3 afterwards/at test.
Schedules follow `shared.STATE` (see shared.py).
"""

import torch
from gsplat import rasterization

from . import schedule
from .shared import STATE

GS_MASK = "trick_mask"
SH_MASK = "trick_sh_mask"
SCHEDULE_CAPABILITY = "render.trick_schedule"
CAPABILITIES = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed",
                SCHEDULE_CAPABILITY)
SH_COUNTS = (3, 5, 7)
DEFAULTS = {"gs_threshold": 0.05, "sh_threshold": 0.1}


def ste_mask(logits, threshold):
    """Eq. 3 / Eq. 5: hard {0,1} forward, sigmoid gradient backward."""
    soft = torch.sigmoid(logits)
    return ((soft > threshold).float() - soft).detach() + soft


def expand_sh(mask3):
    """[N, 3] per-degree values -> [N, 15, 1] per shN coefficient (3/5/7 per degree)."""
    return torch.cat([mask3[:, i : i + 1].expand(-1, c) for i, c in enumerate(SH_COUNTS)],
                     dim=1)[..., None]


def scaled_camera(camera, height, width):
    if (height, width) == (int(camera["height"]), int(camera["width"])):
        return camera
    K = camera["K"].clone()
    K[0] *= width / float(camera["width"])
    K[1] *= height / float(camera["height"])
    return {**camera, "K": K, "width": int(width), "height": int(height)}


def masked_attributes(params, gs_threshold, sh_threshold, sh_active):
    scales = params["scales"].exp()
    opacities = params["opacities"].sigmoid()
    shn = params["shN"]
    if GS_MASK in params:
        m = ste_mask(params[GS_MASK], gs_threshold)
        scales = scales * m[:, None]
        opacities = opacities * m
    if sh_active and SH_MASK in params:
        shn = shn * expand_sh(ste_mask(params[SH_MASK], sh_threshold))
    return scales, opacities, torch.cat((params["sh0"], shn), dim=1)


class Renderer:
    capabilities = set(CAPABILITIES)

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown trick-renderer options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}

    def training_schedule(self, camera, n):
        cfg, step = STATE.cfg, STATE.next_step
        h, w = schedule.ds_size(step, int(camera["height"]), int(camera["width"]),
                                cfg["ds_start"], cfg["ds_until"], cfg["ds_mode"])
        if not cfg["progressive_lowpass"] or step > cfg["lowpass_until"]:
            STATE.lowpass = cfg["lowpass_min"]
        elif schedule.lowpass_due(step, cfg["lowpass_every"], cfg["lowpass_until"]):
            STATE.lowpass = schedule.lowpass_value(h, w, n, cfg["lowpass_min"], cfg["lowpass_max"])
            STATE.lowpass_history.append((step, STATE.lowpass))
        scale = w / float(camera["width"])
        blur = schedule.blur_params(step, scale, cfg["blur_radius"], cfg["blur_sigma"],
                                    cfg["blur_every"], cfg["blur_until"]) if cfg["blur"] else None
        return {"step": step, "height": h, "width": w, "scale": scale, "blur": blur,
                "lowpass": STATE.lowpass, "sh_mask_active": step >= cfg["sh_mask_start"]}

    def render(self, params, camera, sh_degree=3):
        if torch.is_grad_enabled() and STATE.active:
            trick = self.training_schedule(camera, len(params["means"]))
            cam = scaled_camera(camera, trick["height"], trick["width"])
        else:
            trick = {"step": None, "scale": 1.0, "blur": None, "lowpass": 0.3,
                     "sh_mask_active": True}
            cam = camera
        scales, opacities, colors = masked_attributes(
            params, self.cfg["gs_threshold"], self.cfg["sh_threshold"], trick["sh_mask_active"])
        render, alpha, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=cam["viewmat"][None],
            Ks=cam["K"][None],
            width=int(cam["width"]),
            height=int(cam["height"]),
            sh_degree=sh_degree,
            packed=False,
            render_mode="RGB+ED",
            eps2d=float(trick["lowpass"]),
            rasterize_mode="classic",
        )
        return {
            "rgb": render[0, ..., :3] + (1 - alpha[0]),  # dataset convention: white background
            "depth": render[0, ..., 3:4],
            "alpha": alpha[0],
            "info": info,
            "trick": trick,
        }


def create_renderer(spec):
    return Renderer(spec)
