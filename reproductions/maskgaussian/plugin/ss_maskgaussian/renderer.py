"""MaskGaussian renderer: Gumbel-Softmax mask sampling + the official masked rasterizer.

Training (grad enabled): logits = mask_scores + score_offset (official init [10, 1]);
  M = gumbel_softmax(log_softmax(logits), tau, hard=True)[:, 0:1]  (gaussian_model.get_mask,
  straight-through: forward one-hot, backward through the soft sample); the kernel blends with
  C = sum M c alpha T, T_{i+1} = (1 - M alpha) T (Eq. 3-4) and returns dL/dM (Eq. 7 + bg term).
Evaluation (no grad): M = 1 for every Gaussian, which is the stock 3D-GS rasterizer; the
  density component has already reduced the model to one mask sample at the final step
  (official save_ply_default + render_orig_3dgs).

outputs["info"]: `mask` [N,1] (ST sample with graph, consumed by the mask loss), `mask_hard`
[N] bool, `means2d` [N,3] (retained grad), `radii` [N], `num_rendered`, width/height.
"""

import torch
import torch.nn.functional as F

from .raster import ACCUM, rasterize

CAPABILITIES = {"render.rgb", "gradient.screen.signed"}
FIELD = "mask_scores"
MASK_MODES = ("raster", "opacity")
DEFAULTS = {
    "background": [0.0, 0.0, 0.0],  # official white_background False
    "score_offset": [10.0, 1.0],  # official init: ones @ [[10, 0], [0, 1]]
    "tau": 1.0,  # F.gumbel_softmax default
    "accum": "official",  # "exact": supplementary recursion (see README)
    "mask_mode": "raster",  # "opacity": paper ablation "Mask x opacity"
}


def mask_logits(scores, offset):
    return scores + torch.as_tensor(offset, dtype=scores.dtype, device=scores.device)


def sample_mask(scores, offset, tau=1.0):
    """gaussian_model.get_mask: hard Gumbel-Softmax over the 2 scores, channel 0 = present."""
    logits = mask_logits(scores, offset)
    return F.gumbel_softmax(F.log_softmax(logits, dim=-1), tau=tau, hard=True)[:, 0:1]


def presence_probability(scores, offset):
    return F.softmax(mask_logits(scores, offset), dim=-1)[:, 0]


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown MaskGaussian renderer options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        if cfg["accum"] not in ACCUM:
            raise ValueError(f"accum must be one of {ACCUM}")
        if cfg["mask_mode"] not in MASK_MODES:
            raise ValueError(f"mask_mode must be one of {MASK_MODES}")
        if float(cfg["tau"]) <= 0 or len(cfg["score_offset"]) != 2:
            raise ValueError("tau must be > 0 and score_offset must have 2 entries")
        self.cfg = cfg
        self.background = [float(v) for v in cfg["background"]]
        self.capabilities = set(CAPABILITIES)
        self._bg = {}

    def background_tensor(self, device):
        if device not in self._bg:
            self._bg[device] = torch.tensor(self.background, dtype=torch.float32,
                                            device=device)
        return self._bg[device]

    def render(self, params, camera, sh_degree=3, masks=None):
        """``masks`` overrides the sampled mask (probes); default: sample when training."""
        cfg = self.cfg
        means = params["means"]
        bg = self.background_tensor(means.device)
        if masks is None:
            if torch.is_grad_enabled():
                masks = sample_mask(params[FIELD], cfg["score_offset"], float(cfg["tau"]))
            else:
                masks = torch.ones((len(means), 1), dtype=means.dtype, device=means.device)
        color, means2d, radii, stats = rasterize(
            params, camera, sh_degree, bg, masks, cfg["accum"],
            opacity_mask=cfg["mask_mode"] == "opacity")
        return {
            "rgb": color.permute(1, 2, 0),
            "info": {"mask": masks, "mask_hard": masks.detach()[:, 0] > 0.5,
                     "means2d": means2d, "radii": radii, **stats,
                     "width": int(camera["width"]), "height": int(camera["height"])},
        }


def create_renderer(spec):
    return Renderer(spec)
