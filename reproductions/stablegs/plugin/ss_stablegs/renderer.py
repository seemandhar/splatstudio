"""StableGS Dual Opacity renderer (paper Sec. 3.3; official train_pair.py `render_rade` calls).

Geometric opacity σ uses α = sigmoid(opacities) only; rendering opacity σ' = α'·σ with
α' = sigmoid(opacity_aux). Training (gradients enabled and a pair published by the density):
for both views of the pair
  * appearance path: opacity α·α' if `dual` (official: iteration > 15000) else α,
    require_depth False  (train_pair.py:121-122);
  * geometric path (if `geo`, official: iteration > 15000 with --single_loss): opacity α,
    require_depth True -> expected depth, median depth, normal  (train_pair.py:136-137).
The runtime's own camera is ignored in that case (the pair stack drives sampling).
Evaluation: the given camera, appearance path α·α' (official render1.py dual_alpha=True).
Background black (official white_background False); kernel_size 0 (no 2D filter; the release
lacks render_rade, the RaDe-GS default is 0).
"""

import torch
import torch.nn.functional as F

from .link import LINK
from .raster import raster_settings, rasterize

PAIR_CAPABILITY = "render.pair_batch"
CAPABILITIES = {
    "render.rgb", "render.depth", "render.depth.median", "render.alpha", "render.normal.camera",
    "gradient.screen.gof", PAIR_CAPABILITY,
}
DEFAULTS = {"background": [0.0, 0.0, 0.0], "kernel_size": 0.0, "eval_dual": True}


def activations(params):
    return {
        "means": params["means"],
        "scales": params["scales"].exp(),
        "rotations": F.normalize(params["quats"], dim=-1),
        "alpha": params["opacities"].sigmoid()[:, None],
        "alpha_aux": params["opacity_aux"].sigmoid()[:, None],
        "sh": torch.cat((params["sh0"], params["shN"]), dim=1),
    }


def render_path(act, camera, sh_degree, background, kernel_size, dual, require_depth):
    """One rasterization. dual: opacity α·α' (appearance) else α (geometric / pre-dual)."""
    opac = act["alpha"] * act["alpha_aux"] if dual else act["alpha"]
    settings = raster_settings(camera, background, sh_degree, kernel_size, require_depth)
    maps, means2d = rasterize(act["means"], opac, act["scales"], act["rotations"], act["sh"],
                              settings)
    return {
        "rgb": maps["color"].permute(1, 2, 0),
        "depth": maps["depth"][0][..., None],
        "median_depth": maps["median_depth"][0][..., None],
        "alpha": maps["alpha"][0][..., None],
        "normal": maps["normal"].permute(1, 2, 0),
        "means2d": means2d,
        "radii": maps["radii"],
    }


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown stablegs-raster options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        self.background = [float(v) for v in cfg["background"]]
        self.kernel_size = float(cfg["kernel_size"])
        self.eval_dual = bool(cfg["eval_dual"])
        self.capabilities = set(CAPABILITIES)
        self._bg = {}

    def _bgt(self, dev):
        if dev not in self._bg:
            self._bg[dev] = torch.tensor(self.background, dtype=torch.float32, device=dev)
        return self._bg[dev]

    def render(self, params, camera, sh_degree=3):
        bg = self._bgt(params["means"].device)
        batch = LINK.batch() if torch.is_grad_enabled() else None
        act = activations(params)
        if batch is None:
            out = render_path(act, camera, sh_degree, bg, self.kernel_size, self.eval_dual, True)
            out["info"] = {"means2d": [out.pop("means2d")], "radii": [out.pop("radii")]}
            return out
        views = []
        for v in batch["views"]:
            app = render_path(act, v["camera"], sh_degree, bg, self.kernel_size, batch["dual"],
                              False)
            geo = None
            if batch["geo"]:
                geo = render_path(act, v["camera"], sh_degree, bg, self.kernel_size, False, True)
            views.append({"app": app, "geo": geo, "camera": v["camera"], "target": v["target"],
                          "index": v["index"]})
        first = views[0]["app"]
        return {
            "rgb": first["rgb"],
            "depth": (views[0]["geo"] or first)["depth"],
            "alpha": first["alpha"],
            "info": {"means2d": [v["app"]["means2d"] for v in views],
                     "radii": [v["app"]["radii"] for v in views]},
            "pair": {"views": views, "in_P": batch["in_P"], "dual": batch["dual"],
                     "geo": batch["geo"], "step": batch["step"], "indices": batch["indices"]},
        }


def create_renderer(spec):
    return Renderer(spec)
