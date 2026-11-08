"""OMG renderer: gsplat rasterisation of either the per-Gaussian state or the OMG attribute
representation.

- Before net_itr (Mini-Splatting phase), after the final bake, and in a fresh process
  (ss-eval, no bank): the stock gsplat renderer on the base fields (sh0/shN/opacities/scales/
  quats). This is the official `render_imp` with net_enabled False, and the decoded model of
  `decode()`.
- From net_itr until the bake (live bank, net enabled), following the official render_imp net
  branch:
  - sh0 = MLP_t([F,T]), opacity = sigmoid(MLP_o([F,T])), shN = MLP_v([F,V]);
  - F = MLP_s(gamma(contract(means.detach()))).
  - From svq_itr on, scale, rotation and [T,V] come from the SVQ codebooks at the frozen
    indices (get_svq_scale / get_svq_rotation / get_svq_appearance).
  - SH degree follows the official schedule (0 until 15k, +1 per 1000 steps), not the runtime's.

The frame's target image is also forwarded, because Mini-Splatting's depth re-initialisation
colours points with ground-truth pixels (copied from our Mini-Splatting port's renderer).
"""

import torch
from gsplat import rasterization
from splatstudio.backends.gsplat import Renderer

from . import core, model

TARGET_CAPABILITY = "training.target_image"
STATIC = "omg_static"
VIEW = "omg_view"


def live_attributes(bank, params, use_svq):
    """Activated rendering attributes of the OMG representation (official render_imp)."""
    if use_svq:
        scales = bank.svq_values("scale").exp()
        quats = bank.svq_values("rotation")
        app = bank.svq_values("app")
        static, view = app[:, 0:3], app[:, 3:6]
    else:
        scales = params["scales"].exp()
        quats = params["quats"]
        static, view = params[STATIC], params[VIEW]
    sh0, shn, opacity = bank.appearance(params["means"], static, view)
    return {"means": params["means"], "quats": quats, "scales": scales,
            "opacities": opacity.sigmoid(), "colors": torch.cat([sh0, shn], dim=1)}


class OMGRenderer(Renderer):
    def __init__(self, spec):
        super().__init__(spec)
        self.capabilities = set(self.capabilities) | {TARGET_CAPABILITY}
        self.bank = None  # explicit bank (probes); otherwise model.ACTIVE

    def active_bank(self):
        bank = self.bank if self.bank is not None else model.ACTIVE["bank"]
        if bank is None or bank.baked or not bank.net_enabled:
            return None
        return bank

    def render(self, params, camera, sh_degree=3):
        bank = self.active_bank()
        if bank is None:
            outputs = super().render(params, camera, sh_degree)
        else:
            attrs = live_attributes(bank, params, bank.vq_enabled)
            degree = core.sh_degree(bank.step, bank.net_itr)
            outputs = self._rasterize(attrs, camera, degree)
            outputs["info"]["omg"] = {"svq": bank.vq_enabled, "sh_degree": degree}
        outputs["target_image"] = camera.get("image")
        outputs["viewmat"] = camera["viewmat"]
        return outputs

    def _rasterize(self, a, camera, sh_degree):
        render, alpha, info = rasterization(
            means=a["means"], quats=a["quats"], scales=a["scales"], opacities=a["opacities"],
            colors=a["colors"], viewmats=camera["viewmat"][None], Ks=camera["K"][None],
            width=camera["width"], height=camera["height"], sh_degree=sh_degree, packed=False,
            render_mode="RGB+ED", absgrad=self.absgrad,
            rasterize_mode="antialiased" if self.antialiased else "classic",
        )
        # Runtime convention: RGB over a white background (official: black).
        return {"rgb": render[0, ..., :3] + (1 - alpha[0]), "depth": render[0, ..., 3:4],
                "alpha": alpha[0], "info": info}


def create_renderer(spec):
    return OMGRenderer(spec)
