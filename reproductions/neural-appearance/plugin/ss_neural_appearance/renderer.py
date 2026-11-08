"""Renderer: the vendored Faster-GS CUDA rasterizer (``ss_faster_gs.raster``, reproductions/faster-gs)
+ the paper's appearance model + PPISP post-processing.

Neural (C1): the colour c = ReLU(0.5 + c0 + A(d)) with c0 = Y0 * sh0 is produced by the
rasterizer's degree-0 SH path (``0.5 + Y0 * sh0``, clamped at 0 with the ReLU gradient mask), fed
with ``sh0 + A(d) / Y0``; A(d) is computed in PyTorch for all Gaussians (unfused reference path),
so autograd carries dL/dA to the features, the MLP and (through d) to the means.
SH (paper's baseline row): the rasterizer's own degree-``sh_degree`` SH (3DGS formula, Eq. 1).

Training: PPISP with the view's frame parameters. After ``train_steps`` (controller distillation)
the scene is rendered without gradients (frozen, official ``ppisp_controller_distillation``).
Evaluation (no grad): every view is a novel view -> PPISP controller (frame_idx = -1). If no live
Bank exists (fresh ``ss-eval`` process) it is rebuilt from ``params["na_blob"]``.
"""

import torch
from ss_faster_gs.raster import rasterize

from . import bank as bank_mod
from .appearance import DEFAULTS as APPEARANCE_DEFAULTS
from .appearance import SH_C0, neural_residual

CAPABILITIES = {"render.rgb", "appearance.neural-ppisp"}
DEFAULTS = {
    "appearance": "neural",  # MODEL.APPEARANCE.TYPE: "neural" | "sh"
    "background": [0.0, 0.0, 0.0],  # DATASET.BACKGROUND_COLOR
    "near": 0.2,  # DATASET.NEAR_PLANE
    "far": 10000.0,  # DATASET.FAR_PLANE
    **APPEARANCE_DEFAULTS,
}


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown neural-appearance renderer options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        if self.cfg["appearance"] not in ("neural", "sh"):
            raise ValueError("appearance must be 'neural' or 'sh'")
        self.capabilities = set(CAPABILITIES)
        self._bg = {}
        self._loaded = (None, None)  # (blob storage pointer, Bank) for evaluation

    def background(self, device):
        if device not in self._bg:
            self._bg[device] = torch.tensor(self.cfg["background"], dtype=torch.float32,
                                            device=device)
        return self._bg[device]

    def bank(self, params):
        if bank_mod.ACTIVE is not None:
            return bank_mod.ACTIVE
        blob = params["na_blob"]
        ptr = blob.data_ptr()
        if self._loaded[0] != ptr:
            self._loaded = (ptr, bank_mod.unpack(blob.detach().reshape(-1), device=blob.device))
        return self._loaded[1]

    def raw_render(self, params, camera, sh_degree, bank):
        means = params["means"]
        campos = torch.linalg.inv(camera["viewmat"].float())[:3, 3]
        p = {k: params[k] for k in ("means", "scales", "quats", "opacities", "shN")}
        if self.cfg["appearance"] == "neural":
            residual = neural_residual(means, params["na_features"], campos, bank.mlp,
                                       sh_degree, self.cfg)
            sh0 = params["sh0"]
            p["sh0"] = sh0 if residual is None else sh0 + (residual / SH_C0)[:, None, :]
            degree = 0
        else:
            p["sh0"] = params["sh0"]
            degree = sh_degree
        stats = {}
        image = rasterize(p, camera, degree, self.background(means.device), None,
                          self.cfg["near"], self.cfg["far"], stats)
        return image.permute(1, 2, 0), stats

    def render(self, params, camera, sh_degree=3):
        bank = self.bank(params)
        training = torch.is_grad_enabled() and params["means"].requires_grad
        if training and bank.distilling():
            with torch.no_grad():
                rgb, stats = self.raw_render(params, camera, sh_degree, bank)
        else:
            rgb, stats = self.raw_render(params, camera, sh_degree, bank)
        rgb = bank.apply_ppisp(rgb, camera, novel=not training)
        return {"rgb": rgb, "info": {"neural_appearance": bank, **stats,
                                     "width": int(camera["width"]),
                                     "height": int(camera["height"])}}


def create_renderer(spec):
    return Renderer(spec)
