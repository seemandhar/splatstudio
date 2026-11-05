"""SplatFields losses (supp. Eq. A.2): mask loss (lambda_2 = 0.1, object scenes) and splat norm
regularisation (lambda_3 = 0.01, static scenes).

Mask loss, official train.py: `opacity_image = clamp(render_pkg['opacity'], 0, 1)`;
`F.l1_loss(opacity_image.view(-1), gt_mask.view(-1))` with gt_mask = PNG alpha / 255. The
runtime gives a loss only (params, outputs, target) and composites RGBA onto white, so the
ground-truth alpha lives in a sidecar `<data>/masks.npz` (written by
tools/prepare_blender_fewview.py: train-view alpha + the viewmat the native loader computes)
and is looked up by the frame viewmat the renderers return. The data root is taken from
`SPLATFIELDS_DATA` or from ss-train's `--data` argument (bench.py always passes it); a missing
sidecar or an unmatched viewmat raises (no silent fallback).

Splat norm, official train.py: `lambda_norm * means3D.norm(dim=1).mean()` on the predicted
centres p_hat (`outputs["sf_means"]`).
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch


def data_root():
    env = os.environ.get("SPLATFIELDS_DATA")
    if env:
        return Path(env)
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--data" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if a.startswith("--data="):
            return Path(a.split("=", 1)[1])
    raise RuntimeError("mask loss: set SPLATFIELDS_DATA or run through `ss-train --data`")


class MaskTable:
    """GT alpha masks keyed by the native loader's viewmat."""

    def __init__(self, viewmats, masks, tol=1e-4):
        self.viewmats = torch.as_tensor(np.asarray(viewmats), dtype=torch.float32)
        self.masks = torch.as_tensor(np.asarray(masks))
        self.tol = float(tol)
        self.cache = {}

    @classmethod
    def load(cls, root):
        path = Path(root) / "masks.npz"
        if not path.exists():
            raise FileNotFoundError(f"mask loss: {path} not found")
        with np.load(path, allow_pickle=False) as d:
            return cls(d["viewmats"], d["masks"])

    def lookup(self, viewmat, device):
        err = (self.viewmats - viewmat.detach().float().cpu()[None]).abs().amax(dim=(1, 2))
        i = int(err.argmin())
        if float(err[i]) > self.tol:
            raise KeyError(f"mask loss: no mask for this camera (closest err {float(err[i])})")
        key = (i, str(device))
        if key not in self.cache:
            m = self.masks[i]
            m = m.float() / 255.0 if m.dtype == torch.uint8 else m.float()
            self.cache[key] = m.to(device)
        return self.cache[key]


def mask_l1(alpha, gt_mask):
    """Official: F.l1_loss(clamp(opacity, 0, 1).view(-1), gt_mask.view(-1))."""
    return (alpha.clamp(0.0, 1.0).reshape(-1) - gt_mask.reshape(-1)).abs().mean()


def create_mask(spec):
    weight = float(spec.config.get("weight", 0.1))
    state = {"table": None}  # loaded lazily; probes may inject a MaskTable

    def loss(params, outputs, target):
        if state["table"] is None:
            state["table"] = MaskTable.load(data_root())
        gt = state["table"].lookup(outputs["viewmat"], outputs["alpha"].device)
        return weight * mask_l1(outputs["alpha"][..., 0], gt)

    loss.state = state
    return loss


def splat_norm(means):
    return means.norm(dim=1).mean()


def create_norm(spec):
    weight = float(spec.config.get("weight", 0.01))

    def loss(params, outputs, target):
        if "sf_means" not in outputs:
            raise KeyError("splat-norm needs the SplatFields renderer's predicted centres")
        return weight * splat_norm(outputs["sf_means"])

    return loss
