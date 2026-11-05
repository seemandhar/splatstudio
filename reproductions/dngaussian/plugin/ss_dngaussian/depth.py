"""Global-Local Depth Normalization losses (paper §3.3, Eq. 7-9) and the DPT sidecar.

Numerics follow the official utils/loss_utils.py exactly:
  patchify            F.unfold(kernel=stride=p) -> [n_patches, p*p] (remainder cropped)
  normalize(x, std)   (x - mean_dim1(x)) / (std + 1e-2 * std(x.reshape(-1)))   (unbiased std)
      local  (Eq. 7): std = per-patch std (dim 1)
      global (Eq. 8): std = whole-map std, detached
  margin_l2           mean((a - b)[|a - b| > margin]^2)   (error tolerance, §3.5)
  smoothness          edge-aware |grad D| e^{-|grad D~|} (train_llff.py, after iteration 3000)
and the per-pass combination of train_llff.py:94-108 / 118-131:
  R = 0.1 * local + [0.1 * smoothness if it > 3000] + 1.0 * global,
with independent random patch sizes randint(5, 17) for the local and the global term.

The monocular target is the official ``depth_mono = 255 - png16 / 255`` of DPT-Hybrid,
precomputed for the training views only by reproductions/dngaussian/tools/prepare_dng_llff.py
into ``<data>/dng_depth.npz`` and looked up by the frame viewmat (as the SplatFields plugin
does for masks). The data root comes from ``DNG_DATA`` or ss-train's ``--data`` argument.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DEPTH_DEFAULTS = {"error_tolerance": 0.00025, "local_weight": 0.1, "global_weight": 1.0,
                  "smooth_weight": 0.1, "smooth_start": 3000, "patch_min": 5, "patch_max": 17}


def patchify(x, p):
    return F.unfold(x, kernel_size=p, stride=p).permute(0, 2, 1).reshape(-1, p * p)


def normalize(x, std=None):
    mean = x.mean(dim=1, keepdim=True)
    s = x.std(dim=1, keepdim=True) if std is None else std
    return (x - mean) / (s + 1e-2 * x.reshape(-1).std())


def margin_l2(a, b, margin):
    diff = a - b
    sel = diff.abs() > margin
    if not bool(sel.any()):
        # Official: mean of an empty selection = NaN; its gradient contribution is zero.
        return diff.sum() * 0.0
    return diff[sel].square().mean()


def local_loss(depth, mono, p, margin):
    """patch_norm_mse_loss (Eq. 7 + Eq. 9 L2 with tolerance). depth, mono: [1, 1, H, W]."""
    return margin_l2(normalize(patchify(depth, p)), normalize(patchify(mono, p)), margin)


def global_loss(depth, mono, p, margin):
    """patch_norm_mse_loss_global (Eq. 8): patch mean removed, whole-image std."""
    return margin_l2(normalize(patchify(depth, p), std=depth.std().detach()),
                     normalize(patchify(mono, p), std=mono.std().detach()), margin)


def smoothness(depth, img):
    """loss_depth_smoothness(depth, img) with img = the monocular map (raw scale)."""
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]
    wx = torch.exp(-gx.abs().mean(1, keepdim=True))
    wy = torch.exp(-gy.abs().mean(1, keepdim=True))
    num = ((depth[:, :, :, :-1] - depth[:, :, :, 1:]).abs() * wx).sum() + \
          ((depth[:, :, :-1, :] - depth[:, :, 1:, :]).abs() * wy).sum()
    return num / (wx.sum() + wy.sum())


def depth_regularization(depth, mono, iteration, rng, cfg):
    """One official depth pass loss. Returns (total, terms dict of floats)."""
    lo, hi = int(cfg["patch_min"]), int(cfg["patch_max"])
    tol = float(cfg["error_tolerance"])
    loc = local_loss(depth, mono, rng.randint(lo, hi), tol)
    total = float(cfg["local_weight"]) * loc
    terms = {"local": float(loc)}
    if iteration > int(cfg["smooth_start"]):
        sm = smoothness(depth, mono)
        total = total + float(cfg["smooth_weight"]) * sm
        terms["smooth"] = float(sm)
    glo = global_loss(depth, mono, rng.randint(lo, hi), tol)
    total = total + float(cfg["global_weight"]) * glo
    terms["global"] = float(glo)
    return total, terms


# ------------------------------------------------------------------ sidecar
def data_root():
    env = os.environ.get("DNG_DATA")
    if env:
        return Path(env)
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--data" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if a.startswith("--data="):
            return Path(a.split("=", 1)[1])
    raise RuntimeError("DNGaussian depth: set DNG_DATA or run through `ss-train --data`")


class DepthTable:
    """DPT targets (``depth_mono``) of the training views keyed by the native viewmat."""

    def __init__(self, viewmats, depth_mono, spiral_centers, tol=1e-4):
        self.viewmats = torch.as_tensor(np.asarray(viewmats), dtype=torch.float32)
        self.depth = torch.as_tensor(np.asarray(depth_mono), dtype=torch.float32)
        self.spiral = torch.as_tensor(np.asarray(spiral_centers), dtype=torch.float32)
        self.tol = float(tol)
        self.cache = {}

    @classmethod
    def load(cls, root):
        path = Path(root) / "dng_depth.npz"
        if not path.exists():
            raise FileNotFoundError(f"DNGaussian depth sidecar {path} not found (build it with "
                                    "reproductions/dngaussian/tools/prepare_dng_llff.py)")
        with np.load(path, allow_pickle=False) as d:
            return cls(d["viewmats"], d["depth_mono"], d["spiral_centers"])

    def lookup(self, viewmat, device):
        err = (self.viewmats - viewmat.detach().float().cpu()[None]).abs().amax(dim=(1, 2))
        i = int(err.argmin())
        if float(err[i]) > self.tol:
            raise KeyError(f"DNGaussian depth: no DPT map for this camera (closest {float(err[i])})")
        key = (i, str(device))
        if key not in self.cache:
            self.cache[key] = self.depth[i].to(device)[None, None]
        return self.cache[key]
