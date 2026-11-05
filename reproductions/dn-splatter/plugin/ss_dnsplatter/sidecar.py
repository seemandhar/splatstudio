"""Per-training-view priors that the native dataset cannot carry (sensor depth, Omnidata normals)
and the official Gaussian initialization, built by reproductions/dn-splatter/tools/
prepare_mushroom.py into ``<data>/dnsplat/`` and looked up by the frame viewmat.

The data root comes from ``DNSPLAT_DATA`` or ss-train's ``--data`` argument. Without a sidecar
the renderer provides no priors and the prior losses raise (no silent fallback).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_TABLES = {}


def data_root():
    env = os.environ.get("DNSPLAT_DATA")
    if env:
        return Path(env)
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--data" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if a.startswith("--data="):
            return Path(a.split("=", 1)[1])
    return None


class Table:
    def __init__(self, root, tol=1e-4):
        self.root = Path(root) / "dnsplat"
        self.viewmats = torch.from_numpy(np.load(self.root / "viewmats.npy")).float()
        self.names = json.loads((self.root / "names.json").read_text())
        self.depth = np.load(self.root / "depth_mm.npy", mmap_mode="r")
        self.normal = np.load(self.root / "normal_omni_u8.npy", mmap_mode="r")
        self.tol = tol

    def index(self, viewmat):
        err = (self.viewmats - viewmat.detach().float().cpu()[None]).abs().amax(dim=(1, 2))
        i = int(err.argmin())
        return i if float(err[i]) <= self.tol else None

    def view(self, i, device):
        """{"depth": [H,W,1] metres (float32), "normal": [H,W,3] in [0,1]} on device.

        depth: get_depth_image_from_path (uint16 mm x 1e-3, already image resolution);
        normal: GDataset.get_normal_image_from_path png / 255 (the official OpenGL->OpenCV sign
        flip is applied to both sides of the loss in the official code, so it cancels; see
        renderer.py)."""
        d = torch.from_numpy(np.array(self.depth[i], dtype=np.int32)).to(device)
        n = torch.from_numpy(np.array(self.normal[i])).to(device)
        return {"depth": (d.double() * 1e-3).float()[..., None], "normal": n.float() / 255.0}

    def init(self):
        p = self.root / "init.npz"
        if not p.exists():
            return None
        with np.load(p) as d:
            return {k: torch.from_numpy(d[k]) for k in ("points", "log_scales", "quats")}


def table(root=None):
    root = Path(root) if root is not None else data_root()
    if root is None or not (root / "dnsplat" / "viewmats.npy").exists():
        return None
    key = str(root.resolve())
    if key not in _TABLES:
        _TABLES[key] = Table(root)
    return _TABLES[key]
