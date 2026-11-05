"""Per-training-view data of the official loader that the native dataset cannot carry: object
masks, Omnidata normal priors and the PIL-downscaled images used by the coarse-to-fine schedule.

Built by reproductions/gaussian-surfels/tools/prepare_dtu.py into ``<data>/gsurfels/`` and looked
up by the frame viewmat. The data root comes from ``GSURFELS_DATA`` or ss-train/ss-eval's
``--data`` argument; if no sidecar exists the renderer runs without priors (probes, other data).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_TABLES = {}


def data_root():
    env = os.environ.get("GSURFELS_DATA")
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
        self.root = Path(root) / "gsurfels"
        self.viewmats = torch.from_numpy(np.load(self.root / "viewmats.npy")).float()
        self.names = json.loads((self.root / "names.json").read_text())
        self.tol = tol
        self.arrays = {}
        self.cache = {}

    def index(self, viewmat):
        err = (self.viewmats - viewmat.detach().float().cpu()[None]).abs().amax(dim=(1, 2))
        i = int(err.argmin())
        return i if float(err[i]) <= self.tol else None

    def _arr(self, key):
        if key not in self.arrays:
            self.arrays[key] = np.load(self.root / f"{key}.npy", mmap_mode="r")
        return self.arrays[key]

    def view(self, i, scale, device):
        """{"mask": [1,h,w], "mono": [3,h,w], "image": [3,h,w] or None (scale 1)} on device."""
        key = (i, scale, str(device))
        if key not in self.cache:
            out = {
                "mask": torch.from_numpy(np.array(self._arr(f"mask_s{scale}")[i]))[None],
                "mono": torch.from_numpy(np.array(self._arr(f"normal_s{scale}")[i])),
                "image": None,
            }
            if scale != 1:
                img = np.array(self._arr(f"image_s{scale}")[i])
                out["image"] = torch.from_numpy(img).permute(2, 0, 1).contiguous()
            self.cache[key] = {k: (v.to(device) if v is not None else None) for k, v in out.items()}
        c = self.cache[key]  # fp16 / uint8 on device, float32 views per call
        return {"mask": c["mask"].float(), "mono": c["mono"].float(),
                "image": None if c["image"] is None else c["image"].float() / 255.0}

    def init(self):
        p = self.root / "init.npz"
        if not p.exists():
            return None
        with np.load(p) as d:
            return {k: torch.from_numpy(d[k]) for k in ("points", "log_scales", "quats")}


def table(root=None):
    root = Path(root) if root is not None else data_root()
    if root is None or not (root / "gsurfels" / "viewmats.npy").exists():
        return None
    key = str(root.resolve())
    if key not in _TABLES:
        _TABLES[key] = Table(root)
    return _TABLES[key]
