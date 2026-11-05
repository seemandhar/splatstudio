"""Per-training-view priors the native dataset cannot carry. Copied from ss_dnsplatter.sidecar
(reproductions/dn-splatter) and extended with the AGS-Mesh DNC mask:

* ``<data>/dnsplat/`` (reproductions/{dn-splatter,ags-mesh}/tools/prepare_mushroom.py): sensor
  depth (uint16 mm), Omnidata v2 normal pngs (uint8), official TSDF init, viewmats, names;
* ``<data>/agsmesh/dnc_mask_u8.npy`` (reproductions/ags-mesh/tools/prepare_dnc.py): decoded JPEG
  DNC masks (255 = depth rejected), same frame order as ``dnsplat/names.json``.

The data root comes from ``AGSMESH_DATA`` (or ``DNSPLAT_DATA``) or ss-train's ``--data``
argument. Without a sidecar the renderer provides no priors and the prior losses raise.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_TABLES = {}


def data_root():
    for key in ("AGSMESH_DATA", "DNSPLAT_DATA"):
        if os.environ.get(key):
            return Path(os.environ[key])
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
        dnc = Path(root) / "agsmesh" / "dnc_mask_u8.npy"
        self.dnc = np.load(dnc, mmap_mode="r") if dnc.exists() else None
        if self.dnc is not None and self.dnc.shape != self.depth.shape:
            raise ValueError(f"DNC mask {self.dnc.shape} does not match depth {self.depth.shape}")
        self.tol = tol

    def index(self, viewmat):
        err = (self.viewmats - viewmat.detach().float().cpu()[None]).abs().amax(dim=(1, 2))
        i = int(err.argmin())
        return i if float(err[i]) <= self.tol else None

    def view(self, i, device):
        """{"depth": [H,W,1] m, "normal": [H,W,3] png/255, "confidence": [H,W,1] or None}.

        confidence = 1 - mask/255 (dn_model.py:646, nerfstudio get_depth_image_from_path with
        scale 1 on the JPEG mask)."""
        d = torch.from_numpy(np.array(self.depth[i], dtype=np.int32)).to(device)
        n = torch.from_numpy(np.array(self.normal[i])).to(device)
        out = {"depth": (d.double() * 1e-3).float()[..., None], "normal": n.float() / 255.0,
               "confidence": None}
        if self.dnc is not None:
            m = torch.from_numpy(np.array(self.dnc[i])).to(device).float()
            out["confidence"] = (1.0 - m / 255.0)[..., None]
        return out

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
