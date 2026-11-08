"""Offline per-view priors (Depth Anything V2-L disparity, Canny 20/200 edge map) of the training
views, built by reproductions/det-gs/tools/prepare_detgs.py into ``<data>/detgs_priors.npz`` and
looked up by the native viewmat. The data root comes from ``DETGS_DATA`` or ss-train's ``--data``."""

import os
import sys
from pathlib import Path

import numpy as np
import torch

_TABLE = {}


def data_root():
    env = os.environ.get("DETGS_DATA")
    if env:
        return Path(env)
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--data" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if a.startswith("--data="):
            return Path(a.split("=", 1)[1])
    raise RuntimeError("DET-GS: set DETGS_DATA or run through `ss-train --data`")


class PriorTable:
    def __init__(self, viewmats, disparity, canny, tol=1e-4):
        self.viewmats = torch.as_tensor(np.asarray(viewmats), dtype=torch.float32)
        self.disparity = np.asarray(disparity)
        self.canny = np.asarray(canny)
        self.tol = float(tol)
        self.cache = {}

    @classmethod
    def load(cls, root):
        path = Path(root).expanduser() / "detgs_priors.npz"
        if not path.exists():
            raise FileNotFoundError(f"DET-GS priors {path} not found (build them with "
                                    "reproductions/det-gs/tools/prepare_detgs.py)")
        with np.load(path, allow_pickle=False) as d:
            return cls(d["viewmats"], d["disparity"], d["canny"])

    def index(self, viewmat):
        err = (self.viewmats - viewmat.detach().float().cpu()[None]).abs().amax(dim=(1, 2))
        i = int(err.argmin())
        if float(err[i]) > self.tol:
            raise KeyError(f"DET-GS: no prior for this camera (closest {float(err[i]):.2e})")
        return i

    def lookup(self, viewmat, device):
        """(disparity [H,W] float32, canny edge map [H,W] bool) on ``device``."""
        i = self.index(viewmat)
        key = (i, str(device))
        if key not in self.cache:
            disp = torch.from_numpy(self.disparity[i].astype(np.float32)).to(device)
            edge = torch.from_numpy(self.canny[i].astype(bool)).to(device)
            self.cache[key] = (disp, edge)
        return self.cache[key]


def table():
    if "t" not in _TABLE:
        _TABLE["t"] = PriorTable.load(data_root())
    return _TABLE["t"]


def set_table(t):
    """Install a table directly (probes / tests)."""
    _TABLE["t"] = t
