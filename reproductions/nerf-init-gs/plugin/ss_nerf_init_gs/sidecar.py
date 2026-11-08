"""Per-scene NeRF sidecar written by `ss-nerfinit-prep` into `<data>/nerfinit/`.

    viewmats.npy      [N, 4, 4] float32 world-to-camera (OpenCV) of the training views
    names.json        training frame file paths (same order)
    depth_z_f16.npy   [N, H, W] float16 NeRF expected depth (Eq. 6) converted to camera z in
                      world (COLMAP) units: z = t / scale / |K^-1 [u+.5, v+.5, 1]|
    init.npz          points_sha256 + exact 3-NN log scales of points.npz (3DGS init)
    ingp.pt           iNGP weights + normalisation; nerfinit.json  timings / provenance

The data root comes from `NERFINIT_DATA` or ss-train's `--data` argument (the runtime does not
pass the dataset path to components). Frames are matched by their viewmat (pattern from the
SPLATIFY ags-mesh / dn-splatter sidecars).
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SUBDIR = "nerfinit"
_TABLES = {}


def data_root():
    if os.environ.get("NERFINIT_DATA"):
        return Path(os.environ["NERFINIT_DATA"])
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--data" and i + 1 < len(argv):
            return Path(argv[i + 1]).expanduser()
        if a.startswith("--data="):
            return Path(a.split("=", 1)[1]).expanduser()
    return None


def points_sha256(points):
    if isinstance(points, torch.Tensor):
        points = points.detach().float().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(points, dtype=np.float32).tobytes()).hexdigest()


class Table:
    def __init__(self, root, tol=1e-4):
        self.root = Path(root) / SUBDIR
        self.viewmats = torch.from_numpy(np.load(self.root / "viewmats.npy")).float()
        self.names = json.loads((self.root / "names.json").read_text())
        depth = self.root / "depth_z_f16.npy"  # absent when built with --no-depth
        self.depth = np.load(depth, mmap_mode="r") if depth.exists() else None
        n_depth = len(self.names) if self.depth is None else self.depth.shape[0]
        if len(self.names) != len(self.viewmats) or n_depth != len(self.names):
            raise ValueError(f"Inconsistent NeRF sidecar in {self.root}")
        self.tol = tol
        self._cache = {}
        info = self.root / "nerfinit.json"
        self.nerf_scale = (float(json.loads(info.read_text())["normalisation_scale"])
                           if info.exists() else None)

    def index(self, viewmat):
        err = (self.viewmats - viewmat.detach().float().cpu()[None]).abs().amax(dim=(1, 2))
        i = int(err.argmin())
        return i if float(err[i]) <= self.tol else None

    def depth_view(self, i, device):
        """[H, W, 1] float32 NeRF camera-z depth of training view i (fp16 cache on device)."""
        key = (i, str(device))
        if key not in self._cache:
            self._cache[key] = torch.from_numpy(np.array(self.depth[i])).to(device)[..., None]
        return self._cache[key].float()

    def init(self):
        with np.load(self.root / "init.npz", allow_pickle=False) as d:
            return str(d["points_sha256"]), torch.from_numpy(d["log_scales"])


def table(root=None):
    root = Path(root) if root is not None else data_root()
    if root is None:
        return None
    if not (root / SUBDIR / "viewmats.npy").exists():
        raise FileNotFoundError(
            f"{root} has no {SUBDIR}/ sidecar: run ss-nerfinit-prep to build the NeRF-initialized "
            "dataset copy first")
    key = str(root.resolve())
    if key not in _TABLES:
        _TABLES[key] = Table(root)
    return _TABLES[key]
