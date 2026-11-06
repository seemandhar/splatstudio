"""Run-wide shared state: data root, per-training-image SD features / cluster maps (looked up by
the native viewmat, since the runtime exposes only cameras to components), and the step-local
values the two SpotLessSplats losses exchange.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def data_root():
    env = os.environ.get("SLS_DATA")
    if env:
        return Path(env)
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--data" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if a.startswith("--data="):
            return Path(a.split("=", 1)[1])
    raise RuntimeError("SpotLessSplats: set SLS_DATA or run through `ss-train --data <scene>`")


def native_viewmat(transform_matrix):
    """Same conversion as splatstudio.data.load_dataset (OpenGL c2w -> OpenCV w2c)."""
    c2w = torch.tensor(transform_matrix, dtype=torch.float32)
    c2w[:3, 1:3] *= -1
    return torch.linalg.inv(c2w)


class FrameTable:
    """Per training frame: a [C,h,w] float32 array (SD features or one-hot clusters), CPU."""

    def __init__(self, viewmats, arrays, names, tol=1e-4):
        self.viewmats = torch.stack(list(viewmats)).float()
        self.arrays = list(arrays)
        self.names = list(names)
        self.tol = float(tol)

    @classmethod
    def load(cls, root, subdir, loader):
        root = Path(root)
        meta = json.loads((root / "transforms.json").read_text())
        train = set(meta["train_filenames"])
        vms, arrays, names = [], [], []
        for f in meta["frames"]:
            if f["file_path"] not in train:
                continue
            stem = Path(f["file_path"]).stem
            path = root / subdir / f"{stem}.npy"
            if not path.exists():
                raise FileNotFoundError(f"SpotLessSplats: missing {path}")
            vms.append(native_viewmat(f["transform_matrix"]))
            arrays.append(loader(path))
            names.append(f["file_path"])
        if not names:
            raise ValueError("SpotLessSplats: no training frames found")
        return cls(vms, arrays, names)

    def index(self, viewmat):
        err = (self.viewmats - viewmat.detach().float().cpu()[None]).abs().amax(dim=(1, 2))
        i = int(err.argmin())
        if float(err[i]) > self.tol:
            raise KeyError(f"SpotLessSplats: no per-image data for this camera ({float(err[i])})")
        return i

    def lookup(self, viewmat, device):
        return self.arrays[self.index(viewmat)].to(device, non_blocking=True)


def load_features(path):
    """Official SemanticParser: np.load of the DIFT feature map, [1280, 50, 50] float32."""
    a = np.load(path, allow_pickle=False)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 3:
        raise ValueError(f"SD feature {path} has shape {a.shape}, expected [C,h,w]")
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).pin_memory() \
        if torch.cuda.is_available() else torch.from_numpy(np.ascontiguousarray(a, np.float32))


def load_clusters(path):
    """Cluster labels [h,w] int -> official one-hot [n_clusters, h, w] float32."""
    labels = np.load(path, allow_pickle=False).astype(np.int64)
    n = int(labels.max()) + 1
    onehot = (labels[None] == np.arange(n)[:, None, None]).astype(np.float32)
    return torch.from_numpy(onehot)


class Shared:
    """Tables are loaded once per process; probes inject synthetic tables."""

    features = None
    clusters = None
    feature_dir = "SD"
    cluster_dir = "SLS_clusters"

    @classmethod
    def feature_table(cls):
        if cls.features is None:
            cls.features = FrameTable.load(data_root(), cls.feature_dir, load_features)
        return cls.features

    @classmethod
    def cluster_table(cls):
        if cls.clusters is None:
            cls.clusters = FrameTable.load(data_root(), cls.cluster_dir, load_clusters)
        return cls.clusters
