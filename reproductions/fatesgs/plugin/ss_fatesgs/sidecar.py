"""Per-view priors of the training views: Marigold depth, source-view pairs, Vis-MVSNet features.

Built by reproductions/fatesgs/tools/prepare_dtu.py into ``<data>/fatesgs.npz`` (+
``<data>/images_full/``), looked up by the frame viewmat. The data root comes from
``FATESGS_DATA`` or ss-train's ``--data`` argument.
"""

import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

WEIGHTS = Path(os.environ.get("FATESGS_VISMVSNET", "~/models/vismvsnet/vismvsnet.pt")).expanduser()
_TABLES = {}


def data_root():
    env = os.environ.get("FATESGS_DATA")
    if env:
        return Path(env)
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--data" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if a.startswith("--data="):
            return Path(a.split("=", 1)[1])
    raise RuntimeError("FatesGS: set FATESGS_DATA or run through `ss-train --data`")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Table:
    def __init__(self, root, tol=1e-3):
        root = Path(root)
        with np.load(root / "fatesgs.npz", allow_pickle=False) as d:
            self.viewmats = torch.as_tensor(d["viewmats"], dtype=torch.float32)
            self.mono = torch.as_tensor(d["mono_depth"], dtype=torch.float32)
            self.alpha = torch.as_tensor(d["alpha"].astype(np.float32) / 255.0)
            self.pairs = d["pairs"].tolist()
            self.names = [str(n) for n in d["names"]]
            self.full_size = tuple(int(v) for v in d["full_size"])
        self.root = root
        self.tol = tol
        self.cache = {}
        self.feats = None

    def index(self, viewmat):
        err = (self.viewmats - viewmat.detach().float().cpu()[None]).abs().amax(dim=(1, 2))
        i = int(err.argmin())
        if float(err[i]) > self.tol:
            raise KeyError(f"FatesGS: no prior for this camera (closest {float(err[i]):.2e})")
        return i

    def mono_depth(self, i, device):
        key = ("mono", i, str(device))
        if key not in self.cache:
            self.cache[key] = self.mono[i].to(device)
        return self.cache[key]

    def features(self, device):
        """[level][view] -> [32, h, w] Vis-MVSNet features (official dataset_readers.py)."""
        if self.feats is None:
            from .vismvsnet import FeatExt

            net = FeatExt().to(device).eval()
            net.load_state_dict(load_weights())
            for p in net.parameters():
                p.requires_grad_(False)
            ow, oh = 1600, 1200  # official buffer for -r 2
            self.feats = []
            for scale in (1, 2):
                size_w, size_h = ow // scale, oh // scale
                batch = torch.zeros(len(self.names), 3, size_h, size_w)
                for k, name in enumerate(self.names):
                    img = Image.open(self.root / "images_full" / Path(name).name)
                    res = (int(img.size[0] / scale), int(img.size[1] / scale))
                    chans = [np.asarray(c.resize(res, Image.Resampling.BICUBIC), np.float32) / 255
                             for c in img.split()[:3]]
                    t = torch.from_numpy(np.stack(chans))
                    batch[k, :, : t.shape[1], : t.shape[2]] = t
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                batch = (batch / 2 + 0.5 - mean) / std  # official normalisation (sic)
                with torch.no_grad():
                    f = net(batch.to(device))[2]
                f = f[..., : (oh // 2) // scale, : (ow // 2) // scale]
                # scene/cameras.py crop: [:ceil(H/l), :ceil(W/l)] of the training resolution
                hh = math.ceil(round(self.full_size[1] / 2) / scale)
                ww = math.ceil(round(self.full_size[0] / 2) / scale)
                self.feats.append([f[k, :, :hh, :ww].contiguous() for k in range(len(self.names))])
            del net
        return self.feats


def load_weights():
    """Vis-MVSNet feature-extractor weights (shipped in the FatesGS repo as utils/vismvsnet.pt),
    loaded tensor-only."""
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Vis-MVSNet weights {WEIGHTS} not found")
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)["state_dict"]
    return {k[16:]: v for k, v in sd.items() if k.startswith("module.feat_ext")}


def table(root=None):
    root = Path(root) if root is not None else data_root()
    key = str(root.resolve())
    if key not in _TABLES:
        _TABLES[key] = Table(root)
    return _TABLES[key]
