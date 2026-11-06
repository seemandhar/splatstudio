"""Run-wide RobustSplat state: per-training-view tables (DINOv2 GT features at both scales,
coarse GT image and coarse camera), the mask MLP with its Adam, the residual histogram and the
step-local values the two losses exchange.

The runtime hands components only the camera of the sampled view, so per-view data is looked up
by the native world-to-camera matrix (same OpenGL -> OpenCV conversion as
``splatstudio.data.load_dataset``).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import dino
from .masks import MaskMLP


def data_root():
    env = os.environ.get("ROBUSTSPLAT_DATA")
    if env:
        return Path(env)
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--data" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if a.startswith("--data="):
            return Path(a.split("=", 1)[1])
    raise RuntimeError("RobustSplat: set ROBUSTSPLAT_DATA or run through `ss-train --data <scene>`")


def native_viewmat(transform_matrix):
    c2w = torch.tensor(transform_matrix, dtype=torch.float32)
    c2w[:3, 1:3] *= -1
    return torch.linalg.inv(c2w)


def load_rgb(path):
    """Inria ``PILtoTorch`` without resize: [3,H,W] float in [0,1]."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a.copy()).permute(2, 0, 1).contiguous()


class ViewTable:
    """Per training view: fine/coarse GT DINOv2 features, coarse GT image, coarse camera."""

    def __init__(self, viewmats, entries, tol=1e-4):
        self.viewmats = torch.stack(list(viewmats)).float()
        self.entries = list(entries)
        self.tol = float(tol)

    def index(self, viewmat):
        err = (self.viewmats - viewmat.detach().float().cpu()[None]).abs().amax(dim=(1, 2))
        i = int(err.argmin())
        if float(err[i]) > self.tol:
            raise KeyError(f"RobustSplat: no per-view data for this camera ({float(err[i]):.3g})")
        return i

    def lookup(self, viewmat):
        return self.entries[self.index(viewmat)]

    @classmethod
    def build(cls, root, device, fine_grid, coarse_grid, weights=dino.DEFAULT_WEIGHTS):
        """Official ``train.py``: GT features of every training image at both grids, computed
        once before training; coarse cameras from ``scene.getTrainCameras(4)``."""
        root = Path(root)
        meta = json.loads((root / "transforms.json").read_text())
        train = set(meta["train_filenames"])
        model = load_dino(device, weights)
        vms, entries = [], []
        for f in meta["frames"]:
            if f["file_path"] not in train:
                continue
            if "coarse_file_path" not in f:
                raise KeyError("RobustSplat needs coarse views: prepare the scene with "
                               "reproductions/robustsplat/tools/prepare_robustsplat.py")
            image = load_rgb(root / f["file_path"]).to(device)
            coarse = load_rgb(root / f["coarse_file_path"]).to(device)
            cw, ch = int(f["coarse_w"]), int(f["coarse_h"])
            if coarse.shape[1:] != (ch, cw):
                raise ValueError(f"{f['coarse_file_path']}: size differs from transforms.json")
            K = torch.tensor([[float(f["coarse_fl_x"]), 0.0, cw / 2.0],
                              [0.0, float(f["coarse_fl_y"]), ch / 2.0],
                              [0.0, 0.0, 1.0]], device=device)
            vm = native_viewmat(f["transform_matrix"])
            entries.append({
                "name": f["file_path"],
                "feat_fine": dino.extract(model, image, fine_grid),
                "feat_coarse": dino.extract(model, image, coarse_grid),
                "coarse_image": coarse,
                "coarse_camera": {"viewmat": vm.to(device), "K": K, "width": cw, "height": ch},
            })
            vms.append(vm)
        if not entries:
            raise ValueError("RobustSplat: no training frames found")
        return cls(vms, entries)


def load_dino(device, weights=dino.DEFAULT_WEIGHTS):
    return dino.load_backbone(weights, device=device)


class Shared:
    """One instance per process (the runtime builds each component once); probes reset it."""

    table = None
    mlp = None
    optimizer = None
    hist = None
    pending_step = False

    @classmethod
    def reset(cls):
        cls.table = None
        cls.mlp = None
        cls.optimizer = None
        cls.hist = None
        cls.pending_step = False

    @classmethod
    def ensure_mlp(cls, device, cfg):
        if cls.mlp is None:
            cls.mlp = MaskMLP(int(cfg["feature_dim"]), int(cfg["mlp_hidden"])).to(device)
            cls.optimizer = torch.optim.Adam(cls.mlp.parameters(), lr=float(cfg["mlp_lr"]))
            cls.hist = torch.zeros(int(cfg["bins"]), device=device)
        return cls.mlp

    @classmethod
    def ensure_table(cls, device, cfg):
        if cls.table is None:
            cls.table = ViewTable.build(data_root(), device, int(cfg["fine_grid"]),
                                        int(cfg["coarse_grid"]), cfg["dino_weights"])
        return cls.table

    @classmethod
    def apply_pending_step(cls):
        """MLP update of the previous step (official: right after ``mask_loss.backward()``;
        the runtime backpropagates the summed losses after all components ran)."""
        if cls.pending_step and cls.optimizer is not None:
            cls.optimizer.step()
        if cls.optimizer is not None:
            cls.optimizer.zero_grad(set_to_none=True)
        cls.pending_step = False
