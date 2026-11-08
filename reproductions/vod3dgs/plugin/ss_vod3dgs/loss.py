"""View-consistency loss L_vc (Eq. 12) with covisibility-weighted view sampling (§4.4)."""

from pathlib import Path

import numpy as np
import torch

from .covis import DATA_DIR
from .vod import _CONTEXT, view_consistency

DEFAULTS = {"weight": 1.0, "view_sampling": "covisibility", "covis_dir": str(DATA_DIR), "match_tol": 1e-3}


def match_covis(train_centers, covis_dir, tol):
    """Find the covisibility file whose cameras contain every training camera; return the
    [C_train, C_train] match matrix restricted to the training views (and the file name)."""
    tc = train_centers.detach().double().cpu().numpy()
    extent = max(float(np.linalg.norm(tc - tc.mean(0), axis=1).max()), 1e-6)
    for path in sorted(Path(covis_dir).glob("*.npz")):
        d = np.load(path)
        centers = d["centers"]
        dist = np.linalg.norm(tc[:, None, :] - centers[None, :, :], axis=-1)
        idx = dist.argmin(1)
        if (dist[np.arange(len(tc)), idx] < tol * extent).all() and len(set(idx.tolist())) == len(tc):
            return torch.from_numpy(d["matches"][np.ix_(idx, idx)].astype(np.float64)), path.name
    raise FileNotFoundError(
        f"No covisibility file in {covis_dir} matches the training cameras; build one with "
        "`python -m ss_vod3dgs.covis --sparse <scene>/sparse/0 --name <scene>` or set view_sampling=uniform"
    )


class ViewConsistency:
    def __init__(self, spec):
        cfg = {**DEFAULTS, **spec.config}
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown vc-loss options: {sorted(unknown)}")
        if cfg["view_sampling"] not in ("covisibility", "uniform"):
            raise ValueError("view_sampling must be 'covisibility' or 'uniform'")
        self.weight = float(cfg["weight"])
        self.sampling = cfg["view_sampling"]
        self.covis_dir = cfg["covis_dir"]
        self.tol = float(cfg["match_tol"])
        self.centers = None
        self.probs = None
        self.source = None
        self.uniform_rows = 0

    def _setup(self, centers):
        self.centers = centers
        n = len(centers)
        w = torch.ones(n, n, dtype=torch.float64) - torch.eye(n, dtype=torch.float64)
        if self.sampling == "covisibility":
            m, self.source = match_covis(centers, self.covis_dir, self.tol)
            m.fill_diagonal_(0)
            empty = m.sum(1) == 0
            # A training view with no SfM covisibility at all gets uniform sampling (counted).
            self.uniform_rows = int(empty.sum())
            w = torch.where(empty[:, None], w, m)
        self.probs = (w / w.sum(1, keepdim=True)).to(centers.device, torch.float32)

    def sample(self, center_i):
        centers = _CONTEXT["train_centers"]
        if centers is None:
            raise RuntimeError("L_vc needs training cameras registered by the vod-density component")
        if self.centers is not centers:
            self._setup(centers)
        i = int((self.centers - center_i[None]).norm(dim=-1).argmin())
        j = int(torch.multinomial(self.probs[i], 1))
        return i, j

    def __call__(self, params, outputs, target):
        _, j = self.sample(outputs["camera_center"])
        return self.weight * view_consistency(
            params["opacities"], params["vod_S"], params["means"], outputs["camera_center"], self.centers[j]
        )


def create_vc_loss(spec):
    return ViewConsistency(spec)
