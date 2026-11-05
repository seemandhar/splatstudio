"""Trim3DGS normal regularization (Sec. 4.3, Eq. 7-8; code tune.py:normal_regularization).

L = weight * (L_normal + L_smooth), active for step > start_step (fine-tune stage).
"""

import torch
import torch.nn.functional as F

from . import core
from .renderer import render_features

DEFAULTS = {
    "weight": 0.1,
    "start_step": 30000,
    "depth_grad_thresh": 0.05,
    "mask_dilation": 1,
    "dilation": 2,
    "close_thresh": 1.0,
    "knn": 16,
    "knn_interval": 500,
    "smooth": True,
}


class NormalLoss:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown normal-loss options: {sorted(unknown)}")
        self.c = {**DEFAULTS, **spec.config}
        self.knn = None
        self.knn_step = None
        self.last = {}

    def _neighbours(self, params, t):
        n = len(params["means"])
        if self.knn is None or len(self.knn) != n or (t - 1) % self.c["knn_interval"] == 0:
            # official reset_neighbor: tune start, every 500 its and after every densification
            self.knn = core.knn_indices(params["means"], self.c["knn"])
            self.knn_step = t
        return self.knn

    def terms(self, params, outputs):
        c = self.c
        cam = outputs["camera"]
        eps = outputs["flat_eps"]
        viewmat = cam["viewmat"].float()
        c2w = torch.linalg.inv(viewmat)
        center = c2w[:3, 3]
        means = params["means"]
        cam_pts = means @ viewmat[:3, :3].T + viewmat[:3, 3]
        z = cam_pts[:, 2:3]
        gs_n = core.face_camera(core.gaussian_normals(params["quats"]), means, center)
        out, _, info = render_features(params, cam, eps, torch.cat((z, gs_n), dim=1))
        depth, pred_n = out[..., 0], out[..., 1:4]
        mask = core.sobel_mask(depth.detach(), c["depth_grad_thresh"], c["mask_dilation"])
        pts = core.backproject(depth, cam["K"].float(), c2w)
        dn = core.depth_normals(pts, center, c["dilation"])
        l_normal = core.normal_map_loss(pred_n, dn, mask)
        if not c["smooth"]:
            return l_normal, l_normal.new_zeros(()), dn
        radii = info["radii"][0]
        visible = (radii > 0).all(dim=-1) if radii.dim() == 2 else radii > 0
        idx = torch.nonzero(visible).squeeze(1)
        K = cam["K"].float()
        W, H = int(cam["width"]), int(cam["height"])
        p = cam_pts[idx].detach()
        u = (K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]) / W * 2 - 1
        v = (K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]) / H * 2 - 1
        sampled = F.grid_sample(depth.detach()[None, None], torch.stack((u, v), -1)[None, None],
                                align_corners=True).reshape(-1)
        close = (sampled - p[:, 2]).abs() < c["close_thresh"]
        idx = idx[close]
        knn = self._neighbours(params, outputs["step"] - c["start_step"])
        l_smooth = core.smooth_loss(gs_n, idx, knn)
        return l_normal, l_smooth, dn

    def __call__(self, params, outputs, target):
        if outputs["step"] <= self.c["start_step"]:
            return params["means"].sum() * 0.0
        l_normal, l_smooth, _ = self.terms(params, outputs)
        total = l_normal + l_smooth
        self.last = {"normal": float(l_normal), "smooth": float(l_smooth)}
        if not torch.isfinite(total):  # official: 'Got NaN in normal loss, skip'
            return params["means"].sum() * 0.0
        return self.c["weight"] * total


def create_normal_loss(spec):
    return NormalLoss(spec)
