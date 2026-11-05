"""Octree-GS global learned state: the three Scaffold-GS MLPs (with their own Adam and exponential LR schedules)
plus the octree scalars (voxel size, standard distance, origin, K, init level, coarse intervals, visible threshold)
and the training-camera centres used by weed_out. One bank per process, owned by the density component and read by
the renderer through `ACTIVE` (the HAC / ContextGS native-port pattern).

Numerics follow github.com/city-super/Octree-GS @ 9db64fe: scene/gaussian_model.py (__init__ MLPs,
training_setup, update_learning_rate), utils/general_utils.py:get_expon_lr_func.
"""

import numpy as np
import torch
from torch import nn

# The live bank during training (one per process); None in a fresh process such as ss-eval.
ACTIVE = {"bank": None}

RENDER_MLPS = ("mlp_opacity", "mlp_cov", "mlp_color")


def expon_lr(lr_init, lr_final, max_steps):
    """get_expon_lr_func with lr_delay_steps = 0 (the official defaults; delay_mult is then unused)."""

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        t = np.clip(step / max_steps, 0, 1)
        return float(np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t))

    return helper


def scaffold_mlps(feat_dim, n_offsets):
    """mlp_opacity / mlp_cov / mlp_color of Octree-GS gaussian_model.py with the script settings
    (use_feat_bank, add_opacity_dist, add_cov_dist, add_color_dist, add_level all False, appearance_dim 0):
    input = [feature, view direction] (feat_dim + 3; no distance)."""
    d = feat_dim + 3
    return nn.ModuleDict({
        "mlp_opacity": nn.Sequential(nn.Linear(d, feat_dim), nn.ReLU(True),
                                     nn.Linear(feat_dim, n_offsets), nn.Tanh()),
        "mlp_cov": nn.Sequential(nn.Linear(d, feat_dim), nn.ReLU(True),
                                 nn.Linear(feat_dim, 7 * n_offsets)),
        "mlp_color": nn.Sequential(nn.Linear(d, feat_dim), nn.ReLU(True),
                                   nn.Linear(feat_dim, 3 * n_offsets), nn.Sigmoid()),
    })


class OctreeBank(nn.Module):
    """MLPs + octree scalars; `step` is the step of the next training render."""

    def __init__(self, c, device):
        super().__init__()
        self.c = c
        self.feat_dim, self.n_offsets = c["feat_dim"], c["n_offsets"]
        self.mlps = scaffold_mlps(self.feat_dim, self.n_offsets).to(device)
        self.step = 1
        # octree scalars (set by the density when the octree is built)
        self.fork = int(c["fork"])
        self.levels = None
        self.init_level = None
        self.voxel_size = None
        self.standard_dist = None
        self.init_pos = None
        self.visible_threshold = float(c["visible_threshold"])
        self.coarse_intervals = []
        self.cam_centers = None
        t = c["total_steps"]
        lr = c["lr"]
        self.schedules = {k: expon_lr(lr[k][0], lr[k][1], t) for k in RENDER_MLPS}
        groups = [{"params": list(self.mlps[k].parameters()), "name": k, "lr": 0.0} for k in RENDER_MLPS]
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.update_learning_rate(1)

    @property
    def ready(self):
        return self.levels is not None

    def update_learning_rate(self, step):
        for g in self.optimizer.param_groups:
            g["lr"] = self.schedules[g["name"]](step)

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def octree(self):
        """Scalars the renderer needs (also packed into the blob)."""
        return {"levels": self.levels, "fork": self.fork, "voxel_size": self.voxel_size,
                "standard_dist": self.standard_dist, "init_pos": self.init_pos,
                "init_level": self.init_level}
