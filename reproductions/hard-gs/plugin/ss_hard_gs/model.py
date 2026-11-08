"""Scaffold-GS global learned state (the three decoding MLPs) with its own Adam and the official
exponential schedules (github.com/city-super/Scaffold-GS: scene/gaussian_model.py,
utils/general_utils.py:get_expon_lr_func, arguments/__init__.py). Engineering pattern from the
HAC native plugin (reproductions/hac/plugin/ss_hac/model.py)."""

import numpy as np
import torch
from torch import nn

# The live bank during training (one per process); None in a fresh process such as ss-eval.
ACTIVE = {"bank": None}
RENDER_MLPS = ("mlp_opacity", "mlp_cov", "mlp_color")


def expon_lr(lr_init, lr_final, max_steps):
    """get_expon_lr_func with lr_delay_steps = 0 (official defaults)."""

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        t = np.clip(step / max_steps, 0, 1)
        return float(np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t))

    return helper


def scaffold_mlps(feat_dim, n_offsets):
    """mlp_opacity / mlp_cov / mlp_color with add_*_dist = False, appearance_dim = 0,
    use_feat_bank = False (train_mip360.sh / train_tnt.sh / train_db.sh): input [feat, view dir]."""
    d = feat_dim + 3
    return nn.ModuleDict({
        "mlp_opacity": nn.Sequential(nn.Linear(d, feat_dim), nn.ReLU(True),
                                     nn.Linear(feat_dim, n_offsets), nn.Tanh()),
        "mlp_cov": nn.Sequential(nn.Linear(d, feat_dim), nn.ReLU(True),
                                 nn.Linear(feat_dim, 7 * n_offsets)),
        "mlp_color": nn.Sequential(nn.Linear(d, feat_dim), nn.ReLU(True),
                                   nn.Linear(feat_dim, 3 * n_offsets), nn.Sigmoid()),
    })


class MLPBank(nn.Module):
    def __init__(self, c, device):
        super().__init__()
        self.feat_dim, self.n_offsets = c["feat_dim"], c["n_offsets"]
        self.mlps = scaffold_mlps(self.feat_dim, self.n_offsets).to(device)
        t, lr = c["total_steps"], c["lr"]
        self.schedules = {k: expon_lr(lr[k][0], lr[k][1], t) for k in RENDER_MLPS}
        self.optimizer = torch.optim.Adam(
            [{"params": list(self.mlps[k].parameters()), "name": k, "lr": 0.0}
             for k in RENDER_MLPS], lr=0.0, eps=1e-15)
        self.update_learning_rate(1)

    def update_learning_rate(self, step):
        for g in self.optimizer.param_groups:
            g["lr"] = self.schedules[g["name"]](step)

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
