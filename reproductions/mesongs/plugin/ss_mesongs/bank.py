"""Fine-tune state of the compressed model (C8) and its process-local slot.

Pattern of the c3dgs plugin's `CompressedBank` (reproductions/c3dgs/plugin/ss_c3dgs/bank.py): the
runtime cannot hold global learned tensors, so the density owns a `MesonBank` (RAHT tables, block
quantisers, SH codebook + per-Gaussian indices and an Adam for the codebook) registered in
`ACTIVE`; the mesongs renderer renders from it while it is active. Per-Gaussian attributes
(opacity logit, quaternion, SH DC, log scale) stay runtime fields; positions are the frozen voxel
corners.

Forward = official `ft_render` (raht=True, per_block_quant=True, use_indexed=True):
q = r / |r|; e = Euler(q); rf = [o, e, dc] (7 ch) -> RAHT -> C[0] kept, C[1:] block-fake-quantised
per channel -> inverse RAHT -> opacity = sigmoid, covariance from (exp(Q(log s)), e),
colours = [dc, codebook[idx]].
"""

import torch

from . import euler, raht
from .quant import BlockQuantizer

ACTIVE = {"bank": None}
RF_CHANNELS = 7  # opacity, 3 Euler angles, 3 SH DC


def rf_of(params):
    r = params["quats"]
    q = r / torch.sqrt((r * r).sum(-1, keepdim=True))
    e = euler.to_euler(q)
    return torch.cat((params["opacities"][:, None], e, params["sh0"].reshape(-1, 3)), -1)


class MesonBank:
    def __init__(self, tables, n_block, bits, codebook, indices, codebook_lr, device):
        n = tables.n
        self.tables = tables
        self.n_block = n_block
        self.ac_q = BlockQuantizer(n - 1, RF_CHANNELS, n_block, bits, device)
        self.scale_q = BlockQuantizer(n, 3, n_block, bits, device)
        self.codebook = codebook.detach().clone().requires_grad_(True)  # [K, 45]
        self.indices = indices.detach().long()
        # official finetuning_setup: one Adam, eps 1e-15, fresh state; codebook = 'f_rest' group
        self.optimizer = torch.optim.Adam([{"params": [self.codebook], "lr": codebook_lr,
                                            "name": "f_rest"}], lr=0.0, eps=1e-15)
        self.decoded = False
        self.steps = 0

    @property
    def n(self):
        return self.tables.n

    def coefficients(self, params):
        """RAHT coefficients [N, 7] of the current attributes (differentiable)."""
        return raht.raht(rf_of(params), self.tables)

    def attributes(self, params, update=True):
        C = self.coefficients(params)
        quant_ac = self.ac_q(C[1:], update)
        features = raht.iraht(torch.cat((C[:1], quant_ac), 0), self.tables)
        scales = torch.exp(self.scale_q(params["scales"], update))
        e = features[:, 1:4]
        return {
            "means": params["means"],
            "opacities": torch.sigmoid(features[:, 0]),
            "covars": euler.covariance_from_euler(scales, e),
            "colors": torch.cat((features[:, 4:7].reshape(-1, 1, 3),
                                 self.codebook.reshape(-1, 15, 3)[self.indices]), 1),
        }

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.steps += 1

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def summary(self):
        return {"gaussians": int(self.n), "codebook": int(len(self.codebook)),
                "n_block": int(self.n_block), "bank_steps": int(self.steps)}
