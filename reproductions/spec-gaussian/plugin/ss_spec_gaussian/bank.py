"""Process-local bank of the global Spec-Gaussian modules (Theta + Psi) and their optimizer.

The density component creates the bank in ``initialize`` and publishes it as ``ACTIVE``; the
renderer reads the live network and the current step from it during training. After every step
the decoder is packed into the ``spec_blob`` field so the checkpoint carries it to ss-eval.
Optimizer: scene/specular_model.py SpecularModel.train_setting (Adam eps 1e-15, lr linear
feature_lr -> feature_lr/20 over specular_lr_max_steps, updated after every step).
"""

import torch

from .asg import build_network, linear_lr

ACTIVE = None  # live bank of the current training process (None in ss-eval)


class SpecBank:
    def __init__(self, cfg, device):
        self.cfg = dict(cfg)
        self.network = build_network(cfg).to(device)
        self.schedule = linear_lr(cfg["mlp_lr_init"], cfg["mlp_lr_final"], cfg["mlp_lr_max_steps"])
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.schedule(0), eps=1e-15)
        self.step = 0  # last completed runtime step
        self.updates = 0

    def current_step(self):
        return self.step + 1

    def after_step(self, step):
        if any(p.grad is not None for p in self.network.parameters()):
            self.optimizer.step()
            self.updates += 1
        self.optimizer.zero_grad(set_to_none=True)
        lr = self.schedule(step)  # official: update_learning_rate(iteration) after the step
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.step = step
        return lr
