"""Taming-3DGS sparse Adam, as used by Mini-Splatting2 ("3DGS-accel").

Official `SparseGaussianAdam` (mini-splatting2 submodules/diff-gaussian-rasterization_ms,
cuda_rasterizer/adam.cu): every call increments the step count t; rows of Gaussians with
radii > 0 get m = b1 m + (1-b1) g, v = b2 v + (1-b2) g^2, p -= lr' m / (sqrt(v) + eps) with
lr' = lr sqrt(1 - b2^t) / (1 - b1^t); all other rows (moments included) are left untouched.
On CUDA the update runs in gsplat's fused selective-Adam kernel (which has no bias correction,
so lr' is passed in); on CPU the same update is written in torch.
"""

import torch

from .core import sparse_adam_lr


class SparseGaussianAdam(torch.optim.Adam):
    def __init__(self, params, lr, eps=1e-15):
        super().__init__(params, lr=lr, eps=eps, betas=(0.9, 0.999))
        self.visibility = None

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("SparseGaussianAdam does not support closures")
        if self.visibility is None:
            raise RuntimeError("SparseGaussianAdam.step needs the visibility set by the density")
        vis = self.visibility
        for group in self.param_groups:
            (param,) = group["params"]
            if param.grad is None or param.numel() == 0:
                continue
            if vis.shape != (param.shape[0],):
                raise RuntimeError("Visibility mask does not match the Gaussian count")
            state = self.state[param]
            if len(state) == 0:
                state["step"] = torch.tensor(0.0)
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
            state["step"] += 1
            b1, b2 = group["betas"]
            lr = sparse_adam_lr(group["lr"], float(state["step"]), b1, b2)
            m, v, g = state["exp_avg"], state["exp_avg_sq"], param.grad
            if param.is_cuda:
                from gsplat.cuda._wrapper import adam

                adam(param, g.contiguous(), m, v, vis.contiguous(), lr, b1, b2, group["eps"])
            else:
                rows = vis.nonzero().squeeze(1)
                m[rows] = b1 * m[rows] + (1 - b1) * g[rows]
                v[rows] = b2 * v[rows] + (1 - b2) * g[rows] * g[rows]
                param[rows] -= lr * m[rows] / (v[rows].sqrt() + group["eps"])
        self.visibility = None


def sparsify(optimizers):
    """Replace each per-field Adam by a SparseGaussianAdam on the same parameter and lr."""
    for name, opt in list(optimizers.items()):
        group = opt.param_groups[0]
        new = SparseGaussianAdam(group["params"], lr=group["lr"], eps=group["eps"])
        optimizers[name] = new
