"""LiteGS sparse Adam (official litegs/training/optimizer.py SparseGaussianAdam +
submodules/gaussian_raster/compact.cu adamUpdate).

Official numerics: only primitives visible in the current view are updated (moments and
parameter); m = b1 m + (1-b1) g, v = b2 v + (1-b2) g^2, p -= lr * m / (sqrt(v) + eps) with **no
bias correction**; eps = 1e-15. The official visibility unit is a 128-primitive Morton cluster whose
AABB intersects the frustum; here it is the per-primitive gsplat visibility (radii > 0).
"""

import torch


class Visibility:
    """Holder for the current view's [N] visibility mask, shared by all LiteAdam instances."""

    def __init__(self):
        self.mask = None


class LiteAdam(torch.optim.Adam):
    def __init__(self, params, lr, visibility, eps=1e-15, betas=(0.9, 0.999), sparse=True,
                 bias_correction=False):
        super().__init__(params, lr=lr, eps=eps, betas=betas)
        self.visibility = visibility
        self.sparse = bool(sparse)
        self.bias_correction = bool(bias_correction)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("LiteAdam does not support closures")
        for group in self.param_groups:
            (p,) = group["params"]
            if p.grad is None:
                continue
            st = self.state[p]
            if "exp_avg" not in st:
                st["step"] = torch.tensor(0.0)
                st["exp_avg"] = torch.zeros_like(p)
                st["exp_avg_sq"] = torch.zeros_like(p)
            st["step"] += 1
            b1, b2 = group["betas"]
            lr, eps = group["lr"], group["eps"]
            if self.sparse:
                mask = self.visibility.mask
                if mask is None or mask.shape[0] != p.shape[0]:
                    raise RuntimeError("LiteAdam visibility mask missing or misaligned")
                rows = torch.where(mask)[0]
                g = p.grad[rows]
                m = st["exp_avg"][rows].mul_(b1).add_(g, alpha=1 - b1)
                v = st["exp_avg_sq"][rows].mul_(b2).addcmul_(g, g, value=1 - b2)
                st["exp_avg"][rows] = m
                st["exp_avg_sq"][rows] = v
            else:
                g = p.grad
                m = st["exp_avg"].mul_(b1).add_(g, alpha=1 - b1)
                v = st["exp_avg_sq"].mul_(b2).addcmul_(g, g, value=1 - b2)
            if self.bias_correction:
                t = float(st["step"])
                upd = (m / (1 - b1**t)) / ((v / (1 - b2**t)).sqrt() + eps)
            else:
                upd = m / (v.sqrt() + eps)
            if self.sparse:
                p[rows] -= lr * upd
            else:
                p.sub_(lr * upd)
