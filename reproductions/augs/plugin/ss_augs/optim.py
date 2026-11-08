"""Joint-optimisation optimiser (official refine.py): diff_gaussian_2d_rasterization's
SparseGaussianAdam -- Adam *without bias correction* (cuda_rasterizer/adam.cu adamUpdateCUDA)
applied only to selected Gaussian rows; unselected rows keep parameters and moments untouched.

Row selection here: the augmented Gaussians (contiguous tail starting at `n_base`) always; for the
opacity field also the original Gaussians the view used (official `is_used`: contributed to at
least one pixel), detected as a nonzero opacity gradient."""

import torch


class SparseRowAdam(torch.optim.Optimizer):
    def __init__(self, param, lr, n_base, include_used_base=False, eps=1e-15, betas=(0.9, 0.999)):
        super().__init__([param], {"lr": lr, "eps": eps, "betas": betas})
        self.n_base = int(n_base)
        self.include_used_base = bool(include_used_base)

    def rows(self, p):
        n = p.shape[0]
        mask = torch.zeros(n, dtype=torch.bool, device=p.device)
        mask[self.n_base :] = True
        if self.include_used_base:
            g = p.grad.reshape(n, -1)[: self.n_base]
            mask[: self.n_base] = (g != 0).any(1)
        return mask

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            p = group["params"][0]
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            b1, b2 = group["betas"]
            m, v, g = state["exp_avg"], state["exp_avg_sq"], p.grad
            if self.include_used_base:
                mask = self.rows(p).reshape((-1,) + (1,) * (p.dim() - 1))
                m.copy_(torch.where(mask, b1 * m + (1 - b1) * g, m))
                v.copy_(torch.where(mask, b2 * v + (1 - b2) * g * g, v))
                p.copy_(torch.where(mask, p - group["lr"] * m / (v.sqrt() + group["eps"]), p))
            else:
                s = slice(self.n_base, None)
                m[s].mul_(b1).add_(g[s], alpha=1 - b1)
                v[s].mul_(b2).addcmul_(g[s], g[s], value=1 - b2)
                p[s].addcdiv_(m[s], v[s].sqrt().add_(group["eps"]), value=-group["lr"])
