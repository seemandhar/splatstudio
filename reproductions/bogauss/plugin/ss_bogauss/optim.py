"""BOGausS unbiased sparse Adam (sections "Proposed optimizer", "Precision-aware parameters
updates", "Sparse and unbiased states updates"; Eq. adamNewState).

    alpha_{s,t} = max(1/t, 1 - beta_s);  g_t = g_{t-1} + alpha_{1,t} (grad - g_{t-1})
    g2_t = g2_{t-1} + alpha_{2,t} (grad^2 - g2_{t-1});  theta <- theta - eta Delta g_t / (sqrt(g2_t) + eps)

t is per Gaussian and only incremented when the Gaussian contributes to the rendered image
(`visible`, set by the density before each step), so the filtered states are unbiased without
Adam's bias correction. The paper prints "+" and "g2" without square root; we use descent and
the standard Adam square root.

Modes:
- "position": Delta_theta = Delta_i (means).
- "size": sizes are linearly parameterized: the stored log-scale is converted to sigma, the
  gradient w.r.t. sigma is grad_log / sigma, sigma <- max(sigma - eta Delta_i step, m Delta_i)
  (minimal size from the precision model), and log sigma is stored back.
- "plain": Delta_theta = 1 (quats, opacities, SH).
"""

import torch

MODES = ("position", "size", "plain")


class BOGausSAdam(torch.optim.Optimizer):
    def __init__(self, params, lr, mode="plain", betas=(0.9, 0.999), eps=1e-15,
                 min_size_factor=1.0):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        super().__init__(params, dict(lr=float(lr), betas=tuple(betas), eps=float(eps)))
        if len(self.param_groups) != 1 or len(self.param_groups[0]["params"]) != 1:
            raise ValueError("BOGausSAdam manages exactly one per-Gaussian parameter")
        self.mode = mode
        self.min_size_factor = float(min_size_factor)
        self.visible = None  # bool [N], set before every step
        self.delta = None  # float [N] precision, set before every step

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("BOGausSAdam does not support closures")
        group = self.param_groups[0]
        (p,) = group["params"]
        if p.grad is None:
            return
        n = p.shape[0]
        if self.visible is None or self.visible.shape != (n,):
            raise RuntimeError("BOGausSAdam needs the visibility mask of the current render")
        if self.mode != "plain" and (self.delta is None or self.delta.shape != (n,)):
            raise RuntimeError("BOGausSAdam needs the per-Gaussian precision Delta")
        st = self.state[p]
        if not st:
            st["exp_avg"] = torch.zeros_like(p)
            st["exp_avg_sq"] = torch.zeros_like(p)
            st["t"] = torch.zeros(n, device=p.device)
        idx = torch.nonzero(self.visible).flatten()
        if len(idx) == 0:
            return
        b1, b2 = group["betas"]
        shape = (-1,) + (1,) * (p.dim() - 1)
        t = st["t"][idx] + 1.0
        st["t"][idx] = t
        a1 = torch.maximum(1.0 / t, torch.full_like(t, 1.0 - b1)).view(shape)
        a2 = torch.maximum(1.0 / t, torch.full_like(t, 1.0 - b2)).view(shape)
        grad = p.grad[idx]
        if self.mode == "size":
            sigma = p[idx].exp()
            grad = grad / sigma  # d L / d sigma for the linear size parameterization
        m = st["exp_avg"][idx]
        v = st["exp_avg_sq"][idx]
        m += a1 * (grad - m)
        v += a2 * (grad * grad - v)
        st["exp_avg"][idx] = m
        st["exp_avg_sq"][idx] = v
        upd = group["lr"] * m / (v.sqrt() + group["eps"])
        if self.mode == "plain":
            p[idx] = p[idx] - upd
            return
        d = self.delta[idx].view(shape)
        if self.mode == "position":
            p[idx] = p[idx] - upd * d
        else:
            new_sigma = torch.maximum(sigma - upd * d, self.min_size_factor * d)
            p[idx] = new_sigma.log()
