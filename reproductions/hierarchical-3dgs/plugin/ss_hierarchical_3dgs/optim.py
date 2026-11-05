"""Optimizers of the official code, as torch.optim.Adam subclasses (state layout unchanged, so
gsplat.strategy.ops keep moments aligned and checkpoints load into torch Adam).

`SparseAdam`  = scene/OurAdam.py `_single_tensor_adam` (train_single.py, our_adam=True): rows
               whose *opacity* gradient is non-zero ("relevant", visible Gaussians) get an Adam
               update; other rows keep parameters and moments untouched. The step counter is
               global (bias correction uses the iteration count). If no row is relevant the
               official code falls back to a dense step (`_single_tensor_adam2`).
`FrozenAdam`  = torch.optim.Adam (train_post.py, our_adam=False) after zeroing the gradient rows
               of the anchors (train_post.py:180-186, leaves/anchors frozen).

The relevant mask is shared by the per-field optimizers of one iteration: the runtime steps the
optimizers in field order and clears each gradient after its step, so the first optimizer of an
iteration computes the mask from params["opacities"].grad and the others reuse it.
"""

import torch


class StepMask:
    def __init__(self, params):
        self.params = params
        self.step = None
        self.mask = None
        self.any = None

    def begin(self, step):
        self.step = step
        self.mask = None

    def relevant(self):
        if self.mask is None:
            g = self.params["opacities"].grad
            if g is None:
                raise RuntimeError("SparseAdam: opacity gradient missing")
            self.mask = (g.reshape(len(g), -1) != 0).any(dim=1)
            self.any = bool(self.mask.any())
        return self.mask, self.any


class SparseAdam(torch.optim.Adam):
    def __init__(self, params, lr, mask, eps=1e-15):
        super().__init__(params, lr=lr, eps=eps)
        self.shared = mask

    @torch.no_grad()
    def step(self, closure=None):
        rel, some = self.shared.relevant()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = torch.tensor(0.0)
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                st["step"] += 1
                t = float(st["step"])
                bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
                g = p.grad
                m = st["exp_avg"].mul(b1).add_(g, alpha=1 - b1)
                v = st["exp_avg_sq"].mul(b2).addcmul_(g, g, value=1 - b2)
                upd = p.addcdiv(m, (v.sqrt() / (bc2 ** 0.5)).add_(group["eps"]),
                                value=-group["lr"] / bc1)
                if some:  # official: dense step when no row is relevant
                    keep = rel.reshape(-1, *([1] * (p.dim() - 1)))
                    m = torch.where(keep, m, st["exp_avg"])
                    v = torch.where(keep, v, st["exp_avg_sq"])
                    upd = torch.where(keep, upd, p)
                st["exp_avg"].copy_(m)
                st["exp_avg_sq"].copy_(v)
                p.copy_(upd)
        return None


class FrozenAdam(torch.optim.Adam):
    def __init__(self, params, lr, frozen, eps=1e-15):
        super().__init__(params, lr=lr, eps=eps)
        self.frozen = frozen

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and self.frozen is not None:
                    p.grad.masked_fill_(self.frozen.reshape(-1, *([1] * (p.dim() - 1))), 0)
        return super().step(closure)
