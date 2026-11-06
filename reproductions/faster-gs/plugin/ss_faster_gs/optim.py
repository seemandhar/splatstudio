"""Faster-GS CUDA Adam (paper §3.3 "Parameter Updates").

`FusedAdam` = `FasterGSCudaBackend/torch_bindings/adam.py`: a `torch.optim.Adam` subclass with the
same state layout (`step` int, `exp_avg`, `exp_avg_sq`) so that density operations which edit the
moments (gsplat ops, Morton re-ordering) keep it aligned, and whose `step` launches the vendored
`adam_step` kernel (PyTorch Adam numerics, fmaf + fast math).
"""

import torch

from .kernels import extension


class FusedAdam(torch.optim.Adam):
    def __init__(self, params, lr, eps=1e-15, betas=(0.9, 0.999)):
        super().__init__(params=params, lr=lr, eps=eps, betas=betas)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("FusedAdam does not support closures")
        for group in self.param_groups:
            if len(group["params"]) != 1:
                raise ValueError("FusedAdam expects exactly one tensor per parameter group")
            param = group["params"][0]
            if param.grad is None or param.numel() == 0:
                continue
            state = self.state[param]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
            state["step"] += 1
            extension().adam_step(param.grad.contiguous(), param, state["exp_avg"],
                                  state["exp_avg_sq"], int(state["step"]), float(group["lr"]),
                                  *group["betas"], group["eps"])


def replace_optimizers(params, optimizers, lr_overrides=None):
    """Swap the runtime's per-field `torch.optim.Adam` for `FusedAdam` in place (before training)."""
    lr_overrides = lr_overrides or {}
    for name in list(optimizers):
        old = optimizers[name]
        if len(old.param_groups) != 1 or len(old.param_groups[0]["params"]) != 1:
            raise ValueError(f"{name}: expected one parameter per optimizer")
        if len(old.state):
            raise ValueError(f"{name}: optimizer already has state; replace before training")
        (param,) = old.param_groups[0]["params"]
        if param is not params[name]:
            raise ValueError(f"{name}: optimizer does not own the current parameter")
        group = old.param_groups[0]
        optimizers[name] = FusedAdam([param], lr=float(lr_overrides.get(name, group["lr"])),
                                     eps=group["eps"], betas=group["betas"])
