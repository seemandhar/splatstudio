"""Batched SH updates (paper Sec. 4.2; official train.py `--sh_lower`).

Official: SH bands >= 1 (`_features_rest`) live in a separate Adam (`shoptimizer`) whose
`step()` and `zero_grad()` run only when `iteration % 16 == 0`, so the gradients of 16 backward
passes are summed into one Adam step. Densification replaces the parameter by a new
`nn.Parameter` (no .grad), which discards the partial accumulation.

The native runtime calls `step()` and `zero_grad()` on every optimizer each iteration;
`IntervalAdam` keeps its own gradient sum and performs the Adam step on every `every`-th call.
"""

import torch


class IntervalAdam(torch.optim.Adam):
    def __init__(self, params, lr, every=16, eps=1e-15):
        super().__init__(params, lr=lr, eps=eps)
        if int(every) < 1:
            raise ValueError("every must be >= 1")
        self.every = int(every)
        self.calls = 0
        self._acc = None
        self._acc_param = None

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("IntervalAdam does not support closures")
        self.calls += 1
        (param,) = self.param_groups[0]["params"]
        if self._acc_param is not param:  # population changed: official grad is lost
            self._acc, self._acc_param = None, param
        if param.grad is not None:
            self._acc = param.grad.clone() if self._acc is None else self._acc.add_(param.grad)
        if self.calls % self.every == 0:
            if self._acc is not None:
                param.grad = self._acc
                super().step()
            self._acc = None
