"""Multi-view update (paper Sec. 3.5; official training_loop.run_3dgs_parameter_update_method).

Official: on iterations where `should_run_parameter_update` is false neither
`optimizer.step()` nor `optimizer.zero_grad()` runs, so the gradients of the skipped single-view
backward passes are summed in `.grad` and applied in one Adam step. The native runtime calls
`step()`/`zero_grad()` on every optimizer each iteration, so the density controller replaces each
field's Adam with a `GatedAdam` (same parameter, hyper-parameters and state) whose two calls are
no-ops when the shared gate says so.
"""

import torch


class Gate:
    """Holds the current one-based step (set by the density in before_backward)."""

    def __init__(self, rule):
        self.rule = rule
        self.step = 0

    def open(self):
        return bool(self.rule(self.step))


class GatedAdam(torch.optim.Adam):
    def __init__(self, base, gate):
        group = base.param_groups[0]
        super().__init__(group["params"], lr=group["lr"], betas=group["betas"],
                         eps=group["eps"], weight_decay=group["weight_decay"])
        self.load_state_dict(base.state_dict())
        self.gate = gate
        self.skipped = 0

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("GatedAdam does not support closures")
        if not self.gate.open():
            self.skipped += 1
            return None
        return super().step()

    def zero_grad(self, set_to_none=True):
        if self.gate.open():
            super().zero_grad(set_to_none=set_to_none)


def gate_optimizers(optimizers, gate):
    """Replace every optimizer in place (the runtime iterates this same dict)."""
    for name in list(optimizers):
        if not isinstance(optimizers[name], GatedAdam):
            optimizers[name] = GatedAdam(optimizers[name], gate)
    return optimizers
