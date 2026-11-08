"""3DGS-accel optimizer schedule (official `GaussianModel.optimizer_step`, supp. Sec. 8.1).

Official (optimizer_type "default"):
  iteration <= 15000: main Adam (xyz, f_dc, opacity, scaling, rotation) steps every iteration;
                      SH-rest Adam steps when iteration % 16 == 0
  15000 < it <= 20000: both step when iteration % 32 == 0
  it > 20000:          both step when iteration % 64 == 0
zero_grad() runs only after a step, so the gradients of all backward passes in between are
summed into one Adam step. Densification / pruning / opacity resets build new nn.Parameters,
which discards the partial sum.

The native runtime calls step() and zero_grad() on every optimizer each iteration;
`ScheduledAdam` keeps its own gradient sum and performs the Adam step only on scheduled
iterations (generalises the SplatStudio Taming-3DGS port's IntervalAdam). The density sets
`current_step` before the optimizer step; without it the call count is used.
"""

import torch


def main_steps_at(step):
    return step <= 15000 or (step <= 20000 and step % 32 == 0) or (step > 20000 and step % 64 == 0)


def sh_steps_at(step):
    if step <= 15000:
        return step % 16 == 0
    return (step <= 20000 and step % 32 == 0) or (step > 20000 and step % 64 == 0)


SCHEDULES = {"main": main_steps_at, "sh": sh_steps_at, "every": lambda step: True}


class ScheduledAdam(torch.optim.Adam):
    def __init__(self, params, lr, schedule="main", eps=1e-15):
        super().__init__(params, lr=lr, eps=eps)
        if schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {sorted(SCHEDULES)}")
        self.schedule = schedule
        self.calls = 0
        self.current_step = None
        self.adam_steps = 0
        self._acc = None
        self._acc_param = None

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("ScheduledAdam does not support closures")
        self.calls += 1
        step = self.calls if self.current_step is None else int(self.current_step)
        (param,) = self.param_groups[0]["params"]
        if self._acc_param is not param:  # population changed: official grad sum is lost
            self._acc, self._acc_param = None, param
        if param.grad is not None:
            self._acc = param.grad.clone() if self._acc is None else self._acc.add_(param.grad)
        if SCHEDULES[self.schedule](step):
            if self._acc is not None:
                param.grad = self._acc
                super().step()
                self.adam_steps += 1
            self._acc = None


def wrap(optimizers, name, schedule, lr=None):
    old = optimizers[name]
    group = old.param_groups[0]
    new = ScheduledAdam(group["params"], lr=group["lr"] if lr is None else float(lr),
                        schedule=schedule, eps=group["eps"])
    (p,) = group["params"]
    if p in old.state:  # keep any existing moments
        new.state[p] = old.state[p]
    optimizers[name] = new
    return new
