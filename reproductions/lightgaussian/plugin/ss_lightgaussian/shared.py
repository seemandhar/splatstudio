"""Run state shared by the LightGaussian density, renderer and losses (one run per process).

The density's `initialize` activates it; outside a training run (e.g. `ss-eval`) it stays
inactive, so the renderer renders the stored (decoded) model unchanged and the losses are inert.
"""


class RunState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.cfg = None
        self.next_step = 1  # optimizer step whose forward pass happens next
        self.teacher = None  # dict of detached SH-3 params (distillation)
        self.generator = None  # torch.Generator for pseudo-view offsets
        self.last_offset = None  # world-space camera-centre shift of the last pseudo view

    def stage(self):
        from . import schedule

        if not self.active:
            return "done"
        return schedule.stage(self.next_step, self.cfg)


STATE = RunState()
