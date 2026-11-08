"""Run state shared by the Trick-GS density, renderer and losses (one training run per process).

The native runtime calls `renderer.render(params, camera, sh_degree)` without the step, so the
density (which owns the schedules) activates this state in `initialize` and advances
`next_step` after every `after_backward`. Training renders (autograd enabled, state active)
follow the schedules of `next_step`; any other render (ss-eval in a fresh process, no-grad test
renders, probes without a run) is the final model: full resolution, eps2d 0.3, masks applied.
"""


class RunState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.cfg = None
        self.next_step = 1
        self.lowpass = 0.3  # current eps2d (progressive Gaussian scale, RAIN-GS)
        self.lowpass_history = []  # (step, value) at each recomputation

    def activate(self, cfg):
        self.reset()
        self.active = True
        self.cfg = cfg


STATE = RunState()
