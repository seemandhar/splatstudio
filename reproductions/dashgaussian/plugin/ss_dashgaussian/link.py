"""Process-local link from the renderer to the density controller that owns the schedule.

The native runtime calls `renderer.render(params, camera, sh_degree)` without a step, so the
renderer cannot follow a resolution schedule by itself. The density controller created by the
same `build(method)` registers itself here; the renderer asks it for the render scale of the
step about to be rendered. Evaluation renders run under `torch.no_grad()` and always use full
resolution (see renderer.py), so a stale link can never lower evaluation resolution.
"""


class Link:
    def __init__(self):
        self.density = None

    def attach(self, density):
        self.density = density

    def training_scale(self):
        """Render scale for the next training render, or 1 when no active schedule exists."""
        density = self.density
        if density is None or not density.schedule_active:
            return 1
        return density.current_scale


LINK = Link()
