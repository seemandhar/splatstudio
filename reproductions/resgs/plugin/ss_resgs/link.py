"""Process-local link from the renderer to the density controller that owns the schedule.

The native runtime calls `renderer.render(params, camera, sh_degree)` without a step, so the
renderer cannot follow the image-pyramid schedule by itself. The density controller built by
the same `build(method)` registers here; once initialized it reports the pyramid level of the
next training render. Evaluation renders run under `torch.no_grad()` and always use full
resolution (renderer.py), so a stale link never changes evaluation.
"""


class Link:
    def __init__(self):
        self.density = None

    def attach(self, density):
        self.density = density

    def training_level(self):
        """(pyramid level, owning density or None) for the next training render."""
        density = self.density
        if density is None or not density.active:
            return 0, None
        return density.pyramid_level, density


LINK = Link()
