"""Process-local link from the renderer to the density controller that owns the MS sampler.

The native runtime calls `renderer.render(params, camera, sh_degree)` without a step. The MS-GS
density controller (created by the same `build(method)`) registers itself here and decides the
resolution index of every training render. Evaluation renders run under `torch.no_grad()` and
always render the camera they are given (renderer.py), so a stale link never alters evaluation.
"""


class Link:
    def __init__(self):
        self.density = None

    def attach(self, density):
        self.density = density

    def training_level(self):
        density = self.density
        if density is None:
            return 0
        return density.next_training_level()


LINK = Link()
