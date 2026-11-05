"""Process-local link from the renderer to the density controller that owns the MVGS schedule.

The runtime calls `renderer.render(params, camera, sh_degree)` without a step and gives losses a
single target, so the density controller (which has the training images and the stage schedule)
publishes the batch of views for the coming step here and the renderer reads it (DashGaussian
precedent, reproductions/dashgaussian/plugin/ss_dashgaussian/link.py).
"""


class Link:
    def __init__(self):
        self.density = None

    def attach(self, density):
        self.density = density

    def batch(self):
        """Views for the next training render, or None when no multi-view schedule is active."""
        density = self.density
        if density is None:
            return None
        return density.current_batch()


LINK = Link()
