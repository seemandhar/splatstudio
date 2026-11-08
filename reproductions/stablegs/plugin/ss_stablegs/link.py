"""Process-local link from the renderer to the density controller that owns the pair schedule.

The runtime renders one random camera without a step and gives losses a single target; the
density controller (which holds the training images, the pair stack and the step) publishes the
pair of views for the coming step here and the renderer reads it (MVGS / DashGaussian precedent).
"""


class Link:
    def __init__(self):
        self.density = None

    def attach(self, density):
        self.density = density

    def batch(self):
        density = self.density
        if density is None:
            return None
        return density.current_batch()


LINK = Link()
