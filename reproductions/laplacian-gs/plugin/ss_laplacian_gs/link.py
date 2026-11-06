"""Process-local link from the renderer to the density controller that owns the stages.

The native runtime calls `renderer.render(params, camera, sh_degree)` without a step. The
density controller built by the same `build(method)` registers itself here and tells the
training render which pyramid level is active and the stage-local step about to be rendered.
Evaluation renders (torch.no_grad) never consult the link: they compose every level present
in the checkpoint.
"""


class Link:
    def __init__(self):
        self.density = None

    def attach(self, density):
        self.density = density

    def training_stage(self):
        """(active level, stage-local step of the next render)."""
        if self.density is None:
            raise RuntimeError("laplacian-gs renderer used for training without its density")
        return self.density.level, self.density.local_step + 1


LINK = Link()
