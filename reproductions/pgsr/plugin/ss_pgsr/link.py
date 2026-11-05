"""Process-local link between the PGSR components (MVGS/DashGaussian precedent).

The runtime gives losses only ``(params, outputs, target)`` and the renderer only the camera.
The density controller owns the training views (cameras + grey images of the *training split*,
reloaded from ``--data``) and the official neighbour graph; it publishes them here, tells the
renderer whether the next training step needs the plane/normal maps, and steps the optimizers
of component-held parameters (per-image exposure) after each backward pass.
"""

import torch


class Link:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cameras = None      # list of camera dicts (training split, runtime order)
        self.gray = None         # list of [H,W] grey training images (official 0.299/0.587/0.114)
        self.neighbours = None   # list of lists of training indices
        self._viewmats = None
        self.render_geo = True   # geometry for the next render (density switches it per step)
        self.renderer = None     # renderer instance used for neighbour renders
        self.extra_optimizers = []

    def set_views(self, cameras, gray, neighbours):
        if not (len(cameras) == len(gray) == len(neighbours)) or not cameras:
            raise ValueError("cameras, grey images and neighbour lists must be aligned")
        self.cameras = cameras
        self.gray = gray
        self.neighbours = neighbours
        self._viewmats = torch.stack([c["viewmat"].float() for c in cameras])

    @property
    def active(self):
        return self.cameras is not None

    def view_index(self, camera):
        """Index of ``camera`` in the training list (exact viewmat match), else raise."""
        if not self.active:
            raise RuntimeError("PGSR: training views not registered (pgsr-density "
                               "set_training_cameras / set_training_frames not called)")
        vm = camera["viewmat"].float().to(self._viewmats.device)
        err = (self._viewmats - vm[None]).abs().flatten(1).max(dim=1).values
        i = int(err.argmin())
        if float(err[i]) > 1e-5:
            raise RuntimeError("PGSR: the rendered camera is not a registered training view")
        return i

    def step_extra(self):
        for opt in self.extra_optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)


LINK = Link()
