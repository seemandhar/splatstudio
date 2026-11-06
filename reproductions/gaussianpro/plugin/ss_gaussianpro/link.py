"""Process-local state shared by the GaussianPro components (PGSR/MVGS precedent).

The runtime gives the density controller cameras only and the losses only
``(params, outputs, target)``. The density reloads the training split (cameras + images) and
publishes it here together with the official per-view caches (``Camera.depth`` /
``Camera.normal`` in the official code) and the shared-SfM-point neighbour lists. The renderer
uses it to identify the rendered training view and to attach that view's propagated normal.
"""

import torch


class Link:
    def __init__(self):
        self.reset()

    def reset(self):
        self.renderer = None
        self.reset_views()

    def reset_views(self):
        """Forget views and caches (a new density controller); keeps the renderer handle."""
        self.cameras = None        # training cameras (runtime order)
        self.images = None         # [H,W,3] uint8 CPU, round(255 * image) (exact loader values)
        self.neighbours = None     # list of lists of training indices
        self.depth_cache = None    # official viewpoint_cam.depth (propagated depth, mutated)
        self.normal_targets = None  # official viewpoint_cam.normal ([H,W,3], -10 = invalid)
        self.propagated = None     # official propagation_dict
        self._viewmats = None

    def set_views(self, cameras, images, neighbours):
        if not (len(cameras) == len(images) == len(neighbours)) or not cameras:
            raise ValueError("cameras, images and neighbour lists must be aligned and non-empty")
        self.cameras = cameras
        self.images = images
        self.neighbours = neighbours
        n = len(cameras)
        self.depth_cache = [None] * n
        self.normal_targets = [None] * n
        self.propagated = [False] * n
        self._viewmats = torch.stack([c["viewmat"].float() for c in cameras])

    @property
    def active(self):
        return self.cameras is not None

    def view_index(self, camera):
        """Training index of ``camera`` (exact viewmat match) or None (e.g. a test view)."""
        if not self.active:
            return None
        vm = camera["viewmat"].float().to(self._viewmats.device)
        err = (self._viewmats - vm[None]).abs().flatten(1).max(dim=1).values
        i = int(err.argmin())
        if float(err[i]) > 1e-6 or int(camera["width"]) != int(self.cameras[i]["width"]):
            return None
        return i

    def image_float(self, index, device):
        return self.images[index].to(device).float() / 255.0


LINK = Link()
