"""DropGaussian density control, unchanged (ss_dropgaussian.density), plus ownership of what the
DWTGS novel-view HF loss needs: the training cameras (runtime `set_training_cameras` hook) and
the initial points (for pseudo_bounds='points'). The loss reaches it through outputs["dwtgs"]."""

from ss_dropgaussian.density import DropGaussianDensity


class DWTGSDensity(DropGaussianDensity):
    def __init__(self, spec):
        super().__init__(spec)
        self.training_cameras = None
        self.initial_points = None

    def initialize(self, params, optimizers, scene_scale=1.0):
        super().initialize(params, optimizers, scene_scale)
        self.initial_points = params["means"].detach().cpu().double().numpy()

    def set_training_cameras(self, cameras):
        self.training_cameras = list(cameras)

    def before_backward(self, params, optimizers, step, outputs):
        super().before_backward(params, optimizers, step, outputs)
        outputs["dwtgs"] = self


def create_density(spec):
    return DWTGSDensity(spec)
