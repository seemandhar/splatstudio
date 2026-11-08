"""Stock gsplat renderer plus pass-through of the frame's camera and target image.

The sensitivity branch (loss.py) re-renders the current camera with colours 1 - sigma(eps), and
the depth reinitialization (density.py) colours re-sampled points with ground-truth pixels
(official train.py: `gt[indices]`); the native runtime passes neither to components.
"""

from splatstudio.backends.gsplat import Renderer

TARGET_CAPABILITY = "training.target_image"
CAMERA_CAPABILITY = "training.camera"
CAMERA_KEYS = ("K", "viewmat", "width", "height")


class PassThroughRenderer(Renderer):
    def __init__(self, spec):
        super().__init__(spec)
        self.capabilities = set(self.capabilities) | {TARGET_CAPABILITY, CAMERA_CAPABILITY}

    def render(self, params, camera, sh_degree=3):
        outputs = super().render(params, camera, sh_degree)
        outputs["target_image"] = camera.get("image")
        outputs["viewmat"] = camera["viewmat"]
        outputs["camera"] = {k: camera[k] for k in CAMERA_KEYS}
        return outputs


def create_renderer(spec):
    return PassThroughRenderer(spec)
