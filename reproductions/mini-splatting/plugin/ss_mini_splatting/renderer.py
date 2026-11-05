"""Stock gsplat renderer plus a pass-through of the frame's target image.

Mini-Splatting's depth reinitialization colours the re-sampled points with the ground-truth
pixel colours of the training views (official ms/train.py: `gt[indices]`). The native runtime
gives density components cameras only, so the renderer forwards the image of the frame it
renders; the density caches it for training cameras (matched by viewmat) and never sees
evaluation frames, which are rendered only after training.
"""

from splatstudio.backends.gsplat import Renderer

TARGET_CAPABILITY = "training.target_image"


class TargetPassRenderer(Renderer):
    def __init__(self, spec):
        super().__init__(spec)
        self.capabilities = set(self.capabilities) | {TARGET_CAPABILITY}

    def render(self, params, camera, sh_degree=3):
        outputs = super().render(params, camera, sh_degree)
        outputs["target_image"] = camera.get("image")
        outputs["viewmat"] = camera["viewmat"]
        return outputs


def create_renderer(spec):
    return TargetPassRenderer(spec)
