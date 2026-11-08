"""Stock gsplat renderer plus the rendered frame's target image and viewmat.

LPM needs the ground-truth image of the rendered training view (error map, C1) and the training
images for SuperPoint/LightGlue matching (C2); the native runtime gives density components
cameras only, so the renderer forwards them (same pattern as the GaussianSpa / Mini-Splatting2
ports). Rendering itself is unchanged 3DGS.
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
