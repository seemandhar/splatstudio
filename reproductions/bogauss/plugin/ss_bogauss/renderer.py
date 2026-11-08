"""Stock gsplat renderer that also forwards the frame's ground-truth image.

BOGausS's split priority needs the per-pixel squared error (R(p) - I(p))^2 of the training view
being optimized. Densities do not receive the target image, so the renderer passes it on as
`outputs["target_image"]` (same pattern as the SplatStudio Taming 3DGS port).
"""

from splatstudio.backends.gsplat import Renderer

TARGET_CAPABILITY = "training.target_image"


class BOGausSRenderer(Renderer):
    def __init__(self, spec):
        super().__init__(spec)
        self.capabilities = set(self.capabilities) | {TARGET_CAPABILITY}

    def render(self, params, camera, sh_degree=3):
        outputs = super().render(params, camera, sh_degree)
        outputs["target_image"] = camera.get("image")
        return outputs


def create_renderer(spec):
    return BOGausSRenderer(spec)
