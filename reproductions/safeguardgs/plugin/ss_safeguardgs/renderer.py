"""Stock gsplat renderer plus a pass-through of the frame's target image.

SafeguardGS scores Gaussians against the ground-truth pixel colour (CS1 in Score_SG_36). The
native runtime gives density components cameras only, so the renderer forwards the image and
viewmat of the frame it renders; the density caches it for training cameras (matched by viewmat).
Pattern copied from the mini-splatting plugin in this workspace
(reproductions/mini-splatting/plugin/ss_mini_splatting/renderer.py).
"""

from splatstudio.backends.gsplat import Renderer

TARGET_CAPABILITY = "training.target_image"
CAPABILITIES = (
    "render.rgb",
    "render.depth.expected_z",
    "gradient.screen.signed",
    TARGET_CAPABILITY,
)


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
