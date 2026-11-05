"""Stock gsplat renderer that also forwards the frame's ground-truth image and camera.

Taming's score needs the L1 error and edge map of arbitrary training views at densification
time (official `compute_gaussian_score` reads `camera.original_image`). Densities do not see the
training images, so the renderer passes the current frame's target on (`outputs["target_image"]`,
`outputs["camera"]`) and the density caches every training view it has seen.
"""

from splatstudio.backends.gsplat import Renderer

TARGET_CAPABILITY = "training.target_image"


class TamingRenderer(Renderer):
    def __init__(self, spec):
        super().__init__(spec)
        self.capabilities = set(self.capabilities) | {TARGET_CAPABILITY}

    def render(self, params, camera, sh_degree=3):
        outputs = super().render(params, camera, sh_degree)
        outputs["target_image"] = camera.get("image")
        outputs["camera"] = {k: camera[k] for k in ("K", "viewmat", "width", "height")}
        return outputs


def create_renderer(spec):
    return TamingRenderer(spec)
