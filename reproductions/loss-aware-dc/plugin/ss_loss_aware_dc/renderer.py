"""gsplat renderer of the runtime (identical rasterization call, white background) that also
exposes the camera and SH degree the density needs for its surrogate statistics."""

from splatstudio.backends.gsplat import Renderer as _GsplatRenderer

CAMERA_CAPABILITY = "surrogate.camera"
CAPABILITIES = {"render.rgb", "render.depth.expected_z", "gradient.screen.signed",
                CAMERA_CAPABILITY}


class Renderer(_GsplatRenderer):
    def __init__(self, spec):
        super().__init__(spec)
        self.capabilities = set(self.capabilities) | {CAMERA_CAPABILITY}

    def render(self, params, camera, sh_degree=3):
        out = super().render(params, camera, sh_degree)
        out["info"]["ss_camera"] = {k: camera[k] for k in ("K", "viewmat", "width", "height")}
        out["info"]["ss_sh_degree"] = int(sh_degree)
        return out


def create_renderer(spec):
    return Renderer(spec)
