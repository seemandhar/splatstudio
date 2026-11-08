"""gsplat renderer that optimises all SH bands from step 1 and passes the training frame through.

- ConeGS starts with every SH band active (`GaussianModel(sh_degree, start_sh_degree=3)`; paper
  Sec. 4.4), so the runtime's degree schedule is ignored (`sh_degree` config, default 3).
- `conegs-density` needs the rendered view's camera and ground-truth image to build the per-pixel
  L1 error map (official `l1_full.mean(0)`), so the frame dict is exposed as
  `outputs["conegs_frame"]` (capability `conegs.frame`).
"""

from splatstudio.backends.gsplat import Renderer as GsplatRenderer
from splatstudio.specs import ComponentSpec

CAPABILITIES = {"render.rgb", "render.depth.expected_z", "gradient.screen.signed", "conegs.frame"}


class ConeGSRenderer:
    def __init__(self, spec):
        unknown = set(spec.config) - {"sh_degree", "background"}
        if unknown:
            raise ValueError(f"Unknown conegs renderer options: {sorted(unknown)}")
        self.sh_degree = int(spec.config.get("sh_degree", 3))
        # "white": runtime convention; "black": official ConeGS (white_background: false)
        self.background = spec.config.get("background", "white")
        if self.background not in ("white", "black"):
            raise ValueError("background must be 'white' or 'black'")
        self.inner = GsplatRenderer(
            ComponentSpec(spec.name, "renderer", spec.factory, spec.requires, (), {})
        )
        self.capabilities = set(CAPABILITIES)

    def render(self, params, camera, sh_degree=3):
        outputs = self.inner.render(params, camera, sh_degree=self.sh_degree)
        if self.background == "black":
            outputs["rgb"] = outputs["rgb"] - (1 - outputs["alpha"])
        outputs["conegs_frame"] = camera
        return outputs


def create_renderer(spec):
    return ConeGSRenderer(spec)
