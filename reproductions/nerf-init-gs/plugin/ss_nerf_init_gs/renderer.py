"""gsplat renderer + NeRF depth lookup for the depth-distillation loss (paper Sec. 3.3).

Wraps SplatStudio's built-in gsplat renderer (unchanged rasterization, RGB+ED) and adds

- `outputs["depth_accum"]` = D_GS of Eq. 5 = sum_i d_i alpha_i T_i (not alpha-normalised),
  recovered as ED * alpha. d_i is gsplat's camera-z of the Gaussian centre, not the paper's
  ray-conditional mean (Supp. "Depth rendering with Gaussians"); see README fidelity gaps;
- `outputs["nerf_depth"]` = the precomputed D_NeRF (Eq. 6, camera z) of the rendered training
  view, looked up in the `nerfinit/` sidecar by viewmat (training renders only).
"""

from splatstudio.backends.gsplat import Renderer as GsplatRenderer
from splatstudio.specs import ComponentSpec

from . import sidecar

CAPABILITIES = {"render.rgb", "render.depth.expected_z", "gradient.screen.signed",
                "render.depth.accumulated_z", "nerfinit.nerf_depth"}


class NerfInitRenderer:
    def __init__(self, spec):
        if spec.config:
            raise ValueError(f"Unknown nerf-init-gs renderer options: {sorted(spec.config)}")
        self.inner = GsplatRenderer(
            ComponentSpec(spec.name, "renderer", spec.factory, spec.requires, (), {})
        )
        self.capabilities = set(CAPABILITIES)
        self._table = False
        self.table_override = None  # probes

    def table(self):
        if self.table_override is not None:
            return self.table_override
        if self._table is False:
            self._table = sidecar.table()
        return self._table

    def render(self, params, camera, sh_degree=3):
        out = self.inner.render(params, camera, sh_degree=sh_degree)
        out["depth_accum"] = out["depth"] * out["alpha"]  # Eq. 5 (ED * alpha = sum w z)
        training = params["means"].requires_grad and out["rgb"].requires_grad
        tab = self.table() if training else None
        if tab is not None and tab.depth is not None:
            idx = tab.index(camera["viewmat"])
            if idx is None:
                raise KeyError("Training view not found in the nerfinit sidecar (viewmat mismatch)")
            depth = tab.depth_view(idx, out["rgb"].device)
            if depth.shape[:2] != out["rgb"].shape[:2]:
                raise ValueError(f"NeRF depth {tuple(depth.shape)} vs render {tuple(out['rgb'].shape)}")
            out["nerf_depth"] = depth
            out["nerf_scale"] = tab.nerf_scale  # world -> nerfacc normalised scene units
        return out


def create_renderer(spec):
    return NerfInitRenderer(spec)
