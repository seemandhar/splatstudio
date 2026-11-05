"""gsplat renderer with a per-Gaussian SH band count (official Reduced-3DGS `_degrees`).

The official rasterizer evaluates, for Gaussian i, only the SH bands up to `degrees[i]`
(`forward.cu:computeColorFromSH`, `int deg = degs[idx]`), and back-propagates only into those
coefficients. gsplat evaluates one global degree, so the coefficients above each Gaussian's own
degree are multiplied by a 0/1 mask: identical colours, and zero gradient for the dropped
coefficients. The global degree (runtime schedule, one band per 1000 steps) still applies, so
the effective degree is min(own, active), as in the official code where `_degrees` is bumped by
`oneupSHdegree`.
"""

import torch
from splatstudio.backends.gsplat import Renderer

DEGREE_FIELD = "sh_degree"
# index j of shN (coefficient j+1) belongs to band b(j): j < (b+1)^2 - 1
COEFF_BAND = torch.tensor([1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3], dtype=torch.float32)


def coefficient_mask(degree, active=3):
    """[N,15,1] float mask: 1 where coefficient band <= min(own degree, active degree)."""
    bands = COEFF_BAND.to(degree.device)
    own = degree.detach().reshape(-1, 1).clamp(max=active)
    return (bands[None, :] <= own).to(torch.float32)[..., None]


class ReducedRenderer(Renderer):
    def render(self, params, camera, sh_degree=3):
        masked = dict(params.items())
        masked["shN"] = params["shN"] * coefficient_mask(params[DEGREE_FIELD], 3)
        return super().render(masked, camera, sh_degree)


def create_renderer(spec):
    return ReducedRenderer(spec)
