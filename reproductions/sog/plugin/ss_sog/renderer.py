"""gsplat renderer for SOG's log-contracted positions (paper §3.3).

`params["means"]` holds the official `_xyz` (log space); the renderer rasterizes
`get_xyz = sign(y) * (exp|y| - 1)` (official scene/gaussian_model.py:27-32, 123-125).
"""

import torch
from splatstudio.backends.gsplat import Renderer


def log_transform(x):
    """World -> stored coordinates: sign(x) log(1 + |x|) (official `log_transform`)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def inverse_log_transform(y):
    """Stored -> world coordinates, paper §3.3: x = sign(x_log) (exp(|x_log|) - 1)."""
    return torch.sign(y) * torch.expm1(torch.abs(y))


class SOGRenderer(Renderer):
    def render(self, params, camera, sh_degree=3):
        world = dict(params.items())
        world["means"] = inverse_log_transform(params["means"])
        return super().render(world, camera, sh_degree)


def create_renderer(spec):
    return SOGRenderer(spec)
