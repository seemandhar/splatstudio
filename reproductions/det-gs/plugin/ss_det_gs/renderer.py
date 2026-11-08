"""gsplat renderer (unchanged) that also returns the camera, so losses can look up per-view priors."""

import torch
from gsplat import rasterization
from splatstudio.backends.gsplat import Renderer as GsplatRenderer


class Renderer(GsplatRenderer):
    def render(self, params, camera, sh_degree=3):
        out = super().render(params, camera, sh_degree)
        out["camera"] = {k: camera[k] for k in ("viewmat", "K", "width", "height")}
        return out


def create_renderer(spec):
    return Renderer(spec)


def hard_depth(params, camera, omega=0.95):
    """Eq. 4: alpha compositing with every opacity = omega (weights omega (1-omega)^(n-1) G),
    camera-z depth, not alpha-normalised (gsplat "D"). Scales/rotations are detached as in
    DNGaussian's render_for_depth, so the gradient reaches only the centres. Returns [1,1,H,W]."""
    n = len(params["means"])
    dev = params["means"].device
    render, _, _ = rasterization(
        means=params["means"], quats=params["quats"].detach(), scales=params["scales"].detach().exp(),
        opacities=torch.full((n,), float(omega), device=dev), colors=torch.ones(n, 3, device=dev),
        viewmats=camera["viewmat"][None], Ks=camera["K"][None],
        width=int(camera["width"]), height=int(camera["height"]),
        sh_degree=None, packed=False, render_mode="D", rasterize_mode="classic",
    )
    return render[0].permute(2, 0, 1)[None]
