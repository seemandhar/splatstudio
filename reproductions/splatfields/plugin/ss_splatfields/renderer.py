"""gsplat renderers for SplatFields.

`SplatFieldsRenderer`: while a training `SplatFieldsNet` is active and not yet baked, every
splat attribute is predicted by the network from the detached anchors (official train.py
`get_gaussian_dict`: `xyz = get_xyz.detach()`, `scaling = get_scaling.detach()`, `deform.step`)
and rendered with precomputed colours (`colors_precomp`, no SH). Without an active network (any
fresh process, e.g. `ss-eval`) or after baking, the base fields are rendered as-is: they then
hold the baked Gaussians.

Both renderers also return the camera viewmat (mask-loss lookup); `SplatFieldsRenderer`
returns the predicted centres as `outputs["sf_means"]` for the splat-norm loss.
"""

from dataclasses import replace

from gsplat import rasterization
from splatstudio.backends.gsplat import Renderer

from . import network

CAPS = {"render.rgb", "render.depth.expected_z", "gradient.screen.signed", "render.alpha"}


class PlainRenderer(Renderer):
    """gsplat Renderer that also declares render.alpha and returns the frame viewmat."""

    def __init__(self, spec):
        super().__init__(replace(spec, config={}))
        if spec.config:
            raise ValueError("splatfields renderers accept no configuration")
        self.capabilities = set(CAPS)

    def render(self, params, camera, sh_degree=3):
        out = super().render(params, camera, sh_degree)
        out["viewmat"] = camera["viewmat"]
        return out


class SplatFieldsRenderer(PlainRenderer):
    def __init__(self, spec):
        super().__init__(spec)
        self.capabilities = set(CAPS) | {"render.splat_centres"}
        self.net = None  # explicit network (probes); otherwise network.ACTIVE

    def active_net(self):
        net = self.net if self.net is not None else network.ACTIVE["net"]
        if net is None or net.baked:
            return None
        return net

    def render(self, params, camera, sh_degree=3):
        net = self.active_net()
        if net is None:
            return super().render(params, camera, sh_degree)
        pred = net(params["means"].detach(), params["scales"].detach().exp())
        render, alpha, info = rasterization(
            means=pred["means"],
            quats=pred["quats"],
            scales=pred["scales"],
            opacities=pred["opacities"],
            colors=pred["rgb"],
            viewmats=camera["viewmat"][None],
            Ks=camera["K"][None],
            width=camera["width"],
            height=camera["height"],
            sh_degree=None,
            packed=False,
            render_mode="RGB+ED",
            absgrad=False,
            rasterize_mode="classic",
        )
        return {
            "rgb": render[0, ..., :3] + (1 - alpha[0]),  # white background
            "depth": render[0, ..., 3:4],
            "alpha": alpha[0],
            "info": info,
            "viewmat": camera["viewmat"],
            "sf_means": pred["means"],
            "sf_pred": pred,
        }


def create_renderer(spec):
    return SplatFieldsRenderer(spec)


def create_plain_renderer(spec):
    return PlainRenderer(spec)
