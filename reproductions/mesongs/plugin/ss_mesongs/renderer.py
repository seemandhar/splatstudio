"""gsplat renderer that renders the MesonGS bank during the fine-tune.

Without an active, not-yet-decoded `MesonBank` (3DGS phase, after the final decode, and in any
fresh process such as `ss-eval`) it is the unchanged SplatStudio gsplat renderer. With an active
bank it rasterises the bank's fake-quantised attributes with full 3x3 covariances (the official
`ft_render` passes `cov3D_precomp` built from the Euler angles). `rasterize` mirrors
`splatstudio.backends.gsplat.Renderer.render` (pattern of the c3dgs plugin renderer).
"""

from gsplat import rasterization
from splatstudio.backends.gsplat import Renderer

from . import bank as bank_mod


def rasterize(a, camera, sh_degree, white=True):
    render, alpha, info = rasterization(
        means=a["means"],
        quats=None,
        scales=None,
        covars=a["covars"],
        opacities=a["opacities"],
        colors=a["colors"],
        viewmats=camera["viewmat"][None],
        Ks=camera["K"][None],
        width=camera["width"],
        height=camera["height"],
        sh_degree=sh_degree,
        packed=False,
        render_mode="RGB+ED",
    )
    rgb = render[0, ..., :3] + (1 - alpha[0]) if white else render[0, ..., :3]
    return {"rgb": rgb, "depth": render[0, ..., 3:4], "alpha": alpha[0], "info": info}


class MesonRenderer(Renderer):
    def __init__(self, spec):
        super().__init__(spec)
        if self.absgrad or self.antialiased:
            raise ValueError("mesongs-renderer follows the classic 3DGS rasteriser")
        self.bank = None  # explicit bank (probes); otherwise bank.ACTIVE

    def active_bank(self, params):
        bank = self.bank if self.bank is not None else bank_mod.ACTIVE["bank"]
        if bank is None or bank.decoded or bank.n != len(params["means"]):
            return None
        return bank

    def render(self, params, camera, sh_degree=3):
        bank = self.active_bank(params)
        if bank is None:
            return super().render(params, camera, sh_degree)
        return rasterize(bank.attributes(params), camera, sh_degree)


def create_renderer(spec):
    return MesonRenderer(spec)
