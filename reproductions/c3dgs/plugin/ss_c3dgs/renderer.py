"""gsplat renderer that renders the c3dgs compressed bank during quantization-aware fine-tuning.

Without an active, not-yet-decoded `CompressedBank` (steps 1..30000, after the final decode, and
in any fresh process such as `ss-eval`) it is the unchanged SplatStudio gsplat renderer. With an
active bank it rasterises the bank's fake-quantised attributes (see bank.py) with the same gsplat
call as `splatstudio.backends.gsplat.Renderer.render` (whose body `rasterize` mirrors), passing
activated scales and opacities directly (a relu-normalised codebook scale can be exactly 0, so
going through log/exp would produce NaN gradients).
"""

from gsplat import rasterization
from splatstudio.backends.gsplat import Renderer

from . import bank as bank_mod


def rasterize(means, quats, scales, opacities, colors, camera, sh_degree, white=True,
              covars=None):
    """SplatStudio gsplat call on activated attributes; colours are SH coefficients when
    sh_degree is not None (gsplat clamps SH + 0.5 at 0 like 3DGS), else per-Gaussian RGB."""
    render, alpha, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=camera["viewmat"][None],
        Ks=camera["K"][None],
        width=camera["width"],
        height=camera["height"],
        sh_degree=sh_degree,
        packed=False,
        render_mode="RGB+ED",
        covars=covars,
    )
    rgb = render[0, ..., :3] + (1 - alpha[0]) if white else render[0, ..., :3]
    return {"rgb": rgb, "depth": render[0, ..., 3:4], "alpha": alpha[0], "info": info}


class C3DGSRenderer(Renderer):
    def __init__(self, spec):
        super().__init__(spec)
        if self.absgrad or self.antialiased:
            raise ValueError("c3dgs-renderer follows the classic 3DGS rasteriser")
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
        a = bank.attributes(params)
        return rasterize(a["means"], a["quats"], a["scales"], a["opacities"], a["colors"],
                         camera, sh_degree)


def create_renderer(spec):
    return C3DGSRenderer(spec)

