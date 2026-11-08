"""gsplat renderer that renders the CompGS straight-through quantised attributes.

While a VQ state is active (density owns it, steps vq_start+1..total_steps) the DC colour, SH
rest, log-scale and raw quaternion fed to the unchanged SplatStudio gsplat call are
`x - x.detach() + C[idx]` (official `forward_*`); scales are quantised before exp and quaternions
before normalisation (paper §4). Without an active state (before VQ, after the final bake and in
any fresh process such as ss-eval) it is the plain renderer.
"""

from splatstudio.backends.gsplat import Renderer

from . import vq as vq_mod


class CompGSRenderer(Renderer):
    def __init__(self, spec):
        super().__init__(spec)
        if self.absgrad or self.antialiased:
            raise ValueError("compgs-renderer follows the classic 3DGS rasteriser")
        self.vq = None  # explicit state (probes); otherwise vq.ACTIVE

    def active(self):
        vq = self.vq if self.vq is not None else vq_mod.ACTIVE["vq"]
        if vq is None or not vq.ready or vq.decoded:
            return None
        return vq

    def render(self, params, camera, sh_degree=3):
        vq = self.active()
        if vq is None:
            return super().render(params, camera, sh_degree)
        if vq.n != len(params["means"]):
            raise RuntimeError(f"CompGS VQ state has {vq.n} rows, model {len(params['means'])}")
        return super().render(vq.quantized(params), camera, sh_degree)


def create_renderer(spec):
    return CompGSRenderer(spec)
