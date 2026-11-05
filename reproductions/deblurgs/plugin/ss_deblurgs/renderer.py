"""DeblurGS renderer (gsplat).

* Training camera (gradients on, live bank, viewmat matches a bank view): renders the view's
  sub-frames along its learned trajectory in **one batched gsplat call** (C = 21 cameras;
  C = 1 at nu = 0 before curve_start_iter), all over one random background per step
  (motion.query background="random"), and returns
  ``blur_linear`` = mean of the sub-frames (Eq. 6/7 without gamma, official train.py computes
  the loss in linear space against gamma^-1(GT)), ``subframes`` [C,H,W,3], ``rgb`` = gamma(blur)
  and the batched gsplat ``info`` for the density statistics of every sub-frame.
* Any other camera (ss-eval, validation, test-pose optimisation): sharp render over white
  (test.py bg ones), ``rgb`` = gamma(render) = clamp_min(x, 1e-8)^(1/2.2) (tonemapping.py).

Official Gaussian activations: opacity = clamp(raw, 0, 1) (gaussian_activation.Clamp; the raw
value lives in params["opacities"]), colour = ReLU(SH + 0.5), max SH degree 2, near plane 0.2.
gsplat drops opacity/conic/means2d gradients where opac*G > 0.999 (Inria clamps alpha at 0.99
in the forward pass only), so the opacity given to gsplat is capped at 0.999 with a
straight-through gradient (TODO-runtime.md, DropGaussian fix).
"""

import torch
from gsplat import rasterization

from . import bank as bank_mod

CAPABILITIES = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed",
                "render.motion_blur")
DEFAULTS = {"near_plane": 0.2, "max_sh_degree": 2, "gamma": 2.2, "alpha_cap": 0.999,
            "tone_eps": 1e-8}


def tonemap(x, gamma=2.2, eps=1e-8):
    """ToneMapping('gamma'): clamp_min(x, eps) ** (1/2.2)."""
    return x.clamp_min(eps) ** (1.0 / gamma)


def inverse_tonemap(x, gamma=2.2, eps=1e-8):
    """ToneMapping('reverse_gamma'): clamp_min(x, eps) ** 2.2."""
    return x.clamp_min(eps) ** gamma


def activations(params, max_sh_degree=2, alpha_cap=0.999):
    n_coef = (max_sh_degree + 1) ** 2 - 1
    opac = params["opacities"].clamp(0.0, 1.0)
    capped = opac.clamp(max=alpha_cap)
    return {
        "means": params["means"],
        "quats": params["quats"],
        "scales": params["scales"].exp(),
        "opacities": opac + (capped - opac).detach(),
        "colors": torch.cat((params["sh0"], params["shN"][:, :n_coef]), dim=1),
    }


def rasterize(act, viewmats, Ks, width, height, sh_degree, near_plane=0.2):
    """gsplat over C cameras; returns linear RGB [C,H,W,3], depth [C,H,W,1], alpha, info."""
    render, alpha, info = rasterization(
        means=act["means"], quats=act["quats"], scales=act["scales"],
        opacities=act["opacities"], colors=act["colors"],
        viewmats=viewmats, Ks=Ks, width=int(width), height=int(height),
        sh_degree=sh_degree, packed=False, render_mode="RGB+ED", near_plane=near_plane,
        rasterize_mode="classic",
    )
    return render[..., :3], render[..., 3:4], alpha, info


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown deblurgs renderer options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.capabilities = set(CAPABILITIES)

    def _act(self, params):
        return activations(params, int(self.cfg["max_sh_degree"]), float(self.cfg["alpha_cap"]))

    def _degree(self, sh_degree):
        return min(int(sh_degree), int(self.cfg["max_sh_degree"]))

    def render_sharp(self, params, camera, sh_degree=3, viewmat=None):
        """Sharp render over white, tone-mapped (test.py evaluate / optimize_test_pose)."""
        c = self.cfg
        vm = camera["viewmat"] if viewmat is None else viewmat
        rgb, depth, alpha, info = rasterize(self._act(params), vm[None], camera["K"][None],
                                            camera["width"], camera["height"],
                                            self._degree(sh_degree), c["near_plane"])
        linear = rgb[0] + (1 - alpha[0])
        return {"rgb": tonemap(linear, c["gamma"], c["tone_eps"]), "rgb_linear": linear,
                "depth": depth[0], "alpha": alpha[0], "info": info}

    def render(self, params, camera, sh_degree=3):
        bank = bank_mod.ACTIVE
        idx = bank.view_index(camera["viewmat"]) if (bank is not None
                                                     and torch.is_grad_enabled()) else None
        if idx is None:
            return self.render_sharp(params, camera, sh_degree)
        c = self.cfg
        step = bank.step + 1  # the runtime step being rendered
        viewmats = bank.subframe_viewmats(idx, step)  # [C, 4, 4]
        n_sub = viewmats.shape[0]
        Ks = camera["K"][None].expand(n_sub, 3, 3)
        rgb, depth, alpha, info = rasterize(self._act(params), viewmats, Ks, camera["width"],
                                            camera["height"], self._degree(sh_degree),
                                            c["near_plane"])
        bg = torch.rand(3, device=rgb.device)  # one random background per step
        subframes = rgb + (1 - alpha) * bg  # [C, H, W, 3] linear irradiance
        blur = subframes.mean(dim=0)  # Eq. 6 (average of sharp sub-frame renders)
        return {
            "rgb": tonemap(blur, c["gamma"], c["tone_eps"]),  # Eq. 7
            "blur_linear": blur,
            "subframes": subframes,
            "depth": depth.mean(dim=0),
            "alpha": alpha.mean(dim=0),
            "info": info,
            "view_index": idx,
            "subframe_viewmats": viewmats,
            "background": bg,
        }


def create_renderer(spec):
    return Renderer(spec)
