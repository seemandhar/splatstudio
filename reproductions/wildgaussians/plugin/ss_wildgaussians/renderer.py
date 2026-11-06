"""WildGaussians renderer (official method.py `_render_internal`) on gsplat.

* Colours are evaluated explicitly: F = clamp_max(cat(sh0, shN), 1) (every SH coefficient is
  clamped at 1), raw colour = clamp_min(SH(F, mu - campos) + 0.5, 0).
* Appearance (Eq. 4-5): toned F' = clamp_max(MLP-affine(F), 1), toned colour the same way.
  The image embedding is the training view's (bank lookup by viewmat), otherwise the bank's
  evaluation override, otherwise zeros (official get_embedding(None)).
* Two gsplat passes (raw, toned) over a black background, as the official code calls the
  rasterizer twice; the density sums the screen-space gradients of both passes.
* Mip-Splatting: 3D filter on scales / opacities (field `filter_3d`), 2D Mip filter =
  gsplat `rasterize_mode="antialiased"` with eps2d = kernel_size 0.1. Near plane 0.2 (Inria
  frustum cull). GOF absolute gradients via gsplat `absgrad=True`.
* gsplat drops opacity/conic/means2d gradients where opac * G > 0.999 (the Inria/Mip rasterizer
  clamps alpha in the forward pass only), so opacity is capped at 0.999 with a straight-through
  gradient (TODO-runtime.md, DropGaussian fix).
* SH degree during training follows the official schedule (one band more at every 1000th
  iteration, applied before rendering): min(step // 1000, 3).
"""

import torch
from gsplat import rasterization, spherical_harmonics

from . import bank as bank_mod
from .geometry import filtered_scales_opacities

CAPABILITIES = ("render.rgb", "render.rgb_raw", "render.appearance", "gradient.screen.signed",
                "gradient.screen.absolute", "render.antialiased_2d", "render.filter_3d")
DEFAULTS = {"kernel_size": 0.1, "near_plane": 0.2, "alpha_cap": 0.999, "appearance": True,
            "max_sh_degree": 3}


def sh_features(params):
    """[N, 16, 3] SH coefficients (official features = cat(features_dc, features_rest))."""
    return torch.cat((params["sh0"], params["shN"]), dim=1)


def sh_to_rgb(coeffs, dirs, degree):
    """eval_sh(deg, sh, dirs) + 0.5, clamp_min 0 (gsplat SH == Inria eval_sh)."""
    return torch.clamp_min(spherical_harmonics(int(degree), dirs, coeffs) + 0.5, 0.0)


def camera_center(viewmat):
    return torch.linalg.inv(viewmat)[:3, 3]


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown wildgaussians renderer options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.capabilities = set(CAPABILITIES)

    def _rasterize(self, means, quats, scales, opac, colors, camera):
        render, alpha, info = rasterization(
            means=means, quats=quats, scales=scales, opacities=opac, colors=colors,
            viewmats=camera["viewmat"][None], Ks=camera["K"][None],
            width=int(camera["width"]), height=int(camera["height"]),
            near_plane=float(self.cfg["near_plane"]), eps2d=float(self.cfg["kernel_size"]),
            sh_degree=None, packed=False, render_mode="RGB", absgrad=True,
            rasterize_mode="antialiased")
        return render[0], alpha[0], info

    def toned_features(self, params, features, embedding, bank):
        flat = features.reshape(len(features), -1)
        toned = bank.mlp(params["wg_embedding"], embedding[None], flat)
        return toned.reshape(features.shape).clamp_max(1.0)

    def render(self, params, camera, sh_degree=3):
        c = self.cfg
        bank = bank_mod.ACTIVE
        idx = None
        if bank is not None and torch.is_grad_enabled() and bank.training:
            idx = bank.view_index(camera["viewmat"])
        degree = min(int(sh_degree), int(c["max_sh_degree"]))
        if idx is not None:
            degree = min((bank.step + 1) // 1000, int(c["max_sh_degree"]))
        scales, opac = filtered_scales_opacities(params["scales"], params["opacities"],
                                                 params["filter_3d"])
        capped = opac.clamp(max=float(c["alpha_cap"]))
        opac = opac + (capped - opac).detach()
        means, quats = params["means"], params["quats"]
        features = sh_features(params).clamp_max(1.0)
        dirs = means - camera_center(camera["viewmat"])[None]
        raw_colors = sh_to_rgb(features, dirs, degree)
        raw, alpha, info = self._rasterize(means, quats, scales, opac, raw_colors, camera)
        out = {"rgb_raw": raw, "rgb": raw, "alpha": alpha, "info": info, "view_index": idx,
               "infos": [info]}
        if c["appearance"]:
            if bank is None or not bank.appearance:
                return out  # no learned appearance state (ss-eval): raw render
            emb = bank.embedding_for(idx)
            toned_f = self.toned_features(params, features, emb, bank)
            toned_colors = sh_to_rgb(toned_f, dirs, degree)
            toned, _, info2 = self._rasterize(means, quats, scales, opac, toned_colors, camera)
            out.update(rgb=toned, infos=[info, info2], toned_available=True)
        return out


def create_renderer(spec):
    return Renderer(spec)
