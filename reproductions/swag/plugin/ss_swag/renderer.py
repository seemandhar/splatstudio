"""SWAG renderer (paper §4, Fig. 2) on gsplat.

For each render: c = 3DGS SH colour (clamp_min(SH + 0.5, 0)); (o_c, delta_alpha) = F_theta(c,
emb(x), l_I) (Eq. 8); c^I = sigmoid(o_c) (A3); delta_alpha_tilde = Binary-Concrete sample with
location |delta_alpha| and temperature T (Eq. 9, U ~ U(0,1) on training views, U = 0.5 otherwise);
alpha^I = max(alpha - delta_alpha_tilde, 0) (Eq. 11). Rasterised by gsplat with explicit colours
over a black background (3DGS default, A9).

Without a bank (plain ``ss-eval``) it renders the plain 3DGS colour/opacity.
"""

import torch
from gsplat import rasterization, spherical_harmonics

from . import bank as bank_mod

CAPABILITIES = ("render.rgb", "gradient.screen.signed")
DEFAULTS = {"appearance": True, "transient": True, "temperature": 0.1, "eval_u": 0.5,
            "log_eps": 1e-10, "background": 0.0}


def sh_colors(params, camera, degree):
    coeffs = torch.cat((params["sh0"], params["shN"]), dim=1)
    campos = torch.linalg.inv(camera["viewmat"])[:3, 3]
    dirs = params["means"] - campos[None]
    return torch.clamp_min(spherical_harmonics(int(degree), dirs, coeffs) + 0.5, 0.0)


def concrete(delta_alpha, u, temperature, eps):
    """Eq. 9: sigmoid((log|d| + log U - log(1-U)) / T)."""
    return torch.sigmoid((torch.log(delta_alpha.abs() + eps) + torch.log(u) - torch.log1p(-u))
                         / temperature)


def image_opacity(alpha, delta_tilde):
    """Eq. 11."""
    return torch.clamp_min(alpha - delta_tilde, 0.0)


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown swag renderer options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        if not (self.cfg["appearance"] or self.cfg["transient"]):
            raise ValueError("swag renderer needs appearance and/or transient enabled")
        self.capabilities = set(CAPABILITIES)

    def conditioned(self, params, camera, degree, bank, u=None):
        """Returns (colors [N,3], opacities [N], extras) for one view."""
        c = self.cfg
        colors = sh_colors(params, camera, degree)
        alpha = torch.sigmoid(params["opacities"])
        extras = {"view_index": None}
        if bank is None:
            return colors, alpha, extras
        training = torch.is_grad_enabled() and bank.training
        idx = bank.view_index(camera["viewmat"]) if training else None
        emb = bank.embedding_for(idx)
        out = bank.mlp_outputs(colors, params["means"], emb)
        extras.update(view_index=idx, mlp=out)
        if c["appearance"]:
            colors = torch.sigmoid(out[:, :3])  # Eq. 7/8, A3
        if c["transient"]:
            if u is None:
                if idx is not None:  # Eq. 10 on training views
                    u = torch.rand_like(out[:, 3]).clamp(1e-6, 1 - 1e-6)
                else:  # evaluation: U = 0.5
                    u = torch.full_like(out[:, 3], float(c["eval_u"]))
            dt = concrete(out[:, 3], u, float(c["temperature"]), float(c["log_eps"]))
            alpha = image_opacity(alpha, dt)
            extras["delta_tilde"] = dt
        return colors, alpha, extras

    def render(self, params, camera, sh_degree=3, u=None):
        bank = bank_mod.ACTIVE
        colors, opac, extras = self.conditioned(params, camera, sh_degree, bank, u)
        render, alpha, info = rasterization(
            means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
            opacities=opac, colors=colors, viewmats=camera["viewmat"][None],
            Ks=camera["K"][None], width=int(camera["width"]), height=int(camera["height"]),
            sh_degree=None, packed=False, render_mode="RGB", absgrad=False,
            rasterize_mode="classic")
        rgb = render[0] + float(self.cfg["background"]) * (1 - alpha[0])
        return {"rgb": rgb, "alpha": alpha[0], "info": info, **extras}


def create_renderer(spec):
    return Renderer(spec)
