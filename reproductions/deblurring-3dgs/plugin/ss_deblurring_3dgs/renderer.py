"""Deblurring 3DGS renderer (gsplat).

* Training (live bank, gradients on): F_theta deltas from the detached Gaussian attributes and
  the camera centre; transformed covariance Sigma = R(q~) diag(s * ds)^2 R(q~)^T with
  q~ = normalize(q) * dr left **unnormalised**, exactly as the Inria computeCov3D consumes the
  official `rotations * rotations_delta` (it does not renormalise; gsplat would, so Sigma is
  built here and passed as `covars`).
  - defocus: one render of G(x, r^, s^) (Eq. 9-10).
  - motion: M + 1 = 5 renders in one batched gsplat call: render 0 = (x, s*ds[..., M],
    q*dr[..., M]), renders i = 1..M = (x + lambda_p dx_i, s*ds_i, q*dr_i); I_b = mean (Eq. 12).
* Evaluation / any no-grad call / no bank: the sharp Gaussians G(x, r, s) (Sec. 3.2).
Background black (official white_background False).
"""

import torch
import torch.nn.functional as F
from gsplat import rasterization

from . import bank as bank_mod

CAPABILITIES = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed",
                "render.blur_mlp")
DEFAULTS = {"background": 0.0}


def quat_to_rotmat_unnormalized(q):
    """Inria computeCov3D rotation from a (possibly unnormalised) wxyz quaternion (row form,
    equal to glm's transpose): no normalisation."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def covariance(scales, quats):
    """Sigma = R S S^T R^T (Eq. 3) with R from the unnormalised quaternion."""
    m = quat_to_rotmat_unnormalized(quats) * scales[..., None, :]
    return m @ m.transpose(-1, -2)


def camera_center(viewmat):
    return torch.linalg.inv(viewmat)[:3, 3]


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown deblurring-3dgs renderer options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.capabilities = set(CAPABILITIES)

    def _colors(self, params):
        return torch.cat((params["sh0"], params["shN"]), dim=1)

    def _finish(self, render, alpha):
        bg = float(self.cfg["background"])
        return render[..., :3] + bg * (1 - alpha), render[..., 3:4]

    def render_sharp(self, params, camera, sh_degree=3):
        render, alpha, info = rasterization(
            means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
            opacities=params["opacities"].sigmoid(), colors=self._colors(params),
            viewmats=camera["viewmat"][None], Ks=camera["K"][None], width=camera["width"],
            height=camera["height"], sh_degree=sh_degree, packed=False, render_mode="RGB+ED",
            rasterize_mode="classic")
        rgb, depth = self._finish(render, alpha)
        return {"rgb": rgb[0], "depth": depth[0], "alpha": alpha[0], "info": info,
                "n_renders": 1}

    def render_blur(self, params, camera, sh_degree, bank):
        means = params["means"]
        scales = params["scales"].exp()
        quats = F.normalize(params["quats"], dim=-1)  # rotation_activation
        ds, dr, dx = bank.deltas(means, scales, quats, camera_center(camera["viewmat"]))
        opac = params["opacities"].sigmoid()
        colors = self._colors(params)
        if not bank.motion:  # defocus (dashed arrows of Fig. 2)
            mb, cov = means[None], covariance(scales * ds, quats * dr)[None]
        else:  # camera motion (dotted arrows), render 0 unshifted with the last factors
            m = int(bank.cfg["num_moments"])
            dx = dx.view(-1, 3, m)
            ds = ds.view(-1, 3, m + 1)
            dr = dr.view(-1, 4, m + 1)
            mb = torch.stack([means] + [means + dx[..., i] for i in range(m)])
            order = [m] + list(range(m))
            cov = torch.stack([covariance(scales * ds[..., i], quats * dr[..., i])
                               for i in order])
        b = mb.shape[0]
        render, alpha, info = rasterization(
            means=mb, quats=None, scales=None, covars=cov,
            opacities=opac[None].expand(b, -1), colors=colors[None].expand(b, -1, -1, -1),
            viewmats=camera["viewmat"][None, None].expand(b, 1, 4, 4),
            Ks=camera["K"][None, None].expand(b, 1, 3, 3), width=camera["width"],
            height=camera["height"], sh_degree=sh_degree, packed=False, render_mode="RGB+ED",
            rasterize_mode="classic")
        rgb, depth = self._finish(render[:, 0], alpha[:, 0])  # [B, H, W, *]
        return {"rgb": rgb.mean(dim=0), "depth": depth.mean(dim=0),
                "alpha": alpha[:, 0].mean(dim=0), "info": info, "renders": rgb,
                "n_renders": b, "deltas": {"ds": ds, "dr": dr, "dx": dx}}

    def render(self, params, camera, sh_degree=3):
        bank = bank_mod.ACTIVE
        if bank is None or not torch.is_grad_enabled():
            return self.render_sharp(params, camera, sh_degree)
        return self.render_blur(params, camera, sh_degree, bank)


def create_renderer(spec):
    return Renderer(spec)
