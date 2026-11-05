"""C1 parameter sensitivity (paper Eq. sensitivity; official `compress.py:calc_importance`).

S(p) = 1 / sum_i P_i * sum_i |dE_i / dp|,  E_i = sum of the RGB values of rendered training view i.

As in the official code:
* the model is the loaded 3DGS model re-parameterised with s = eta * s_hat (`load_ply`) and seen
  through the fake quantisers (their observers initialise here): normalised covariance from
  FQ(normalize(relu(s_hat))) and normalize(FQ(q)), computed once; per view
  cov = cov_hat * exp(FQ(log eta))^2, opacity FQ(sigmoid(o)), SH FQ(dc), FQ(rest), half positions;
* colours are NOT clamped (supp. "Sensitivity Calculation": SH + 0.5 without clamp_min(0)),
* black background, |grad| per view (the official `register_hook(grad.abs())`), summed over views,
  divided by the total pixel count;
* SH sensitivity of a Gaussian = max over its 48 coefficients, shape sensitivity = max over the
  6 normalised-covariance entries (`.amax(-1)`).
"""

import torch
import torch.nn.functional as F
from gsplat import spherical_harmonics

from . import vq
from .quant import half_ste
from .renderer import rasterize


def reparameterize(params):
    """Official load_ply: scaling = exp(log s); _scaling = s / |s|; _scaling_factor = log |s|."""
    scaling = params["scales"].detach().exp()
    norm = scaling.norm(2, -1, keepdim=True)
    return scaling / norm, norm.log()


def camera_center(camera):
    return torch.linalg.inv(camera["viewmat"])[:3, 3]


def unclamped_colors(means, sh, camera, degree=3):
    dirs = means - camera_center(camera)[None]
    return spherical_harmonics(degree, dirs, sh) + 0.5


def compute(params, quantizers, cameras, white_background=False):
    """Returns (colour sensitivity [N, 48], shape sensitivity [N, 6], total pixels)."""
    q = quantizers
    s_hat, log_eta = reparameterize(params)
    with torch.no_grad():
        scaling = q["scaling"](F.normalize(F.relu(s_hat)))
        rotation = F.normalize(q["rotation"](params["quats"].detach()))
        cov_hat = vq.covariance(scaling, rotation)
        eta = torch.exp(q["scaling_factor"](log_eta))
    cov_hat = cov_hat.clone().requires_grad_(True)
    sh0 = params["sh0"].detach().clone().requires_grad_(True)
    shN = params["shN"].detach().clone().requires_grad_(True)
    means = params["means"].detach()
    opacity_logit = params["opacities"].detach()
    acc = [torch.zeros_like(sh0), torch.zeros_like(shN), torch.zeros_like(cov_hat)]
    num_pixels = 0
    for camera in cameras:
        cov = vq.to_full_cov(cov_hat * eta.square())
        m = half_ste(means)
        opacities = q["opacity"](torch.sigmoid(opacity_logit)[:, None])[:, 0]
        sh = torch.cat((q["features_dc"](sh0), q["features_rest"](shN)), dim=1)
        colors = unclamped_colors(m, sh, camera)
        out = rasterize(m, None, None, opacities, colors, camera, None, white=white_background,
                        covars=cov)
        energy = out["rgb"].sum()
        grads = torch.autograd.grad(energy, [sh0, shN, cov_hat])
        for a, g in zip(acc, grads):
            a.add_(g.abs())
        num_pixels += int(camera["width"]) * int(camera["height"])
    color = torch.cat((acc[0], acc[1]), dim=1).flatten(-2) / num_pixels
    shape = acc[2] / num_pixels
    return color, shape, num_pixels
