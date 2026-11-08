import math

import torch

from ss_vol3dgs import vol3dgs
from ss_vol3dgs.density import expon_lr
from ss_vol3dgs.probes import densify_rules, density_reparam
from ss_vol3dgs.reference import (
    alpha2density,
    camera_gaussians,
    dense_render,
    halve_theta,
    quat_to_rotmat,
    volumetric_alpha,
)


def test_reparam_centre_alpha():
    # code normalisation: a centre ray through an isotropic Gaussian gives alpha = 0.99 theta
    theta = torch.tensor([0.3], dtype=torch.float64)
    s = torch.full((1, 3), 0.2, dtype=torch.float64)
    kappa = alpha2density(theta, s)
    alpha = 1 - torch.exp(-2.506 * kappa * 0.2)
    assert torch.allclose(alpha, 0.99 * theta)


def test_halve():
    theta = torch.tensor([0.5, 0.99999], dtype=torch.float64)
    s = torch.ones(2, 3, dtype=torch.float64)
    # official expression as written (general_utils.py:56): 1 - exp(-x) / 0.99
    assert torch.allclose(halve_theta(theta, s), 1 - torch.sqrt(1 - 0.99 * theta) / 0.99)
    assert torch.allclose(halve_theta(theta, s, "inverse"),
                          (1 - torch.sqrt(1 - 0.99 * theta)) / 0.99)


def test_rotmat_orthonormal():
    q = torch.randn(10, 4, dtype=torch.float64)
    R = quat_to_rotmat(q)
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3, dtype=torch.float64).expand(10, 3, 3))


def test_line_integral_numeric():
    # analytic alpha equals 1 - exp(-kappa * numerical integral of G along the ray)
    g = torch.Generator().manual_seed(0)
    means = torch.tensor([[0.2, -0.1, 4.0]], dtype=torch.float64)
    quats = torch.randn(1, 4, generator=g, dtype=torch.float64)
    log_s = torch.log(torch.tensor([[0.3, 0.1, 0.5]], dtype=torch.float64))
    mean_c, precis = camera_gaussians(means, quats, log_s, torch.eye(4, dtype=torch.float64))
    d = torch.tensor([[0.03, -0.02, 1.0]], dtype=torch.float64)
    d = d / d.norm()
    kappa = torch.tensor([1.7], dtype=torch.float64)
    alpha, _ = volumetric_alpha(mean_c, precis, kappa, d, clamp=False)
    t = torch.linspace(0, 10, 200001, dtype=torch.float64)
    x = t[:, None] * d - mean_c
    G = torch.exp(-0.5 * torch.einsum("ti,ij,tj->t", x, precis[0], x))
    integral = torch.trapz(G, t)
    assert abs(alpha.item() - (1 - math.exp(-kappa.item() * integral.item() * 2.506 / math.sqrt(2 * math.pi)))) < 1e-6


def test_dense_render_single_opaque_stop():
    # two coincident-ray Gaussians; the second is dropped once T would fall below 1e-4
    means_c = torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 4.0]], dtype=torch.float64)
    precis = torch.eye(3, dtype=torch.float64).expand(2, 3, 3) / 0.04
    kappa = torch.tensor([100.0, 100.0], dtype=torch.float64)
    colors = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]], dtype=torch.float64)
    K = torch.tensor([[10.0, 0, 0.5], [0, 10.0, 0.5], [0, 0, 1]], dtype=torch.float64)
    rgb, alpha = dense_render(means_c, precis, kappa, colors, K, 1, 1)
    assert min(abs(alpha.item() - 0.99), abs(alpha.item() - 0.9999)) < 1e-9
    assert rgb[0, 0, 0] > 0.98


def test_expon_lr():
    assert abs(expon_lr(0, 1.6e-4, 1e-5, 30000) - 1.6e-4) < 1e-15
    assert abs(expon_lr(30000, 1.6e-4, 1e-5, 30000) - 1e-5) < 1e-15


def test_cpu_probes():
    from ss_vol3dgs import vol3dgs_black

    assert vol3dgs_black().components[0].config["background"] == "black"
    m = vol3dgs()
    density_reparam(m, "cpu")
    densify_rules(m, "cpu")
