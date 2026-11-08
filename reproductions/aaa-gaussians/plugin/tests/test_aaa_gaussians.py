import math

import torch

from ss_aaa_gaussians import reference as ref
from ss_aaa_gaussians.density import CAPS, resolve_cap, resolve_total_steps
from ss_aaa_gaussians.filters import perpendicular_scaling, view_filter_variance, volume_scaling
from ss_aaa_gaussians.raster import AAA, make_settings


def test_perpendicular_equals_schur_definition():
    # Eq. 10: sqrt(|Sigma| d^T Sigma^-1 d / (|Sigma^| d^T Sigma^^-1 d)) == Eq. 11
    g = torch.Generator().manual_seed(0)
    s = torch.rand(5, 3, generator=g, dtype=torch.float64) * 0.3 + 0.01
    R = ref.rotmat(torch.randn(5, 4, generator=g))
    d = torch.nn.functional.normalize(torch.randn(5, 3, generator=g, dtype=torch.float64), dim=-1)
    v = torch.rand(5, generator=g, dtype=torch.float64) * 0.01
    S = R @ torch.diag_embed(s ** 2) @ R.transpose(1, 2)
    Sh = S + v[:, None, None] * torch.eye(3, dtype=torch.float64)
    q = torch.einsum("ni,nij,nj->n", d, torch.linalg.inv(S), d)
    qh = torch.einsum("ni,nij,nj->n", d, torch.linalg.inv(Sh), d)
    want = torch.sqrt(torch.linalg.det(S) * q / (torch.linalg.det(Sh) * qh))
    assert torch.allclose(perpendicular_scaling(s, R, d, v), want, rtol=1e-10)


def test_perpendicular_vs_volume_on_disk():
    s = torch.tensor([[0.3, 0.3, 1e-3]], dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64)[None]
    v = torch.tensor([1e-4], dtype=torch.float64)
    face_on = perpendicular_scaling(s, R, torch.tensor([[0, 0, 1.0]], dtype=torch.float64), v)
    assert face_on.item() > 0.99
    assert volume_scaling(s, v).item() < 0.2


def test_filter_variance_floor():
    z = torch.tensor([2.0, 8.0])
    v = view_filter_variance(z, 100.0, 0.3, torch.tensor([0.0, 0.0]))
    assert torch.allclose(v, 0.3 * (z / 100) ** 2)
    floor = torch.tensor([math.sqrt(0.3) * 4 / 100] * 2)
    v = view_filter_variance(z, 100.0, 0.3, floor)
    assert torch.allclose(v, torch.tensor([0.3 * 16 / 1e4, 0.3 * 64 / 1e4]))


def test_settings_and_caps():
    s = make_settings(AAA, eval_3D=False, per_pixel=4)
    assert s["eval_3D"] is False and s["sort_settings"]["sort_mode"] == 3
    assert AAA["proper_ewa_scaling"] and AAA["culling_settings"]["tile_based_culling"]
    assert CAPS["garden"] == 5834784 and CAPS["room"] == 1593376
    assert resolve_cap(7) == 7 and resolve_total_steps(11) == 11


def test_reference_line_minimum():
    # rho^2_min on the axis ray through an isotropic Gaussian is zero -> alpha = o * a
    p = {"means": torch.tensor([[0.0, 0.0, 3.0]]), "scales": torch.full((1, 3), math.log(0.2)),
         "quats": torch.tensor([[1.0, 0, 0, 0]]), "opacities": torch.logit(torch.tensor([0.5])),
         "sh0": ((torch.ones(1, 3) - 0.5) / ref.C0)[:, None]}
    cam = {"viewmat": torch.eye(4), "width": 2, "height": 2,
           "K": torch.tensor([[1e6, 0, 1.0], [0, 1e6, 1.0], [0, 0, 1]])}
    img = ref.composite(ref.leaves_from(p), cam, torch.zeros(3))
    assert abs(img[0, 0, 0].item() - 0.5) < 1e-3
