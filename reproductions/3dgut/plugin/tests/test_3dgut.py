import math

import torch

from ss_3dgut.cameras import distortion_keys, project_points
from ss_3dgut.density import densification_statistic, official_init_values, step_condition
from ss_3dgut.probes import _rotmat, _ut_reference


def test_step_condition_matches_official():
    assert not step_condition(500, 500, 15000, 300)
    assert step_condition(600, 500, 15000, 300)
    assert not step_condition(700, 500, 15000, 300)
    assert not step_condition(15000, 500, 15000, 300)
    assert step_condition(3000, 0, 15000, 3000)
    assert not step_condition(0, 0, 15000, 3000)


def test_statistic_is_scaled_3d_gradient():
    grad = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    means = torch.tensor([[0.0, 0.0, 2.0], [1.0, 1.0, 1.0]])
    mask, inc = densification_statistic(grad, means, torch.zeros(3))
    assert mask.tolist() == [True, False]
    assert torch.allclose(inc, torch.tensor([5.0 * 2.0 / 2]))


def test_official_init():
    means = torch.tensor([[0.0, 0.0, 2.0], [5.0, 0.0, 0.0]])
    cams = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    log_s, q = official_init_values(means, cams, 0.01, 42)
    assert torch.allclose(log_s.exp(), torch.tensor([[0.02] * 3, [0.01] * 3]))
    assert torch.allclose(q.norm(dim=-1), torch.ones(2))


def test_fisheye_equidistant():
    K = torch.tensor([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]], dtype=torch.float64)
    p = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64)
    uv = project_points(p, K, "fisheye", [0.0] * 4)
    assert torch.allclose(uv, torch.tensor([[50 + 100 * math.pi / 4, 40.0]], dtype=torch.float64))


def test_distortion_keys():
    assert distortion_keys({"camera_model": "OPENCV_FISHEYE", "k1": 0.1})["camera_model"] == "fisheye"
    assert "radial_coeffs" not in distortion_keys({"camera_model": "PINHOLE"})
    assert distortion_keys({"camera_model": "OPENCV", "k1": -0.2})["radial_coeffs"][0] == -0.2


def test_ut_exact_for_affine_projection():
    # Points on a plane z=const project affinely under pinhole only along x,y; use a flat
    # Gaussian (zero z-extent) so the UT must reproduce J Sigma J^T exactly.
    mean = torch.tensor([0.2, -0.1, 2.0], dtype=torch.float64)
    scale = torch.tensor([0.3, 0.1, 0.0], dtype=torch.float64)
    quat = torch.tensor([math.cos(0.3), 0.0, 0.0, math.sin(0.3)], dtype=torch.float64)
    K = torch.tensor([[100.0, 0, 0], [0, 100.0, 0], [0, 0, 1]], dtype=torch.float64)
    mu, cov = _ut_reference(mean, scale, quat, K, "pinhole", None, None, 1.0, 2.0, 0.0)
    R = _rotmat(quat)
    sig = R @ torch.diag(scale**2) @ R.T
    J = torch.tensor([[50.0, 0, 0], [0, 50.0, 0]], dtype=torch.float64)
    assert torch.allclose(mu, torch.tensor([10.0, -5.0], dtype=torch.float64))
    assert torch.allclose(cov, J @ sig @ J.T)
