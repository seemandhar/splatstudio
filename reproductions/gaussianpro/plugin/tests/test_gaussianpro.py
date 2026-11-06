import math

import torch

from ss_gaussianpro import geometry as G
from ss_gaussianpro import losses as L
from ss_gaussianpro import method, probes


def test_threshold_schedule():
    assert G.threshold(1000, 1000, 12000, 1.0, 0.8) == 1.0
    assert math.isclose(G.threshold(12000, 1000, 12000, 1.0, 0.8), 0.8)
    assert math.isclose(G.threshold(6500, 1000, 12000, 1.0, 0.8), 0.9)


def test_shortest_axis_and_flip():
    quats = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    scales = torch.tensor([[0.1, 0.01, 0.2], [0.001, 0.1, 0.2]]).log()
    n = G.shortest_axis_normals(quats, scales)
    assert torch.allclose(n, torch.tensor([[0.0, 1, 0], [1.0, 0, 0]]))
    means = torch.tensor([[0.0, 1.0, 2.0], [-1.0, 0.0, 2.0]])
    nc = G.camera_normals(means, quats, scales, torch.eye(4))
    # (mean - cam) . n > 0 for both -> flipped
    assert torch.allclose(nc, torch.tensor([[0.0, -1, 0], [1.0, 0, 0]]))


def test_backproject_roundtrip():
    K = torch.tensor([[50.0, 0, 16], [0, 50.0, 12], [0, 0, 1]])
    vm = torch.eye(4)
    vm[:3, 3] = torch.tensor([0.1, -0.2, 0.3])
    depth = torch.full((24, 32), 2.0)
    mask = torch.zeros(24, 32, dtype=torch.bool)
    mask[8, 16] = True
    pts, sub = G.backproject(depth, K, vm, mask, 8)
    assert pts.shape == (1, 3) and sub.shape == (3, 4)
    cam = vm[:3, :3] @ pts[0] + vm[:3, 3]
    assert torch.allclose(cam, torch.tensor([0.0, 2.0 * (8 - 12) / 50.0, 2.0]), atol=1e-6)


def test_neighbours_rank_shared_points():
    obs = {"a": {1, 2, 3}, "b": {1, 2}, "c": {3}, "d": set()}
    nb = G.shared_point_neighbours(["a", "b", "c", "d"], obs, 4)
    assert nb == [[1, 2], [0], [0], []]


def test_consistency_and_losses_cpu():
    m = method()
    assert probes.consistency_check(m, "cpu")["offset_frac"] < 0.05
    assert probes.planar_losses(m, "cpu")["grad_invalid"] == 0
    assert probes.opacity_mask(m, "cpu")["early_grad_masked"] == 0


def test_photometric_unmasked_equals_runtime():
    from splatstudio.losses import ssim

    rgb, tgt = torch.rand(20, 24, 3), torch.rand(20, 24, 3)
    alpha = torch.ones(20, 24, 1)
    val = L.masked_photometric(rgb, tgt, alpha, 5000, dict(L.PHOTO_DEFAULTS))
    ref = 0.8 * (rgb - tgt).abs().mean() + 0.2 * (1 - ssim(rgb, tgt))
    assert torch.allclose(val, ref, atol=1e-6)


def test_finite_full_opacity_logit():
    x = torch.tensor([20.0], requires_grad=True)
    s = x.sigmoid()
    s.sum().backward()
    assert float(s) == 1.0 and float(x.grad) == 0.0
