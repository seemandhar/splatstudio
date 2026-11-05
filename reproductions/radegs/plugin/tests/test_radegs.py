import math

import torch

from ss_radegs import CODE_COMMIT, method
from ss_radegs.density import expon_lr, is_filter_step
from ss_radegs.losses import edge_map, ssim_same
from ss_radegs.probes import density_recipe, regularizer_schedule
from ss_radegs.raster import check_centered, depth_double_to_normal


def test_density_recipe_cpu():
    out = density_recipe(method(), "cpu")
    assert out["last"]["clone"] == 2


def test_regularizer_schedule_cpu():
    out = regularizer_schedule(method(), "cpu")
    assert out["loss"][14999] == 0.0 and out["loss"][15000] > 0


def test_ssim_same_padding_identity():
    torch.manual_seed(0)
    a = torch.rand(20, 24, 3)
    assert math.isclose(ssim_same(a, a).item(), 1.0, rel_tol=1e-5)
    b = a.clone()
    b[0, 0] += 0.5
    assert ssim_same(a, b).item() < 1.0


def test_edge_map_constant_image_and_border():
    e = edge_map(torch.full((8, 9, 3), 0.4))
    assert torch.all(e[1:-1, 1:-1] == 1.0) and e[0].abs().sum() == 0 and e[:, -1].sum() == 0


def test_depth_normal_border_and_frontal_plane():
    cam = {"K": torch.tensor([[10.0, 0, 6], [0, 10.0, 5], [0, 0, 1]]), "width": 12, "height": 10,
           "viewmat": torch.eye(4)}
    n = depth_double_to_normal(cam, torch.full((10, 12), 3.0), torch.full((10, 12), 3.0))
    assert torch.allclose(n[:, 1:-1, 1:-1], torch.tensor([0.0, 0.0, -1.0]), atol=1e-6)
    assert n[:, 0].abs().sum() == 0


def test_principal_point_check():
    cam = {"K": torch.tensor([[10.0, 0, 7.0], [0, 10.0, 5], [0, 0, 1]]), "width": 12,
           "height": 10}
    try:
        check_centered(cam)
    except ValueError:
        return
    raise AssertionError("off-centre principal point must be rejected")


def test_schedules():
    assert expon_lr(0, 1.6e-4, 1.6e-6) == 1.6e-4
    assert math.isclose(expon_lr(30000, 1.6e-4, 1.6e-6), 1.6e-6, rel_tol=1e-9)
    assert is_filter_step(14900, 100, 500, 15000, 30000)
    assert not is_filter_step(29900, 100, 500, 15000, 30000)


def test_spec_defaults():
    m = method()
    cfg = {c.name: c.config for c in m.components}
    assert cfg["radegs-reg"]["lambda_distortion"] == 100.0
    assert cfg["radegs-reg"]["lambda_depth_normal"] == 0.05
    assert cfg["radegs-reg"]["depth_ratio"] == 0.6
    assert cfg["radegs-reg"]["regularization_from_iter"] == 15000
    assert cfg["radegs-density"]["min_opacity"] == 0.05
    assert cfg["radegs-raster"]["background"] == [0.0, 0.0, 0.0]
    assert CODE_COMMIT.startswith("2d4bc08")
