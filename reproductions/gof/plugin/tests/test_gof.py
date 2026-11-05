import math

import torch

from ss_gof import method
from ss_gof.density import expon_lr, is_filter_step
from ss_gof.losses import ssim_same
from ss_gof.probes import density_recipe, normal_consistency


def test_density_recipe_cpu():
    out = density_recipe(method(), "cpu")
    assert out["last"]["clone"] == 2


def test_normal_consistency_cpu():
    out = normal_consistency(method(), "cpu")
    assert out["loss"][14999] == 0.0


def test_ssim_same_padding_identity_and_border():
    torch.manual_seed(0)
    a = torch.rand(20, 24, 3)
    assert math.isclose(ssim_same(a, a).item(), 1.0, rel_tol=1e-5)
    # same padding: zero-padded borders make a constant image != 1 only if it differs
    b = a.clone()
    b[0, 0] += 0.5
    assert ssim_same(a, b).item() < 1.0


def test_schedules():
    assert expon_lr(0, 1.6e-4, 1.6e-6) == 1.6e-4
    assert math.isclose(expon_lr(30000, 1.6e-4, 1.6e-6), 1.6e-6, rel_tol=1e-9)
    assert not is_filter_step(500, 100, 500, 15000, 30000)
    assert is_filter_step(14900, 100, 500, 15000, 30000)
    assert not is_filter_step(29900, 100, 500, 15000, 30000)


def test_spec_defaults():
    m = method()
    cfg = {c.name: c.config for c in m.components}
    assert cfg["gof-reg"]["lambda_distortion"] == 100.0
    assert cfg["gof-reg"]["lambda_depth_normal"] == 0.05
    assert cfg["gof-density"]["min_opacity"] == 0.05
    assert cfg["gof-raytrace"]["background"] == [0.0, 0.0, 0.0]
