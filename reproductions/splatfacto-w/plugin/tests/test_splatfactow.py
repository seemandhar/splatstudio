import math

import pytest
import torch

from ss_splatfactow import method, method_a, method_t, probes
from ss_splatfactow.losses import PHOTOMETRIC_DEFAULTS, RobustMask, downscale_target
from ss_splatfactow.schedule import exp_decay, rotation_matrix_between


@pytest.mark.parametrize("fn", ["robust_mask", "masked_view_padding", "alpha_loss",
                                "splatfacto_schedule", "splatfacto_refine", "ns_normalisation"])
def test_cpu_probes(fn):
    assert getattr(probes, fn)(method(), "cpu")


def test_variants():
    names = {c.name for c in method_a().components}
    assert "sfw-alpha" not in names
    r = next(c for c in method_t().components if c.name == "sfw-photometric")
    assert r.config["robust_mask"] is True
    a = next(c for c in method_a().components if c.name == "sfw-photometric")
    assert a.config["robust_mask"] is False
    ren = next(c for c in method_a().components if c.name == "sfw-renderer")
    assert ren.config["bg_model"] is False


def test_exp_decay_endpoints():
    assert exp_decay(1.0, 0.01, 100, 0) == 1.0
    assert math.isclose(exp_decay(1.0, 0.01, 100, 50), 0.1)
    assert math.isclose(exp_decay(1.0, 0.01, 100, 500), 0.01)


def test_rotation_between():
    a = torch.tensor([0.3, -0.2, 0.9], dtype=torch.float64)
    R = rotation_matrix_between(a, torch.tensor([0.0, 0, 1], dtype=torch.float64))
    z = R @ (a / a.norm())
    assert torch.allclose(z, torch.tensor([0.0, 0, 1], dtype=torch.float64), atol=1e-9)
    assert torch.allclose(R @ R.T, torch.eye(3, dtype=torch.float64), atol=1e-9)


def test_downscale_target_is_area_average():
    t = torch.arange(2 * 9 * 10 * 3, dtype=torch.float32).reshape(18, 10, 3)
    d = downscale_target(t, 2, (9, 5, 3))
    assert d.shape == (9, 5, 3)
    assert torch.allclose(d[0, 0], t[:2, :2].reshape(-1, 3).mean(0))


def test_robust_mask_zero_percentage_keeps_everything():
    rm = RobustMask(PHOTOMETRIC_DEFAULTS)
    m, pct = rm(torch.rand(20, 20, 3), 6000)  # first call: max == min -> 0 %
    assert pct == 0.0 and bool((m == 1).all())


def test_masked_view_statistics_use_full_width():
    from splatstudio.components import build

    _, d, _ = build(method())
    means2d = torch.zeros(1, 3, 2)
    means2d.absgrad = torch.ones(1, 3, 2)
    radii = torch.tensor([[[10, 10], [0, 0], [30, 20]]])
    info = {"radii": radii, "means2d": means2d, "height": 100, "width": 60}
    d.accumulate(info, masked_view=True)
    assert d.last_size == (100, 120)
    assert torch.allclose(d.max_2d, torch.tensor([10 / 120, 0.0, 30 / 120]))
    d.reset_accumulators()
    d.accumulate(info, masked_view=False)
    assert d.last_size == (100, 60)
