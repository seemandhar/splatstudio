import math

import pytest
import torch

from ss_revising_densification import core, method, probes


def test_corrected_opacity_solves_center_pixel_equation():
    a = torch.linspace(0.01, 0.99, 50, dtype=torch.float64)
    h = core.corrected_opacity(a)
    torch.testing.assert_close((1 - h) ** 2, 1 - a)
    # Sec. 3.3 inequality for all G in (0, 1]
    g = torch.linspace(0.05, 1.0, 20, dtype=torch.float64)[:, None]
    assert torch.all(1 - a * g >= (1 - h * g) ** 2 - 1e-12)
    assert torch.all((1 - h * g) ** 2 > (1 - a * g) ** 2)


def test_growth_budget_and_selection():
    assert core.growth_budget(1000, 10**6, 0.05) == 50
    assert core.growth_budget(1000, 1020, 0.05) == 20
    assert core.growth_budget(1000, 900, 0.05) == 0
    s = torch.tensor([0.05, 0.5, 0.2, 0.9, 0.11, 0.1])
    sel, n = core.select_growth(s, 0.1, 2)
    assert n == 4 and sel.tolist() == [False, True, False, True, False, False]
    sel, n = core.select_growth(s, 0.1, 10)
    assert sel.tolist() == [False, True, True, True, True, False]


def test_decrement_and_cap():
    o = torch.tensor([0.5, 0.0005])
    torch.testing.assert_close(core.decrement_opacity(o, 0.001), torch.tensor([0.499, 1e-6]))
    assert core.primitive_cap("baseline", 136029) == (2494648, "tandt/truck")
    with pytest.raises(ValueError):
        core.primitive_cap("baseline", 1)


def test_error_map_cpu():
    probes.error_map(method(), "cpu")


def test_schedule_and_selection_cpu():
    m = method()
    assert probes.adc_schedule(m, "cpu")["runs"] == 264
    probes.error_selection(m, "cpu")
    probes.growth_budget(m, "cpu")


def test_ssim_map_constant_images():
    x = torch.full((16, 16, 3), 0.3)
    assert math.isclose(float(core.ssim_map(x, x)[8, 8]), 1.0, rel_tol=1e-6)
