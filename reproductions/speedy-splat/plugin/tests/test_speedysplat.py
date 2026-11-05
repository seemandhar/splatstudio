import torch

from ss_speedysplat import accutile, method, stock
from ss_speedysplat.density import DEFAULTS, prune_mask, prune_steps
from ss_speedysplat.probes import _box_min, adc_schedule, prune_schedule


def test_prune_steps_match_official_conditions():
    soft, hard = prune_steps(DEFAULTS)
    assert soft == [6000, 9000, 12000]
    assert hard == [15000, 18000, 21000, 24000, 27000]


def test_prune_mask_percentile_rule():
    s = torch.tensor([5.0, 1.0, 3.0, 2.0, 4.0])
    # int(0.5 * 4) = 2 -> threshold = 3.0 -> prune {1, 2, 3}
    assert prune_mask(s, 0.5).tolist() == [False, True, True, True, False]
    assert prune_mask(torch.zeros(0), 0.8).numel() == 0


def test_box_min_inside_and_outside():
    A = [[1.0, 0.0], [0.0, 4.0]]
    assert _box_min(A, [0.5, 0.5], 0, 1, 0, 1) == 0.0
    # nearest point (2, 0.5) -> dx = -1 -> q = 1
    assert abs(_box_min(A, [3.0, 0.5], 0, 2, 0, 1) - 1.0) < 1e-12


def test_methods_and_cpu_probes():
    m = method()
    assert {c.kind for c in m.components} == {"renderer", "density", "loss"}
    assert next(c for c in accutile().components if c.kind == "density").config["prune"] is False
    assert next(c for c in stock().components if c.kind == "renderer").config[
        "rasterizer"] == "stock"
    assert adc_schedule(m, "cpu")["count"] == 144
    assert prune_schedule(m, "cpu")["final"] == 2
