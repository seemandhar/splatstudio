import numpy as np
import pytest
import torch

from ss_taming_3dgs import budget, core, method, probes
from ss_taming_3dgs.optim import IntervalAdam


def test_count_array_matches_official_formula():
    v = core.count_array(54275, 5987095, 500, 15000, 100)
    assert len(v) == 145 and v[0] == 54275
    assert v[144] < 5987095 and v[144] > 5987095 - 1000
    assert all(b >= a for a, b in zip(v, v[1:]))
    # budget mode: 29 entries, interval 500
    assert len(core.count_array(1000, 15000, 500, 15000, 500)) == 29


def test_resolve_budget():
    table = core.table_by_count(core.BIG_BUDGETS)
    assert core.resolve_budget(138766, "final_count", None, table) == 5728191
    mt = core.table_by_count(core.BUDGET_MULTIPLIERS)
    assert core.resolve_budget(138766, "multiplier", None, mt) == 138766 * 15
    assert core.resolve_budget(10, "final_count", 77, table) == 77
    with pytest.raises(ValueError):
        core.resolve_budget(12345, "final_count", None, table)


def test_normalize_median():
    x = torch.tensor([0.0, 2.0, 4.0, float("nan"), 6.0, 8.0])
    out = core.normalize(10.0, x)
    # torch.median of [2,4,6,8] is the lower middle value 4
    assert torch.allclose(out, torch.tensor([0.0, 5.0, 10.0, 0.0, 15.0, 20.0]))
    assert torch.equal(core.normalize(3.0, torch.zeros(4)), torch.zeros(4))


def test_split_budgets():
    assert core.split_budgets(1100, 1000, 30, 70) == (30, 70)
    assert core.split_budgets(1100, 1000, 10, 10) == (10, 10)  # capped by candidates
    assert core.split_budgets(1100, 1000, 1, 2) == (1, 2)
    assert core.split_budgets(1100, 1000, 0, 0) == (0, 0)
    assert core.split_budgets(1011, 1000, 200, 200) == (5, 5)  # official floor division


def test_sample_without_replacement_fills_with_zero_weights():
    w = torch.tensor([0.0, 1.0, 0.0, 2.0])
    idx = core.sample_without_replacement(w, 3)
    assert len(set(idx.tolist())) == 3 and {1, 3} <= set(idx.tolist())
    assert len(core.sample_without_replacement(w, 0)) == 0
    assert len(core.sample_without_replacement(torch.zeros(4), 2)) == 0


def test_ssim_same_identity_and_padding():
    x = torch.rand(20, 24, 3)
    assert abs(float(core.ssim_same(x, x)) - 1.0) < 1e-5
    assert float(core.ssim_same(x, torch.rand(20, 24, 3))) < 0.5


def test_edge_map_is_pil_find_edges():
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:, 8:] = 200
    e = core.edge_map(torch.from_numpy(img))
    assert e.shape == (16, 16) and e.dtype == torch.uint8
    assert int(e[5, 8]) > 0 and int(e[5, 3]) == 0


def test_interval_adam_cpu():
    probes.sh_interval(method(), "cpu")


def test_cpu_probes():
    for fn in (probes.growth_curve, probes.high_opacity, probes.schedule,
               probes.guided_prune, probes.sampled_densify):
        fn(method(), "cpu")


def test_interval_adam_every_one_is_adam():
    p = torch.nn.Parameter(torch.zeros(4))
    q = torch.nn.Parameter(torch.zeros(4))
    a, b = IntervalAdam([p], lr=0.1, every=1), torch.optim.Adam([q], lr=0.1, eps=1e-15)
    for _ in range(5):
        g = torch.randn(4)
        p.grad, q.grad = g.clone(), g.clone()
        a.step()
        b.step()
    assert torch.allclose(p, q)


def test_specs():
    m, b = method(), budget()
    dm = next(c for c in m.components if c.kind == "density").config
    db = next(c for c in b.components if c.kind == "density").config
    assert dm["mode"] == "final_count" and dm["densification_interval"] == 100
    assert db["mode"] == "multiplier" and db["densification_interval"] == 500
    assert dm["lr"] == {"opacities": 0.025, "shN": 0.00025}
