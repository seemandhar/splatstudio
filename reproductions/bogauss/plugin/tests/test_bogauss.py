import math

import torch

from ss_bogauss import core, method, probes


def test_count_target_reaches_budget():
    steps = core.densify_steps(500, 15000, 500)
    t = [core.count_target(s, 1000, 50000, 500, 15000, 500) for s in steps]
    assert len(steps) == 29 and t[-1] == 50000
    assert all(b >= a for a, b in zip(t, t[1:]))
    # parabola through (0, S) with vertex at (n, B)
    n = 29
    ref = [1000 + 49000 * (1 - (1 - x / n) ** 2) for x in range(1, n + 1)]
    assert max(abs(a - b) for a, b in zip(t, ref)) <= 1.0


def test_budget_lookup():
    table = core.budget_table()
    assert core.resolve_budget(136029, None, table) == 2584171  # truck
    assert core.resolve_budget(182686, None, table) == 1085480  # train
    assert core.resolve_budget(5, 77, table) == 77


def test_effective_opacity_and_priority():
    blend = torch.tensor([2.0, 0.0, 1.0])
    count = torch.tensor([4.0, 0.0, 10.0])
    assert torch.allclose(core.effective_opacity(blend, count), torch.tensor([0.5, 0.0, 0.1]))
    se = torch.tensor([1.0, 3.0, 0.0])
    assert torch.allclose(core.snr_priority_increment(se), torch.tensor([0.25, 0.75, 0.0]))
    assert torch.equal(core.snr_priority_increment(torch.zeros(3)), torch.zeros(3))


def test_prune_selection_cap():
    pmax = torch.tensor([0.001, 0.5, 0.01, 0.015, 0.3] * 40)
    sel = core.prune_selection(pmax, 0.02, 0.01)
    assert len(sel) == 2 and torch.all(pmax[sel] == 0.001)


def test_split_adaptation_monotone():
    grid, ratio, oc = core.split_adaptation_table()
    assert torch.all(oc < grid) and torch.all(ratio < 1.0) and torch.all(ratio > 0.6)
    # low opacity: additive regime, children ~ half opacity, moment-matched width sqrt(1-0.09)
    assert abs(float(ratio[0]) - math.sqrt(1 - 0.09)) < 0.02
    r, o = core.split_adaptation(torch.tensor([0.5]))
    assert 0.25 < float(o) < 0.5 and 0.8 < float(r) < 1.0


def test_lr_calibration():
    delta = torch.tensor([1e-3, 2e-3, 3e-3])
    sigma = torch.full((3, 3), 0.01)
    em, es = core.lr_calibration(delta, sigma, extent=4.0)
    assert math.isclose(em * 2e-3, 1.6e-4 * 4.0, rel_tol=1e-6)
    assert math.isclose(es, 0.005 * 5.0, rel_tol=1e-6)


def test_cpu_probes():
    m = method()
    for fn in (probes.precision, probes.unbiased_sparse_adam, probes.linear_size,
               probes.state_inheritance, probes.prune, probes.split, probes.split_priority,
               probes.schedule):
        fn(m, "cpu")
