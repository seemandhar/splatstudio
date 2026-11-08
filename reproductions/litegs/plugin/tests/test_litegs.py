"""CPU unit tests of the pure LiteGS pieces (schedules, score, budget, optimizer, densify)."""

import ss_litegs
from ss_litegs import probes, schedule
from ss_litegs.density import growth_budget, growth_target


def test_schedule_garden():
    assert schedule.total_epochs(30000, 161) == 186
    assert schedule.densify_until(186, 10) == 141
    assert schedule.is_densify_epoch(5, 3, 141, 5)
    assert not schedule.is_densify_epoch(3, 3, 141, 5)
    assert not schedule.is_densify_epoch(145, 3, 141, 5)
    assert schedule.sh_degree(1, 161) == 0
    assert schedule.sh_degree(161 * 5 + 1, 161) == 1
    assert schedule.sh_degree(161 * 15 + 1, 161) == 3


def test_growth():
    assert growth_target(3, 100, 1100, 3, 13) == 100
    assert growth_target(13, 100, 1100, 3, 13) == 1100
    assert growth_budget(8, 400, 7, 100, 1100, 3, 13) == 200 + 7
    assert growth_budget(8, 700, 0, 100, 1100, 3, 13) == 1  # at least one


def test_probes_cpu():
    m = ss_litegs.method()
    probes.variance_score_probe(m, "cpu")
    probes.adam_probe(m, "cpu")
    probes.decay_schedule(m, "cpu")
    probes.densify_probe(m, "cpu")
