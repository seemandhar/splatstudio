import torch

from ss_resgs import method, probes
from ss_resgs.density import DEFAULTS, thresholds
from ss_resgs.pyramid import pyr_size
from ss_resgs.schedule import ResGSSchedule


def test_residual_split_cpu():
    out = probes.residual_split_probe(method(), "cpu")
    assert out["gaussians_after"] == 6000


def test_selection_cpu():
    probes.selection_probe(method(), "cpu")


def test_schedule_matches_official_loop():
    out = probes.schedule_probe(method(), "cpu")
    assert out["level_first_step"][1] == 800
    assert out["level_first_step"][4] == 3000  # first densify after the 1st warm-up


def test_pyramid_cpu():
    probes.pyramid_probe(method(), "cpu")


def test_opacity_reduce_cpu():
    probes.opacity_reduce_probe(method(), "cpu")


def test_pyr_size():
    assert pyr_size(839, 1297, 2) == (210, 325)
    assert pyr_size(840, 1297, 1) == (420, 649)


def test_threshold_lower_for_coarse():
    thr = thresholds(torch.tensor([0.0, 3.0, 6.0]), 6, 1.0, 2 ** (1 / 3))
    torch.testing.assert_close(thr, torch.tensor([0.25, 0.5, 1.0]))


def test_schedule_levels_at_stage_starts():
    s = ResGSSchedule(DEFAULTS)
    seen = {}
    for t in range(1, 12001):
        if s.densify_step(t):
            seen.setdefault(s.cur_level, t)
        s.change_stage(t)
        s.end_iteration(t)
    assert seen[4] == 3000 and seen[7] == 6000  # counter reaches 7 at 5998
