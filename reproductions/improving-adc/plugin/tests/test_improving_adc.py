import math

import numpy as np
import torch

from ss_improving_adc import core, method, probes


def test_threshold_matches_eq14_and_official_steps():
    f = (4e-4 / 1e-4) ** (1 / 145)
    cur = 1e-4
    for k in range(1, 145):
        cur *= f  # official train.py loop
        assert math.isclose(core.threshold(k), cur, rel_tol=1e-12)
        assert math.isclose(core.threshold(k), core.eq14(k, 145), rel_tol=1e-12)
    assert core.threshold(0) == 1e-4 and math.isclose(core.threshold(145), 4e-4)


def test_extents_match_numpy():
    rng = np.random.default_rng(0)
    pts, cams = rng.normal(size=(200, 3)) * 5, rng.normal(size=(10, 3)) * 0.3
    c = cams.mean(0)
    assert math.isclose(core.corrected_extent(torch.tensor(pts), torch.tensor(cams)),
                        np.linalg.norm(pts - c, axis=1).mean(), rel_tol=1e-9)
    assert math.isclose(core.camera_extent(torch.tensor(cams)),
                        1.1 * np.linalg.norm(cams - c, axis=1).max(), rel_tol=1e-6)


def test_depth_factor_squared_and_clipped():
    d = torch.tensor([0.0, 1.85, 3.7, 10.0])
    torch.testing.assert_close(core.depth_factor(d, 3.7), torch.tensor([0.0, 0.25, 1.0, 1.0]))


def test_revised_opacity_preserves_transmittance():
    o = torch.tensor([0.1, 0.5, 0.9])
    n = core.revised_opacity(o)
    torch.testing.assert_close((1 - n) ** 2, 1 - o)


def test_significance_mask():
    sigma = torch.tensor([5.0, 0.0, 0.1, 9.0, 0.2, 3.0])
    split = torch.tensor([False, False, False, True, False, False])
    # post-densify layout: 5 old non-split, then 2 children
    opa = torch.tensor([0.001, 0.001, 0.5, 0.001, 0.001, 0.001, 0.9])
    m = core.significance_prune_mask(sigma, split, opa, 0.005)
    q = torch.quantile(torch.tensor([5.0, 0.0, 0.1, 0.2, 3.0]), 5 / 6)
    want = torch.tensor([5.0, 0.0, 0.1, 0.2, 3.0]) < q
    want = torch.cat([want, torch.tensor([False, False])]) & (opa < 0.005)
    assert torch.equal(m, want)
    # nothing low -> nothing pruned; new children (index 5, 6) never pruned
    assert not core.significance_prune_mask(sigma, split, opa * 0 + 1, 0.005).any()
    assert not m[5:].any()


def test_cpu_probes():
    m = method()
    for fn in (probes.scene_extent, probes.threshold_schedule, probes.revised_opacity,
               probes.schedule):
        fn(m, "cpu")
