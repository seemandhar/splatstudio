import numpy as np
import pytest

from ss_ad_gs import method, probes
from ss_ad_gs.pseudo import random_poses_llff
from ss_ad_gs.schedule import Schedule


@pytest.fixture(scope="module")
def m():
    return method()


def test_schedule_matches_official():
    s = Schedule()
    assert not s.regularize(1500) and not s.regularize(1600) and s.regularize(1700)
    assert s.phase(1600) == "high" and s.phase(1700) == "low" and s.phase(800) == "warmup"


def test_alternating_densification(m):
    out = probes.alternating_densification(m, "cpu")
    assert out["low"]["pruned"] == 1


def test_gated_losses(m):
    probes.phase_gated_losses(m, "cpu")


def test_edge_aware(m):
    probes.edge_aware_depth(m, "cpu")


def test_pseudo_poses_reproduce_official_horns_focus():
    # Bounds of the horns 3-view training cameras (official pose_utils.py comment value).
    rng = np.random.default_rng(0)
    views = [np.eye(4), np.eye(4)]
    views[1][0, 3] = -1.0
    bounds = np.array([[12.47, 168.81], [16.21, 153.86]])
    poses, info = random_poses_llff(views, bounds, 5, rng)
    scale = 1 / (12.47 * 0.75)
    inf = 168.81 * 5 * scale
    focal = 1 / (0.25 / (12.47 * 0.9 * scale) + 0.75 / inf)
    assert info["focus_distance"] == pytest.approx(focal / scale)
    assert poses.shape == (5, 4, 4)


def test_sh_schedule(m):
    assert probes.sh_schedule(m, "cpu")["degrees"] == {"499": 0, "500": 1, "1000": 2, "1500": 3}


def test_official_bounds_lookup():
    from ss_ad_gs.density import DEFAULTS, official_bounds

    c2w = np.eye(4)
    c2w[:3, 3] = DEFAULTS["scene_bounds"]["horns"]["train_centers"][0]
    views = []
    for c in DEFAULTS["scene_bounds"]["horns"]["train_centers"]:
        m4 = np.eye(4)
        m4[:3, 3] = c
        views.append(np.linalg.inv(m4))
    scene, b = official_bounds(DEFAULTS["scene_bounds"], views)
    assert scene == "horns" and b.tolist() == [[12.473701480698228, 168.81094102559487]]
    with pytest.raises(ValueError):
        official_bounds(DEFAULTS["scene_bounds"], [np.eye(4)] * 3)
