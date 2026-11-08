import math

import torch

from ss_lpm import core, probes


def test_error_map_probe():
    probes.error_map_probe(None, "cpu")


def test_zone_probe():
    probes.zone_projection_probe(None, "cpu")


def test_densify_probe():
    probes.localized_densify_probe(None, "cpu")


def test_reset_probe():
    probes.localized_reset_probe(None, "cpu")


def test_parallel_rays_give_nan_zone_and_no_points():
    # official behaviour when a view is paired with itself: identical rays -> NaN zone -> no points
    cam = probes._camera([0, 0, 0])
    zones = core.zones3d(cam, cam, [(10, 10, 20, 20)], [(10, 10, 20, 20)])
    assert len(zones) == 1 and torch.isnan(zones[0]).any()
    xyz = torch.randn(50, 3)
    assert not core.in_zones(xyz, zones).any()
    assert not core.reset_mask(torch.zeros(3), zones, xyz, torch.zeros(50)).any()


def test_neighbour_angles():
    cams = [probes._camera([0, 0, 0])]
    c = probes._camera([1, 0, 0])
    a = math.radians(60)
    R = torch.tensor([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]])
    c["viewmat"][:3, :3] = R
    cams.append(c)
    d = core.center_ray_dirs(cams)
    assert not core.neighbour_mask(d, 45)[0, 1]
    assert core.neighbour_mask(d, 90)[0, 1]
    assert not core.neighbour_mask(d, 90)[0, 0]


def test_zero_area_regions_skipped():
    cam = probes._camera([0, 0, 0])
    assert core.zones3d(cam, cam, [(5, 5, 5, 9)], [(1, 1, 4, 4)]) == []
