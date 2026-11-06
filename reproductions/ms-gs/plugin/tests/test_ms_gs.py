import torch

from ss_ms_gs import method, probes
from ss_ms_gs.losses import resize_target, ssim_same
from ss_ms_gs.renderer import scaled_camera


def test_pixel_coverage():
    probes.pixel_coverage_probe(method(), "cpu")


def test_selection_rule():
    probes.selection_rule_probe(method(), "cpu")


def test_range_update():
    probes.range_update_probe(method(), "cpu")


def test_aggregation():
    probes.aggregation_probe(method(), "cpu")


def test_schedule():
    probes.ms_schedule_probe(method(), "cpu")


def test_densify_fields():
    probes.densify_fields_probe(method(), "cpu")


def test_scaled_camera_official_sizes():
    cam = {"viewmat": torch.eye(4), "K": torch.tensor([[1186.0, 0, 800], [0, 1187.0, 518], [0, 0, 1]]),
           "width": 1600, "height": 1036}
    c = scaled_camera(cam, 7)
    assert (c["width"], c["height"]) == (12, 8)
    assert abs(c["K"][0, 2].item() - 6.0) < 1e-5 and abs(c["K"][1, 2].item() - 4.0) < 1e-5


def test_ssim_same_identity_and_tiny():
    x = torch.rand(8, 12, 3)
    assert abs(ssim_same(x, x).item() - 1) < 1e-5
    assert resize_target(torch.rand(1036, 1600, 3), 8, 12).shape == (8, 12, 3)
