"""CPU unit tests of the pure WildGaussians functions and CPU probes."""

import math

import pytest
import torch

from ss_wildgaussians import method, method_noapp, method_pt, probes
from ss_wildgaussians.density import expon_lr
from ss_wildgaussians.evaluate import dmpix_ssim, left_half_camera, to_uint8_float
from ss_wildgaussians.geometry import knn_log_scales
from ss_wildgaussians.uncertainty import dino_downsample, pads_for

M = method()


@pytest.mark.parametrize("name", ["appearance_toning", "split_objective", "uncertainty_loss",
                                  "sky_init", "densify_official", "filter_3d"])
def test_cpu_probes(name):
    assert getattr(probes, name)(M, "cpu")


def test_noapp_has_no_appearance_state():
    m = method_noapp()
    assert {f.name for f in m.fields} == {"filter_3d"}
    r = next(c for c in m.components if c.kind == "renderer")
    assert r.config["appearance"] is False
    probes.sky_init(m, "cpu")


def test_pt_schedule():
    d = next(c for c in method_pt().components if c.kind == "density").config
    assert (d["iterations"], d["densify_from_iter"], d["densify_until_iter"],
            d["densification_interval"], d["opacity_reset_interval"]) == (
        200000, 4000, 100000, 400, 15000)
    u = next(c for c in method_pt().components if c.name == "wg-uncertainty").config
    assert (u["warmup_start"], u["warmup_iters"], u["protected_iters"]) == (35000, 5000, 1500)


def test_expon_lr_endpoints():
    assert expon_lr(0, 1.6e-4, 1.6e-6, 30000) == pytest.approx(1.6e-4)
    assert expon_lr(30000, 1.6e-4, 1.6e-6, 30000) == pytest.approx(1.6e-6)
    assert expon_lr(15000, 1.6e-4, 1.6e-6, 30000) == pytest.approx(1.6e-5)


def test_dino_downsample_and_pads():
    x = torch.zeros(1, 3, 1095, 730)
    y = dino_downsample(x, 350)
    assert y.shape[2] % 14 == 0 and y.shape[3] % 14 == 0 and y.shape[2] == 350
    assert pads_for(torch.zeros(1, 3, 271, 483)) == [3, 4, 4, 5]  # 483->490, 271->280


def test_knn_scales():
    pts = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
    s = knn_log_scales(pts)
    assert s[0, 0].item() == pytest.approx(math.log(math.sqrt((1 + 4 + 9) / 3)), rel=1e-5)


def test_metrics_conventions():
    x = torch.full((4, 4, 3), 0.5039)  # 128.49 -> truncated to 128
    assert to_uint8_float(x)[0, 0, 0].item() == pytest.approx(128 / 255)
    a = torch.rand(32, 40, 3)
    assert dmpix_ssim(a, a) == pytest.approx(1.0, abs=1e-5)
    cam, gt = left_half_camera({"width": 65, "height": 4, "K": torch.eye(3),
                                "viewmat": torch.eye(4), "image": torch.zeros(4, 65, 3)})
    assert cam["width"] == 33 and gt.shape[1] == 33  # W - W//2 columns
