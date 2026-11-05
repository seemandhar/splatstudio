import math

import pytest
import torch

from ss_dashgaussian import method, probes
from ss_dashgaussian import schedule as S
from ss_dashgaussian.renderer import scaled_camera, scaled_size


@pytest.mark.parametrize("name", ["reso_schedule", "target_downsample", "primitive_schedule",
                                  "topk_densify", "lr_schedule"])
def test_cpu_probes(name):
    assert getattr(probes, name)(method(), "cpu")


def test_scaled_size_matches_interpolate():
    for h, w, r in ((822, 1237, 5), (545, 980, 3), (61, 83, 7.37), (100, 100, 1)):
        x = torch.zeros(1, 1, h, w)
        y = torch.nn.functional.interpolate(x, scale_factor=1 / r, mode="bilinear",
                                            recompute_scale_factor=True, antialias=True)
        assert scaled_size(h, w, r) == tuple(y.shape[-2:])


def test_scaled_camera_keeps_field_of_view():
    cam = {"K": torch.tensor([[500.0, 0, 400], [0, 510.0, 300], [0, 0, 1]]),
           "viewmat": torch.eye(4), "width": 801, "height": 601}
    low = scaled_camera(cam, 4)
    assert (low["width"], low["height"]) == (200, 150)
    # tan(fov/2) = (W/2)/fx is preserved up to the floor of the size.
    assert math.isclose(low["width"] / low["K"][0, 0].item(), 801 / 500.0, rel_tol=1e-6)
    assert math.isclose(low["height"] / low["K"][1, 1].item(), 601 / 510.0, rel_tol=1e-6)
    assert scaled_camera(cam, 1) is cam


def test_position_lr_endpoints():
    assert S.position_lr(10, 1.0, 0.01, 30000, 5000) == 1.0
    assert math.isclose(S.position_lr(30000, 1.0, 0.01, 30000, 5000), 0.01)
    assert math.isclose(S.position_lr(17500, 1.0, 0.01, 30000, 5000), 0.1)


def test_low_frequency_image_hits_cap():
    yy, xx = torch.meshgrid(torch.arange(64.0), torch.arange(80.0), indexing="ij")
    img = 0.5 + 0.3 * torch.cos(2 * math.pi * xx / 80)
    sched = S.build_schedule([img.expand(3, -1, -1).clone()], increase_until=27000)
    assert sched["max_scale"] == 8.0


def _reference_ssim_same(img, ref):
    """Independent 3DGS loss_utils.ssim (CHW, create_window via outer product, padding 5)."""
    g = torch.tensor([math.exp(-((x - 5) ** 2) / (2 * 1.5**2)) for x in range(11)])
    g = (g / g.sum())[:, None]
    w = (g @ g.t())[None, None].expand(3, 1, 11, 11).contiguous()
    a, b = img.permute(2, 0, 1)[None], ref.permute(2, 0, 1)[None]
    mu1 = torch.nn.functional.conv2d(a, w, padding=5, groups=3)
    mu2 = torch.nn.functional.conv2d(b, w, padding=5, groups=3)
    s11 = torch.nn.functional.conv2d(a * a, w, padding=5, groups=3) - mu1.pow(2)
    s22 = torch.nn.functional.conv2d(b * b, w, padding=5, groups=3) - mu2.pow(2)
    s12 = torch.nn.functional.conv2d(a * b, w, padding=5, groups=3) - mu1 * mu2
    m = ((2 * mu1 * mu2 + 1e-4) * (2 * s12 + 9e-4)) / ((mu1.pow(2) + mu2.pow(2) + 1e-4) * (s11 + s22 + 9e-4))
    return m.mean()


@pytest.mark.parametrize("padding,should_match", [("same", True), ("valid", False)])
def test_loss_ssim_padding(padding, should_match):
    from dataclasses import replace

    from ss_dashgaussian.losses import create_photometric

    spec = next(c for c in method().components if c.kind == "loss")
    assert spec.config["ssim_padding"] == "same"
    loss = create_photometric(replace(spec, config={**spec.config, "ssim_padding": padding}))
    g = torch.Generator().manual_seed(0)
    target = torch.rand(40, 52, 3, generator=g)
    rgb = (target + 0.2 * torch.randn(40, 52, 3, generator=g)).clamp(0, 1)
    expected = 0.8 * (rgb - target).abs().mean() + 0.2 * (1 - _reference_ssim_same(rgb, target))
    got = loss(None, {"rgb": rgb}, target)
    assert math.isclose(float(got), float(expected), rel_tol=1e-5) is should_match
