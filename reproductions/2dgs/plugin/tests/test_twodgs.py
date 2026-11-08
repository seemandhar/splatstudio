import torch

from ss_twodgs import method, method_paperdist, probes
from ss_twodgs.losses import normal_error


def test_normal_loss_probe():
    probes.normal_loss(method())


def test_distortion_loss_probe():
    probes.distortion_loss(method())
    probes.distortion_loss(method_paperdist())


def test_density_recipe_probe():
    probes.density_recipe(method())


def test_normal_error_zero_for_aligned_opaque():
    n = torch.nn.functional.normalize(torch.randn(4, 5, 3), dim=-1)
    assert normal_error(n, n).abs().max() < 1e-6


def test_spec_values():
    m = method()
    cfg = {c.name: c.config for c in m.components}
    assert cfg["2dgs-normal"] == {"lambda_normal": 0.05, "start": 7000}
    assert cfg["2dgs-distortion"]["lambda_dist"] == 0.0
    assert cfg["2dgs-density"]["opacity_cull"] == 0.05
    assert m.representation == "surfel2d"
