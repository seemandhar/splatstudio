import math

import torch

from ss_3dhgs import core, method, probes, reference


def test_clone_opacity_logit():
    a = torch.tensor([0.1, 0.5, 0.9])
    out = core.clone_opacity_logit(torch.logit(a)).sigmoid()
    torch.testing.assert_close(out, 1 - torch.sqrt(1 - a))
    # two clones of 1 - sqrt(1 - a) composite to the original alpha
    torch.testing.assert_close(1 - (1 - out) ** 2, a)


def test_averaged_nan_to_zero():
    out = core.averaged(torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0]))
    assert out.tolist() == [0.5, 0.0]


def test_split_offsets_follow_rotation():
    g = torch.Generator().manual_seed(0)
    q = torch.tensor([[math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]])  # 90 deg about z
    s = torch.tensor([[1.0, 1e-6, 1e-6]])
    off = core.split_offsets(s, q, g)
    assert off.shape == (2, 3)
    assert torch.all(off[:, 0].abs() < 1e-4)  # x-axis scale rotated onto y
    assert torch.all(off[:, 1].abs() > 0)


def test_density_probe_cpu():
    out = probes.density(method(), "cpu")
    assert out["after_first_densify"] == 5


def test_reference_equal_opacity_is_3dgs():
    cam = probes._camera(W=24, H=20, f=24.0)
    p = probes._scene(n=5, seed=1)
    p["opacity2"] = p["opacities"].clone()
    P = probes._to(p, "cpu", torch.float64)
    c = probes._to(cam, "cpu", torch.float64)
    half = reference.render(P, c, 3)
    full = reference.render(P, c, 3, kernel="full")
    torch.testing.assert_close(half, full)


def test_method_fields():
    m = method()
    names = {f.name: f for f in m.fields}
    assert names["hgs_normal"].learning_rate == 0.003
    assert names["hgs_normal"].initial_value == 0.6
    assert abs(names["opacity2"].initial_value - math.log(0.1 / 0.9)) < 1e-12
    dens = next(c for c in m.components if c.kind == "density")
    assert dens.config["densify_until"] == 20000
    assert dens.config["min_opacity"] == 0.01
    assert dens.config["reset_value"] == 0.02


def test_clone_opacity_logit_finite_for_extremes():
    out = core.clone_opacity_logit(torch.tensor([-30.0, -100.0, 50.0, 0.0]))
    assert torch.isfinite(out).all()
    assert out[1].sigmoid() < 1e-15 and out[2].sigmoid() > 0.999
