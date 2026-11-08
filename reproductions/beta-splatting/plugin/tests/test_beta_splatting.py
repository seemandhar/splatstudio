import math

import torch
from splatstudio.contracts import validate
from splatstudio.specs import ComponentSpec

from ss_beta_splatting import beta_splatting, beta_splatting_garden, beta_splatting_room
from ss_beta_splatting.density import split_opacity
from ss_beta_splatting.losses import create_regularizer
from ss_beta_splatting.reference import dense_render, spherical_beta_ref
from ss_beta_splatting.renderer import beta_activation, beta_radii, sb_activation


def test_specs_valid_and_caps():
    for fn, cap in ((beta_splatting, 1_000_000), (beta_splatting_garden, 5_000_000),
                    (beta_splatting_room, 1_500_000)):
        m = fn()
        assert validate(m) == []
        density = next(c for c in m.components if c.kind == "density")
        assert density.config["cap_max"] == cap
        assert {f.name: f.learning_rate for f in m.fields} == {"betas": 0.001, "sb_params": 0.0025}


def test_beta_activation_and_sb_activation():
    assert float(beta_activation(torch.zeros(1))) == 4.0
    raw = torch.zeros(1, 2, 6)
    act = sb_activation(raw)
    assert torch.allclose(act[..., :3], torch.full((1, 2, 3), math.log(2) / (10 * math.log(2))))
    assert torch.equal(act[..., 3:], raw[..., 3:])


def test_dense_single_splat_beta():
    m = torch.tensor([[10.0, 10.0]], dtype=torch.float64)
    s = 3.0
    q = torch.tensor([[1 / s**2, 0.0, 1 / s**2]], dtype=torch.float64)
    _, alpha = dense_render(m, q, torch.ones(1, 3, dtype=torch.float64),
                            torch.tensor([0.5], dtype=torch.float64),
                            torch.tensor([4.0], dtype=torch.float64), torch.ones(1), 20, 20)
    ys, xs = torch.meshgrid(torch.arange(20) + 0.5, torch.arange(20) + 0.5, indexing="ij")
    r2 = (((xs - 10) ** 2 + (ys - 10) ** 2) / s**2).double()
    want = torch.where(r2 < 1, 0.5 * (1 - r2).clamp_min(0) ** 4, torch.zeros_like(r2))
    assert torch.allclose(alpha[..., 0], want, atol=1e-12)


def test_beta_radii_one_sigma():
    conics = torch.tensor([[[1 / 9.0, 0.0, 1 / 4.0]]])
    r = beta_radii(torch.tensor([[[50.0, 50.0]]]), conics, torch.ones(1, 1, 2, dtype=torch.int32),
                   100, 100)
    assert r[0, 0].tolist() == [3, 2]
    r = beta_radii(torch.tensor([[[-5.0, 50.0]]]), conics, torch.ones(1, 1, 2, dtype=torch.int32),
                   100, 100)
    assert r.abs().sum() == 0


def test_spherical_beta_reference():
    dirs = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64)
    c0 = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64)
    co = torch.zeros(1, 2, 6, dtype=torch.float64)
    co[0, 0, :3] = 1.0  # lobe 0 along +z (theta 0), b 0 -> dot^4 = 1
    co[0, 1, :3] = 1.0
    co[0, 1, 3] = math.pi  # lobe 1 along -z: dot < 0 -> 0
    out = spherical_beta_ref(dirs, c0, co)
    assert torch.allclose(out, c0 + 1.0)


def test_split_opacity():
    o = torch.tensor([0.5, 0.5, 0.004])
    k = torch.tensor([0.0, 1.0, 0.0])
    got = split_opacity(o, k, 0.005)
    assert torch.allclose(got[:2], torch.tensor([0.5, 1 - 0.5 ** 0.5]))
    assert got[2] == 0.005


def test_regularizer_window():
    fn = create_regularizer(ComponentSpec("r", "loss", "x", (), (), {}))
    p = {"opacities": torch.zeros(4, requires_grad=True), "scales": torch.zeros(4, 3)}
    assert float(fn(p, {"step": 500}, None)) == 0.0
    assert abs(float(fn(p, {"step": 501}, None)) - (0.01 * 0.5 + 0.01)) < 1e-9
    assert float(fn(p, {"step": 25000}, None)) == 0.0
