import torch
from splatstudio.losses import create_photometric as base_photometric

import ss_spec_gaussian as sg
from ss_spec_gaussian import probes
from ss_spec_gaussian.asg import DEFAULTS, build_network, linear_lr, pack, unpack
from ss_spec_gaussian.losses import create_photometric


def test_variants_config():
    for name, (indoor, c2f) in sg.VARIANTS.items():
        m = sg.method(name)
        r = next(c for c in m.components if c.kind == "renderer").config
        d = next(c for c in m.components if c.kind == "density").config
        assert m.name == name and r["indoor"] == indoor and r["c2f"] == c2f
        for k in DEFAULTS:  # renderer and density build the same decoder
            assert r[k] == d[k]
        assert d["grow_grad2d"] == 0.0005 and d["absgrad"] and d["spatial_lr_scale"] == 5.0
        assert r["spec_start"] == 3000 and r["c2f_until"] == 5000


def test_cpu_probes():
    for name in sg.VARIANTS:
        m = sg.method(name)
        probes.asg_basis(m)
        out = probes.decoder_architecture(m)
        assert len(out["mlp_shapes"]) == (3 if m.name.startswith("spec-gaussian-indoor") else 2)


def test_blob_roundtrip_and_checksum():
    cfg = {**DEFAULTS, "indoor": True}
    torch.manual_seed(0)
    net = build_network(cfg)
    flat = pack(net)
    blob = torch.zeros(5000)
    blob[: flat.numel()] = flat
    net2 = unpack(blob, cfg)
    for a, b in zip(net.state_dict().values(), net2.state_dict().values()):
        assert torch.equal(a, b)
    try:
        unpack(blob, {**cfg, "indoor": False})
    except ValueError:
        pass
    else:
        raise AssertionError("outdoor layout accepted an indoor blob")


def test_linear_lr():
    f = linear_lr(0.0025, 0.000125, 30000)
    assert f(0) == 0.0025 and abs(f(30000) - 0.000125) < 1e-15 and f(60000) == f(30000)


def test_photometric_matches_base_at_full_res():
    class S:
        config = {"ssim_weight": 0.2}

    rgb = torch.rand(32, 40, 3, requires_grad=True)
    tgt = torch.rand(32, 40, 3)
    a = create_photometric(S)(None, {"rgb": rgb}, tgt)
    b = base_photometric(S)(None, {"rgb": rgb}, tgt)
    assert torch.allclose(a, b)
