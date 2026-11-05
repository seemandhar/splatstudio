import math

import pytest
import torch

from ss_hac import method, model
from ss_hac.density import DEFAULTS, merge
from ss_hac.renderer import BLOB_WIDTH, blob_floats, pack_blob, unpack_blob


def test_spec_valid():
    from splatstudio.contracts import validate

    assert validate(method()) == []


def test_expon_lr_endpoints():
    f = model.expon_lr(0.01, 0.0001, 30000)
    assert f(0) == pytest.approx(0.01)
    assert f(30000) == pytest.approx(0.0001)
    assert f(15000) == pytest.approx(math.sqrt(0.01 * 0.0001))
    assert model.expon_lr(0.0, 0.0, 10)(5) == 0.0


def test_ste_multistep_and_binary():
    x = torch.tensor([0.26, -0.74, 1.01], requires_grad=True)
    y = model.STEMultistep.apply(x, torch.tensor(0.5), torch.tensor(0.0))
    assert torch.allclose(y, torch.tensor([0.5, -0.5, 1.0]))
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones(3))
    t = torch.tensor([-2.0, -0.3, 0.0, 0.7], requires_grad=True)
    b = model.STEBinary.apply(t)
    assert torch.equal(b, torch.tensor([-1.0, -1.0, 1.0, 1.0]))
    b.sum().backward()
    assert torch.equal(t.grad, torch.tensor([0.0, 1.0, 1.0, 1.0]))


def test_gaussian_bits_formula():
    x, m, s, q = torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([1.0])
    p = math.erf(0.5 / math.sqrt(2))
    assert model.gaussian_bits(x, m, s, q).item() == pytest.approx(-math.log2(p), rel=1e-5)


def test_quantize_anchor_16bit():
    a = torch.rand(100, 3)
    lo, hi = torch.zeros(1, 3), torch.ones(1, 3)
    q, v = model.QuantizeAnchor.apply(a, lo, hi)
    assert v.max() <= 2**16 - 1 and (q - a).abs().max() <= 1 / (2**16 - 1) + 1e-6


def test_blob_roundtrip():
    torch.manual_seed(0)
    mlps = model.scaffold_mlps(50, 10)
    n = blob_floats(50, 10) // BLOB_WIDTH + 3
    blob = pack_blob(mlps, n, 50, 10, "cpu")
    back = unpack_blob(blob, 50, 10)
    for k in model.RENDER_MLPS:
        for a, b in zip(mlps[k].parameters(), back[k].parameters()):
            assert torch.equal(a, b)
    with pytest.raises(ValueError):
        pack_blob(mlps, 10, 50, 10, "cpu")
    with pytest.raises(ValueError):
        unpack_blob(torch.zeros_like(blob), 50, 10)


def test_config_merge():
    c = merge(DEFAULTS, {"lr": {"feat": 1.0}, "voxel_size": 0.01})
    assert c["lr"]["feat"] == 1.0 and c["lr"]["scaling"] == 0.007 and c["voxel_size"] == 0.01
    with pytest.raises(ValueError):
        merge(DEFAULTS, {"nope": 1})
