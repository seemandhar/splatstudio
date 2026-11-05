import math

import pytest
import torch

from ss_contextgs import levels, method, model
from ss_contextgs.density import DEFAULTS, merge
from ss_contextgs.renderer import BLOB_WIDTH, blob_floats, pack_blob, unpack_blob


def test_spec_valid():
    from splatstudio.contracts import validate

    assert validate(method()) == []


def test_expon_lr_endpoints():
    f = model.expon_lr(0.005, 0.00001, 30000)
    assert f(0) == pytest.approx(0.005)
    assert f(30000) == pytest.approx(0.00001)
    assert f(15000) == pytest.approx(math.sqrt(0.005 * 0.00001))


def test_ste_multistep_clamp():
    x = torch.tensor([0.26, -0.74, 1.01, -1e9], requires_grad=True)
    y = model.STEMultistep.apply(x, torch.tensor(0.5))
    assert torch.allclose(y, torch.tensor([0.5, -0.5, 1.0, -7500.0]))
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones(4))


def test_quantize_anchor_16bit():
    a = torch.rand(100, 3)
    q, v = model.QuantizeAnchor.apply(a, torch.zeros(1, 3), torch.ones(1, 3))
    assert v.max() <= 2**16 - 1 and (q - a).abs().max() <= 1 / (2**16 - 1) + 1e-6


def test_unique_min_index():
    t = torch.tensor([[1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [2, 2, 2]]).float()
    _, inv, idx, counts = levels.torch_unique_with_indices(t)
    assert idx.tolist() == [1, 0, 4] and counts.tolist() == [2, 2, 1]


def test_mapping_chain():
    mapping = [torch.tensor([0, 3, 5, 7]), torch.tensor([1, 3])]
    assert levels.mapping_to_orign(mapping, 2).tolist() == [3, 7]
    assert levels.mapping_to_orign(mapping, 1, torch.tensor([True, False, True, False])).tolist() == [0, 5]


def test_factorized_prior_normalised():
    prior = model.FactorizedPrior(4)
    cdf = prior.cdf_table(-200, 200)
    assert cdf[:, 0].max() < 1e-6 and cdf[:, -1].min() > 1 - 1e-6
    assert bool((cdf[:, 1:] >= cdf[:, :-1]).all())


def test_level_mlp_dims():
    mlps = model.level_mlps(50, 10, 3, 12)
    assert mlps[2][0].in_features == 15 and mlps[0][0].in_features == 71
    assert mlps[0][2].out_features == (50 + 6 + 30) * 2 + 3
    n = sum(p.numel() for m in mlps for p in m.parameters()) + 13860
    assert n * 4 / 2**20 == pytest.approx(0.316, abs=1e-3)  # MLPs column of tab:coding_anchor


def test_blob_roundtrip():
    torch.manual_seed(0)
    mlps = model.scaffold_mlps(50, 10)
    n = blob_floats(50, 10) // BLOB_WIDTH + 3
    back = unpack_blob(pack_blob(mlps, n, 50, 10, "cpu"), 50, 10)
    for k in model.RENDER_MLPS:
        for a, b in zip(mlps[k].parameters(), back[k].parameters()):
            assert torch.equal(a, b)


def test_config_merge():
    c = merge(DEFAULTS, {"lr": {"feat": 1.0}, "voxel_size": 0.01})
    assert c["lr"]["feat"] == 1.0 and c["lr"]["hyper"] == 0.0075 and c["voxel_size"] == 0.01
    with pytest.raises(ValueError):
        merge(DEFAULTS, {"nope": 1})
