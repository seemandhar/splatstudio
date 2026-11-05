"""CPU unit tests of pure SplatFields functions."""

import math

import numpy as np
import pytest
import torch
from splatstudio.registry import load as load_method

from ss_splatfields import density, losses, network


def test_embed_matches_nerf_order():
    x = torch.tensor([[0.3, -0.2, 0.7]])
    e = network.embed(x, 2)
    ref = torch.cat([x, torch.sin(x), torch.cos(x), torch.sin(2 * x), torch.cos(2 * x)], -1)
    assert torch.allclose(e, ref)
    assert network.embed_dim(6) == 39


def test_general_mlp_skip_and_last_activation():
    torch.manual_seed(0)
    m = network.GeneralMLP(5, 3, 16, 6, [3], 2, "none")
    assert m.net[4].in_features == 16 + network.embed_dim(2) + 5  # layer after the skip
    assert len(m.net) == 8
    out = m(torch.randn(10, 3), torch.randn(10, 5))
    # leaky_relu is applied to the last layer too: negative outputs are scaled by 0.01
    pre = m.net[-1].weight
    assert out.shape == (10, 3) and pre.shape == (3, 16)


def test_exp_lr_endpoints():
    assert math.isclose(network.exp_lr(0, 8e-4, 1.6e-6, 40000), 8e-4)
    assert math.isclose(network.exp_lr(40000, 8e-4, 1.6e-6, 40000), 1.6e-6)
    assert math.isclose(network.exp_lr(80000, 8e-4, 1.6e-6, 40000), 1.6e-6)


def test_network_size_about_1m():
    n = sum(p.numel() for p in network.SplatFieldsNet().parameters())
    assert 0.9e6 < n < 1.1e6  # supp. "~1M parameters for the static case"


def test_mlp_only_ablation_has_no_triplane():
    net = network.SplatFieldsNet({"use_triplane": False})
    assert net.encoder is None and net.feat_dim == 0
    out = net(torch.rand(7, 3), torch.full((7, 3), 0.01))
    assert out["means"].shape == (7, 3)


def test_mask_table_lookup_and_l1():
    vm = np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32) * 2])
    vm[1, 3, 3] = 1
    masks = np.zeros((2, 4, 4), np.uint8)
    masks[1] = 255
    t = losses.MaskTable(vm, masks)
    m = t.lookup(torch.tensor(vm[1]), "cpu")
    assert float(m.mean()) == 1.0
    with pytest.raises(KeyError):
        t.lookup(torch.eye(4) * 3, "cpu")
    alpha = torch.full((4, 4), 1.5)
    assert float(losses.mask_l1(alpha, m)) == 0.0


def test_norm_value():
    p = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 1.0]])
    assert float(losses.splat_norm(p)) == 3.0


def test_method_configs():
    m = load_method("splatfields")
    names = [c.name for c in m.components]
    assert names == ["splatfields-renderer", "splatfields-density", "photometric", "mask",
                     "splat-norm"]
    cfg = {c.name: c.config for c in m.components}
    assert cfg["mask"]["weight"] == 0.1 and cfg["splat-norm"]["weight"] == 0.01
    assert cfg["splatfields-density"]["total_steps"] == 40000
    assert density.BASE["refine_stop_iter"] == 40000
    s1 = load_method("splatfields-3dgs")
    assert [c.name for c in s1.components] == ["sf3dgs-renderer", "sf3dgs-density",
                                              "photometric", "mask"]


def test_bake_writes_baseline_compatible_fields():
    m = load_method("splatfields")
    spec = next(c for c in m.components if c.kind == "density")
    d = density.SplatFieldsDensity(spec)
    from splatstudio.state import sample_scene

    params, opts = sample_scene(m, "cpu", n=20)
    try:
        d.initialize(params, opts, 1.0)
        with torch.no_grad():
            pred = d.net(params["means"], params["scales"].exp())
            d.bake(params)
        assert torch.allclose(params["means"], pred["means"])
        assert torch.allclose(params["scales"].exp(), pred["scales"].abs(), rtol=1e-5)
        assert torch.allclose(params["opacities"].sigmoid(), pred["opacities"], atol=1e-6)
        rgb = params["sh0"][:, 0] * density.C0 + 0.5
        assert torch.allclose(rgb, pred["rgb"], atol=1e-6)
        assert float(params["shN"].abs().max()) == 0 and d.net.baked
    finally:
        network.ACTIVE["net"] = None


def test_steps_mismatch_raises(monkeypatch):
    monkeypatch.setattr("sys.argv", ["ss-train", "splatfields", "--steps", "2000"])
    with pytest.raises(ValueError):
        density.check_steps(1, 40000)
    density.check_steps(2, 40000)  # only checked at step 1
    monkeypatch.setattr("sys.argv", ["ss-train", "--steps", "40000"])
    density.check_steps(1, 40000)
