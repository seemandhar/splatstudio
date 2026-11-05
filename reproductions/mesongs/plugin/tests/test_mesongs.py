import math

import numpy as np
import torch

import ss_mesongs
from ss_mesongs import euler, octree, probes, quant, raht, schedule


def _density_cfg(m):
    return next(c for c in m.components if c.kind == "density").config


def test_events():
    assert schedule.events(30000, 30000, 38000) == ["base", "compress"]
    assert schedule.events(30001, 30000, 38000) == ["bank_step"]
    assert schedule.events(37999, 30000, 38000) == ["bank_step", "freeze"]
    assert schedule.events(38000, 30000, 38000) == ["encode"]
    assert sum("bank_step" in schedule.events(s, 30000, 38000) for s in range(1, 38001)) == 7999
    assert schedule.events(30000, 30000, 30000) == ["base", "compress", "encode"]
    assert math.isclose(schedule.inferred_run_length(1.0, 0.01 ** (1 / 38000)), 38000)


def test_scene_tables():
    g = _density_cfg(ss_mesongs.method_garden())
    assert (g["prune_percent"], g["codebook_size"], g["depth"], g["n_block"]) == (0.28, 2048, 20, 57)
    assert math.isclose(g["lr_scale"], 0.4) and g["beta"] == 0.1 and g["total_steps"] == 38000
    b = _density_cfg(ss_mesongs.method_bicycle())
    assert (b["prune_percent"], b["beta"]) == (0.4, 0.03) and math.isclose(b["lr_scale"], 0.6)
    c = _density_cfg(ss_mesongs.method_counter())
    assert (c["prune_percent"], c["depth"]) == (0.22, 19)
    c1 = _density_cfg(ss_mesongs.method_c1_garden())
    assert (c1["prune_percent"], c1["n_block"]) == (0.18, 66)
    assert _density_cfg(ss_mesongs.method_noft_garden())["total_steps"] == 30000
    assert _density_cfg(ss_mesongs.method_kitchen())["codebook_size"] == 8192
    assert (g["vq_batch_size"], g["vq_steps"], g["vq_decay"]) == (262144, 1000, 0.8)


def test_cpu_probes():
    for fn in (probes.octree, probes.block_quant, probes.euler, probes.raht):
        fn(ss_mesongs.method_garden(), "cpu")
    probes.schedule(ss_mesongs.method_garden(), "cpu")
    probes.schedule(ss_mesongs.method_noft_garden(), "cpu")


def test_raht_edge_point_roundtrip():
    # official edge case: the maximum coordinate maps to code 2^depth
    rng = np.random.default_rng(1)
    V = np.unique(rng.integers(0, 2**5, (400, 3)), axis=0)
    one = np.concatenate([V, [[2**5, 0, 0]]])
    T = raht.RAHTTables(one, 5, "cpu")
    x = torch.randn(len(one), 7)
    assert torch.allclose(raht.inverse_transform(raht.forward_transform(x, T), T), x, atol=1e-5)
    assert T.roundtrip_bad_rows == 0
    # two edge codes: the official 9*depth inverse tree breaks the round trip for those rows
    two = np.concatenate([V, [[2**5, 0, 0], [0, 0, 2**5]]])
    assert raht.RAHTTables(two, 5, "cpu").roundtrip_bad_rows == 2


def test_octree_keys_official_formula():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.25, 0.75]], np.float32)
    ki, par = octree.octree_codes(pts, 2)
    assert ki.tolist() == [0, 4 * 16 + 4 * 4 + 4, 2 * 16 + 1 * 4 + 3]


def test_quant_formula():
    s, z = quant.calc_scale_zero_point(torch.tensor(-1.0), torch.tensor(3.0))
    assert math.isclose(float(s), 4 / 255, rel_tol=1e-6) and float(z) == round(255 - 3 / (4 / 255))
    assert torch.equal(quant.quantize_tensor(torch.tensor([3.0, -1.0]), s, z),
                       torch.tensor([255.0, 0.0]).clamp(0, 255).round())


def test_euler_identity():
    e = euler.to_euler(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(e, torch.zeros(1, 3), atol=1e-5)
