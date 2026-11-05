import io
import math

import numpy as np
import torch
import torch.nn.functional as F

from ss_c3dgs import codec, method, quant, schedule, smoke, vq
from ss_c3dgs.bank import DEFAULT_LRS, CompressedBank


def test_events_and_lrs():
    assert schedule.events(30000, 30000, 35000) == ["base", "compress"]
    assert schedule.events(30001, 30000, 35000) == ["bank_step"]
    assert schedule.events(35000, 30000, 35000) == ["encode"]
    assert sum("bank_step" in schedule.events(s, 30000, 35000) for s in range(1, 35001)) == 4999
    for s in (1, 15000, 29999, 30000):
        assert math.isclose(schedule.pretrain_means_lr(2.0, s, 30000), 2.0 * 0.01 ** (s / 30000))
    assert math.isclose(schedule.inferred_run_length(1.0, 0.01 ** (1 / 35000)), 35000)
    assert schedule.inferred_run_length(1.0, 1.0) is None


def test_spec_defaults():
    d = next(c for c in method().components if c.kind == "density").config
    assert d["base_steps"] == 30000 and d["total_steps"] == 35000
    assert d["color_codebook_size"] == d["gaussian_codebook_size"] == 4096
    assert d["color_importance_include"] == 6e-7 and d["gaussian_importance_include"] == 3e-6
    assert (d["color_cluster_iterations"], d["color_batch_size"]) == (100, 2**18)
    assert (d["gaussian_cluster_iterations"], d["gaussian_batch_size"]) == (800, 2**20)
    assert d["decay"] == 0.8 and d["prune_threshold"] == 0.0 and d["freeze_means"]
    assert d["base"] == {"absgrad": False, "grow_grad2d": 0.0002}
    assert DEFAULT_LRS == {"cb_dc": 0.0025, "cb_rest": 0.000125, "cb_scale": 0.005,
                           "log_eta": 0.005, "cb_rot": 0.001}
    s = next(c for c in smoke().components if c.kind == "density").config
    assert (s["base_steps"], s["total_steps"]) == (1500, 2000)


def test_morton():
    pos = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [3, 0, 0], [1, 1, 1]])
    assert quant.morton_encode(pos).tolist() == [1, 2, 4, 9, 7]


def test_weighted_update_and_vq():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(500, 6, generator=g)
    w = torch.rand(500, generator=g)
    cb = torch.randn(8, 6, generator=g)
    cb0 = cb.clone()
    vq.update(cb, x, w, 0.8, assign=vq.nearest_reference)
    _, idx = vq.nearest_reference(x, cb0)
    for k in range(8):
        m = idx == k
        new = (x[m] * w[m, None]).sum(0) / (w[m].sum() + 1e-5)
        assert torch.allclose(cb[k], 0.8 * cb0[k] + 0.2 * new, atol=1e-5)
    q = F.normalize(torch.randn(300, 4, generator=g))
    s = F.normalize(torch.rand(300, 3, generator=g) + 0.1)
    cov = vq.covariance(s, q)
    assert torch.allclose(cov[:, [0, 3, 5]].sum(-1), torch.ones(300), atol=1e-5)
    book, ind = vq.vq_features(cov, w[:300], 16, 64, 10, scale_normalize=True, generator=g,
                               assign=vq.nearest_reference)
    assert torch.allclose(book[:, [0, 3, 5]].sum(-1), torch.ones(16), atol=1e-5)
    keep = w[:300] > 0.9
    joined, jidx = vq.join_features(cov, keep, book, ind[~keep])
    assert torch.equal(joined[jidx[keep]], cov[keep]) and len(joined) == 16 + int(keep.sum())


def test_eigh_split_reconstructs():
    g = torch.Generator().manual_seed(1)
    q = F.normalize(torch.randn(200, 4, generator=g))
    s = F.normalize(torch.rand(200, 3, generator=g) + 0.05)
    cov = vq.covariance(s, q)
    rot, sc = vq.extract_rot_scale(vq.to_full_cov(cov))
    assert torch.allclose(vq.covariance(sc, rot), cov, atol=1e-5)
    assert torch.allclose(sc.norm(dim=-1), torch.ones(200), atol=1e-4)


def test_quantize_matches_fake_quant():
    fq = torch.ao.quantization.FakeQuantize(dtype=torch.qint8)
    x = torch.randn(1000) * 3
    y = fq(x)
    deq = quant.dequantize(quant.quantize_int8(x, fq), fq.scale, fq.zero_point)
    assert torch.allclose(deq, y, atol=1e-6)
    assert fq.quant_min == -128 and fq.quant_max == 127


def _bank(n=300, kc=12, kg=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    params = {"means": torch.rand(n, 3, generator=g), "opacities": torch.randn(n, generator=g)}
    bank = CompressedBank(
        torch.randn(kc, 1, 3, generator=g), torch.randn(kc, 15, 3, generator=g) * 0.2,
        torch.randint(0, kc, (n,), generator=g), F.normalize(torch.rand(kg, 3, generator=g)),
        F.normalize(torch.randn(kg, 4, generator=g)), torch.randint(0, kg, (n,), generator=g),
        torch.randn(n, 1, generator=g) - 3, quant.fake_quantizers("cpu"))
    return bank, params


def test_encode_decode_cpu():
    bank, p = _bank()
    with torch.no_grad():
        a = bank.attributes(p)
    data, order = codec.encode_npz(bank, p)
    dec = codec.decode_npz(data, "cpu")
    assert torch.allclose(dec["sh0"], a["colors"][order][:, :1], atol=1e-6)
    assert torch.allclose(dec["shN"], a["colors"][order][:, 1:], atol=1e-6)
    assert torch.allclose(dec["scales"].exp(), a["scales"][order], rtol=1e-5)
    assert torch.allclose(torch.sigmoid(dec["opacities"]), a["opacities"][order], atol=1e-5)
    assert torch.equal(dec["means"], p["means"][order].half().float())
    z = np.load(io.BytesIO(data))
    assert z["feature_indices"].dtype == np.int32 and z["xyz"].dtype == np.float16
    assert z["opacity"].shape == (300, 1) and z["features_rest"].shape == (12, 15, 3)
    rep = codec.size_report(data, 400, 300)
    assert rep["size_MB"] == len(data) / 2**20
    assert math.isclose(rep["3dgs_ply_MB_equivalent"], 400 * 248 / 2**20)
