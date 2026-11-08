import math

import numpy as np
import torch

from ss_omg import codec, core, indoor, medium, method, model


def test_methods_build_specs():
    for fn, metric, tau in ((method, "outdoor", 0.96), (indoor, "indoor", 0.96),
                            (medium, "outdoor", 0.99)):
        spec = fn()
        density = next(c for c in spec.components if c.kind == "density")
        assert density.config["imp_metric"] == metric
        assert density.config["importance_thresh"] == tau
        assert {f.name for f in spec.fields} == {"omg_static", "omg_view"}


def test_morton_matches_bit_loop():
    q = torch.randint(0, 2**21, (50, 3))
    got = core.morton_encode(q)
    for (x, y, z), c in zip(q.tolist(), got.tolist()):
        ref = 0
        for b in range(21):
            ref |= ((x >> b) & 1) << 3 * b | ((y >> b) & 1) << 3 * b + 1 | ((z >> b) & 1) << 3 * b + 2
        assert ref == c


def test_ld_residual_official_and_morton():
    t = torch.tensor([[0.0], [1.0], [3.0], [6.0]])
    order = torch.tensor([2, 0, 3, 1])  # rank: 0->1, 1->3, 2->0, 3->2
    off = core.ld_residual(t, order, "official")
    # official: neighbours of i are order[order[i] -/+ 1] (clamped)
    for i in range(4):
        left = order[max(order[i] - 1, 0)]
        right = order[min(order[i] + 1, 3)]
        assert math.isclose(off[i].item(), abs(t[left] - t[i]).item() + abs(t[right] - t[i]).item())
    mor = core.ld_residual(t, order, "morton")
    # Gaussian 0 sits between 2 and 3 in Morton order
    assert math.isclose(mor[0].item(), 3.0 + 6.0)


def test_frequency_encoding_layout():
    x = torch.tensor([[0.3, 0.6, 0.9]])
    enc = core.frequency_encoding(x, 4)
    assert enc.shape == (1, 24)
    assert math.isclose(enc[0, 0].item(), math.sin(0.3 * math.pi), abs_tol=1e-6)
    assert math.isclose(enc[0, 1].item(), math.cos(0.3 * math.pi), abs_tol=1e-6)
    assert math.isclose(enc[0, 2].item(), math.sin(0.3 * 2 * math.pi), abs_tol=1e-6)
    assert math.isclose(enc[0, 8].item(), math.sin(0.6 * math.pi), abs_tol=1e-6)


def test_contract_to_unisphere():
    x = torch.tensor([[0.5, 0.0, 0.0], [4.0, 0.0, 0.0]])
    y = core.contract_to_unisphere(x)
    assert torch.allclose(y[0], torch.tensor([0.625, 0.5, 0.5]))
    assert torch.allclose(y[1], torch.tensor([(2 - 0.25) / 4 + 0.5, 0.5, 0.5]))


def test_net_lr_closed_form():
    assert math.isclose(core.net_lr(0.01, 0), 1e-4)
    assert math.isclose(core.net_lr(0.01, 100), 0.01)
    assert math.isclose(core.net_lr(0.01, 1000), 0.0033)
    assert math.isclose(core.net_lr(0.01, 6000), 0.01 * 0.33**3)


def test_sh_degree():
    assert [core.sh_degree(s) for s in (14999, 15999, 16000, 17000, 18000, 29000)] == \
        [0, 0, 1, 2, 3, 3]


def test_kmeans_labels_are_nearest():
    g = torch.Generator().manual_seed(0)
    x = torch.cat([torch.randn(500, 2, generator=g) + 4, torch.randn(500, 2, generator=g) - 4])
    c, lab = core.kmeans(x, 8, 100, 1e-4, g)
    d = torch.cdist(x, c)
    assert torch.equal(lab, d.argmin(1))
    assert c.shape == (8, 2)


def test_fp16_uint16_roundtrip_and_order():
    x = torch.randn(1000, 3) * 10
    u = core.float16_to_uint16(x)
    assert int(u.min()) >= 0 and int(u.max()) <= 65535
    assert torch.equal(core.uint16_to_float16(u), x.half())
    o = core.position_order(u)
    key = (u - u.min(0).values).long()
    b = int(key.max()) + 1
    k = key[:, 0] + key[:, 1] * b + key[:, 2] * b * b
    assert bool((k[o][1:] >= k[o][:-1]).all())


def test_huffman_roundtrip():
    rng = np.random.default_rng(1)
    sym = rng.integers(0, 64, 5000) ** 2 % 512
    h = codec.huffman_encode(sym)
    assert np.array_equal(codec.huffman_decode(h), sym)
    assert h["official_bytes"] >= (h["bits"] + 7) // 8


def test_container_roundtrip():
    blob = codec.write_container({"a": b"xyz", "b": b"\x00" * 10}, {"n": 3})
    meta, s = codec.read_container(blob)
    assert meta == {"n": 3} and s["a"] == b"xyz" and s["b"] == b"\x00" * 10


def test_bank_layout():
    bank = model.OMGBank()
    assert bank.param_count() == 15360  # tcnn params: 7168 + 4096 + 2048 + 2048
    dc, shn, opa = bank.appearance(torch.randn(10, 3), torch.randn(10, 3), torch.zeros(10, 3))
    assert dc.shape == (10, 1, 3) and shn.shape == (10, 15, 3) and opa.shape == (10,)
