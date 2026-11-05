import io

import numpy as np
import torch
from splatstudio.specs import ComponentSpec

from ss_rdo_gaussian import RATES, bank, codec, density, losses, method, rans, smoke


def _bits(n, seed=0):
    rng = np.random.default_rng(seed)
    return {k: rng.random(n) > p for k, p in zip(("sh1", "sh2", "sh3"), (0.3, 0.5, 0.7))}


def test_shmask_sort_boundaries_and_inverse():
    bits = _bits(500)
    order, b = codec.shmask_sort(bits)
    value = bits["sh1"] * 4 + bits["sh2"] * 2 + bits["sh3"]
    assert list(b) == [int((value <= j).sum()) for j in range(7)] + [500]
    assert np.all(np.diff(value[order]) >= 0)
    back = codec.sh_bits_from_boundaries(b)
    for k in bits:
        assert np.array_equal(back[k], bits[k][order])
    counts = codec.index_counts(b)
    for k in bits:
        assert counts[k] == int(bits[k].sum())


def test_shmask_sort_degenerate():
    for fill in (False, True):
        bits = {k: np.full(10, fill) for k in ("sh1", "sh2", "sh3")}
        _, b = codec.shmask_sort(bits)
        assert len(b) == 8 and b[-1] == 10


def test_opacity_quant():
    o = np.linspace(-3, 5, 1000, dtype=np.float32)[:, None]
    q, logp, step, lo = codec.opacity_quant(o)
    assert q.dtype == np.uint8 and q.min() == 0 and q.max() == 255
    assert np.abs(q * step + lo - o).max() <= step / 2 + 1e-6
    assert logp.shape == (1, 256)


def test_shrink():
    cb = np.arange(20, dtype=np.float32).reshape(1, 10, 2)
    lg = np.arange(10, dtype=np.float32)[None]
    idx = np.array([7, 2, 7, 9])
    c2, l2, i2 = codec.shrink(cb, lg, idx)
    assert np.array_equal(c2[0, i2], cb[0, idx]) and l2.shape == (1, 3)


def test_rans_roundtrip_cpu():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(1, 40, generator=g)
    sym = torch.multinomial(torch.softmax(logits, -1)[0], 3000, replacement=True, generator=g)
    s = rans.encode(sym, logits)
    assert torch.equal(rans.decode(s, len(sym), logits), sym)
    packed = rans.pack_strings([b"ab", b"", s])
    out, rest = rans.unpack_strings(packed, 3)
    assert out == [b"ab", b"", s] and rest == b""
    assert rans.encode(torch.zeros(0), logits) == b"" and len(rans.decode(b"", 0, logits)) == 0
    hdr = rans.pack_header(list(range(8)), 0.5, -1.25)
    assert len(hdr) == 40 and rans.unpack_header(hdr) == (list(range(8)), 0.5, -1.25)


def test_codec_roundtrip_cpu():
    rng = np.random.default_rng(1)
    n = 300
    bits = _bits(n, 2)
    attrs = {"xyz": rng.normal(size=(n, 3)).astype(np.float32),
             "opacity": rng.normal(size=(n, 1)).astype(np.float32),
             "sh_bits": bits, "index": {}, "codebook": {}, "logits": {}}
    for k in bank.KEYS:
        m = int(bits[k].sum()) if k in bank.SH_KEYS else n
        attrs["index"][k] = rng.integers(0, 16, m)
        attrs["codebook"][k] = rng.normal(size=(1, 16, bank.DIMS[k])).astype(np.float32)
        attrs["logits"][k] = rng.normal(size=(1, 16)).astype(np.float32)
    files, info = codec.encode(attrs)
    dec = codec.decode(files)
    order = info["order"]
    assert dec["n"] == n
    assert np.allclose(dec["xyz"], attrs["xyz"][order], atol=1e-2)
    for k in ("scale", "rot"):
        assert np.array_equal(dec[k], attrs["codebook"][k][0, attrs["index"][k][order]])
    sh1_rows = np.flatnonzero(bits["sh1"])
    full = np.zeros((n, 9), dtype=np.float32)
    full[sh1_rows] = attrs["codebook"]["sh1"][0, attrs["index"]["sh1"]]
    assert np.array_equal(dec["shN"][:, 0:3].reshape(n, 9), full[order])
    np.load(io.BytesIO(files["codebook.npz"]))
    rep = codec.size_report(files, info["index_bytes"])
    assert rep["total_bytes"] == sum(len(v) for v in files.values())
    assert rep["files_bytes"]["index_bitstream.bin"] == sum(info["index_bytes"].values())


def test_kmeans_cpu():
    torch.manual_seed(0)
    x = torch.cat([torch.randn(100, 2) * 0.05 + c for c in (torch.tensor([0., 0.]),
                                                              torch.tensor([5., 5.]))])
    means, bins = bank.kmeans(x, 2, 10, 64)
    assert sorted(bins.tolist()) == [100, 100]
    assert torch.allclose(means.sort(0).values, torch.tensor([[0., 0.], [5., 5.]]), atol=0.05)


def test_schedule_events():
    cfg = density.merged(density.DEFAULTS, {})
    ev = {s: density.schedule_events(s, cfg) for s in range(1, 30001)}
    hits = {}
    for s, es in ev.items():
        for e in es:
            hits.setdefault(e, []).append(s)
    assert hits["vq_on"] == [15000] and hits["rate_on"] == [20000]
    assert hits["reactivate"] == [16000, 17000, 18000, 19000, 20000]
    assert hits["encode"] == [30000]


def test_merged_config_and_methods():
    cfg = density.merged(density.DEFAULTS, {"lmbda": {"scale": 1.0}})
    assert cfg["lmbda"]["rot"] == 256.0 and cfg["lmbda"]["scale"] == 1.0
    m = method()
    w = {c.name: c.config.get("weight") for c in m.components if c.kind == "loss"}
    assert w["rdo-gs-prune"] == RATES[1][0] and w["rdo-sh-prune"] == RATES[1][1]
    s = smoke()
    d = next(c for c in s.components if c.kind == "density")
    assert d.config["total_steps"] == 2000 and d.config["base"]["refine_stop_iter"] == 1000


def test_losses_zero_before_activation():
    spec = ComponentSpec("rdo-gs-prune", "loss", "x:y", (), (), {"weight": 0.5})
    fn = losses.create_gs_prune(spec)
    rgb = torch.ones(2, 2, 3, requires_grad=True)
    v = fn(None, {"rgb": rgb}, None)
    assert float(v) == 0 and v.requires_grad
    t = torch.tensor(2.0, requires_grad=True)
    assert float(fn(None, {"rgb": rgb, "rdo": {"gs_prune": t}}, None)) == 1.0


def test_codec_empty_sh_degree():
    rng = np.random.default_rng(3)
    n = 50
    bits = {"sh1": rng.random(n) > 0.5, "sh2": np.zeros(n, bool), "sh3": np.zeros(n, bool)}
    attrs = {"xyz": rng.normal(size=(n, 3)).astype(np.float32),
             "opacity": rng.normal(size=(n, 1)).astype(np.float32),
             "sh_bits": bits, "index": {}, "codebook": {}, "logits": {}}
    for k in bank.KEYS:
        m = int(bits[k].sum()) if k in bank.SH_KEYS else n
        attrs["index"][k] = rng.integers(0, 8, m)
        attrs["codebook"][k] = rng.normal(size=(1, 8, bank.DIMS[k])).astype(np.float32)
        attrs["logits"][k] = rng.normal(size=(1, 8)).astype(np.float32)
    files, info = codec.encode(attrs)
    dec = codec.decode(files)
    assert np.all(dec["shN"][:, 3:] == 0) and len(dec["index"]["sh3"]) == 0
