import math
from heapq import heapify, heappop, heappush

import numpy as np
import torch

from ss_compact_3dgs import codec, hashgrid, model
from ss_compact_3dgs.density import DEFAULTS, schedule_events
from ss_compact_3dgs.renderer import straight_through_mask


def test_level_table_matches_tcnn_and_paper_storage():
    scales, res, offsets = hashgrid.level_table()
    assert res.tolist()[0] == 16 and res.tolist()[-1] == 4085  # "16 to 4096"
    assert int(offsets[-1]) * 2 == 12_599_920
    assert abs(int(offsets[-1]) * 2 * 2 / 1e6 - 25.2) < 0.01  # Table tab:size, Hash(16f)


def test_reference_dense_level_is_trilinear():
    s, r, o = hashgrid.level_table(n_levels=1)
    params = torch.zeros(int(o[-1]) * 2)
    res = int(r[0])
    # feature 0 of the grid vertex (i, j, k) = i + 10 j + 100 k  -> linear field
    for k in range(res):
        for j in range(res):
            for i in range(res):
                params[(i + j * res + k * res * res) * 2] = i + 10 * j + 100 * k
    x = torch.tensor([[0.3, 0.4, 0.5]])
    pos = x * float(s[0]) + 0.5
    out = hashgrid.reference_encode(x, params, s, r, o)
    expected = pos[0, 0] + 10 * pos[0, 1] + 100 * pos[0, 2]
    assert torch.allclose(out[0, 0], expected, atol=1e-3)


def test_contraction_eq8():
    p = torch.tensor([[0.5, 0.0, 0.0], [4.0, 0.0, 0.0]])
    c = model.contract_to_unisphere(p)
    assert torch.allclose(c[0], torch.tensor([0.5 / 4 + 0.5, 0.5, 0.5]))
    assert torch.allclose(c[1], torch.tensor([(2 - 0.25) / 4 + 0.5, 0.5, 0.5]))


def test_sh_encoding_quirk_and_size():
    d = torch.nn.functional.normalize(torch.randn(10, 3), dim=-1)
    e = model.sh_encoding_tcnn(d, 3, quirk=True)
    assert e.shape == (10, 9)
    v = 2 * d - 1
    assert torch.allclose(e[:, 2], 0.48860251190291987 * v[:, 2])
    assert torch.allclose(model.sh_encoding_tcnn(d, 3, quirk=False)[:, 2],
                          0.48860251190291987 * d[:, 2])


def test_mlp_shapes_and_padding_with_ones():
    shapes = model.mlp_shapes(41)
    assert shapes == [(64, 48), (64, 64), (16, 64)]
    assert sum(a * b for a, b in shapes) == 8192  # 0.016 MB at fp16
    w = [torch.zeros(s) for s in shapes]
    w[0][0, 41:] = 1.0  # padded inputs act as a bias of 7
    w[1][0, 0] = 1.0
    w[2][0, 0] = 1.0
    out = model.mlp_forward(w, torch.zeros(2, 41))
    assert torch.allclose(out[:, 0], torch.full((2,), 7.0))


def test_straight_through_mask():
    m = torch.tensor([-10.0, 0.0, 2.0], requires_grad=True)
    v = straight_through_mask(m, 0.01)
    assert v.tolist() == [0.0, 1.0, 1.0]
    v.sum().backward()
    s = torch.sigmoid(m.detach())
    assert torch.allclose(m.grad, s * (1 - s))


def test_net_lr_closed_form():
    assert math.isclose(model.net_lr(0.01, 0), 1e-4)
    assert math.isclose(model.net_lr(0.01, 100), 0.01)
    assert math.isclose(model.net_lr(0.01, 5000), 0.0033)
    assert math.isclose(model.net_lr(0.01, 29999), 0.01 * 0.33**3)


def test_schedule():
    assert schedule_events(600, DEFAULTS) == ["mask_prune"]
    assert schedule_events(500, DEFAULTS) == []
    assert schedule_events(15000, DEFAULTS) == ["mask_prune"]
    assert schedule_events(15100, DEFAULTS) == []
    assert schedule_events(30000, DEFAULTS) == ["mask_prune", "encode"]


def test_rvq_cascade_and_ste():
    torch.manual_seed(0)
    vq = model.ResidualVQ(4, 16, 6)
    x = torch.randn(300, 4, requires_grad=True)
    q, idx = vq(x, train=True)
    assert idx.shape == (300, 6) and vq.initted
    (g,) = torch.autograd.grad(q.sum(), x)
    assert torch.allclose(g, torch.full_like(g, 6.0))
    qe, idx2 = vq(x.detach(), train=False)
    assert torch.allclose(vq.decode(idx2), qe)
    assert float((qe - x.detach()).square().mean()) < float(x.detach().square().mean())


def _optimal_cost(freqs):
    heap = list(freqs)
    heapify(heap)
    cost = 0
    while len(heap) > 1:
        a, b = heappop(heap), heappop(heap)
        cost += a + b
        heappush(heap, a + b)
    return cost


def test_huffman_lengths_optimal_and_eof():
    rng = np.random.default_rng(1)
    sym = rng.integers(0, 20, size=5000) ** 2 % 37
    lengths, counts, total = codec.huffman_lengths(sym)
    table = codec.dahuffman_table(counts)
    assert total + table[codec.EOF][0] == _optimal_cost(list(counts.values()) + [1])
    codes, order = codec.canonical_codes(lengths)
    payload, nbits = codec.pack_bits(sym, lengths, codes)
    assert nbits == total and len(payload) == (total + 7) // 8


def test_post_quant_and_paper_sizes():
    v = torch.tensor([-1.0, -0.05, 0.05, 1.0])
    out, q, mm = codec.post_quant(v, prune=True)
    assert q.tolist() == [0, 128, 128, 255]  # pruned values sit at round(0.5 * 255)
    assert torch.allclose(codec.dequant(q, mm), out)
    s = codec.paper_sizes(1_000_000, 12_599_920, 8192)
    assert math.isclose(s["position"], 6.0) and math.isclose(s["opacity"], 2.0)
    assert math.isclose(s["scale"], 4.5 + 64 * 6 * 3 * 4 / 1e6)
    assert math.isclose(s["hash"], 25.19984)


def test_index_packing_roundtrip():
    idx = np.arange(1000) % 64
    assert np.array_equal(codec.unpack_indices(codec.pack_indices(idx, 6), 1000, 6), idx)
    assert len(codec.pack_indices(idx, 6)) == 750
