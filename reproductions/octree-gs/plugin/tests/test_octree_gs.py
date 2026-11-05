"""CPU unit tests of the pure Octree-GS functions (octree, LOD selection, schedules, blob, decode)."""

import math

import numpy as np
import pytest
import torch

from ss_octree_gs import model, octree, probes
from ss_octree_gs.density import DEFAULTS, membership, merge, scatter_max
from ss_octree_gs.renderer import BLOB_HEADER, blob_floats, pack_blob_flat, unpack_blob, write_blob


def test_set_level_matches_numpy():
    g = np.random.default_rng(0)
    pts = g.normal(0, 3, (2000, 3)).astype(np.float32)
    cams = g.normal(0, 2, (7, 3)).astype(np.float32)
    std, K, init = octree.set_level(torch.tensor(pts), torch.tensor(cams))
    pairs = []
    for c in cams:
        d = np.sqrt(((pts.astype(np.float64) - c) ** 2).sum(1))
        pairs += [np.quantile(d, 0.001), np.quantile(d, 0.999)]
    dmax, dmin = np.quantile(pairs, 0.999), np.quantile(pairs, 0.001)
    assert K == int(np.round(np.log2(dmax / dmin))) + 1
    assert abs(std - dmax) < 1e-4 * dmax
    assert init == K // 2


def test_octree_sample_brute_force():
    g = torch.Generator().manual_seed(0)
    pts = torch.rand(500, 3, generator=g) * 4 - 2
    voxel, init_pos = octree.octree_box(pts, 1.1, 2, 3)
    pos, lev = octree.octree_sample(pts, init_pos, voxel, 3, 2)
    for L in range(3):
        size = voxel / 2**L
        ref = torch.unique(torch.round((pts - init_pos) / size), dim=0)
        got = torch.round((pos[lev == L] - init_pos) / size)
        assert len(ref) == len(got)
        assert torch.equal(torch.unique(got, dim=0), ref)


def test_coarse_intervals_and_index():
    iv = octree.coarse_intervals(6, 3, 10000, 1.5)
    assert len(iv) == 2 and abs(iv[-1] - 10000) < 1e-6
    assert abs(iv[0] / (iv[1] - iv[0]) - 1.5) < 1e-9
    assert octree.coarse_index(iv, 1, 3, 6) == 4
    assert octree.coarse_index(iv, 10001, 3, 6) == 6
    assert octree.coarse_index(iv, 5, 3, 6, training=False) == 6
    assert octree.coarse_intervals(3, 2) == []
    assert octree.coarse_index([], 1, 2, 3) == 3


def test_anchor_mask_matches_loop():
    probes.lod_selection(None, "cpu")


def test_weed_out_view_frequency():
    pos = torch.tensor([[0.0, 0, 1], [0.0, 0, 100]])
    lev = torch.tensor([1.0, 1.0])
    cams = torch.zeros(4, 3)
    keep, mean = octree.weed_out(pos, lev, cams, 4.0, 3, 0.5)
    # d=1: round(log2 4) = 2 >= 1 -> selected; d=100: negative -> level 0 only
    assert keep.tolist() == [True, False] and mean == 0.5


def test_membership_and_scatter_max():
    table = torch.tensor([[0, 0, 0], [1, 2, 3]])
    q = torch.tensor([[1, 2, 3], [4, 4, 4]])
    assert membership(q, table).tolist() == [True, False]
    src = torch.tensor([[1.0, 5.0], [3.0, 2.0], [0.0, 0.0]])
    out = scatter_max(src, torch.tensor([0, 0, 1]), 2)
    assert out.tolist() == [[3.0, 5.0], [0.0, 0.0]]


def test_blob_roundtrip():
    torch.manual_seed(0)
    mlps = model.scaffold_mlps(32, 10)
    oct_ = {"levels": 6, "fork": 2, "voxel_size": 0.1234, "standard_dist": 17.5, "init_pos": -3.3, "init_level": 3}
    need = blob_floats(32, 10)
    blob = torch.zeros(math.ceil(need / 8) + 5, 8)
    write_blob(blob, pack_blob_flat(mlps, oct_, 32, 10, 77, "cpu"))
    got, o2 = unpack_blob(blob, 32, 10)
    for k in model.RENDER_MLPS:
        for a, b in zip(got[k].parameters(), mlps[k].parameters()):
            assert torch.equal(a, b)
    assert o2["levels"] == 6 and o2["init_level"] == 3 and abs(o2["standard_dist"] - 17.5) < 1e-6
    blob.view(-1)[BLOB_HEADER + 3] += 1.0
    with pytest.raises(ValueError):
        unpack_blob(blob, 32, 10)


def test_neural_gaussians_cpu():
    probes.neural_gaussians(None, "cpu")


def test_lr_schedule_and_config():
    f = model.expon_lr(0.008, 0.00005, 40000)
    assert abs(f(0) - 0.008) < 1e-12 and abs(f(40000) - 0.00005) < 1e-12
    assert abs(f(20000) - math.sqrt(0.008 * 0.00005)) < 1e-9
    c = merge(DEFAULTS, {"lr": {"feat": 0.1}})
    assert c["lr"]["feat"] == 0.1 and c["lr"]["scaling"] == 0.007
    with pytest.raises(ValueError):
        merge(DEFAULTS, {"dist2level": "floor"})
