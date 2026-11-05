import math

import torch
from splatstudio.contracts import validate

import ss_trimgs
from ss_trimgs import core, probes


def test_specs_valid():
    for m in (ss_trimgs.method(), ss_trimgs.indoor()):
        assert validate(m) == []
    w = {m.name: next(c for c in m.components if c.name == "trim-normal").config["weight"]
         for m in (ss_trimgs.method(), ss_trimgs.indoor())}
    assert w == {"trimgs": 0.1, "trimgs-indoor": 0.01}


def test_flat_schedule_probe_cpu():
    probes.flat_schedule(ss_trimgs.method(), "cpu")


def test_official_radii_matches_3dgs_formula():
    cov = torch.tensor([[4.0, 1.0], [1.0, 2.0]])
    inv = torch.linalg.inv(cov)
    conic = torch.tensor([[inv[0, 0], inv[0, 1], inv[1, 1]]])
    det = torch.det(cov)
    mid = 0.5 * (cov[0, 0] + cov[1, 1])
    lam = max(mid + math.sqrt(max(0.1, mid * mid - det)), mid - math.sqrt(max(0.1, mid * mid - det)))
    assert core.official_radii(conic).item() == math.ceil(3 * math.sqrt(lam))


def test_culling_box_uses_x_span_for_z():
    centers = torch.tensor([[0.0, 0, 0], [4.0, 1.0, 0.0]])
    # x span 8 (after x2), y span 2, z span 0 -> z bounds use the x span (+-4 around 0)
    pts = torch.tensor([[2.0, 0.5, 3.9], [2.0, 0.5, 4.1], [2.0, 1.6, 0.0], [-2.1, 0.5, 0.0]])
    assert core.culling_mask(pts, centers, 2.0).tolist() == [True, False, False, False]


def test_split_n_cpu():
    params, opts = probes._params(torch.zeros(2, 3), torch.tensor([[1e-8, 0.6, 0.3]] * 2),
                                  torch.tensor([0.5, 0.5]), None, "cpu")
    core.pin_flat_axis(params, 1e-8)
    n = core.split_n(params, opts, torch.tensor([True, False]), 3, 1e-8)
    assert n == 1 and len(params["means"]) == 4
    s = params["scales"][1:].exp()
    assert torch.allclose(s[:, 1:], torch.tensor([[0.25, 0.125]] * 3))
    assert params["means"][1:, 0].abs().max() < 1e-6  # no offset along the flat axis


def test_depth_normals_tilted_plane():
    H = W = 32
    K = torch.tensor([[30.0, 0, 16], [0, 30.0, 16], [0, 0, 1]])
    v, u = torch.meshgrid(torch.arange(H) + 0.5, torch.arange(W) + 0.5, indexing="ij")
    # plane z = 3 + 0.5 y  -> along a ray z = 3 / (1 - 0.5 (v - cy)/fy)
    ry = (v - 16) / 30
    depth = 3.0 / (1 - 0.5 * ry)
    pts = core.backproject(depth, K, torch.eye(4))
    n = core.depth_normals(pts, torch.zeros(3), 2)[8:24, 8:24].reshape(-1, 3)
    want = torch.nn.functional.normalize(torch.tensor([0.0, 0.5, -1.0]), dim=0)
    assert (n - want).abs().max() < 1e-4


def test_sobel_mask_masks_edges():
    d = torch.ones(16, 16)
    d[:, 8:] = 2.0
    m = core.sobel_mask(d, 0.05, 1)
    assert m[:, 6:10].sum() == 0 and m[:, :6].all() and m[:, 10:].all()


def test_trim_mask_quantile():
    c = torch.arange(20.0)
    assert core.trim_mask(c, 0.1).sum() == 2  # quantile 1.9 -> values 0, 1


def test_smooth_loss_zero_for_aligned():
    n = torch.tensor([[0.0, 0, 1]]).repeat(5, 1)
    n[2] = -n[2]  # sign-aligned neighbours do not count as disagreement
    knn = torch.arange(5)[None].repeat(5, 1)
    assert core.smooth_loss(n, torch.tensor([0, 1, 3]), knn).item() < 1e-6
    n2 = n.clone()
    n2[4] = torch.tensor([0.6, 0, 0.8])
    assert core.smooth_loss(n2, torch.tensor([0]), knn).item() > 1e-3


def test_knn_includes_self():
    pts = torch.rand(50, 3)
    idx = core.knn_indices(pts, 4)
    assert torch.equal(idx[:, 0], torch.arange(50))
