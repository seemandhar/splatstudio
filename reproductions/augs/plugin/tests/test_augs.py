import math

import numpy as np
import torch

from ss_augs.density import DEFAULTS, allocate, lobe_t_init
from ss_augs.fit2d import TinyAdam, cov2d, loss_map, ssim_map
from ss_augs.inverse import (
    calibrate_scale,
    cluster_select,
    cluster_select_sparse,
    components_edges,
    connected_components,
    rotmat_to_quat,
    sparse_edges,
    weighted_pca,
)
from ss_augs.probes import inverse_probe, joint_probe, lobe_probe


def bfs_components(points, thr):
    n = len(points)
    lab = [-1] * n
    for s in range(n):
        if lab[s] >= 0:
            continue
        stack = [s]
        lab[s] = s
        while stack:
            i = stack.pop()
            for j in range(n):
                if lab[j] < 0 and np.linalg.norm(points[i] - points[j]) < thr:
                    lab[j] = s
                    stack.append(j)
    return np.array(lab)


def test_connected_components_match_single_linkage():
    rng = np.random.default_rng(0)
    pts = np.concatenate([rng.normal(0, 0.05, (30, 3)), rng.normal(1, 0.05, (25, 3)),
                          np.linspace([3, 0, 0], [4, 0, 0], 20)])
    thr = 0.08
    ref = bfs_components(pts, thr)
    t = torch.tensor(pts)[None]
    valid = torch.ones(1, len(pts), dtype=torch.bool)
    adj = (torch.cdist(t, t) < thr) & valid[:, :, None] & valid[:, None, :]
    lab = connected_components(adj, valid)[0].numpy()
    np.testing.assert_array_equal(lab, ref)
    lab2 = components_edges(len(pts), sparse_edges(torch.tensor(pts), thr)).numpy()
    np.testing.assert_array_equal(lab2, ref)


def test_cluster_select_dense_and_sparse_agree_and_pick_pivot_cluster():
    rng = np.random.default_rng(1)
    near = rng.normal([0, 0, 3], 0.01, (40, 3))
    far = rng.normal([0, 0, 9], 0.01, (60, 3))
    pts = torch.tensor(np.concatenate([near, far]))
    w = torch.rand(100, dtype=torch.float64) * 0.1
    w[5] = 0.9  # pivot in the near (smaller) cluster
    depth = pts[:, 2].float()
    sel_s, nd_s, _ = cluster_select_sparse(pts, w, depth, 0.05)
    sel_d, nd_d, _ = cluster_select(pts[None], w[None], depth[None],
                                    torch.ones(1, 100, dtype=torch.bool), torch.tensor([0.05]))
    assert bool(sel_s[:40].all()) and not bool(sel_s[40:].any())
    assert torch.equal(sel_s, sel_d[0])
    assert abs(float(nd_s) - float(depth[5])) < 1e-9 and abs(float(nd_d[0]) - float(depth[5])) < 1e-9


def test_cluster_threshold_grows_until_cluster_of_three():
    pts = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]], dtype=torch.float64)
    w = torch.tensor([0.5, 0.2, 0.1], dtype=torch.float64)
    sel, _, retries = cluster_select_sparse(pts, w, pts[:, 2], 0.5)
    assert retries == 2 and bool(sel.all())  # 0.5 -> 0.75 -> 1.125 > 1


def test_weighted_pca_matches_wpca_formula():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 3)) @ np.diag([3.0, 1.0, 0.1])
    w = rng.uniform(0.1, 1.0, 50)
    W = np.repeat(w[:, None], 3, 1)
    mean = (W * x).sum(0) / W.sum(0)
    X = (x - mean) * W
    cov = X.T @ X / (W.T @ W)
    ev, evec = np.linalg.eigh(cov)
    R, var = weighted_pca(torch.tensor(x)[None], torch.tensor(w)[None],
                          torch.ones(1, 50, dtype=torch.bool))
    np.testing.assert_allclose(var[0].numpy(), ev[::-1], rtol=1e-10)
    for k in range(3):
        assert abs(abs(float(np.dot(R[0, :, k].numpy(), evec[:, ::-1][:, k]))) - 1) < 1e-8


def test_rotmat_quat_roundtrip():
    from gsplat.utils import normalized_quat_to_rotmat

    q = torch.nn.functional.normalize(torch.randn(100, 4), dim=-1)
    R = normalized_quat_to_rotmat(q)
    q2 = rotmat_to_quat(R)
    torch.testing.assert_close(normalized_quat_to_rotmat(q2), R, atol=1e-5, rtol=1e-5)


def test_scale_calibration_minimises_frobenius():
    torch.manual_seed(0)
    from gsplat.utils import normalized_quat_to_rotmat

    R = normalized_quat_to_rotmat(torch.nn.functional.normalize(torch.randn(1, 4), dim=-1)).double()
    var = torch.tensor([[2.0, 0.7, 0.1]], dtype=torch.float64) * 1e-3
    pv = torch.tensor([[0.2, -0.1, 3.0]], dtype=torch.float64)
    K = torch.tensor([[300.0, 0, 200], [0, 300.0, 150], [0, 0, 1]])
    vm = torch.eye(4)
    target = torch.tensor([[9.0, 2.0, 5.0]], dtype=torch.float64)
    s, bad = calibrate_scale(pv, R, var, target, K, vm, 400, 300)
    assert not bool(bad.any())

    def frob(scale):
        J = torch.tensor([[300 / 3.0, 0, -300 * 0.2 / 9], [0, 300 / 3.0, 300 * 0.1 / 9]],
                         dtype=torch.float64)
        M = J @ R[0]
        c = M @ torch.diag(scale**2) @ M.T
        t = torch.tensor([[9.0, 2.0], [2.0, 5.0]], dtype=torch.float64)
        return float(((c - t) ** 2).sum())

    best = frob(s[0].double())
    for f in (0.97, 1.03):
        assert frob(s[0].double() * f) > best


def test_lobe_t_init_and_allocation():
    th = torch.tensor([0.0, 0.1, 0.3, math.pi / 2])
    T = lobe_t_init(th, DEFAULTS)
    torch.testing.assert_close(T, torch.clamp(th * 7 / math.pi, 0.01, 1.0))
    assert allocate([1.0, 3.0], 10.0) == [3, 8]


def test_ssim_map_same_padding_and_loss_map():
    x = torch.rand(3, 20, 24)
    assert ssim_map(x, x).shape == (3, 20, 24)
    torch.testing.assert_close(ssim_map(x, x), torch.ones(3, 20, 24))
    lm = loss_map(x, x)
    assert float(lm.abs().max()) < 1e-6 and lm.shape == (20, 24)


def test_cov2d_rotation():
    c = cov2d(torch.tensor([[2.0, 1.0]]), torch.tensor([math.pi / 2]))
    torch.testing.assert_close(c, torch.tensor([[1.0, 0.0, 4.0]]), atol=1e-6, rtol=0)


def test_tiny_adam_matches_torch_adam_and_keeps_step_on_cat():
    p = torch.randn(4, 2, requires_grad=True)
    q = torch.nn.Parameter(p.detach().clone())
    ref = torch.optim.Adam([q], lr=0.01, eps=1e-15)
    opt = TinyAdam({"x": 0.01})
    for _ in range(3):
        g = torch.randn(4, 2)
        p.grad, q.grad = g.clone(), g.clone()
        opt.step({"x": p})
        ref.step()
    torch.testing.assert_close(p.detach(), q.detach())
    opt.cat(2)
    assert opt.state["x"][0] == 3 and opt.state["x"][1].shape == (6, 2)


def test_cpu_probes():
    lobe_probe(None, "cpu")
    inverse_probe(None, "cpu")
    joint_probe(None, "cpu")


def test_edges_match_float64_reference_far_from_origin():
    from ss_augs.inverse import adjacency

    rng = np.random.default_rng(3)
    pts = np.array([120.0, -80.0, 300.0]) + rng.normal(0, 0.01, (300, 3))
    thr = 0.008
    ref = np.linalg.norm(pts[:, None] - pts[None], axis=-1) < thr
    t = torch.tensor(pts)
    valid = torch.ones(1, 300, dtype=torch.bool)
    adj = adjacency(t[None], valid, torch.tensor([thr], dtype=torch.float64))[0].numpy()
    np.testing.assert_array_equal(adj, ref)
    e = sparse_edges(t, thr, chunk=64).numpy()
    sp = np.zeros_like(ref)
    sp[e[:, 0], e[:, 1]] = sp[e[:, 1], e[:, 0]] = True
    np.fill_diagonal(sp, True)
    np.testing.assert_array_equal(sp, ref)
