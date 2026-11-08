import numpy as np
import torch

from ss_corgs import method, method_single
from ss_corgs.density import knn, nearest_distance, nonmatching_masks
from ss_corgs.losses import coreg_active, coreg_terms, photometric
from ss_corgs.probes import co_pruning, corgs_schedule


def test_nearest_distance_bruteforce():
    g = torch.Generator().manual_seed(0)
    a, b = torch.rand(57, 3, generator=g) * 10, torch.rand(31, 3, generator=g) * 10
    d = nearest_distance(a, b, budget=40)  # forces chunking
    ref = torch.cdist(a.double(), b.double()).min(1).values
    assert torch.allclose(d.double(), ref, atol=1e-5)


def test_nonmatching_masks_threshold():
    a = torch.tensor([[0.0, 0, 0], [10.0, 0, 0], [0, 0, 4.9]])
    b = torch.tensor([[0.0, 0, 5.0], [0.0, 0, 30.0]])
    ma, mb = nonmatching_masks(a, b, 5.0)
    assert ma.tolist() == [False, True, False]  # 5.0 is not > tau
    assert mb.tolist() == [False, True]


def test_knn_matches_bruteforce():
    g = torch.Generator().manual_seed(1)
    pts = torch.rand(100, 3, generator=g)
    q = torch.tensor([0, 7, 99])
    dist, _ = knn(q, pts, 3, chunk=2)
    d2 = ((pts[q][:, None] - pts[None]) ** 2).sum(-1)
    d2[torch.arange(3), q] = float("inf")
    assert torch.allclose(dist, d2.topk(3, largest=False).values.mean(1), atol=1e-6)


def test_coreg_gate_and_detach():
    assert not coreg_active(499, 500, 10000, 1)
    assert coreg_active(500, 500, 10000, 1) and coreg_active(10000, 500, 10000, 1)
    assert not coreg_active(10001, 500, 10000, 1)
    x = torch.rand(16, 16, 3, requires_grad=True)
    y = torch.rand(16, 16, 3, requires_grad=True)
    r1, r2 = coreg_terms(x, y)
    r1.backward()
    assert y.grad is None and x.grad.abs().sum() > 0
    assert torch.isclose(r1, r2)
    assert torch.isclose(r1, photometric(x, y.detach()))


def test_cpu_probes():
    co_pruning(method(), "cpu")
    corgs_schedule(method(), "cpu")
    corgs_schedule(method_single(), "cpu")


def test_specs():
    m = method()
    names = [c.name for c in m.components]
    assert names == ["corgs-renderer", "corgs-density", "photometric", "corgs-field2",
                     "corgs-coreg"]
    assert next(c for c in m.components if c.name == "corgs-density").config["n_fields"] == 2
    s = method_single()
    assert [c.name for c in s.components] == ["corgs-renderer", "corgs-density", "photometric"]
    assert np.isclose(next(c for c in m.components if c.name == "corgs-coreg").config["weight"],
                      1.0)
