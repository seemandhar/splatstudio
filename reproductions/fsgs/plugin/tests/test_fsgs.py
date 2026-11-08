import numpy as np
import torch

from ss_fsgs import method
from ss_fsgs.density import knn, unpool_values
from ss_fsgs.losses import depth_weight, pearson, pseudo_scale, train_depth_loss
from ss_fsgs.probes import fsgs_densify, pearson_depth, proximity_unpooling, sh_schedule
from ss_fsgs.pseudo import random_poses_llff


def test_knn_matches_bruteforce():
    g = torch.Generator().manual_seed(0)
    pts = torch.rand(200, 3, generator=g)
    q = torch.tensor([0, 5, 199])
    dist, idx = knn(q, pts, 3, chunk=2)
    d2 = ((pts[q][:, None] - pts[None]) ** 2).sum(-1)
    d2[torch.arange(3), q] = float("inf")
    ref = d2.topk(3, largest=False)
    assert torch.allclose(dist, ref.values.mean(1), atol=1e-5)
    assert torch.equal(idx.sort(1).values, ref.indices.sort(1).values)


def test_unpool_values_order():
    p = {"means": torch.tensor([[0.0, 0, 0], [2.0, 0, 0], [0, 4.0, 0]]),
         "scales": torch.tensor([[1.0] * 3, [2.0] * 3, [3.0] * 3]),
         "opacities": torch.tensor([0.1, 0.2, 0.3]),
         "quats": torch.randn(3, 4), "sh0": torch.randn(3, 1, 3), "shN": torch.randn(3, 15, 3)}
    new = unpool_values(p, torch.tensor([0]), torch.tensor([[1, 2]]))
    assert torch.allclose(new["means"], torch.tensor([[1.0, 0, 0], [0, 2.0, 0]]))
    assert torch.equal(new["opacities"], torch.tensor([0.2, 0.3]))
    assert torch.equal(new["quats"][:, 0], torch.ones(2)) and new["sh0"].abs().sum() == 0


def test_pearson_and_schedules():
    x = torch.rand(100, dtype=torch.float64)
    assert abs(float(pearson(x, -2 * x + 1)) + 1) < 1e-9
    d = torch.rand(10, 10, dtype=torch.float64) + 1
    assert float(train_depth_loss(-d, d)) < 1e-9  # rendered = -DPT is perfectly correlated
    assert depth_weight(9501, 0.05, 0.001, 9500) == 0.05 and depth_weight(9502, 0.05, 0.001, 9500) == 0.001
    assert pseudo_scale(2000, 2000, 9500, 1, 500) == 0 and pseudo_scale(2500, 2000, 9500, 1, 500) == 1
    assert pseudo_scale(2010, 2000, 9500, 10, 500) == 0.02 and pseudo_scale(2011, 2000, 9500, 10, 500) == 0


def test_random_poses_center_box():
    vm = []
    for x in (-1.0, 0.0, 1.0):
        v = np.eye(4)
        v[0, 3] = -x
        vm.append(v)
    poses, _info = random_poses_llff(vm, np.array([[2.0, 20.0]]), 50, np.random.default_rng(0))
    c = np.stack([np.linalg.inv(p)[:3, 3] for p in poses])
    assert np.abs(c[:, 0]).max() <= 1 + 1e-9 and np.abs(c[:, 1:]).max() < 1e-9


def test_cpu_probes():
    m = method()
    for probe in (proximity_unpooling, fsgs_densify, pearson_depth, sh_schedule):
        probe(m, "cpu")
