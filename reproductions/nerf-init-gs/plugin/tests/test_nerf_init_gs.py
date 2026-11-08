import numpy as np
import torch

from ss_nerf_init_gs import method, nodepth, probes
from ss_nerf_init_gs.ingp import inverse_cdf, s_to_t, t_to_s
from ss_nerf_init_gs.losses import depth_weight
from ss_nerf_init_gs.preprocess import knn_log_scales, random_box_points
from ss_nerf_init_gs.sidecar import points_sha256


def test_cpu_probes():
    m = method()
    for fn in (probes.proposal_sampler, probes.surface_sampling, probes.depth_loss,
               probes.init_loading):
        fn(m, "cpu")


def test_variants():
    names = {c.name for c in nodepth().components}
    assert "depth-distillation" not in names
    assert "depth-distillation" in {c.name for c in method().components}


def test_schedule():
    assert abs(depth_weight(0) - 0.9) < 1e-12
    assert abs(depth_weight(200) - 0.9 * 0.81) < 1e-12
    assert depth_weight(5000) < 0.005


def test_lindisp_roundtrip():
    s = torch.rand(100)
    torch.testing.assert_close(t_to_s(s_to_t(s, 0.2, 1e3), 0.2, 1e3), s, rtol=1e-4, atol=1e-5)


def test_inverse_cdf_uniform():
    edges = torch.linspace(0, 1, 5)[None]
    cdf = torch.linspace(0, 1, 5)[None]
    u = torch.tensor([[0.1], [0.6]]).reshape(2, 1)
    out = inverse_cdf(edges.expand(2, -1), cdf.expand(2, -1), u)
    torch.testing.assert_close(out, u)


def test_knn_and_sha():
    pts = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3.0]])
    ls = knn_log_scales(pts, k=3, chunk=2)
    ref = np.log(np.sqrt((1 + 4 + 9) / 3))
    assert abs(float(ls[0]) - ref) < 1e-5
    assert points_sha256(pts) == points_sha256(pts.numpy())


def test_random_box():
    p, c = random_box_points(1000, 50.0, 0)
    assert p.shape == (1000, 3) and np.abs(p).max() <= 25
    assert np.all((c > 0.49) & (c < 0.502))
