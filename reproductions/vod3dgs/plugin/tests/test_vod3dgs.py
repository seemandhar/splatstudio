import math

import torch

from ss_vod3dgs import method_l, method_s, probes, vod
from ss_vod3dgs.loss import match_covis


def test_quad_form_matches_matrix():
    S6 = torch.randn(50, 6, dtype=torch.float64)
    w = torch.nn.functional.normalize(torch.randn(50, 3, dtype=torch.float64), dim=-1)
    ref = torch.einsum("ni,nij,nj->n", w, vod.to_matrix(S6), w)
    assert torch.allclose(vod.quad_form(S6, w), ref)
    assert torch.allclose(vod.from_matrix(vod.to_matrix(S6)), S6)


def test_eigen_reset_variants_differ_and_rank_one():
    S6 = torch.randn(20, 6)
    s, l_ = vod.eigen_reset(S6, "S"), vod.eigen_reset(S6, "L")
    ev = torch.linalg.eigvalsh(vod.to_matrix(S6.double()))
    assert torch.allclose(vod.to_matrix(s.double()).diagonal(dim1=-2, dim2=-1).sum(-1), ev[:, 0], atol=1e-4)
    assert torch.allclose(vod.to_matrix(l_.double()).diagonal(dim1=-2, dim2=-1).sum(-1), ev[:, 2], atol=1e-4)


def test_max_view_logits():
    gamma = torch.zeros(4)
    S6 = torch.zeros(4, 6)
    S6[:, 0] = 5.0
    means = torch.zeros(4, 3)
    centers = torch.tensor([[0.0, 0, 1], [1.0, 0, 0]])
    assert torch.allclose(vod.max_view_logits(gamma, S6, means, centers), torch.full((4,), 5.0))


def test_view_consistency_zero_for_identical_views():
    gamma = torch.randn(10)
    S6 = torch.randn(10, 6)
    means = torch.randn(10, 3)
    c = torch.tensor([5.0, 0, 0])
    assert float(vod.view_consistency(gamma, S6, means, c, c)) == 0.0


def test_cpu_probes():
    for m in (method_s(), method_l()):
        probes.max_view_prune_probe(m, "cpu")
        probes.eigen_reset_probe(m, "cpu")
        probes.vc_loss_probe(m, "cpu")
        probes.opacity_lr_probe(m, "cpu")


def test_covis_matches_truck_cameras():
    d = __import__("numpy").load(__import__("ss_vod3dgs.covis", fromlist=["x"]).DATA_DIR / "truck.npz")
    centers = torch.from_numpy(d["centers"][::2].copy()).float()
    m, name = match_covis(centers, __import__("ss_vod3dgs.covis", fromlist=["x"]).DATA_DIR, 1e-3)
    assert name == "truck.npz" and m.shape == (len(centers), len(centers))
    assert math.isclose(float(m[0, 1]), float(d["matches"][0, 2]))
