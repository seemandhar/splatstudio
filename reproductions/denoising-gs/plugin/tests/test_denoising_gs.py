import torch

from ss_denoising_gs import core, mcmc_control, method, probes, quick


def test_momentum_drift_lagged():
    m = torch.zeros(2, 3)
    g1, g2 = torch.ones(2, 3), 2 * torch.ones(2, 3)
    off, m = core.momentum_drift(m, g1, lr=1.0, alpha=0.05, beta1=0.9)
    assert torch.equal(off, torch.zeros(2, 3))
    off, m = core.momentum_drift(m, g2, lr=1.0, alpha=0.05, beta1=0.9)
    assert torch.allclose(off, torch.full((2, 3), -0.05 * 0.1))
    assert torch.allclose(m, torch.full((2, 3), 0.9 * 0.1 + 0.1 * 2))


def test_denoise_identity_rotation():
    q = torch.tensor([[1.0, 0, 0, 0]])
    w, al = core.denoise_term(torch.tensor([[0.0, 0.5, -0.1]]), torch.tensor([[0.0, -0.3, 0.1]]), q)
    assert bool(al[0]) and torch.allclose(w, torch.tensor([[0.0, -0.3, 0.0]]))
    w2, _ = core.denoise_term(torch.tensor([[0.0, 0.5, -0.1]]), torch.tensor([[0.0, -0.3, 0.1]]), q,
                              sign_mode="literal")
    assert torch.allclose(w2, torch.tensor([[0.0, 0.3, 0.0]]))


def test_prune_mask_and_sparsest():
    u = torch.tensor([5.0, 0.1, 3.0, 0.2, 9.0, 1.0, 2.0, 4.0, 6.0, 7.0])
    assert core.prune_mask(u, 0.2).nonzero().flatten().tolist() == [1, 3]
    pts = torch.tensor([[0.0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [5.0, 5, 5]])
    d = core.knn_sparsity(pts, 3)
    assert int(core.sparsest(d, 0.2)[0]) == 4


def test_schedule():
    assert core.schedule(35000) == (30000, [30500, 31000, 31500, 32000, 32500])


def test_cpu_probes():
    for m in (method(), quick()):
        probes.momentum_drift(m, "cpu")
        probes.spatial_denoise(m, "cpu")
        probes.schedule_probe(m, "cpu")
    probes.schedule_probe(mcmc_control(), "cpu")


def test_split_in_two_stable():
    o = torch.tensor([0.0, 1e-9, 1e-3, 0.5, 1.0])
    on, sn = core.split_in_two(o, torch.ones(5, 3))
    assert torch.isfinite(sn).all() and torch.isfinite(on).all()
    ref = 1 - (1 - o[3]).sqrt()
    assert torch.allclose(on[3], ref)
    assert torch.allclose(sn[3, 0], o[3] / (2 * ref - ref**2 / 2**0.5))
    assert torch.allclose(sn[1], torch.ones(3), atol=1e-6)  # o -> 0: scale unchanged
