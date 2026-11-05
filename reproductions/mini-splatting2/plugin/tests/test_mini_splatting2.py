import math

import torch

from ss_mini_splatting2 import core, dense, dense_indoor, indoor, method
from ss_mini_splatting2.density import DEFAULTS
from ss_mini_splatting2.optim import SparseGaussianAdam


def _dens(m):
    return next(c for c in m.components if c.kind == "density").config


def test_methods_build_specs():
    assert _dens(method())["imp_metric"] == "outdoor" and _dens(method())["simplify"] == "sampling"
    assert _dens(indoor())["imp_metric"] == "indoor"
    assert _dens(dense())["simplify"] == "importance"
    assert _dens(dense_indoor())["imp_metric"] == "indoor"
    loss = next(c for c in method().components if c.kind == "loss")
    assert loss.name == "photometric" and loss.factory.startswith("ss_mini_splatting2")


def test_events_match_config_fast():
    ev = {}
    for s in range(1, 18001):
        for e in core.events(s, DEFAULTS):
            ev.setdefault(e, []).append(s)
    assert ev["culling_with_clone"] == [500, 750, 1000, 1250, 1500, 1750, 2250, 2500, 2750]
    assert ev["depth_reinit"] == [2000] and 2000 not in ev["densify"]
    assert ev["densify"][0] == 600 and ev["densify"][-1] == 2900
    assert ev["simplify1"] == [3000] and ev["simplify2"] == [8000]
    assert ev["culling_off"] == [13000]


def test_lr_and_sh():
    assert math.isclose(core.expon_lr(18000, 1.0), 0.01**0.6)
    assert core.sh_degree(3000) == 0 and core.sh_degree(3999) == 0
    assert core.sh_degree(4000) == 1 and core.sh_degree(6000) == 3 and core.sh_degree(17000) == 3


def test_clone_matches_mcmc():
    a = torch.tensor([0.05, 0.3, 0.7, 0.99], dtype=torch.float64)
    s = torch.ones(4, 3, dtype=torch.float64)
    a_new, s_new = core.clone_values(a, s, min_opacity=0.0)
    a_ref, ratio = core.mcmc_reference(a, 2)
    assert torch.allclose(a_new, a_ref) and torch.allclose(s_new[:, 0], ratio)
    # two clones of alpha_new composite to alpha_old
    assert torch.allclose(1 - (1 - a_new) ** 2, a)


def test_clone_opacity_clamp_after_scale():
    a = torch.tensor([0.005])
    a_new, s_new = core.clone_values(a, torch.ones(1, 3))
    assert math.isclose(float(a_new), 0.0051, rel_tol=1e-6)
    raw = 1 - (1 - a) ** 0.5
    assert torch.allclose(s_new[:, 0], a / (2 * raw - 0.5**0.5 * raw**2))


def test_cdf_keep_mask_drops_zero_and_tail():
    imp = torch.tensor([0.0, 0.001, 1.0, 2.0, 5.0])
    keep = core.cdf_keep_mask(imp, 0.99)
    # official init_cdf_mask keeps values strictly above the crossing value (here 1.0)
    assert keep.tolist() == [False, False, False, True, True]


def test_sparse_adam_cpu_matches_adam_on_visible_rows():
    g = torch.Generator().manual_seed(0)
    p0 = torch.randn(6, 2, generator=g)
    vis = torch.tensor([True, True, False, True, False, False])
    p = torch.nn.Parameter(p0.clone())
    q = torch.nn.Parameter(p0[vis].clone())
    sp = SparseGaussianAdam([p], lr=0.1)
    ad = torch.optim.Adam([q], lr=0.1, eps=1e-15)
    for _ in range(4):
        grad = torch.randn(6, 2, generator=g)
        p.grad, q.grad = grad.clone(), grad[vis].clone()
        sp.visibility = vis
        sp.step()
        ad.step()
    assert torch.allclose(p.detach()[vis], q.detach(), atol=1e-6)
    assert torch.equal(p.detach()[~vis], p0[~vis])


def test_half_res_intrinsics():
    K = torch.tensor([[100.0, 0, 64.5], [0, 90.0, 40.0], [0, 0, 1]])
    w2, h2 = core.half_res_size(129, 80)
    assert (w2, h2) == (64, 40)
    K2 = core.scale_intrinsics(K, 129, 80, w2, h2)
    assert math.isclose(float(K2[0, 0]), 100 * 64 / 129, rel_tol=1e-6) and math.isclose(float(K2[1, 2]), 20.0)


def test_weighted_sample_without_replacement():
    g = torch.Generator().manual_seed(0)
    w = torch.tensor([0.0, 1.0, 2.0, 7.0])
    idx = core.weighted_sample(w, 3, generator=g)
    assert sorted(idx.tolist()) == [1, 2, 3]
    assert core.sampling_count(torch.tensor([0.0, 1, 1, 1, 1]), 0.6) == int(5 * 0.6 * 4 / 5)
