import math

import pytest
import torch

from ss_efa_gs import method, method_base, probes
from ss_efa_gs.lfcf import (
    anneal_fraction,
    anneal_lower_bound,
    enlarge_delta,
    is_lfcf_step,
    lfcf_masks,
    quantile_linear,
    shrink_delta,
    split_probability,
    splitting_lower_bound,
)


def test_anneal_matches_official_formula():
    # official: di = 1 - (it-500)/14500; exp(log(di)*pow)*(1-lb)+lb
    for it in (600, 4000, 14800):
        di = 1 - (it - 500) / 14500
        assert math.isclose(anneal_fraction(it, 500, 15000, 2.0, 0.25),
                            math.exp(math.log(di) * 2.0) * 0.75 + 0.25)
    assert anneal_lower_bound(1.0, 1.5) == 0.0
    assert anneal_lower_bound(1.2, 1.0) == 1.0
    with pytest.raises(ValueError):
        anneal_fraction(15000)


def test_split_lower_bound_and_probability():
    assert splitting_lower_bound(500) == 1.0
    assert math.isclose(splitting_lower_bound(7750), 0.5)
    p = split_probability(torch.tensor([0.0, 1.0]), 0.3, 1.0)
    assert torch.allclose(p, torch.tensor([0.3, 1.0]))


def test_lfcf_step_schedule():
    assert is_lfcf_step(600) and is_lfcf_step(800) and not is_lfcf_step(700)
    assert is_lfcf_step(500, interval_times=5) and not is_lfcf_step(600, interval_times=5)


def test_quantile_matches_torch():
    g = torch.Generator().manual_seed(0)
    v = torch.rand(1001, generator=g)
    for q in (0.0, 0.13, 0.5, 0.97, 1.0):
        assert torch.isclose(quantile_linear(v, q), torch.quantile(v, q))


def test_masks_first_lfcf_step_enlarges_large_gradients():
    # Initially prev_sel = True and prev_g = 0: decent iff g <= tol.
    g = torch.tensor([3e-4, 3e-4, 1e-4])
    ga = torch.tensor([1e-3, 1e-3, 0.0])  # official OR: g_abs <= tol would count as decent
    sel, e, s, both = lfcf_masks(g, ga, 2e-4, 1.0e9, torch.ones(3, dtype=torch.bool),
                                 torch.zeros(3), torch.zeros(3), 1e-5)
    assert sel.tolist() == [True, True, False]
    assert e.tolist() == [True, True, False] and not s.any()


def test_masks_abs_branch():
    g = torch.tensor([3e-4, 3e-4])
    ga = torch.tensor([5.0, 0.5])
    prev = torch.tensor([4e-4, 4e-4])  # signed gradient decreased -> decent
    sel, e, s, _ = lfcf_masks(g, ga, 2e-4, 10.0, torch.ones(2, dtype=torch.bool), prev,
                              torch.tensor([1.0, 1.0]), 1e-5)
    assert s.tolist() == [True, True]
    prev = torch.tensor([1e-4, 1e-4])  # signed rose; abs rose for #0 only
    _, e, s, _ = lfcf_masks(g, ga, 2e-4, 10.0, torch.ones(2, dtype=torch.bool), prev,
                            torch.tensor([1.0, 1.0]), 1e-5)
    assert e.tolist() == [True, False] and s.tolist() == [False, True]


def test_deltas_without_diffscale():
    ls = torch.tensor([[-3.0, -2.0, -1.0]])
    lc = torch.tensor([[0.2]])
    assert torch.allclose(enlarge_delta(ls, lc, False), torch.full((1, 3), 0.2))
    assert torch.allclose(shrink_delta(ls, lc, 2.0, False),
                          torch.full((1, 3), -0.5 * (0.2 + math.log(2))))


@pytest.mark.parametrize("name", ["lfcf_decision", "depth_factor", "scale_strategy",
                                  "anneal_schedule", "filter3d", "gof_densify"])
def test_cpu_probes(name):
    getattr(probes, name)(method(), "cpu")


def test_variants_differ_only_in_lfcf():
    a = next(c for c in method().components if c.kind == "density").config
    b = next(c for c in method_base().components if c.kind == "density").config
    assert a["lfcf"] and not b["lfcf"]
    assert {k: v for k, v in a.items() if k != "lfcf"} == {k: v for k, v in b.items()
                                                           if k != "lfcf"}


def test_renderer_background_black_by_default():
    from ss_efa_gs.renderer import create_renderer

    spec = next(c for c in method().components if c.kind == "renderer")
    assert spec.config["background"] == [0.0, 0.0, 0.0]
    assert create_renderer(spec).background == [0.0, 0.0, 0.0]
