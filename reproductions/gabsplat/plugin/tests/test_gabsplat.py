"""CPU unit tests of the pure functions (frequency projection, dense reference, recipe)."""

import math

import torch

from ss_gabsplat import gabsplat, gabsplat_noreset
from ss_gabsplat.probes import frequency_projection, recipe
from ss_gabsplat.reference import dense_render


def test_frequency_projection_matches_paper_and_integration():
    out = frequency_projection(gabsplat(), "cpu")
    assert out["max_err_integration"] < 1e-4


def test_recipe():
    assert recipe(gabsplat(), "cpu")["lr_freq"] == 0.02


def test_noreset_ablation_config():
    d = next(c for c in gabsplat_noreset().components if c.kind == "density").config
    assert d["child_reset"] is False and d["weight_reset"] is False


def test_dense_zero_weight_is_gaussian():
    torch.manual_seed(0)
    n = 5
    m2 = torch.rand(n, 2, dtype=torch.float64) * 10
    con = torch.tensor([[0.3, 0.05, 0.2]], dtype=torch.float64).expand(n, 3)
    col = torch.rand(n, 3, dtype=torch.float64)
    op = torch.full((n,), 0.6, dtype=torch.float64)
    dep = torch.rand(n, dtype=torch.float64)
    gabs = torch.tensor([[0.2, 0.1, 0.0, -0.1, 0.3, 0.0]], dtype=torch.float64).expand(n, 6)
    a, _ = dense_render(m2, con, col, op, dep, 12, 10, gabs)
    b, _ = dense_render(m2, con, col, op, dep, 12, 10, None)
    assert torch.allclose(a, b)


def test_dense_modulation_value():
    """One primitive at a pixel centre offset: alpha = o G [(1-w) + w cos(2 pi f.d)]."""
    m2 = torch.tensor([[2.5, 0.5]], dtype=torch.float64)  # pixel (0,0) centre at (0.5,0.5)
    con = torch.tensor([[0.1, 0.0, 0.1]], dtype=torch.float64)
    gabs = torch.tensor([[0.1, 0.0, 0.4]], dtype=torch.float64)
    _, alpha = dense_render(m2, con, torch.ones(1, 3, dtype=torch.float64),
                            torch.tensor([0.5], dtype=torch.float64),
                            torch.zeros(1, dtype=torch.float64), 1, 1, gabs)
    d = 2.0
    expect = 0.5 * math.exp(-0.5 * 0.1 * d * d) * (0.6 + 0.4 * math.cos(2 * math.pi * 0.1 * d))
    assert abs(alpha.item() - expect) < 1e-12
