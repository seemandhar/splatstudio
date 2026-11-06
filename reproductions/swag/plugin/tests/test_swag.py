import math

import torch

from ss_swag import method, method_a, method_t
from ss_swag.bank import robust_aabb
from ss_swag.density import position_lr
from ss_swag.hashgrid import HashGrid, level_resolutions
from ss_swag.probes import concrete_opacity, hash_grid
from ss_swag.renderer import concrete, image_opacity


def test_levels():
    r = level_resolutions(12, 16, 2048)
    assert r[0] == 16 and r[-1] in (2047, 2048) and len(r) == 12


def test_hash_probe():
    assert hash_grid(method(), "cpu")["max_abs_error"] < 1e-3


def test_concrete_probe():
    concrete_opacity(method(), "cpu")


def test_concrete_eval_closed_form():
    d = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    got = concrete(d, torch.full_like(d, 0.5), 0.1, 0.0)
    assert torch.allclose(got, d**10 / (1 + d**10))


def test_image_opacity_clamps():
    assert float(image_opacity(torch.tensor(0.2), torch.tensor(0.9))) == 0.0


def test_position_lr():
    assert math.isclose(position_lr(2.0, 30000, 30000, 0.01), 0.02)
    assert math.isclose(position_lr(2.0, 45000, 30000, 0.01), 0.02)


def test_hashgrid_grad_and_shape():
    g = HashGrid(log2_table=12)
    x = torch.rand(10, 3)
    y = g(x)
    assert y.shape == (10, 24)
    y.sum().backward()
    assert g.table.grad is not None and float(g.table.grad.abs().sum()) > 0


def test_aabb():
    p = torch.randn(1000, 3)
    box = robust_aabb(p, 0.005, 0.05)
    assert (box[0] < box[1]).all()


def test_variants():
    a = {c.name: c.config for c in method_a().components}
    t = {c.name: c.config for c in method_t().components}
    assert a["swag-renderer"]["transient"] is False and a["swag-renderer"]["appearance"]
    assert t["swag-renderer"]["appearance"] is False and t["swag-renderer"]["transient"]
    d = {c.name: c.config for c in method().components}["swag-density"]
    assert d["refine_every"] == 2000 and d["refine_stop_iter"] == 30000
    assert d["bank_embedding_dim"] == 24 and d["bank_hash_log2_table"] == 19
