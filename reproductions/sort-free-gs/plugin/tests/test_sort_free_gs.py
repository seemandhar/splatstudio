import math

import torch

from ss_sort_free_gs.wsr import (
    C0,
    compose,
    depth_weight,
    eval_sh_scalar,
    expon_lr,
    view_dependent_opacity,
)


def test_sh_degree0_and_offset():
    dirs = torch.randn(5, 3)
    h0 = torch.rand(5, 1)
    u = view_dependent_opacity(0, h0, torch.zeros(5, 15), dirs, 0.1)
    torch.testing.assert_close(u, 0.1 + C0 * h0[:, 0])


def test_sh_degree1_direction_dependence():
    c = torch.zeros(2, 16)
    c[:, 2] = 1.0  # z basis function
    d = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, -1.0]])
    out = eval_sh_scalar(1, c, d)
    torch.testing.assert_close(out, torch.tensor([0.4886025119029199, -0.4886025119029199]))


def test_sh_orthonormal_monte_carlo():
    # real SH basis is orthonormal on the sphere: E[Y_a Y_b] * 4 pi = delta_ab
    torch.manual_seed(0)
    d = torch.nn.functional.normalize(torch.randn(400000, 3), dim=-1)
    basis = torch.stack([eval_sh_scalar(3, torch.eye(16)[k].expand(len(d), 16), d)
                         for k in range(16)], dim=-1)
    gram = basis.T @ basis / len(d) * 4 * math.pi
    torch.testing.assert_close(gram, torch.eye(16), atol=0.03, rtol=0)


def test_lc_weight():
    d = torch.tensor([1.0, 5.0, 10.0, 12.0])
    v = torch.full((4,), 0.1)
    torch.testing.assert_close(depth_weight("lc", d, v, torch.tensor(10.0)),
                               torch.tensor([0.09, 0.05, 0.0, 0.0]))
    torch.testing.assert_close(depth_weight("dir", d, v, torch.tensor(10.0)), v)


def test_compose_eq7():
    feat = torch.tensor([[[0.2, 0.4, 0.6]]])
    ws = torch.tensor([[[0.5]]])
    rgb, a = compose(feat, ws, torch.tensor(0.5), torch.ones(3))
    torch.testing.assert_close(rgb, torch.tensor([[[0.7, 0.9, 1.1]]]))
    torch.testing.assert_close(a, torch.tensor([[[0.5]]]))
    rgb, _ = compose(torch.zeros(1, 1, 3), torch.zeros(1, 1, 1), torch.tensor(0.0), torch.ones(3))
    assert torch.isfinite(rgb).all()


def test_expon_lr():
    assert math.isclose(expon_lr(0, 0.1, 0.001, 30000), 0.1)
    assert math.isclose(expon_lr(30000, 0.1, 0.001, 30000), 0.001)
    assert math.isclose(expon_lr(15000, 0.1, 0.001, 30000), 0.01)
    assert math.isclose(expon_lr(60000, 0.1, 0.001, 30000), 0.001)
