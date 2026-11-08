import torch

from ss_det_gs import method, probes
from ss_det_gs.losses import edge_aware_depth_loss, edge_preserving_tv, hierarchical_depth_loss


def test_probes_cpu():
    m = method()
    for fn in (probes.depth_formula, probes.edge_formula, probes.tv_formula):
        fn(m, "cpu")


def test_depth_loss_gradient_finite():
    d = (torch.rand(1, 1, 40, 60) * 5 + 1).requires_grad_()
    t = torch.rand(1, 1, 40, 60)
    v = hierarchical_depth_loss(d, t, 5, 20, 0.1, 1.0, 0.2)
    v.backward()
    assert torch.isfinite(d.grad).all() and d.grad.abs().sum() > 0


def test_edge_and_tv_constant_inputs_zero():
    assert float(edge_aware_depth_loss(torch.ones(8, 8), torch.zeros(8, 8, dtype=torch.bool))) < 1e-12
    x = torch.full((8, 8, 3), 0.3)
    assert float(edge_preserving_tv(x, x)) == 0.0
