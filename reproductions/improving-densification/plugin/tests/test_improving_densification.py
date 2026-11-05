import math

import pytest
import torch

from ss_improving_densification import core, method, probes


def test_budget_curve():
    assert core.budget(500, 1000) == 1  # max(int(0), 1)
    assert core.budget(14500, 1000) == 1000
    assert core.budget(3999, 10**6) == int(math.sqrt(3499 / 14000) * 10**6)


def test_expon_lr_endpoints():
    assert core.expon_lr(0, 4e-5, 2e-6) == pytest.approx(4e-5)
    assert core.expon_lr(30000, 4e-5, 2e-6) == pytest.approx(2e-6)


def test_minmax_flat_and_nan():
    assert torch.count_nonzero(core.minmax01(torch.ones(5))) == 0
    out = core.minmax01(torch.tensor([float("nan"), 1.0, 3.0]))
    assert out.tolist() == pytest.approx([0.0, 1 / 3, 1.0])


def test_long_axis_children_identity_rotation():
    m, s, o = core.long_axis_children(
        torch.zeros(1, 3), torch.log(torch.tensor([[0.1, 0.4, 0.2]])),
        torch.tensor([[1.0, 0, 0, 0]]), torch.zeros(1))
    assert torch.allclose(m[0], torch.tensor([0.0, 3 * 0.45 * 0.4, 0.0]))
    assert torch.allclose(m[1], -m[0])
    assert torch.allclose(s.exp()[0, 1], torch.tensor(0.4 * 0.55))
    assert torch.allclose(torch.sigmoid(o), torch.full((2,), 0.3))


@pytest.mark.parametrize("name", ["eas_sampling", "las_geometry", "growth_control",
                                  "rap_schedule", "mu_gated_adam", "lr_schedule"])
def test_cpu_probes(name):
    getattr(probes, name)(method(), "cpu")


def test_edge_map_matches_loop():
    img = torch.rand(9, 13, 3)
    assert torch.allclose(core.edge_map(img).double(), probes._edge_loop(img), atol=1e-6)
