import math

import torch

from ss_crim_gs import method
from ss_crim_gs.kernel import WeightCNN
from ss_crim_gs.ode import odeint_dopri5
from ss_crim_gs.probes import dopri5, objective, screw_se3
from ss_crim_gs.se3 import g_matrix, normalize_axis, rodrigues, screw_exp


def test_probes_cpu():
    m = method()
    assert dopri5(m, "cpu")["rel_err"] < 1e-3
    assert screw_se3(m, "cpu")["max_err"] < 1e-8
    objective(m, "cpu")


def test_dopri5_backward_and_reverse():
    a = torch.tensor(-0.7, dtype=torch.float64, requires_grad=True)
    y = odeint_dopri5(lambda t, y: a * y, torch.ones(1, dtype=torch.float64), [0.0, 1.0, 2.0],
                      rtol=1e-8, atol=1e-10)
    assert abs(float(y[-1]) - math.exp(-1.4)) < 1e-6
    y[-1].sum().backward()  # d/da e^{2a} = 2 e^{2a}
    assert abs(float(a.grad) - 2 * math.exp(-1.4)) < 1e-5
    yb = odeint_dopri5(lambda t, y: -0.7 * y, torch.ones(1, dtype=torch.float64), [2.0, 0.0],
                       rtol=1e-8, atol=1e-10)
    assert abs(float(yb[-1]) - math.exp(1.4)) < 1e-5


def test_screw_identity_and_small_angle():
    w = normalize_axis(torch.tensor([[0.0, 0.0, 1e-5]]))
    th = torch.zeros(1, 1)
    T = screw_exp(w, th, torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.allclose(T, torch.eye(4)[None])
    th = torch.tensor([[math.pi / 2]])
    R = rodrigues(normalize_axis(torch.tensor([[0.0, 0.0, 1.0]])), th)[0]
    assert torch.allclose(R @ torch.tensor([1.0, 0, 0]), torch.tensor([0.0, 1, 0]), atol=1e-6)
    G = g_matrix(normalize_axis(torch.tensor([[0.0, 0.0, 1.0]])), th)[0]
    assert torch.allclose(G[2, 2], th[0, 0])


def test_cnn_shapes():
    cnn = WeightCNN()
    P, M = cnn(torch.rand(9, 20, 24, 3))
    assert P.shape == (9, 20, 24, 1) and M.shape == (20, 24, 1)
    assert torch.allclose(P.sum(0), torch.ones(20, 24, 1), atol=1e-6)
    convs = [m for m in cnn.features if isinstance(m, torch.nn.Conv2d)]
    assert [c.kernel_size[0] for c in convs] == [5, 3, 3]
    assert all(c.out_channels == 64 for c in convs)


def test_spec_values():
    m = method()
    dens = next(c for c in m.components if c.name == "crimgs-density").config
    assert dens["kernel_num_poses"] == 9 and dens["kernel_hidden"] == 64
    assert dens["total_steps"] == 40000 and dens["refine_stop_iter"] == 20000
    reg = next(c for c in m.components if c.name == "crimgs-regularizers").config
    assert reg == {"lambda_o": 1e-4, "lambda_m": 1e-3}
