import math

import torch

from ss_analytic_splatting import method, reference
from ss_analytic_splatting.filters import log_filter_coefficient, reset_logits
from ss_analytic_splatting.probes import density_schedule, logistic_cdf


def test_logistic_symmetry():
    x = torch.linspace(-5, 5, 101, dtype=torch.float64)
    s = reference.logistic_cdf(x) + reference.logistic_cdf(-x)
    assert torch.allclose(s, torch.ones_like(s))


def test_window_integral_large_sigma_matches_point_sample():
    # for sigma >> 1 px the pixel integral tends to the centre sample
    d = torch.tensor([[0.3, -1.2], [2.0, 1.0]], dtype=torch.float64)
    th = 0.4
    R = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]],
                     dtype=torch.float64)
    cov = (R @ torch.diag(torch.tensor([30.0 ** 2, 20.0 ** 2], dtype=torch.float64)) @ R.T)
    cov = cov.expand(2, 2, 2)
    a = reference.window_integral_2d(d, cov)
    b = reference.point_sample_2d(d, cov)
    assert torch.allclose(a, b, rtol=1e-2)  # S(0)' = 0.4 vs phi(0) = 0.3989


def test_reset_logits_matches_official_formula():
    ls = torch.log(torch.tensor([[0.05, 0.02, 0.01], [1.0, 1.0, 1.0]]))
    o = torch.logit(torch.tensor([0.6, 0.6]))
    f = torch.tensor([0.03, 0.03])
    coef = torch.exp(log_filter_coefficient(ls, f))
    official = torch.minimum(o.sigmoid() * coef, torch.tensor(0.01)) / coef
    assert torch.allclose(reset_logits(ls, o, f).sigmoid(), official, atol=1e-6)


def test_cpu_probes():
    m = method()
    assert logistic_cdf(m, "cpu")["max_abs_cdf_err"] < 5e-4
    assert density_schedule(m, "cpu")["event_600"]["split"] == 1
