import math

import torch

from ss_confidence_densify import density as D
from ss_confidence_densify import mcmc_control, method, probes
from ss_confidence_densify.losses import confidence_loss_value, ssim_map


def test_rank_ties_and_range():
    r = D.normalized_rank(torch.tensor([0.0, 0.0, 1.0, 3.0, 2.0]))
    assert r.tolist() == [0.0, 0.0, 0.5, 1.0, 0.75]


def test_eta_and_pdf_bounds():
    assert D.eta(2500, 5000) == 0.5 and D.eta(9000, 5000) == 1.0
    p = D.sampling_pdf(torch.arange(10.0), torch.arange(10.0), 10000)
    assert abs(p[-1].item() - 0.2) < 1e-6 and p[0].item() == 0.0


def test_share_schedule():
    assert abs(D.relocation_share(15000)[0] - 0.1 * math.exp(-0.5)) < 1e-9
    assert D.relocation_share(100)[1] is True and D.relocation_share(5000)[1] is False


def test_ssim_map_identity_and_loss_at_unit_confidence():
    x = torch.rand(16, 16, 3)
    assert torch.allclose(ssim_map(x, x), torch.ones(16, 16, 3), atol=1e-4)
    y = torch.rand(16, 16, 3)
    c = torch.ones(16, 16, 1)
    v = confidence_loss_value(x, y, c, 0.075, 0.2, 1e-3, 5.0)
    ref = (0.8 * (x - y).abs() + 0.2 * (1 - ssim_map(x, y))).mean()
    assert torch.allclose(v, ref)


def test_cpu_probes():
    m = method()
    for fn in (probes.confidence_loss, probes.sampling_pdf, probes.relocation_schedule):
        fn(m, "cpu")


def test_specs():
    m = method()
    assert [f.name for f in m.fields] == ["confidence"]
    assert {c.name for c in m.components} == {"cad-renderer", "cad-density",
                                              "confidence-photometric", "mcmc-reg"}
    assert mcmc_control().name == "cad-mcmc-control"
