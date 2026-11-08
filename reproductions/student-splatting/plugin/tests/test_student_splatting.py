"""CPU unit tests of the pure functions (CUDA probes run through ss-verify)."""

import math

import torch

from ss_student_splatting import CONFIGS, student_splatting_bicycle, student_splatting_room
from ss_student_splatting.density import (
    binomials,
    expon_lr,
    nu_lr,
    op_sigmoid,
    relocation_student_t,
    sghmc_lr,
)
from ss_student_splatting.losses import regularizer
from ss_student_splatting.probes import _reloc_reference, regularizer_probe, sghmc_step
from ss_student_splatting.projection import empirical_factor, inria_conic, project
from ss_student_splatting.renderer import nu_activation


def test_configs_follow_official_json():
    assert CONFIGS["student-splatting-bicycle"]["C_burnin"] == 5e8
    assert CONFIGS["student-splatting-bicycle"]["burnin_iterations"] == 15000
    assert CONFIGS["student-splatting-room"]["cap_max"] == 1_500_000
    d = {c.name: c for c in student_splatting_room().components}["sghmc-recycling"]
    assert d.config["cap_max"] == 1_500_000
    assert student_splatting_bicycle().fields[0].initial_value == 100.0


def test_relocation_matches_float64_formula():
    b = binomials(51, "cpu")
    for o, s, nu, n in ((0.9, 1.0, 100.0, 2), (-0.7, 0.3, 2.5, 5), (0.4, 2.0, 1.0, 3)):
        on, sn = relocation_student_t(torch.tensor([o]), torch.tensor([[s] * 3]),
                                      torch.tensor([nu]), torch.tensor([n]), b)
        ro, rs = _reloc_reference(o, s, nu, n)
        assert abs(on.item() - ro) < 1e-5
        assert abs(sn[0, 0].item() - rs) / rs < 1e-4


def test_relocation_identity_for_one_copy():
    b = binomials(51, "cpu")
    on, sn = relocation_student_t(torch.tensor([0.6]), torch.ones(1, 3), torch.tensor([7.0]),
                                  torch.tensor([1]), b)
    assert abs(on.item() - 0.6) < 1e-6 and (sn - 1).abs().max() < 1e-5


def test_schedules():
    s = 3.0
    hi = math.sqrt(0.001226442339330813 * s / 7.45176315307617)
    lo = math.sqrt(1.226630619638022e-05 * s / 7.45176315307617)
    cfg = {"lr_init_base": 0.001226442339330813, "lr_final_base": 1.226630619638022e-05,
           "lr_reference_scale": 7.45176315307617, "lr_max_steps": 30000}
    assert abs(sghmc_lr(0, s, cfg) - hi) < 1e-12
    assert abs(sghmc_lr(40000, s, cfg) - lo) < 1e-12  # clamped after 30k
    assert abs(expon_lr(15000, 1.0, 0.01, 30000) - 0.1) < 1e-12
    nl = {"nu_lr": 10.0, "nu_lr_gamma": 0.5, "nu_lr_step": 5000, "nu_lr_min": 0.01}
    assert nu_lr(9999, nl) == 5.0 and nu_lr(10 ** 6, nl) == 0.01


def test_sigmoid_gate():
    o = torch.tensor([0.0, 0.005, 0.05, 0.9])
    sig = op_sigmoid(1 - o)
    assert sig[0] > 0.6 and abs(sig[1] - 0.5) < 1e-6 and sig[2] < 0.02 and sig[3] < 1e-30


def test_empirical_factor_and_nu_activation():
    e = empirical_factor(torch.tensor([1.0, 2.0, 12.0, 12.999, 13.0, 50.0]))
    assert torch.allclose(e[:3], torch.tensor([9.925, 5.841, 3.012]))
    assert e[4] == 3.0 and e[5] == 3.0
    assert nu_activation(torch.tensor([-5.0, 0.5, 3.0, 2e4])).tolist() == [1.0, 1.0, 3.0, 1e4]


def test_projection_cpu_matches_exact_inverse():
    g = torch.Generator().manual_seed(0)
    means = torch.stack([torch.rand(50, generator=g) - 0.5, torch.rand(50, generator=g) - 0.5,
                         2 + torch.rand(50, generator=g)], -1)
    quats = torch.nn.functional.normalize(torch.randn(50, 4, generator=g), dim=-1)
    scales = -2.0 + torch.rand(50, 3, generator=g)
    K = torch.tensor([[40.0, 0, 16], [0, 40.0, 12], [0, 0, 1]])
    p = project(means, quats, scales, torch.full((50,), 5.0), torch.eye(4), K, 32, 24)
    q = p["conics"][p["valid"]]
    conic = torch.stack((torch.stack((q[:, 0], q[:, 1]), -1), torch.stack((q[:, 1], q[:, 2]), -1)), 1)
    cov = torch.linalg.inv(conic)
    assert (cov[:, 0, 1] - cov[:, 1, 0]).abs().max() < 1e-5
    a = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    c = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(inria_conic, (a, b, c), atol=1e-6)


def test_regularizer_and_sghmc_probe_cpu():
    m = student_splatting_room()
    assert regularizer_probe(m, "cpu")["value"] > 0
    if not torch.cuda.is_available():
        out = sghmc_step(m, "cpu")
        assert out["init"] == "ok"
    params = {"opacities": torch.zeros(4, requires_grad=True), "scales": torch.zeros(4, 3)}
    assert abs(regularizer(params, 0.01, 0.01).item() - 0.01) < 1e-7
