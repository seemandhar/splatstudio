import math

import torch

from ss_neural_appearance import CAPS, build, probes
from ss_neural_appearance.appearance import DEFAULTS, frequency_encode, n_inputs, sh_basis
from ss_neural_appearance.mcmc import relocation_adjustment
from ss_neural_appearance.schedule import cosine_decay, lr_decay


def test_input_width_is_32():
    assert n_inputs(DEFAULTS) == 32


def test_frequency_encoding_order():
    f = torch.tensor([[0.1, 0.2]])
    enc = frequency_encode(f, 1)
    assert torch.allclose(enc, torch.tensor([[math.sin(0.1), math.cos(0.1), math.sin(0.2), math.cos(0.2)]]))


def test_sh_basis_unit_norm_degree0():
    d = torch.nn.functional.normalize(torch.randn(5, 3), dim=-1)
    b = sh_basis(d, 0)
    assert b.shape == (5, 16) and torch.all(b[:, 1:] == 0)


def test_relocation_single_sample_identity():
    o, s = relocation_adjustment(torch.tensor([0.4]), torch.ones(1, 3), torch.tensor([1]))
    assert torch.allclose(o, torch.tensor([0.4])) and torch.allclose(s, torch.ones(1, 3), atol=1e-6)


def test_schedules():
    assert cosine_decay(37000, 1.0, 7000, 37000) == 0.0
    assert abs(lr_decay(30000, 3e-3, 3e-4, 30000, 3000, 1e-8) - 3e-4) < 1e-12


def test_methods_and_caps():
    assert CAPS["garden"] == 5_000_000 and CAPS["room"] == 1_500_000
    m = build("sh", "room")
    density = [c for c in m.components if c.kind == "density"][0].config
    assert density["cap_max"] == 1_500_000 and density["appearance"] == "sh"


def test_cpu_probes():
    for fn in (probes.neural_decoder, probes.mcmc, probes.schedules):
        fn(None, "cpu")
