"""CPU unit tests of the EAGLES port's pure functions and CPU probes."""

import math

import torch

from ss_eagles import codec, decoders, density, method, probes


def test_spec_valid():
    from splatstudio.contracts import validate

    assert validate(method()) == []


def test_cpu_probes_pass():
    m = method()
    for probe in (probes.decoder_ste, probes.latent_init_lr, probes.entropy_size,
                  probes.progressive_schedule, probes.opacity_reset, probes.schedule):
        probe(m, "cpu")
    assert decoders.ACTIVE["bank"] is None


def test_resize_scale_monotone_and_bounds():
    cfg = density.DEFAULTS
    values = [density.resize_scale(t, cfg) for t in range(1, 30001, 50)]
    assert all(b >= a for a, b in zip(values, values[1:]))
    assert 0.3 <= values[0] < 0.3001 and values[-1] == 1.0


def test_paper_size_constant_latents_cost_nothing():
    bank = decoders.DecoderBank(seed=0)
    q = {"q_rot": torch.zeros(10, 8), "q_opa": torch.full((10, 1), 3.0)}
    size = codec.paper_size_bits(q, bank, 10)
    assert size["latent_bits"] == {"q_rot": 0.0, "q_opa": 0.0}
    assert size["total_bits"] == size["decoder_bits"] + 32 * 9 * 10


def test_codec_single_value_dimension_roundtrip():
    q = torch.tensor([[1.0, -2.0], [1.0, 5.0], [1.0, -2.0]])
    assert torch.equal(codec.decode(codec.encode({"q": q}))["q"], q)


def test_dft_orthonormal_columns_first_16():
    f = decoders.dft_matrix(16, 45)
    assert f.shape == (16, 45)
    assert math.isclose(float(f[0, 0]), 1 / math.sqrt(45), rel_tol=1e-6)
