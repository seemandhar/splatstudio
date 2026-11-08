import math

import numpy as np
import torch

from ss_decomp_gs import method
from ss_decomp_gs import schedule as S
from ss_decomp_gs.losses import ssim_same


def test_energy_scaling_white_noise_decreases():
    g = torch.Generator().manual_seed(0)
    ims = [torch.randn(3, 128, 160, generator=g)]
    e = S.scale_energies(ims, 4)
    ratios = [e[r] / e[1] for r in range(1, 5)]
    assert ratios[0] == 1 and all(a > b for a, b in zip(ratios, ratios[1:]))
    # White noise: (roughly) only the 1/r^2 low-frequency window survives -> ratio well below 1/r.
    assert ratios[1] < 0.5 and ratios[3] < 0.25


def test_spectrum_energy_matches_numpy():
    g = torch.Generator().manual_seed(1)
    im = torch.rand(3, 33, 47, generator=g)
    ref = np.abs(np.fft.fft2(im.numpy().astype(np.float64))).sum()
    assert math.isclose(S.spectrum_energy(im), ref, rel_tol=1e-9)


def test_end_iterations_and_scale_lookup():
    ends = S.end_iterations({1: 10.0, 2: 5.0, 3: 2.5}, 1000)
    assert ends == {1: 1000, 2: 500, 3: 250}
    assert [S.scale_at(ends, t) for t in (0, 249, 250, 499, 500, 999, 1000, 5000)] == \
        [3, 3, 2, 2, 1, 1, 1, 1]
    assert S.full_resolution_step(ends) == 500


def test_downscaled_size_floor():
    assert S.downscaled_size(546, 979, 8) == (68, 122)
    assert S.downscale(torch.rand(3, 61, 83), 3).shape == (3, 20, 27)


def test_position_lr():
    assert S.position_lr(10, 1.0, 0.01, 100, 50) == 1.0
    assert math.isclose(S.position_lr(75, 1.0, 0.01, 100, 50), 0.1)
    assert math.isclose(S.position_lr(100, 1.0, 0.01, 100, 50), 0.01)


def test_adaptive_threshold():
    a = torch.tensor([0.5, 0.01, 0.2, 0.03, 0.9, 0.04, 0.6, 0.7, 0.8, 0.02])
    assert math.isclose(S.adaptive_threshold(a, 0.2, 0.05), 0.03, rel_tol=1e-6)  # sorted[2]
    assert S.adaptive_threshold(a, 0.5, 0.05) == 0.05  # sorted[5] = 0.5 capped by tau_u


def test_ssim_same_identity():
    x = torch.rand(20, 24, 3)
    assert abs(float(ssim_same(x, x)) - 1) < 1e-5


def test_method_spec():
    m = method()
    kinds = [c.kind for c in m.components]
    assert kinds == ["renderer", "density", "loss"]
    d = next(c for c in m.components if c.kind == "density")
    assert d.config["densify_until"] == 25000 and d.config["max_scale_factor"] == 8
    assert d.config["prune_ratio"] == 0.03 and d.config["prune_upper"] == 0.05
