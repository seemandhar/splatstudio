import math

import torch

from ss_rain_gs import density as D
from ss_rain_gs import renderer as R


def test_lowpass_value():
    assert R.lowpass_value(840, 1297, 10) == 300.0
    assert R.lowpass_value(840, 1297, 10**8) == 0.3
    assert abs(R.lowpass_value(100, 100, 50) - 1e4 / (9 * math.pi * 50)) < 1e-12


def test_lowpass_due():
    assert R.lowpass_due(1) and R.lowpass_due(1000) and R.lowpass_due(24000)
    assert not R.lowpass_due(25000) and not R.lowpass_due(1500)


def test_sh_degree():
    assert [R.sh_degree_at(s) for s in (1, 4999, 5000, 6000, 7000, 9000)] == [0, 0, 1, 2, 3, 3]


def test_slv_cube():
    vm = [torch.eye(4) for _ in range(2)]
    vm[0][:3, 3] = torch.tensor([-1.0, 0.0, 2.0])
    vm[1][:3, 3] = torch.tensor([3.0, 1.0, 0.0])
    lo, side = D.slv_cube(vm)
    assert side == 12.0 and lo == 1.0 - 6.0


def test_official_sh0_roundtrip():
    shs = torch.rand(100, 3, dtype=torch.float64)
    sh0 = D.official_sh0(shs)
    assert (sh0.double() - shs).abs().max() < 1 / 255 / D.SH_C0 + 1e-6


def test_means_lr():
    assert D.means_lr(9999, 1.0, 10000) == 1.0
    assert abs(D.means_lr(40000, 1.0, 10000) - 0.01) < 1e-12
    assert abs(D.means_lr(25000, 1.0, 10000) - 0.1) < 1e-12


def test_knn_scales():
    p = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
    s = D.knn3_log_scales(p)
    assert abs(float(s[0, 0]) - math.log(math.sqrt((1 + 4 + 9) / 3))) < 1e-6
