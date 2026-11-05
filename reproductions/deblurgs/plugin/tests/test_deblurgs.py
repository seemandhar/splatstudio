import math

import numpy as np
import pytest
import torch

from ss_deblurgs import method, method_nu0, probes
from ss_deblurgs.bank import (
    TrajectoryBank,
    alignment_lr,
    bezier_coefficients,
    curve_lr,
    expon,
    row_c2w_to_viewmat,
    viewmat_to_row_c2w,
)
from ss_deblurgs.density import official_extent
from ss_deblurgs.evaluate import quat_wxyz_from_rotmat, rotmat_from_quat_wxyz, ssim_same
from ss_deblurgs.losses import hinge_l2, smoothness
from ss_deblurgs.renderer import inverse_tonemap, tonemap


def test_cpu_probes_pass():
    m = method()
    probes.bezier_se3(m, "cpu")
    probes.smoothness(m, "cpu")


def test_bezier_coefficients_partition_of_unity_and_endpoints():
    t = torch.linspace(0, 1, 11, dtype=torch.float64)
    c = bezier_coefficients(t, 9)
    assert torch.allclose(c.sum(1), torch.ones(11, dtype=torch.float64))
    assert c[0, 9] == 1 and c[-1, 0] == 1  # t=0 -> c_9, t=1 -> c_0 (official convention)


def test_row_convention_round_trip():
    g = torch.Generator().manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(4, generator=g), dim=0)
    vm = torch.eye(4)
    vm[:3, :3] = rotmat_from_quat_wxyz(q)
    vm[:3, 3] = torch.tensor([0.3, -0.2, 1.5])
    m = viewmat_to_row_c2w(vm[None])
    center = torch.linalg.inv(vm)[:3, 3]
    assert torch.allclose(m[0, 3, :3], center, atol=1e-6)
    assert torch.allclose(row_c2w_to_viewmat(m)[0], vm, atol=1e-6)


def test_expon_matches_official_helper():
    # get_expon_lr_func(4e-4, 2e-4, max_steps=25000)
    assert expon(0, 4e-4, 2e-4, 25000) == pytest.approx(4e-4)
    assert expon(25001, 4e-4, 2e-4, 25000) == 2e-4
    assert expon(10, 0.0, 1e-5, 100) == 0.0
    assert expon(7500, 1e-3, 1e-5, 150000) == pytest.approx(1e-3 * 0.01 ** 0.05)


def test_alignment_and_curve_lr_schedules():
    cfg = TrajectoryBank.__init__.__globals__["BANK_DEFAULTS"]
    assert alignment_lr(30000, cfg) == 0.0
    assert alignment_lr(90000, cfg) == pytest.approx(3e-3 * (1e-7 / 3e-3) ** 0.5)
    assert alignment_lr(10 ** 6, cfg) == 1e-7
    assert alignment_lr(50000, {**cfg, "curve_alignment_lr": 0.0}) == 0.0
    assert curve_lr(500, 1e-3, cfg) == 1e-3
    assert curve_lr(1000, 1e-3, cfg) == pytest.approx(1e-3 * 0.5 ** (1 / 15000))


def test_bank_initial_times_and_nu0_variant():
    cam = {"viewmat": torch.eye(4), "K": torch.eye(3), "width": 8, "height": 6}
    bank = TrajectoryBank([cam, cam], {}, "cpu")
    t = bank.times(1)
    assert torch.allclose(t, torch.linspace(0, 1, 21), atol=1e-6)
    assert bank.subframe_times(0, 999).tolist() == [0.0]
    assert len(bank.subframe_times(0, 1000)) == 21
    assert bank.trainable(1000) and not bank.trainable(999) and not bank.trainable(100000)
    dens = next(c for c in method_nu0().components if c.name == "deblurgs-density")
    assert dens.config["bank_curve_alignment_lr"] == 0.0


def test_tonemap_and_losses():
    x = torch.rand(5, 4, 3)
    assert torch.allclose(inverse_tonemap(tonemap(x)), x.clamp_min(1e-8), atol=1e-5)
    assert torch.allclose(tonemap(x), x.clamp_min(1e-8) ** (1 / 2.2))
    sub = torch.rand(3, 4, 5, 3)
    ref = (sub[1:] - sub[:-1]).abs().mean()
    assert torch.allclose(smoothness(sub), ref)
    assert float(smoothness(sub[:1])) == 0.0
    h = hinge_l2(torch.tensor([-1.0, 0.5, 2.0]))
    assert float(h) == pytest.approx((1 + 0 + 1) / 3)


def test_official_extent_numpy():
    g = np.random.default_rng(0)
    c, p = g.normal(size=(9, 3)), g.normal(size=(100, 3)) + 2
    r1 = np.percentile(np.linalg.norm(c - p.mean(0), axis=1), 10.0)
    r2 = 1.1 * np.linalg.norm(c - c.mean(0), axis=1).max()
    assert official_extent(torch.tensor(c), torch.tensor(p)) == pytest.approx(min(r1, r2))


def test_quaternion_round_trip():
    for seed in range(5):
        g = torch.Generator().manual_seed(seed)
        q = torch.nn.functional.normalize(torch.randn(4, generator=g), dim=0)
        R = rotmat_from_quat_wxyz(q)
        R2 = rotmat_from_quat_wxyz(quat_wxyz_from_rotmat(R))
        assert torch.allclose(R, R2, atol=1e-5)


def test_ssim_same_identity():
    x = torch.rand(20, 24, 3)
    assert float(ssim_same(x, x)) == pytest.approx(1.0, abs=1e-5)
    assert math.isfinite(float(ssim_same(x, torch.rand(20, 24, 3))))
