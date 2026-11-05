import inspect
import math

import torch

from ss_mip_splatting.filters import (
    compute_filter_3d,
    filter_coefficient,
    filtered_scales_opacities,
    is_filter_step,
    is_reset_step,
    reset_logits,
)


def _cam(f=100.0, w=100, h=100, viewmat=None):
    K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    return {"viewmat": torch.eye(4) if viewmat is None else viewmat, "K": K, "width": w,
            "height": h}


def test_filter_analytic_and_min_over_cameras():
    near_cam = torch.eye(4)
    near_cam[2, 3] = -3.0  # camera centre at z = 3 -> depth 2 for a point at z = 5
    pts = torch.tensor([[0.0, 0.0, 5.0]])
    f = compute_filter_3d(pts, [_cam(100.0), _cam(200.0, viewmat=near_cam)])
    # min depth 2, max focal 200 (official: global max focal, not per camera)
    assert torch.allclose(f, torch.tensor([math.sqrt(0.2) * 2 / 200]))


def test_filter_visibility_rules():
    pts = torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 4.0], [0.0, 0.0, -4.0], [0.6, 0.0, 1.0],
                        [0.7, 0.0, 1.0]])
    f = compute_filter_3d(pts, [_cam()])
    # z <= 0.2 and behind the camera are invisible -> max visible depth (4.0).
    # x = 0.6 -> u = 110 <= 115 visible; x = 0.7 -> u = 120 > 115 invisible.
    d = torch.tensor([4.0, 4.0, 4.0, 1.0, 4.0])
    assert torch.allclose(f, d / 100 * math.sqrt(0.2))


def test_filtered_covariance_matches_determinants():
    g = torch.Generator().manual_seed(0)
    log_s = torch.randn(6, 3, generator=g) - 3
    logits = torch.randn(6, generator=g)
    filt = torch.rand(6, generator=g) * 0.05
    scales, opac = filtered_scales_opacities(log_s, logits, filt)
    s2 = log_s.exp().square()
    for i in range(6):
        cov = torch.diag(s2[i])
        cov_f = cov + filt[i] ** 2 * torch.eye(3)
        assert torch.allclose(torch.diag(scales[i]) ** 2, cov_f, rtol=1e-5, atol=1e-10)
        coef = torch.sqrt(torch.linalg.det(cov.double()) / torch.linalg.det(cov_f.double()))
        assert torch.allclose(opac[i].double(), logits[i].sigmoid().double() * coef, rtol=1e-4)
    # zero filter -> identity
    scales0, opac0 = filtered_scales_opacities(log_s, logits, torch.zeros(6))
    assert torch.allclose(scales0, log_s.exp()) and torch.allclose(opac0, logits.sigmoid())


def test_filter_gradient_flows_to_scales_and_opacity_not_filter():
    log_s = torch.full((2, 3), -3.0, requires_grad=True)
    logits = torch.zeros(2, requires_grad=True)
    filt = torch.full((2,), 0.03, requires_grad=True)
    s, o = filtered_scales_opacities(log_s, logits, filt)
    (s.sum() + o.sum()).backward()
    assert log_s.grad.abs().sum() > 0 and logits.grad.abs().sum() > 0 and filt.grad is None


def test_reset_clamps_filtered_opacity():
    log_s = torch.full((3, 3), math.log(0.02))
    filt = torch.tensor([0.0, 0.02, 0.04])
    logits = torch.tensor([2.0, 2.0, -8.0])
    new = reset_logits(log_s, logits, filt, 0.01)
    coef = filter_coefficient(log_s, filt)
    got = new.sigmoid() * coef
    want = torch.minimum(logits.sigmoid() * coef, torch.tensor(0.01))
    assert torch.allclose(got, want, rtol=1e-5)


def test_schedule_boundaries():
    assert not is_filter_step(500) and is_filter_step(600) and is_filter_step(14900)
    assert not is_filter_step(15000) and is_filter_step(15100) and is_filter_step(29800)
    assert not is_filter_step(29900) and not is_filter_step(30000) and not is_filter_step(650)
    assert [s for s in range(1, 30001) if is_reset_step(s)] == [3000, 6000, 9000, 12000]
    assert not any(is_reset_step(s, every=0) for s in range(1, 30001))


def test_gsplat_default_strategy_reset_is_unreachable():
    # Pins the gsplat 1.5.3 behaviour this plugin relies on (no double opacity reset):
    # `step % self.reset_every == 0 & step > 0` parses as `(... == 0) and (0 > 0)`.
    from gsplat import DefaultStrategy

    src = inspect.getsource(DefaultStrategy.step_post_backward)
    assert "step % self.reset_every == 0 & step > 0" in src
    step, every = 3000, 3000
    assert not (step % every == 0 & step > 0)


def _official_reset(log_s, logits, filt, value=0.01):
    # Verbatim official formula (gaussian_model.py reset_opacity) in float32.
    s2 = log_s.exp().square()
    coef = torch.sqrt(s2.prod(dim=1) / (s2 + filt.square()[:, None]).prod(dim=1))
    target = torch.minimum(logits.sigmoid() * coef, torch.full_like(coef, value))
    return torch.logit(target / coef)


def test_reset_finite_in_failure_regime():
    # Real failure regime (garden/bicycle/counter at the 3000/6000 reset): Gaussians far
    # smaller than filter_3d (~4e-3), and saturated opacities.
    filt = torch.full((5,), 0.004)
    log_s = torch.log(torch.tensor([[1e-8] * 3, [1e-4] * 3, [1e-12, 1e-2, 1e-2], [1e-4] * 3,
                                    [1e-2] * 3]))
    logits = torch.tensor([0.0, 20.0, 1.0, -3.0, 2.0])
    official = _official_reset(log_s, logits, filt)
    assert not torch.isfinite(official).all()  # the unclamped official formula fails here
    new = reset_logits(log_s, logits, filt)
    assert torch.isfinite(new).all()
    # Where the official formula is finite, both agree.
    ok = torch.isfinite(official)
    assert torch.allclose(new[ok], official[ok], atol=1e-4)
    # Filtered opacity after reset never exceeds the reset value.
    assert torch.all(new.sigmoid() * filter_coefficient(log_s, filt) <= 0.01 + 1e-6)


def test_filtered_render_inputs_finite_and_differentiable_for_tiny_scales():
    log_s = torch.log(torch.tensor([[1e-8] * 3, [1e-20, 1e-2, 1e-2]])).requires_grad_(True)
    logits = torch.zeros(2, requires_grad=True)
    s, o = filtered_scales_opacities(log_s, logits, torch.full((2,), 0.004))
    (s.sum() + o.sum()).backward()
    assert torch.isfinite(s).all() and torch.isfinite(o).all()
    assert torch.isfinite(log_s.grad).all() and torch.isfinite(logits.grad).all()
