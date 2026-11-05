import torch
from splatstudio.components import build

import ss_absgs
from ss_absgs import probes
from ss_absgs.density import DEFAULTS


def test_paper_defaults():
    assert DEFAULTS["clone_grad2d"] == 0.0002
    assert DEFAULTS["split_grad2d"] == 0.0004
    assert DEFAULTS["grow_scale3d"] == 0.001
    assert DEFAULTS["refine_stop_iter"] == 15000 and DEFAULTS["reduce_every"] == 0


def test_criteria_cpu():
    for factory in (ss_absgs.method, ss_absgs.absgs_0008, ss_absgs.absgs_noreset):
        out = probes.split_clone_criteria(factory(), "cpu")
        assert out["split_mask"] == [False, False, True, False, False, False, True]


def test_criteria_fails_with_gsplat_absgrad_thresholds():
    # Mechanism removal: gsplat-absgrad style (abs for both, 0.0008, tau_S 0.01) must fail.
    m = ss_absgs._make("absgs", "x", clone_grad2d=0.0008, split_grad2d=0.0008, grow_scale3d=0.01)
    try:
        probes.split_clone_criteria(m, "cpu")
    except AssertionError:
        return
    raise AssertionError("probe did not detect wrong thresholds")


def test_reset_disabled_detected():
    m = ss_absgs._make("absgs", "x", opacity_reset=False)
    try:
        probes.split_clone_criteria(m, "cpu")
    except AssertionError:
        return
    raise AssertionError("probe did not detect missing opacity reset")


def test_reduce_opacity_optional():
    m = ss_absgs._make("absgs", "x", reduce_every=500)
    _, density, _ = build(m)
    params, opts = probes._scene(2, "cpu", [[0, 0, 0], [1, 0, 0]], [[0.1] * 3] * 2, 0.95)
    density.initialize(params, opts, 1.0)
    m2d = torch.zeros((1, 2, 2))
    m2d.grad = torch.zeros_like(m2d)
    m2d.absgrad = torch.zeros_like(m2d)
    info = {"means2d": m2d, "width": 4, "height": 4, "n_cameras": 1,
            "radii": torch.ones((1, 2, 2)), "gaussian_ids": None}
    density.after_backward(params, opts, 499, {"info": info})
    assert torch.sigmoid(params["opacities"]).min() > 0.9
    density.after_backward(params, opts, 500, {"info": info})
    assert torch.sigmoid(params["opacities"]).max() <= 0.8 + 1e-6
