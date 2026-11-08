import torch
from splatstudio.methods import make_method  # noqa: F401  (import check)

import ss_turbo_gs
from ss_turbo_gs import probes
from ss_turbo_gs.budget import BudgetScheduler, budget_curve, progress_index, resolve_budget
from ss_turbo_gs.renderer import dilated_camera, dilated_target


def test_cpu_probes():
    m = ss_turbo_gs.method()
    probes.dilated_target(m, "cpu")
    probes.budget_schedule(m, "cpu")
    probes.color_densify(m, "cpu")
    probes.refine_allowance(m, "cpu")


def test_progress_and_curve():
    assert progress_index(500, 500, 15000) == 1.0
    assert progress_index(15000, 500, 15000) == 100.0
    assert budget_curve(100.0, 0.4, 10, 110) == 110
    assert budget_curve(1.0, 0.4, 10, 110) == 10


def test_budget_lookup():
    assert resolve_budget(80861, None, ss_turbo_gs.DEFAULTS["budget_table"]) == 3273600
    assert resolve_budget(5, 42, {}) == 42
    try:
        resolve_budget(5, None, {})
    except ValueError:
        pass
    else:
        raise AssertionError


def test_scheduler_default_alpha_before_warmup():
    s = BudgetScheduler(10, 100)
    for t in range(1, 150):
        s.record(t, 1.0 / t)
    assert s.update() == 1.0  # <= 100 records after warm-up


def test_dilated_camera_pixel_centres():
    K = torch.tensor([[100.0, 0, 50.0], [0, 90.0, 40.0], [0, 0, 1]])
    cam = dilated_camera({"K": K, "width": 101, "height": 80}, 2, 1, 0)
    assert (cam["width"], cam["height"]) == (50, 40)
    # A point projecting to high-res centre of pixel x=2*j+1 lands on ds centre j+0.5.
    j = 7
    u_hr = 2 * j + 1 + 0.5
    x_over_z = (u_hr - 50.0) / 100.0
    u_ds = cam["K"][0, 0] * x_over_z + cam["K"][0, 2]
    assert abs(float(u_ds) - (j + 0.5)) < 1e-5
    img = torch.arange(80 * 101 * 3).reshape(80, 101, 3)
    sub = dilated_target(img, 2, 1, 0, 40, 50)
    assert torch.equal(sub[3, 7], img[6, 15])
