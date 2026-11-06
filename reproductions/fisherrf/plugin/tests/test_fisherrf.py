import math

import pytest
import torch
from torch import nn

from ss_fisherrf import active, probes, selection


def test_eig_selection_probe():
    probes.eig_selection(None, "cpu")


def test_schedule_probe():
    probes.active_schedule(None, "cpu")


def test_schedule_rejects_uneven():
    with pytest.raises(ValueError):
        selection.schedule(20, 3, 4, 100)


class _Density:
    def __init__(self):
        self.resets = 0

    def reset(self, params, optimizers):
        self.resets += 1


def _controller(selector="random", **kw):
    c = active.ActiveController({"selector": selector, "total_views": 4, "views_per_round": 1,
                                 "num_init_views": 2, "interval_epochs": 2,
                                 "total_iterations": 20, "position_lr_max_steps": 20, **kw})
    cams = []
    for i in range(6):
        a = 2 * math.pi * i / 6
        v = torch.eye(4)
        v[:3, 3] = torch.tensor([math.cos(a), math.sin(a), 3.0])
        cams.append({"viewmat": v, "K": torch.eye(3), "width": 8, "height": 8})
    c.register_cameras(cams)
    params = nn.ParameterDict({"means": nn.Parameter(torch.zeros(3, 3))})
    opts = {"means": torch.optim.Adam([params["means"]], lr=0.5)}
    dens = _Density()
    c.register_state(params, opts, dens)
    return c, cams, params, opts, dens


def test_controller_gating_restore_and_budget():
    c, cams, params, opts, dens = _controller()
    assert c.selected == [0, 3]  # farthest camera from view 0 on a hexagon
    assert c.load_its == {4: 1, 10: 1}
    modes = []
    for _ in range(400):
        if c.done:
            break
        for i in range(6):
            with torch.no_grad():
                params["means"].add_(1.0)  # any training drift must be undone by restore
            m = c.begin(params, cams[i])
            modes.append((i, m))
            if m == "train":
                assert i in c.selected
                lr = opts["means"].param_groups[0]["lr"]
                assert lr == pytest.approx(0.5 * 0.01 ** (c.cur_iter / 20))
    assert c.done and len(c.selected) == 4 and c.restores == 2 and dens.resets == 2
    assert all(m == "skip" for i, m in modes if i not in c.selected)
    with pytest.raises(RuntimeError):
        c2 = _controller()[0]
        held = torch.eye(4)
        held[:3, 3] = torch.tensor([0.0, 0.0, 9.0])
        c2.begin(params, {"viewmat": held})


def test_sh_schedule_resets_on_restore():
    c, cams, params, _, _ = _controller(sh_up_every=3)
    seen = []
    while not c.done:
        for i in c.selected:
            if c.begin(params, cams[i]) != "skip" or c.done:
                seen.append((c.it, c.sh_degree))
    d = dict(seen)
    assert d[3] == 1 and d[4] == 0  # restored at it=4 (acquisition), degree back to 0
    assert d[6] == 1 and d[10] == 0 and d[12] == 1
