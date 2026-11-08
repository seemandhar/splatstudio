import torch

from ss_lp_3dgs import core, method
from ss_lp_3dgs.probes import gumbel, mask_loss_probe


def test_window_and_score_steps():
    assert not core.in_mask_window(19500, 19500, 500)
    assert core.in_mask_window(19501, 19500, 500)
    assert core.in_mask_window(19999, 19500, 500)
    assert not core.in_mask_window(20000, 19500, 500)
    steps = [s for s in range(19000, 21000) if core.is_score_step(s, 19500, 500, 20)]
    assert steps == list(range(19520, 20000, 20))


def test_effective_opacity_phases():
    params = torch.nn.ParameterDict({
        "opacities": torch.nn.Parameter(torch.zeros(4)),
        core.FIELD: torch.nn.Parameter(torch.ones(4)),
    })
    assert torch.allclose(core.effective_opacity(params, 0.5), torch.full((4,), 0.5))
    setattr(params, core.SCORE_ATTR, torch.tensor([0.0, 1.0, 2.0, 3.0]))
    setattr(params, core.PHASE_ATTR, "train")
    noise = torch.zeros(4)
    got = core.effective_opacity(params, 0.5, noise)
    want = 0.5 * torch.sigmoid(torch.tensor([0.0, 1.0, 2.0, 3.0]) / 0.5)
    assert torch.allclose(got, want)


def test_prune_selection_sign():
    m = torch.tensor([-1.0, 1.0])
    assert core.prune_selection(m, 0.5, torch.zeros(2)).tolist() == [True, False]


def test_cpu_probes():
    spec = method()
    gumbel(spec, "cpu")
    mask_loss_probe(spec, "cpu")
