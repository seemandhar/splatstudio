import torch

from ss_fast_gs.density import combine_scores, densify_steps, error_mask, prune_weights
from ss_fast_gs.optim import main_steps_at, sh_steps_at
from ss_fast_gs.scenes import ARG_DEFAULTS, resolve


def test_error_mask_minmax_threshold():
    img = torch.zeros(2, 2, 3)
    gt = torch.tensor([0.0, 0.05, 0.5, 1.0]).reshape(2, 2, 1).expand(2, 2, 3)
    m = error_mask(img, gt, 0.1)
    assert m.tolist() == [[0, 0], [1, 1]]


def test_combine_scores_floor_and_minmax():
    counts = torch.tensor([[10, 0, 3], [2, 0, 9]])
    imp, pr = combine_scores(counts, [1.0, 0.5], 2)
    assert imp.tolist() == [6, 0, 6]
    raw = torch.tensor([10 + 1.0, 0.0, 3 + 4.5])
    assert torch.allclose(pr, raw / raw.max())


def test_prune_weights_positional():
    w = prune_weights(torch.tensor([0.0, 1.0, 0.5]), 5)
    assert w[3] == 0 and w[4] == 0 and w[1] > 1e5 and abs(float(w[2]) - 2.0) < 1e-4


def test_schedules():
    assert densify_steps({"densify_until": 15000, "densify_from": 500,
                          "densification_interval": 500})[:2] == [1000, 1500]
    assert main_steps_at(15000) and not main_steps_at(15001) and main_steps_at(15008)
    assert sh_steps_at(16) and not sh_steps_at(17) and sh_steps_at(20032) is True
    assert sh_steps_at(20048) is False and sh_steps_at(20096)


def test_scene_lookup():
    assert resolve(138766, "base")[0] == "garden"
    assert resolve(1, "big") == (None, ARG_DEFAULTS)
