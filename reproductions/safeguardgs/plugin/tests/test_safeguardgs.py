import pytest
import torch

from ss_safeguardgs import core, method
from ss_safeguardgs.density import DEFAULTS, runtime_sh_degree


def test_sg36_score_formula():
    gt = torch.tensor([[1.0, 0.0, 0.0]])
    c = torch.tensor([[0.0, 0.0, 1.0]])
    a, T = torch.tensor([0.6]), torch.tensor([0.5])
    s = core.score(0x24, torch.tensor([0.7]), a, T, torch.zeros(1), torch.zeros(1), gt, c)
    assert torch.allclose(s, torch.tensor([0.6 * 0.5 * (1 - 2 / 3)]))
    s4 = core.score(0x04, torch.tensor([0.7]), a, T, torch.zeros(1), torch.zeros(1), gt, c)
    assert torch.allclose(s4, torch.tensor([0.3]))
    # CS2 and pixel closeness
    s52 = core.score(0x34, torch.tensor([0.7]), a, T, torch.zeros(1), torch.zeros(1), gt, c)
    assert torch.allclose(s52, 0.3 * torch.exp(torch.tensor([-2 / 3])))
    s9 = core.score(0x09, torch.tensor([0.7]), a, T, torch.tensor([3.0]), torch.tensor([4.0]),
                    gt, c)
    assert torch.allclose(s9, 0.3 * torch.exp(torch.tensor([-5.0])))
    with pytest.raises(ValueError):
        core.score(0x60, a, a, T, a, a, gt, c)


def test_quantize_matches_official():
    q = core.quantize(torch.tensor([0.0, 1.0, 0.5, 1.5, -0.1]))
    assert q.tolist() == [65535, 0, 32767, 0, 72088]


def test_padded_offsets():
    offs = torch.tensor([0, 10, 10, 300])
    pad, slots, n = core.padded_offsets(offs, 301, 256)
    assert slots.tolist() == [256, 0, 512, 256]
    assert pad.tolist() == [0, 256, 256, 768]
    assert n == 1024


def test_tile_topk_keep():
    # two tiles of 4 slots; lower q = better; ties broken by slot order (stable sort)
    q = torch.tensor([5, -1, 3, 3, 7, 2, -1, 9])
    ids = torch.tensor([0, -1, 1, 2, 3, 4, -1, 5])
    slots = torch.tensor([4, 4])
    k1 = core.tile_topk_keep(q, ids, slots, 1, 6)
    assert k1.tolist() == [False, True, False, False, True, False]
    k2 = core.tile_topk_keep(q, ids, slots, 2, 6)
    assert k2.tolist() == [False, True, True, True, True, False]


def test_defaults_and_spec():
    assert DEFAULTS["prune_steps"] == [20000]
    assert DEFAULTS["top_k"] == 1 and DEFAULTS["score_function"] == 0x24
    assert runtime_sh_degree(20000) == 3 and runtime_sh_degree(1) == 0
    spec = method()
    density = next(c for c in spec.components if c.kind == "density")
    assert "training.target_image" in density.requires
    assert density.config["selection"] == "tile"
