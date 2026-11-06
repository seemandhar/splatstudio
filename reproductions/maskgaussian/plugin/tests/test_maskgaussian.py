import math

import pytest
import torch

import ss_maskgaussian as mg
from ss_maskgaussian import probes
from ss_maskgaussian.density import mask_prune_steps, never_sampled
from ss_maskgaussian.losses import lambda_at
from ss_maskgaussian.renderer import presence_probability, sample_mask


@pytest.mark.parametrize("factory", [mg.method, mg.beta, mg.gamma, mg.opacitymask, mg.exactgrad])
def test_specs_build(factory):
    m = factory()
    assert [f.name for f in m.fields] == ["mask_scores"]
    assert m.fields[0].learning_rate == 0.01


def test_settings():
    cfg = {c.name: c.config for c in mg.method().components}["maskgs-mask-loss"]
    assert (cfg["lambda_mask"], cfg["mask_from"], cfg["mask_until"]) == (0.1, 19000, 20000)
    assert lambda_at(18999, cfg) == 0 and lambda_at(19000, cfg) == 0.1
    assert lambda_at(20000, cfg) == 0.1 and lambda_at(20001, cfg) == 0


def test_init_probability():
    p = presence_probability(torch.zeros(3, 2), [10.0, 1.0])
    assert torch.allclose(p, torch.full((3,), 1 / (1 + math.exp(-9))))


def test_sample_binary_and_never_sampled():
    torch.manual_seed(0)
    s = torch.tensor([[0.0, 0.0], [-40.0, 0.0]]) - torch.tensor([10.0, 1.0])
    m = sample_mask(s.repeat(50, 1), [10.0, 1.0])
    assert set(m.unique().tolist()) <= {0.0, 1.0}
    ns = never_sampled(s, [10.0, 1.0], 1.0, 10)
    assert ns.tolist() == [False, True]


def test_mask_prune_steps():
    cfg = {c.name: c.config for c in mg.method().components}["maskgs-density"]
    steps = mask_prune_steps(cfg, 30000)
    assert steps[0] == 15000 and steps[-1] == 30000 and len(steps) == 16


def test_cpu_probes():
    for fn in (probes.gumbel, probes.mask_loss, probes.prune_schedule):
        fn(mg.method(), "cpu")
