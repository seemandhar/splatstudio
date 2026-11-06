import torch

from ss_group_training import method, probes, smoke2k
from ss_group_training.density import grouping_iterations, is_merge_only, under_training_mask


def test_schedule_constants():
    c = next(x for x in method().components if x.kind == "density").config
    its = grouping_iterations(c)
    assert min(its) == 1000 and max(its) == 29000 and len(its) == 57
    assert is_merge_only(c, 14500) and is_merge_only(c, 29000) and not is_merge_only(c, 15000)


def test_ops_exact_count_and_bias():
    logits = torch.logit(torch.tensor([0.99] * 50 + [0.01] * 50))
    m = under_training_mask(logits, 0.6)
    assert int(m.sum()) == 60 and int(m[:50].sum()) == 50


def test_random_sampling_is_bernoulli():
    m = under_training_mask(torch.zeros(10000), 0.6, "random")
    assert abs(float(m.float().mean()) - 0.6) < 0.03


def test_probes_cpu():
    probes.ops_sampling(method(), "cpu")
    probes.subset_state(method(), "cpu")
    probes.schedule(smoke2k(), "cpu")


def test_full_schedule_cpu():
    out = probes.schedule(method(), "cpu")
    assert out["merge_only"] == [14500, 29000]
    assert out["sh_frozen_steps"] == 1 + 1001
