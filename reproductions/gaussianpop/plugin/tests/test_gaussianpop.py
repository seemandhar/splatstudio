import math

import pytest
import torch

from ss_gaussianpop import schedule


def test_keep_counts_compound():
    k = schedule.keep_counts(1000, 0.8, 8)
    assert len(k) == 8 and k[-1] == 200
    r = 1 - 0.2 ** (1 / 8)
    assert k[0] == round(1000 * (1 - r))
    assert all(a > b for a, b in zip(k, k[1:]))


def test_keep_counts_linear():
    assert schedule.keep_counts(1000, 0.8, 4, "linear") == [800, 600, 400, 200]


def test_keep_counts_invalid():
    with pytest.raises(ValueError):
        schedule.keep_counts(10, 1.0, 8)
    with pytest.raises(ValueError):
        schedule.keep_counts(10, 0.5, 2, "nope")


def test_lowest_mask_stable_ties():
    s = torch.tensor([0.0, 3.0, 0.0, 1.0, 2.0])
    m = schedule.lowest_mask(s, 3)
    assert m.tolist() == [True, False, True, True, False]
    assert not schedule.lowest_mask(s, 0).any()


def test_means_lr():
    assert math.isclose(schedule.pretrain_means_lr(1.0, 15000, 30000), 0.1)
    assert math.isclose(schedule.pretrain_means_lr(1.0, 33000, 30000), 0.01)


def test_eq4_identity():
    # ||C - C'||^2 == ||T a (c - b)||^2 for a random front-to-back list (pure torch).
    g = torch.Generator().manual_seed(0)
    a = torch.rand(6, generator=g) * 0.6
    c = torch.rand(6, 3, generator=g)
    bg = torch.ones(3)

    def comp(alpha, col):
        T, out = 1.0, torch.zeros(3)
        for ai, ci in zip(alpha, col):
            out = out + T * ai * ci
            T = T * (1 - ai)
        return out + T * bg

    C = comp(a, c)
    T = torch.cumprod(torch.cat((torch.ones(1), 1 - a)), 0)
    P = torch.cumsum((T[:-1] * a)[:, None] * c, 0)
    for k in range(6):
        b = (C - P[k]) / T[k + 1]
        se = ((T[k] * a[k] * (c[k] - b)) ** 2).sum()
        keep = [i for i in range(6) if i != k]
        ref = ((C - comp(a[keep], c[keep])) ** 2).sum()
        assert torch.allclose(se, ref, atol=1e-6)
