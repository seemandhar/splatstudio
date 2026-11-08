import math

import torch

from ss_lightgaussian import codec, method, released, schedule, significance, smoke, vq
from ss_lightgaussian.density import DEFAULTS


def test_schedule_boundaries():
    c = DEFAULTS
    assert schedule.stage(30000, c) == "3dgs"
    assert schedule.stage(30001, c) == "coadapt"
    assert schedule.stage(35001, c) == "distill"
    assert schedule.stage(40001, c) == "vqft"
    assert schedule.stage(45001, c) == "done"
    assert schedule.events(30000, c) == ["prune"]
    assert schedule.events(40000, c) == ["vq", "start_vqft"]
    assert schedule.events(45000, c) == ["encode"]
    r = {**c, "total_steps": 40000}
    assert schedule.events(40000, r) == ["vq", "encode"]


def test_lr_factors_and_decay():
    c = DEFAULTS
    assert schedule.other_lr_factor(30399, c) == 1.0
    assert math.isclose(schedule.other_lr_factor(35000, c), 0.95 ** 12)
    assert math.isclose(schedule.other_lr_factor(40000, c), 0.9 ** 10)
    assert schedule.weight_decay(30500, c) == 0.0
    assert schedule.weight_decay(36000, c) == 0.01
    assert math.isclose(schedule.means_lr(1.0, 35000, c), 0.01)
    assert math.isclose(schedule.means_lr(1.0, 1, c), 1.0)


def test_pseudo_view_cadence():
    c = DEFAULTS
    assert [schedule.pseudo_view(k, c) for k in (35001, 35002, 35003)] == [False, True, True]
    assert not schedule.pseudo_view(34999, c)


def test_v_importance_and_prune():
    score = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    log_scales = torch.zeros(10, 3)
    v = significance.v_importance(score, log_scales, 0.1)
    assert torch.allclose(v, score)
    mask = significance.prune_mask(v, 0.66)
    assert int(mask.sum()) == int(0.66 * 9) + 1


def test_sh_feature_roundtrip():
    sh0 = torch.randn(5, 1, 3)
    shN = torch.randn(5, 15, 3)
    f = vq.sh_features(sh0, shN, 8)
    assert f.shape == (5, 27)
    assert torch.equal(f[:, 3:11], shN[:, :8, 0])  # PLY order: channel-major f_rest
    a, b = vq.from_sh_features(f, 8)
    assert torch.equal(a, sh0) and torch.equal(b, shN[:, :8])


def test_ema_step_fixed_point():
    embed = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    cs = torch.tensor([5.0, 5.0])
    x = torch.tensor([[0.0, 0.0]] * 5 + [[10.0, 10.0]] * 5)
    vq.ema_step(embed, cs, x, torch.ones(10), 0.8)
    assert torch.allclose(embed, torch.tensor([[0.0, 0.0], [10.0, 10.0]]), atol=1e-4)


def test_codec_roundtrip_cpu():
    g = torch.Generator().manual_seed(0)
    n, k = 100, 16
    non_vq = torch.zeros(n, dtype=torch.bool)
    non_vq[:40] = True
    idx = torch.randint(0, k, (60,), generator=g)
    data, files = codec.encode(torch.randn(n, 3), torch.randn(n, 27), torch.randn(n, 8),
                               torch.randn(k, 27), idx, non_vq)
    dec = codec.decode(data, "cpu")
    assert torch.equal(dec["indices"], idx) and torch.equal(dec["non_vq_mask"], non_vq)
    assert len(files) == 7


def test_specs():
    for m in (method(), released(), smoke()):
        names = [c.name for c in m.components]
        assert names == ["lightgaussian-renderer", "lightgaussian-density", "photometric",
                         "lightgaussian-distill"]
    assert next(c for c in released().components if c.kind == "density").config[
        "total_steps"] == 40000
