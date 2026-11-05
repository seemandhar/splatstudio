import math

import numpy as np
import torch

from ss_pup3dgs import fisher, method, schedule, v1


def official_expon(lr_init, lr_final, max_steps, step):
    t = np.clip(step / max_steps, 0, 1)
    return float(np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t))


def test_pretrain_lr_matches_30k_run():
    for s in (1, 500, 15000, 29999, 30000):
        assert math.isclose(schedule.pretrain_means_lr(2.0, s, 30000), 2.0 * 0.01 ** (s / 30000))
        assert math.isclose(
            schedule.pretrain_means_lr(1.6e-4, s, 30000), official_expon(1.6e-4, 1.6e-6, 30000, s)
        )


def test_finetune_lrs():
    lr0 = 1.6e-4 * 3.7
    for it in (30001, 30399, 30401, 34999, 35000):
        want = official_expon(lr0, lr0 * 0.01, 35000, it) * (0.95 if it % 400 == 0 else 1)
        assert math.isclose(schedule.finetune_means_lr(lr0, it, 35000), want, rel_tol=1e-12)
    assert schedule.exp_decay_count(30001, 30000, 400) == 0
    assert schedule.exp_decay_count(30399, 30000, 400) == 0
    assert schedule.exp_decay_count(30400, 30000, 400) == 1
    assert schedule.exp_decay_count(35000, 30000, 400) == 12
    assert math.isclose(schedule.finetune_other_lr(0.05, 35000, 30000), 0.05 * 0.95**12)
    assert schedule.finetune_lr_iter(35001, 35000, 30000) == 30001


def test_prune_mask_official_rule():
    scores = torch.tensor([5.0, -math.inf, 1.0, 3.0, 2.0, 4.0, 0.0, 6.0, 7.0, 8.0])
    mask = fisher.prune_mask(scores, 0.8)  # k = int(0.8*9) = 7 -> prune 8 lowest
    assert int(mask.sum()) == 8 and not mask[9] and not mask[8]
    tied = torch.tensor([1.0, 1.0, 1.0, 2.0])
    assert int(fisher.prune_mask(tied, 0.5).sum()) == 3  # ties at the threshold are pruned


def test_expand_sym_order():
    m = torch.randn(6, 6)
    m = m @ m.T
    sym = torch.stack([m[r, c] for r, c in fisher.TRIU])[None]
    assert torch.allclose(fisher.expand_sym(sym)[0], m)
    score = fisher.log_det_scores(sym)
    assert torch.allclose(score, torch.logdet(m)[None], atol=1e-3)


def test_camera_conversion_matches_pinhole():
    K = torch.tensor([[300.0, 0, 161.3], [0, 290.0, 99.2], [0, 0, 1]])
    c, s = math.cos(0.3), math.sin(0.3)
    viewmat = torch.eye(4)
    viewmat[:3, :3] = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    viewmat[:3, 3] = torch.tensor([0.2, -0.1, 2.0])
    cam = {"K": K, "viewmat": viewmat, "width": 320, "height": 200}
    world_view, full_proj, campos, tx, ty = fisher.raster_matrices(cam)
    x = torch.tensor([0.3, -0.2, 1.5, 1.0])
    p = x @ full_proj
    ndc = p[:2] / p[3]
    pix3dgs = ((ndc + 1) * torch.tensor([320.0, 200.0]) - 1) / 2  # 3DGS ndc2Pix, integer centres
    xc = (viewmat @ x)[:3]
    pinhole = (K @ (xc / xc[2]))[:2] - 0.5  # gsplat convention, centres at +0.5
    assert torch.allclose(pix3dgs, pinhole, atol=1e-3)
    assert torch.allclose(campos, torch.linalg.inv(viewmat)[:3, 3])
    assert math.isclose(tx, 320 / 600, rel_tol=1e-6) and math.isclose(ty, 200 / 580, rel_tol=1e-6)
    low = fisher.low_res_camera(cam, 4)
    assert (low["width"], low["height"]) == (80, 50)
    assert torch.allclose(low["K"][0], K[0] * 0.25) and torch.allclose(low["K"][1], K[1] * 0.25)


def test_specs():
    for m, ratios in ((method(), [0.8, 0.5]), (v1(), [0.66, 0.66])):
        d = next(c for c in m.components if c.kind == "density")
        assert d.config["prune_ratios"] == ratios
        assert d.config["prune_steps"] == [30000, 35000] and d.config["fisher_resolution"] == 4
