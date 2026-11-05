import numpy as np
import torch

from ss_sparsegs import method, method_3dgs, probes
from ss_sparsegs.context import Context, construct_rotation_matrix, forward_warp
from ss_sparsegs.density import calc_diff, expon_lr, prune_percentile
from ss_sparsegs.losses import local_pearson_loss, n_patches, pearson_loss
from ss_sparsegs.sds import fixed_timestep


def test_cpu_probes():
    m = method()
    probes.pearson(m, "cpu")
    probes.warp(m, "cpu")
    probes.sds(m, "cpu")


def test_specs():
    m = method()
    names = [c.name for c in m.components]
    assert names == ["sparsegs-renderer", "sparsegs-density", "photometric", "sparsegs-pearson",
                     "sparsegs-warp", "sparsegs-sds"]
    c = method_3dgs()
    assert [x.name for x in c.components] == ["sparsegs-renderer", "sparsegs-density",
                                               "photometric"]
    assert dict(c.components[1].config)["prune"] is False


def test_pearson_numerics():
    a = torch.rand(40, 50)
    assert abs(float(pearson_loss(2 * a + 1, a)) - 1 / a.numel()) < 1e-5
    assert n_patches(378, 504, 32, 0.5) == int(0.5 * 11 * 15)
    g = torch.Generator().manual_seed(0)
    assert float(local_pearson_loss(-a, a, 8, 0.5, g)) > 1.9


def test_lr_and_schedule_helpers():
    assert abs(expon_lr(0, 1.6e-4, 1.6e-6, 30000) - 1.6e-4) < 1e-12
    assert abs(expon_lr(30000, 1.6e-4, 1.6e-6, 30000) - 1.6e-6) < 1e-12
    assert abs(expon_lr(10000, 1.0, 0.01, 30000) - 0.01 ** (1 / 3)) < 1e-9
    assert fixed_timestep(0.99) == 20
    assert prune_percentile(0.0, 0.97, 7.5, 80) == 97.0
    assert prune_percentile(0.5, 0.97, 7.5, 80) == 80.0


def test_calc_diff_sign():
    depth = torch.full((4, 4), 3.0)
    mode = torch.full((4, 4), 3.0)
    mode[0, 0] = 1.0  # floater in front -> positive diff
    d = calc_diff(mode, depth)
    assert float(d[0, 0]) > 0 and float(d[1, 1]) == 0


def test_rotation_is_orthonormal():
    R = construct_rotation_matrix(np.array([0.0, 1.0, 0.0]), np.pi / 7)
    assert np.allclose(R @ R.T, np.eye(3))


def test_warp_last_write_wins():
    # pixels 0 (far) and 1 (near) both land on target pixel 0; raster order -> pixel 1 wins
    img = torch.tensor([[[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]])
    depth = torch.tensor([[1.0, 0.0, 1.0]])
    K = torch.tensor([[1.0, 0, 1.5], [0, 1.0, 0.5], [0, 0, 1]])
    trg = torch.eye(4)
    trg[0, 3] = -1.0
    out, mask = forward_warp(img, depth, K, torch.eye(4), trg, 1.0, 100.0)
    assert torch.equal(out[0, 0], img[0, 1])
    assert mask.tolist() == [[1.0, 0.0, 1.0]]


def test_draws_order_and_gating():
    ctx = Context()
    ctx.draw_cfg.update(sds_enabled=True, sds_start=10, warp_enabled=True, warp_start=5,
                        sds_freq=1.0, warp_prob=1.0)
    assert ctx.draws(3) == (False, False)
    assert ctx.draws(6) == (False, True)
    assert ctx.draws(11) == (True, True)
    assert ctx.draws(11) == (True, True)  # cached per step
