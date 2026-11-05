import importlib.util
import math
import random
from pathlib import Path

import numpy as np
import torch

from ss_dngaussian import method, method_3dgs
from ss_dngaussian.density import expon_lr
from ss_dngaussian.depth import (
    DepthTable,
    depth_regularization,
    global_loss,
    local_loss,
    margin_l2,
    normalize,
    patchify,
)
from ss_dngaussian.losses import penalties
from ss_dngaussian.neural import HEADER, PACK_WIDTH

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "prepare_dng_llff.py"


def _tools():
    spec = importlib.util.spec_from_file_location("prep", TOOLS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_patchify_crops_remainder():
    x = torch.arange(7 * 11, dtype=torch.float32).view(1, 1, 7, 11)
    p = patchify(x, 5)
    assert p.shape == (2, 25)
    assert torch.equal(p[1].view(5, 5), x[0, 0, :5, 5:10])


def test_normalize_epsilon_is_relative():
    x = torch.randn(10, 25) * 7 + 3
    n = normalize(x)
    ref = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 0.01 * x.reshape(-1).std())
    assert torch.allclose(n, ref)


def test_margin_l2_tolerance_and_empty():
    a = torch.tensor([0.0, 1.0, 0.0001], requires_grad=True)
    b = torch.zeros(3)
    assert math.isclose(float(margin_l2(a, b, 0.001)), 1.0)
    z = margin_l2(a, b, 10.0)
    assert float(z) == 0.0 and z.requires_grad


def test_losses_affine_invariant():
    g = torch.Generator().manual_seed(0)
    d = torch.rand(1, 1, 30, 40, generator=g) * 10
    m = torch.rand(1, 1, 30, 40, generator=g)
    for fn in (local_loss, global_loss):
        assert math.isclose(float(fn(d, m, 6, 0.00025)), float(fn(5 * d + 2, -m * 0 + m * 9 - 1, 6, 0.00025)),
                            rel_tol=1e-3)


def test_depth_regularization_schedule():
    d = torch.rand(1, 1, 30, 40)
    m = torch.rand(1, 1, 30, 40)
    cfg = dict(next(c for c in method().components if c.name == "dngaussian-density").config)
    _, t1 = depth_regularization(d, m, 3000, random.Random(0), cfg)
    _, t2 = depth_regularization(d, m, 3001, random.Random(0), cfg)
    assert "smooth" not in t1 and "smooth" in t2


def test_expon_lr():
    assert math.isclose(expon_lr(0, 0.016, 0.00016, 5500), 0.016)
    assert math.isclose(expon_lr(5500, 0.016, 0.00016, 5500), 0.00016)
    assert math.isclose(expon_lr(9999, 0.016, 0.00016, 5500), 0.00016)
    assert math.isclose(expon_lr(2750, 0.016, 0.00016, 5500), 0.0016)


def test_penalties_empty_selection_finite():
    s = torch.zeros(3, 3, requires_grad=True)
    shape, scale, opa = penalties(s, torch.full((3,), 0.1))
    assert all(math.isfinite(float(v)) for v in (shape, scale, opa))


def test_depth_table_lookup():
    vm = np.eye(4, dtype=np.float32)[None]
    t = DepthTable(vm, np.ones((1, 4, 5), np.float32), np.zeros((2, 3)))
    assert t.lookup(torch.eye(4), "cpu").shape == (1, 1, 4, 5)
    try:
        t.lookup(torch.eye(4) * 2, "cpu")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown camera must raise")


def test_pack_capacity():
    assert HEADER + 64 * 80 + 3 * 64 <= 665 * PACK_WIDTH


def test_specs():
    for m in (method(), method_3dgs()):
        names = {c.name for c in m.components}
        assert {"dngaussian-renderer", "dngaussian-density", "photometric"} <= names
    dens = next(c for c in method_3dgs().components if c.kind == "density").config
    assert dens["hard_depth"] is False and dens["soft_depth"] is False


def test_official_random_init():
    prep = _tools()
    rng = np.random.default_rng(1)
    xyz = rng.normal(size=(500, 3)) * [10, 5, 20]
    pts, colors, n = prep.official_random_init(xyz)
    shape = xyz.max(0) - xyz.min(0)
    assert n == int(shape.max() * 50) and len(pts) == n
    lo = np.sort(xyz, axis=0)[19]
    assert np.all(pts >= lo - 1e-9) and np.all(pts <= lo + 1.3 * shape + 1e-9)
    np.random.seed(0)
    assert np.allclose(pts, np.random.random((n, 3)) * shape * 1.3 + lo)
    assert np.all((colors > 0.49) & (colors < 0.51))


def test_spiral_shape():
    prep = _tools()
    n = 6
    poses = np.zeros((n, 3, 5))
    for i in range(n):
        poses[i, :, :3] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])  # LLFF [down,right,back]
        poses[i, :, 3] = [i * 0.1, 0.0, 0.0]
        poses[i, :, 4] = [378, 504, 400]
    pb = np.concatenate([poses.reshape(n, 15), np.tile([[10.0, 100.0]], (n, 1))], 1)
    c = prep.official_spiral_centers(pb)
    assert c.shape == (180, 3) and np.isfinite(c).all()
    assert np.linalg.norm(c - poses[:, :, 3].mean(0), axis=1).max() < 1.0
