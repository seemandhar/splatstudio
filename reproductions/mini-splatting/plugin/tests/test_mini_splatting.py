import math

import torch

from ss_mini_splatting import core, densified, indoor, method
from ss_mini_splatting.kernels import quat_to_rotmat


def test_methods_build_specs():
    for m in (method(), indoor(), densified()):
        dens = next(c for c in m.components if c.kind == "density")
        assert set(dens.requires) == {"gradient.screen.signed", "training.target_image"}
    assert next(c for c in indoor().components if c.kind == "density").config["imp_metric"] == (
        "indoor"
    )
    d = next(c for c in densified().components if c.kind == "density").config
    assert d["simplify"] is False and d["num_max"] is None and d["depth_with_replacement"]


def test_lr_schedule_matches_official():
    lr0 = 1.0
    assert math.isclose(core.means_lr(0, lr0), 1.0)
    assert math.isclose(core.means_lr(14999, lr0), 0.01 ** (14999 / 30000))
    # iteration >= simp_iteration1 uses lr(iteration - 15000 + 5000)
    assert math.isclose(core.means_lr(15000, lr0), 0.01 ** (5000 / 30000))
    assert math.isclose(core.means_lr(30000, lr0), 0.01 ** (20000 / 30000))
    assert math.isclose(core.expon_lr(60000, lr0), 0.01)


def test_sh_degree_schedule():
    assert core.sh_degree(1) == 0 and core.sh_degree(15000) == 0 and core.sh_degree(15999) == 0
    assert core.sh_degree(16000) == 1 and core.sh_degree(17000) == 2
    assert core.sh_degree(18000) == 3 and core.sh_degree(29999) == 3
    assert [core.active_rest_coeffs(d) for d in range(4)] == [0, 3, 8, 15]


def test_weighted_sample_without_replacement_distribution():
    g = torch.Generator().manual_seed(0)
    w = torch.tensor([0.0, 1.0, 2.0, 7.0])
    counts = torch.zeros(4)
    first = torch.zeros(4)
    for _ in range(4000):
        idx = core.weighted_sample(w, 2, generator=g)
        assert len(set(idx.tolist())) == 2 and 0 not in idx.tolist()
        counts[idx] += 1
    # successive sampling: P(first draw = i) = w_i / sum
    for _ in range(4000):
        first[core.weighted_sample(w, 1, generator=g)] += 1
    assert torch.allclose(first / 4000, w / w.sum(), atol=0.03)
    # P(i in sample of 2) for weights 1,2,7 (exact successive-sampling inclusion probabilities)
    p = w / w.sum()
    incl = [p[i] + sum(p[j] * p[i] / (1 - p[j]) for j in range(1, 4) if j != i) for i in range(4)]
    assert torch.allclose(counts[1:] / 4000, torch.tensor(incl[1:]), atol=0.03)
    assert len(core.weighted_sample(w, 10)) == 3  # clipped to the positive entries


def test_weighted_sample_with_replacement_unique():
    idx = core.weighted_sample(torch.tensor([0.0, 1.0, 1.0]), 50, replace=True)
    assert idx.tolist() == [1, 2]


def test_cdf_keep_mask_official():
    imp = torch.tensor([0.0, 0.001, 0.004, 0.3, 0.7, 5.0])
    keep = core.cdf_keep_mask(imp, 0.99)
    vals = torch.sort(imp + 1e-6).values
    frac = torch.cumsum(vals, 0) / vals.sum()
    split = vals[int(torch.nonzero(frac > 0.01)[0])]
    assert torch.equal(keep, imp > split)
    assert not keep[0] and keep[-1]
    assert core.cdf_keep_mask(imp, 1.0).all()


def test_importance_metrics():
    view = {
        "accum_weights": torch.tensor([2.0, 1.0, 0.5]),
        "area_proj": torch.tensor([4, 2, 1], dtype=torch.int32),
        "area_max": torch.tensor([3, 0, 1], dtype=torch.int32),
    }
    imp, seen = core.accumulate_importance(torch.zeros(3), torch.zeros(3), view, "outdoor")
    assert torch.allclose(imp, torch.tensor([0.5, 0.0, 0.5])) and seen.tolist() == [3, 0, 1]
    imp, _ = core.accumulate_importance(torch.zeros(3), torch.zeros(3), view, "indoor")
    assert torch.allclose(imp, torch.tensor([2.0, 1.0, 0.5]))
    assert core.sampling_count(torch.tensor([0.0, 1, 2, 3]), 0.5) == 1


def test_depth_samples_per_view():
    assert core.depth_samples_per_view(840, 1297, 161, 3_500_000) == int(3_500_000 / 161)


def test_midpoint_reference_is_density_maximum():
    rot = quat_to_rotmat(torch.nn.functional.normalize(torch.tensor([[0.9, 0.2, -0.3, 0.1]])))[0]
    rot = rot.double()
    scale = torch.tensor([0.5, 0.2, 0.05], dtype=torch.float64)
    mean = torch.tensor([0.0, 0.1, 3.0], dtype=torch.float64)
    o = torch.zeros(3, dtype=torch.float64)
    d = torch.nn.functional.normalize(torch.tensor([0.1, 0.05, 1.0], dtype=torch.float64), dim=0)
    p, t = core.midpoint_reference(o, d, mean, rot, scale)
    prec = rot @ torch.diag(scale.pow(-2)) @ rot.T

    def q(tt):
        x = o + tt * d - mean
        return float(x @ prec @ x)

    assert q(float(t)) <= min(q(float(t) - 1e-3), q(float(t) + 1e-3))
    # ray / ellipsoid mid-point (App. eq.): t = -b / 2a in the Gaussian frame
    lo, ld = rot.T @ (o - mean) / scale, rot.T @ d / scale
    assert math.isclose(float(t), float(-(2 * lo @ ld) / (2 * ld @ ld)), rel_tol=1e-9)


def test_blend_reference_single_gaussian():
    out = core.blend_reference(torch.tensor([[1.5, 1.5]]), torch.tensor([[1.0, 0.0, 1.0]]),
                               torch.tensor([0.5]), torch.tensor([1.0]), 3, 3)
    assert out["area_max"][0] == out["area_proj"][0] == 9
    assert math.isclose(float(out["T"][1, 1]), 0.5, rel_tol=1e-9)
