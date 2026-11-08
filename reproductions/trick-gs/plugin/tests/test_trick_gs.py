import numpy as np
import torch

from ss_trick_gs import method, probes, schedule
from ss_trick_gs.density import DEFAULTS, knn3_log_scales, lowest_error_subset
from ss_trick_gs.losses import blur_image, downsample_target, sh_mask_term
from ss_trick_gs.renderer import expand_sh, ste_mask


def test_spec_shape():
    m = method()
    kinds = [c.kind for c in m.components]
    assert kinds.count("renderer") == 1 and kinds.count("density") == 1
    assert {f.name for f in m.fields} == {"trick_mask", "trick_sh_mask"}
    assert [f.learning_rate for f in m.fields] == [0.5, 0.05]


def test_ds_schedule_eq6():
    assert schedule.ds_size(0, 800, 1600) == (100, 200)
    assert schedule.ds_size(19500, 800, 1600) == (800, 1600)
    w = [schedule.ds_size(s, 800, 1600)[1] for s in range(0, 19500, 500)]
    assert w == sorted(w)
    assert schedule.ds_scale(9750) == 0.125 + 0.875 * 0.5


def test_blur_schedule():
    assert schedule.blur_params(1, 1.0) == (4, 2.4)
    assert schedule.blur_params(19499, 1.0) is None or schedule.blur_params(19499, 1.0)[0] >= 1
    assert schedule.blur_params(20000, 1.0) is None
    img = torch.rand(12, 14, 3)
    out = blur_image(img, 1, 0.8)
    assert out.shape == img.shape and abs(out.mean() - img.mean()) < 0.05
    assert downsample_target(img, 6, 7).shape == (6, 7, 3)


def test_masks_and_terms():
    logits = torch.tensor([-5.0, 0.0, 3.0])
    assert ste_mask(logits, 0.05).tolist() == [0.0, 1.0, 1.0]
    assert expand_sh(torch.ones(2, 3)).shape == (2, 15, 1)
    p = {"trick_sh_mask": torch.zeros(4, 3)}
    assert abs(float(sh_mask_term(p)) - 0.5) < 1e-6


def test_init_subset_and_knn():
    keep = lowest_error_subset(np.array([5.0, 1.0, 3.0, 0.5, 2.0]), 0.4)
    assert keep.tolist() == [1, 3]
    pts = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [5, 5, 5]])
    s = knn3_log_scales(pts, budget=7)
    assert s.shape == (5, 3) and torch.isfinite(s).all()


def test_cpu_probes():
    m = method()
    probes.accelerated(m, "cpu")
    probes.redensify(m, "cpu")
    assert DEFAULTS["sg_steps"][0] == 5000
