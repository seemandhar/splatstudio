import numpy as np
import pytest
import torch
from scipy.ndimage import uniform_filter
from splatstudio import ComponentSpec

from ss_opti3dgs import method, method_filter
from ss_opti3dgs.filtering import kernel_at, mean_blur
from ss_opti3dgs.losses import DEFAULTS, create_photometric


def test_mean_blur_matches_scipy():
    img = torch.rand(30, 40, 3, generator=torch.Generator().manual_seed(0))
    for k in (3, 5, 9, 15):
        ref = uniform_filter(img.numpy().astype(np.float64), size=(k, k, 1), mode="mirror")
        assert np.abs(mean_blur(img, k).numpy() - ref).max() < 1e-5


def test_mean_blur_identity_and_errors():
    img = torch.rand(20, 20, 3)
    assert mean_blur(img, 1) is img
    with pytest.raises(ValueError):
        mean_blur(img, 4)
    with pytest.raises(ValueError):
        mean_blur(torch.rand(10, 10, 3), 21)


def test_zero_padding_darkens_borders():
    img = torch.ones(20, 20, 3)
    assert torch.allclose(mean_blur(img, 5, "reflect"), img)
    assert mean_blur(img, 5, "zeros")[0, 0, 0] < 1


def test_schedule():
    s, b = DEFAULTS["kernel_sizes"], DEFAULTS["boundaries"]
    assert [kernel_at(t, s, b) for t in (1, 1000, 1001, 4001, 8001, 12000, 12001)] == \
        [15, 15, 9, 5, 3, 3, 1]


def test_loss_requires_step_and_blurs():
    loss = create_photometric(ComponentSpec("photometric", "loss", "x", (), (), {}))
    gt = torch.rand(32, 32, 3)
    rgb = mean_blur(gt, 15)
    with pytest.raises(KeyError):
        loss({}, {"rgb": rgb}, gt)
    assert loss({}, {"rgb": rgb, "step": 1}, gt) < 1e-6
    assert loss({}, {"rgb": rgb, "step": 12001}, gt) > 1e-3


def test_specs():
    a, b = method(), method_filter()
    da = next(c for c in a.components if c.kind == "density").config
    db = next(c for c in b.components if c.kind == "density").config
    assert da == {"scales_lr": 0.01, "refine_every": 500, "grow_grad2d": 0.0001}
    assert db == {"scales_lr": 0.005, "refine_every": 100, "grow_grad2d": 0.0002}
