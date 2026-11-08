import math

import numpy as np
import pytest
import torch

from ss_perceptual_gs import core, probes, sidecar


def test_sensitivity_map_matches_reference():
    rng = np.random.default_rng(1)
    lum = rng.uniform(0.3, 0.7, size=(23, 31))
    lum[:, 15:] += 0.2
    got = core.sensitivity_map(torch.tensor(lum, dtype=torch.float32)[None, None])[0, 0].numpy()
    ref, _ = probes._reference_map(lum, 0.05, 0.3, 5)
    assert (got == ref).all()


def test_opacity_decline_composites_to_power():
    a = torch.tensor([0.05, 0.3, 0.6, 0.99])
    hat = core.opacity_decline(a, 1.2)
    assert torch.allclose(hat, 1 - torch.sqrt(1 - a**1.2), atol=1e-6)
    assert torch.allclose(hat + (1 - hat) * hat, a**1.2, atol=1e-6)
    assert (hat <= a).all()


def test_sparse_indicator_and_beta():
    m = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    s = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.95, 0.5])
    assert core.sparse_indicator(m, s) == 0.5  # top 2: {8 -> 0.5 mixed, 7 -> 0.95}
    maps = [torch.tensor([[0, 1], [1, 1]]), torch.tensor([[0, 0], [0, 3]])]
    assert core.scene_sensitivity(maps) == pytest.approx((0.75 + 0.25) / 2)


def test_depth_sample_count():
    assert core.depth_samples_per_view(822, 1236, 169, 3_500_000) == int(3_500_000 / 169)


def test_table_lookup():
    vm = torch.eye(4)[None].repeat(2, 1, 1)
    vm[1, 0, 3] = 1.0
    maps = torch.zeros(2, 3, 4, dtype=torch.uint8)
    maps[1] = 255
    t = sidecar.Table(vm, maps)
    assert t.sens(vm[1]).min() == 1.0 and t.sens(vm[0]).max() == 0.0
    with pytest.raises(KeyError):
        t.sens(vm[0] + 0.1)


def test_logit():
    assert math.isclose(1 / (1 + math.exp(-core.logit(0.1))), 0.1)


def test_sidecar_dir_relative_to_datasets():
    d = sidecar.sidecar_dir(sidecar.DATASETS_HOME / "mipnerf360/pilr/bicycle", "/x")
    assert str(d) == "/x/mipnerf360/pilr/bicycle"
