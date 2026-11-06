import math

import numpy as np
import pytest
import torch

from ss_spotlesssplats import method, method_agg, method_noubp, probes
from ss_spotlesssplats.losses import histogram_threshold, positional_encoding, robust_mask


@pytest.mark.parametrize("build", [method, method_noubp, method_agg])
def test_cpu_probes(build):
    m = build()
    for p in m.probes:
        if p.requires_cuda:
            continue
        fn = getattr(probes, p.function.split(":")[1], None)
        if fn is None:
            continue  # built-in splatstudio probe
        assert isinstance(fn(m, "cpu"), dict)


def test_threshold_is_left_edge():
    h = torch.zeros(10)
    h[3] = 5
    h[7] = 5
    assert float(histogram_threshold(h, 0.5)) == pytest.approx(0.3)
    assert float(histogram_threshold(h, 0.51)) == pytest.approx(0.7)


def test_initial_thresholds_mask_everything_or_nothing():
    err = torch.rand(8, 8, 3) * 0.5
    assert robust_mask(err, torch.tensor(1.0)).all()
    assert not robust_mask(err, torch.tensor(0.0)).any()


def test_positional_encoding_layout():
    pe = positional_encoding(5, 7, 20, "cpu")
    assert pe.shape == (80, 5, 7)
    # channel 0 = sin(pi * y), y normalised to [0, 1]
    assert float(pe[0, 4, 0]) == pytest.approx(math.sin(math.pi), abs=1e-6)
    assert float(pe[40 + 20, 0, 6]) == pytest.approx(math.cos(math.pi), abs=1e-6)


def test_variants_differ():
    names = {c.name for c in method_agg().components}
    assert "sls-mlp" not in names
    d = next(c for c in method().components if c.name == "sls-density")
    assert d.config["ubp"] and "render.rgb_grad" in d.requires
    assert not next(c for c in method_noubp().components if c.name == "sls-density").config["ubp"]
    assert np.isclose(next(c for c in method().components if c.name == "sls-robust")
                      .config["robust_percentile"], 0.7)
