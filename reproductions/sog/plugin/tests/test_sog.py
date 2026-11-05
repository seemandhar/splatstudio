import math

import numpy as np
import pytest
import torch

from ss_sog import codec, method, smoke
from ss_sog.density import DEFAULTS, schedule_events
from ss_sog.losses import as_grid_img, grid_side
from ss_sog.probes import neighbor_loss, schedule
from ss_sog.renderer import inverse_log_transform, log_transform
from ss_sog.sorting import neighbour_distance, sort_features, standardize


def test_log_transform_roundtrip_and_formula():
    x = torch.linspace(-100, 100, 2001, dtype=torch.float64)
    torch.testing.assert_close(inverse_log_transform(log_transform(x)), x)
    assert float(inverse_log_transform(torch.tensor(1.0))) == pytest.approx(math.e - 1)
    assert float(log_transform(torch.tensor(-(math.e - 1)))) == pytest.approx(-1.0)


def test_schedule_events():
    cfg = dict(DEFAULTS)
    assert schedule_events(500, cfg) == []
    assert schedule_events(1000, cfg) == ["sort"]
    assert schedule_events(15000, cfg) == ["sort"]
    assert schedule_events(29000, cfg) == ["sort"]
    assert schedule_events(30000, cfg) == ["encode"]
    assert schedule_events(1500, cfg) == []
    assert schedule_events(2000, {**cfg, "total_steps": 2000}) == ["encode"]


def test_quantize_matches_official_formula():
    a = np.linspace(-3, 5, 1000, dtype=np.float32).reshape(10, 10, 10)
    q = codec.quantize(a, 6)
    assert q.dtype == np.float32
    levels = np.unique(np.round((q - a.min()) / (a.max() - a.min()) * 64, 4))
    assert len(levels) <= 65 and np.allclose(levels, np.round(levels))
    assert np.abs(q - a).max() <= (a.max() - a.min()) / 128 + 1e-6


def test_normalize_thresholds():
    img = np.array([-10.0, 0.0, 20.0], dtype=np.float32)
    out, lo, hi = codec.normalize_to_thresholds(img, "_opacity")
    assert (lo, hi) == (-6, 12)
    np.testing.assert_allclose(out, [0.0, 6 / 18, 1.0], rtol=1e-6)
    out, lo, hi = codec.normalize_to_thresholds(img, "_xyz")
    assert lo == -10 and hi == 20


def test_compression_info_csv_format():
    rows = [("_xyz", np.float32(-1.5), np.float32(2.25), "_xyz.jxl"),
            ("_scaling", None, None, "_scaling.jxl"), ("_opacity", -6, 12, "_opacity.jxl")]
    text = codec.compression_info_csv(rows)
    assert text == (",min,max,file\n_xyz,-1.5,2.25,_xyz.jxl\n_scaling,,,_scaling.jxl\n"
                    "_opacity,-6,12,_opacity.jxl\n")
    assert codec.csv_float(np.float32(0.1)) == float("0.1")


def test_grid_helpers():
    assert grid_side(49) == 7
    with pytest.raises(ValueError):
        grid_side(50)
    t = torch.arange(16 * 3).reshape(16, 3)
    assert as_grid_img(t)[1, 2].tolist() == t[6].tolist()


def test_standardize_and_sort_features():
    t = torch.randn(100, 3) * 5 + 2
    s = standardize(t)
    assert abs(float(s.mean())) < 1e-5 and abs(float(s.std()) - 1) < 1e-5
    n = 64
    params = {"means": torch.randn(n, 3), "sh0": torch.randn(n, 1, 3), "shN": torch.randn(n, 15, 3),
              "opacities": torch.randn(n), "scales": torch.randn(n, 3), "quats": torch.randn(n, 4)}
    f = sort_features(params, DEFAULTS["sort_weights"])
    assert f.shape == (n, 9)
    ordered = torch.arange(n, dtype=torch.float32)[:, None]
    assert neighbour_distance(ordered, 8) < neighbour_distance(ordered[torch.randperm(n)], 8)


def test_neighbor_loss_probe_cpu():
    ev = neighbor_loss(method(), "cpu")
    assert ev["loss"] > 0


def test_schedule_probe_cpu():
    ev = schedule(method(), "cpu")
    assert ev["encode_step"] == [30000]
    assert ev["densify_steps"] == [1000, 14000, 14]
    assert ev["sort_steps"] == [1000, 29000, 29]
    assert schedule(smoke(), "cpu")["encode_step"] == [2000]
