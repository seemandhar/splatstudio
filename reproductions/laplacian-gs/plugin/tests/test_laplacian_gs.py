import torch

from ss_laplacian_gs import method, probes, smoke_method
from ss_laplacian_gs import pyramid as P
from ss_laplacian_gs.density import equidistant_indices, knn_log_scales
from ss_laplacian_gs.renderer import final_degree, level_camera


def test_pyramid_probe():
    assert probes.lap_pyramid(method(), "cpu")["down_max_err"] < 1e-10


def test_staging_probe():
    ev = probes.lap_staging(method(), "cpu")
    assert ev["archived_drift"] == 0.0 and ev["inherited"] == 12


def test_loss_probe():
    assert probes.lap_loss(method(), "cpu")["exact_level_loss"] < 1e-5


def test_equidistant_and_knn():
    assert equidistant_indices(10, 10).tolist() == list(range(10))
    assert equidistant_indices(10, 5).tolist() == [0, 2, 4, 6, 8]
    pts = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
    s = knn_log_scales(pts)
    # point 0: neighbours at 1, 2, 3 -> mean sq = 14/3
    assert torch.allclose(s[0, 0].exp() ** 2, torch.tensor(14 / 3), atol=1e-5)


def test_schedules_and_camera():
    assert [final_degree(i) for i in (999, 1000, 2999, 10000)] == [0, 1, 2, 3]
    cam = {"K": torch.tensor([[100.0, 0, 50], [0, 100, 40], [0, 0, 1]]), "width": 100,
           "height": 80, "viewmat": torch.eye(4)}
    c = level_camera(cam, P.down_size(80, 100))
    assert (c["width"], c["height"]) == (50, 40) and float(c["K"][0, 0]) == 50.0


def test_smoke_variant_schedule():
    d = next(c for c in smoke_method().components if c.kind == "density").config
    assert d["iter_plan"] == [1000, 1000] and d["densify_until"] == [900, 900]
    d = next(c for c in method().components if c.kind == "density").config
    assert d["iter_plan"] == [10000, 20000] and d["densify_until"] is None
