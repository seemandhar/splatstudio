import pytest
import torch
from splatstudio.specs import ComponentSpec

from ss_gradient_upscaling import bicubic4x, control3dgs, probes, spline2x, spline4x
from ss_gradient_upscaling.renderer import UpscaleRenderer, low_res_camera
from ss_gradient_upscaling.upscale import central_differences, hermite_basis, spline_upscale


def test_paper_matrix():
    assert probes.spline_equals_paper_matrix(None, "cpu")["max_abs_diff"] < 1e-10


def test_catmull_rom():
    assert probes.spline_equals_catmull_rom(None, "cpu")["max_abs_diff_interior"] < 1e-10


def test_hermite_partition_of_unity():
    t = torch.linspace(-0.5, 1.5, 11, dtype=torch.float64)
    h00, h01, _, _ = hermite_basis(t)
    assert torch.allclose(h00 + h01, torch.ones_like(t))


def test_linear_ramp_is_exact():
    # f = 3x + 2y on low-res pixel indices with exact derivatives -> exact everywhere, incl. border
    h, w, s = 5, 7, 4
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float64),
                            torch.arange(w, dtype=torch.float64), indexing="ij")
    f = (3 * xx + 2 * yy)[..., None]
    out = spline_upscale(f, torch.full_like(f, 3.0), torch.full_like(f, 2.0),
                         torch.zeros_like(f), h * s, w * s)
    Y, X = torch.meshgrid(torch.arange(h * s, dtype=torch.float64),
                          torch.arange(w * s, dtype=torch.float64), indexing="ij")
    u, v = (X + 0.5) / s - 0.5, (Y + 0.5) / s - 0.5
    assert torch.allclose(out[..., 0], 3 * u + 2 * v)


def test_central_differences_interior():
    f = torch.arange(20, dtype=torch.float64).reshape(4, 5, 1) ** 2
    fx, fy, _ = central_differences(f)
    assert fx[1, 2, 0] == (f[1, 3, 0] - f[1, 1, 0]) / 2
    assert fy[2, 1, 0] == (f[3, 1, 0] - f[1, 1, 0]) / 2


def test_low_res_camera():
    cam = {"viewmat": torch.eye(4), "K": torch.tensor([[500.0, 0, 489.5], [0, 500.0, 273.0],
                                                       [0, 0, 1]]), "width": 979, "height": 546}
    low = low_res_camera(cam, 2)
    assert (low["width"], low["height"]) == (489, 273)
    assert torch.isclose(low["K"][0, 2], torch.tensor(489.5 * 489 / 979))
    assert low["K"][2, 2] == 1


def test_renderer_config():
    with pytest.raises(ValueError):
        UpscaleRenderer(ComponentSpec("r", "renderer", "x", (), (), {"upscaler": "lanczos"}))
    with pytest.raises(ValueError):
        UpscaleRenderer(ComponentSpec("r", "renderer", "x", (), (), {"bogus": 1}))


def test_methods():
    for m in (spline2x(), spline4x(), bicubic4x(), control3dgs()):
        density = next(c for c in m.components if c.kind == "density")
        assert density.config["budget"] == 1000000
    r = next(c for c in spline2x().components if c.kind == "renderer")
    assert r.config["factor"] == 2.0 and r.config["upscaler"] == "spline"
