import math

import torch

from ss_stopthepop import reference as ref
from ss_stopthepop.raster import HIERARCHICAL, make_settings


def test_topt_isotropic_is_projection():
    # Sigma^-1 = I/s^2: t_opt = d . (mu - o) (paper's 2D example: cos(theta) * mu_y)
    mu = torch.tensor([[0.0, 0.0, 3.0]])
    d = torch.tensor([[math.sin(0.3), 0.0, math.cos(0.3)]], dtype=torch.float64)
    t = ref.t_opt(mu, torch.full((1, 3), math.log(0.2)), torch.tensor([[1.0, 0, 0, 0]]), d,
                  torch.zeros(3, dtype=torch.float64))
    assert abs(t.item() - 3.0 * math.cos(0.3)) < 1e-12


def test_topt_maximises_gaussian_along_ray():
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(1, 3, generator=g) + torch.tensor([0, 0, 4.0])
    ls = (torch.rand(1, 3, generator=g) * 0.5 + 0.05).log()
    q = torch.randn(1, 4, generator=g)
    d = torch.nn.functional.normalize(torch.randn(1, 3, generator=g, dtype=torch.float64), dim=-1)
    o = torch.zeros(3, dtype=torch.float64)
    t = ref.t_opt(mu, ls, q, d, o).item()
    R = ref.rotmat(q)[0]
    Sinv = R @ torch.diag(1 / ls[0].double().exp() ** 2) @ R.T
    def G(tt):
        x = o + tt * d[0] - mu[0].double()
        return -(x @ Sinv @ x).item()
    assert G(t) >= G(t + 1e-3) and G(t) >= G(t - 1e-3)


def test_settings_builder():
    s = make_settings(sort_mode="GLOBAL", per_pixel=8, tile_based_culling=True)
    assert s["sort_settings"]["sort_mode"] == 0
    assert s["sort_settings"]["queue_sizes"]["per_pixel"] == 8
    assert s["culling_settings"]["tile_based_culling"] is True
    assert HIERARCHICAL["sort_settings"]["sort_mode"] == 3  # default untouched


def test_global_vs_perpixel_reference_differ():
    from ss_stopthepop.probes import _camera, _crossing_scene
    p = _crossing_scene("cpu")
    cam = _camera(48, 48, 120.0, "cpu")
    with torch.no_grad():
        a, _ = ref.composite(p, cam, torch.zeros(3), "perpixel")
        b, _ = ref.composite(p, cam, torch.zeros(3), "global")
    assert (a - b).abs().max() > 0.1


def test_schedule_probe_cpu():
    import ss_stopthepop
    from ss_stopthepop.probes import schedule
    assert schedule(ss_stopthepop.method(), "cpu")["decay_ok"]
