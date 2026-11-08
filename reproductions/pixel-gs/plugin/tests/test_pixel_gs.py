import torch

from ss_pixel_gs import method, method_depth_only, method_pixel_only
from ss_pixel_gs.core import depth_factor, view_contribution
from ss_pixel_gs.probes import depth_scaling


def test_depth_factor_clip():
    f = depth_factor(torch.tensor([0.0, 0.5, 1.0, 10.0]), 0.5, 2.0)
    assert torch.allclose(f, torch.tensor([0.0, 0.25, 1.0, 1.0]))


def test_eq11_two_views():
    # Gaussian seen in two views: big coverage + big gradient, small coverage + small gradient
    g = [torch.tensor([1e-3]), torch.tensor([1e-5])]
    m = [torch.tensor([100.0]), torch.tensor([2.0])]
    z = torch.tensor([10.0])
    num = den = 0
    for gi, mi in zip(g, m):
        n, d = view_contribution(gi, mi, z, pixel_aware=True, depth_scaling=True,
                                 gamma_depth=0.37, radius=1.0)
        num, den = num + n, den + d
    assert torch.allclose(num / den, torch.tensor([(100 * 1e-3 + 2 * 1e-5) / 102]))
    # 3DGS mean would be (1e-3 + 1e-5)/2
    n1, d1 = view_contribution(g[0], m[0], z, pixel_aware=False, depth_scaling=False,
                               gamma_depth=0.37, radius=1.0)
    assert d1.item() == 1.0 and n1.item() == g[0].item()


def test_depth_probe_cpu():
    depth_scaling(method(), "cpu")


def test_specs():
    for fn, flags in ((method, (True, True)), (method_pixel_only, (True, False)),
                      (method_depth_only, (False, True))):
        spec = fn()
        d = next(c for c in spec.components if c.kind == "density")
        assert (d.config["pixel_aware"], d.config["depth_scaling"]) == flags
        assert d.config["gamma_depth"] == 0.37 and d.config["grow_grad2d"] == 2e-4
