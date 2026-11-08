import math

import torch
from splatstudio.specs import ComponentSpec

from ss_linear_splatting import linear_splatting, linear_splatting_noags
from ss_linear_splatting.losses import PAPER_WEIGHTS, combined_loss, create_loss
from ss_linear_splatting.reference import dense_render, kernel_value
from ss_linear_splatting.renderer import linear_radii


def test_kernel_value_and_ags_gradient():
    D = torch.linspace(0.1, 3.0, 30, dtype=torch.float64, requires_grad=True)
    k = kernel_value(D, 2.5, ags=False)
    assert torch.allclose(k, 1 - D / 2.5)
    (g,) = torch.autograd.grad(k.sum(), D)
    assert torch.allclose(g, torch.full_like(g, -0.4))  # Fig. 3b native gradient 1/lam
    k2 = kernel_value(D, 2.5, ags=True)
    assert torch.allclose(k2, k)  # AGS leaves the forward value unchanged
    (g2,) = torch.autograd.grad(k2.sum(), D)
    assert torch.allclose(g2, -0.4 * torch.exp(-D.detach() ** 2))  # Eq. 13-14


def test_dense_single_splat_support():
    m = torch.tensor([[10.0, 10.0]], dtype=torch.float64)
    s = 2.0
    q = torch.tensor([[1 / s**2, 0.0, 1 / s**2]], dtype=torch.float64)
    c = torch.ones(1, 3, dtype=torch.float64)
    o = torch.tensor([0.5], dtype=torch.float64)
    _, alpha = dense_render(m, q, c, o, torch.ones(1), 20, 20, lam=2.5)
    ys, xs = torch.meshgrid(torch.arange(20) + 0.5, torch.arange(20) + 0.5, indexing="ij")
    D = (((xs - 10) ** 2 + (ys - 10) ** 2).sqrt() / s).double()
    want = (0.5 * (1 - D / 2.5)).clamp_min(0)
    want = torch.where(want >= 1 / 255, want, torch.zeros_like(want))
    assert torch.allclose(alpha[..., 0], want, atol=1e-12)


def test_linear_radii_matches_support():
    conics = torch.tensor([[[1 / 9.0, 0.0, 1 / 4.0]]])  # sigma_x 3, sigma_y 2
    means = torch.tensor([[[50.0, 50.0]]])
    op = torch.tensor([[0.8]])
    r = linear_radii(means, conics, op, torch.ones(1, 1, 2, dtype=torch.int32), 100, 100, 2.5)
    ext = 2.5 * (1 - (1 / 255) / 0.8)
    assert r[0, 0].tolist() == [math.ceil(ext * 3), math.ceil(ext * 2)]
    # opacity below the alpha threshold and off-screen boxes are culled
    r = linear_radii(means, conics, torch.tensor([[0.003]]),
                     torch.ones(1, 1, 2, dtype=torch.int32), 100, 100, 2.5)
    assert r.abs().sum() == 0
    r = linear_radii(torch.tensor([[[-20.0, 50.0]]]), conics, op,
                     torch.ones(1, 1, 2, dtype=torch.int32), 100, 100, 2.5)
    assert r.abs().sum() == 0


def test_loss_terms_sum_to_paper_objective():
    g = torch.Generator().manual_seed(0)
    rgb, target = torch.rand(32, 32, 3, generator=g), torch.rand(32, 32, 3, generator=g)
    total = sum(create_loss(ComponentSpec(f"l-{t}", "loss", "x", (), (), {"term": t}))(
        None, {"rgb": rgb}, target) for t in PAPER_WEIGHTS)
    assert torch.allclose(total, combined_loss(rgb, target))
    assert PAPER_WEIGHTS == {"l1": 0.6, "l2": 0.2, "ssim": 0.2}


def test_method_specs():
    m = linear_splatting()
    assert m.name == "3dls"
    r = next(c for c in m.components if c.kind == "renderer")
    assert r.config["lam"] == 2.5 and r.config["ags"] is True
    d = next(c for c in m.components if c.kind == "density")
    assert d.config["grow_scale3d"] == 0.006 and d.config["prune_scale3d"] == 0.4
    assert d.config["prune_opa"] == 0.025
    assert [c.name for c in m.components if c.kind == "loss"] == ["loss-l1", "loss-l2",
                                                                    "loss-ssim"]
    n = linear_splatting_noags()
    assert next(c for c in n.components if c.kind == "renderer").config["ags"] is False
