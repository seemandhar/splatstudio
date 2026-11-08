import torch
from splatstudio.specs import ComponentSpec

from ss_msaa_gs import aws_gdc_only, msaa_gs, msaa_only
from ss_msaa_gs.kernels import PIXEL_CENTRE, ROTATED_GRID_4X
from ss_msaa_gs.losses import aws_weights, create_gdc, create_recon, gradient_difference
from ss_msaa_gs.reference import composite_at, dense_msaa


def test_pattern_centroid_is_pixel_centre():
    xs = sum(o[0] for o in ROTATED_GRID_4X) / 4
    ys = sum(o[1] for o in ROTATED_GRID_4X) / 4
    assert (xs, ys) == (0.5, 0.5)
    assert len({o[0] for o in ROTATED_GRID_4X}) == 4  # rotated grid: distinct columns/rows
    assert len({o[1] for o in ROTATED_GRID_4X}) == 4


def test_dense_msaa_is_mean_of_sample_composites():
    g = torch.Generator().manual_seed(0)
    m = torch.rand(6, 2, generator=g, dtype=torch.float64) * 10
    q = torch.tensor([[0.8, 0.1, 0.6]], dtype=torch.float64).expand(6, 3)
    c = torch.rand(6, 3, generator=g, dtype=torch.float64)
    o = torch.rand(6, generator=g, dtype=torch.float64)
    d = torch.rand(6, generator=g, dtype=torch.float64)
    rgb, _ = dense_msaa(m, q, c, o, d, 10, 8, ROTATED_GRID_4X)
    jj, ii = torch.meshgrid(torch.arange(10.0, dtype=torch.float64),
                            torch.arange(8.0, dtype=torch.float64), indexing="xy")
    parts = [composite_at(m, q, c, o, d, jj + dx, ii + dy)[0] for dx, dy in ROTATED_GRID_4X]
    assert torch.allclose(rgb, sum(parts) / 4)
    one, _ = dense_msaa(m, q, c, o, d, 10, 8, PIXEL_CENTRE)
    assert not torch.allclose(one, rgb)


def test_aws_weight_range_and_detached():
    g = torch.Generator().manual_seed(1)
    p = torch.rand(9, 11, 3, generator=g, requires_grad=True)
    t = torch.rand(9, 11, 3, generator=g)
    w = aws_weights(p, t)
    assert not w.requires_grad and w.shape == (9, 11, 1)
    assert float(w.min()) >= 0.5 and float(w.max()) < 1.0
    e = (p - t).abs().mean(-1)
    assert torch.isclose(w.view(-1)[e.view(-1).argmax()], torch.tensor(1.0), atol=1e-4)


def test_gdc_ramp_and_offset():
    jj = torch.arange(12.0)[None, :, None].expand(7, 12, 3)
    assert torch.isclose(gradient_difference(0.05 * jj, 0.02 * jj), torch.tensor(0.03))
    ii = torch.arange(7.0)[:, None, None].expand(7, 12, 3)
    assert torch.isclose(gradient_difference(0.04 * ii, 0 * ii), torch.tensor(0.04))
    assert float(gradient_difference(jj + 1.0, jj)) == 0.0


def test_loss_factories_reject_unknown_options():
    import pytest

    for f in (create_recon, create_gdc):
        with pytest.raises(ValueError):
            f(ComponentSpec("x", "loss", "m:f", ("render.rgb",), (), {"bogus": 1}))


def test_method_structure():
    full = msaa_gs()
    names = [c.name for c in full.components]
    assert names == ["msaa-renderer", "adaptive-density", "aws-recon", "gdc"]
    assert full.components[2].config["lambda_w"] == 0.8
    assert full.components[2].config["lambda_ssim"] == 0.2
    assert full.components[3].config["weight"] == 0.1
    assert [c.name for c in msaa_only().components] == [
        "msaa-renderer", "adaptive-density", "photometric"]
    assert [c.name for c in aws_gdc_only().components] == [
        "gsplat", "adaptive-density", "aws-recon", "gdc"]


def test_it1_normalised_weights_keep_l1_scale():
    from ss_msaa_gs import msaa_gs_it1
    from ss_msaa_gs.losses import weighted_l1

    g = torch.Generator().manual_seed(5)
    p = torch.rand(9, 11, 3, generator=g)
    t = torch.rand(9, 11, 3, generator=g)
    w = aws_weights(p, t)
    assert torch.isclose(weighted_l1(p, t, normalize_mean=True),
                         ((w / w.mean()) * (p - t).abs()).mean())
    assert msaa_gs_it1().components[2].config["normalize_mean"] is True
    assert msaa_gs().components[2].config["normalize_mean"] is False
