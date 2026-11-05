import math

import numpy as np
import pytest
import torch

from ss_edgs import core, method, nosh, probes
from ss_edgs.density import find_sidecar, means_sha256
from ss_edgs.preprocess import points_sha256


def test_triangulation_probe():
    out = probes.triangulation(method(), "cpu")
    assert out["max_point_error"] < 1e-6


def test_sampling_probe():
    probes.sampling(method(), "cpu")


@pytest.mark.parametrize("factory", [method, nosh])
def test_no_densify_probe(factory):
    probes.no_densify(factory(), "cpu")


def test_pixel_ndc_roundtrip():
    idx = torch.tensor([0, 5, 37, 99])
    ndc = core.pixel_centres_ndc(idx, 10, 10)
    col, row = core.ndc_to_pixel(ndc, 10, 10)
    assert torch.equal(row * 10 + col, idx)


def test_lr_floor():
    assert core.lr_floored(1.0, 1) == pytest.approx(0.01 ** (8000 / 30000))
    assert core.lr_floored(1.0, 30000) == pytest.approx(0.01)
    assert core.lr_floored(1.0, 40000) == pytest.approx(0.01)


def test_isotropic_scale():
    pts = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64)
    s = core.log_isotropic_scale(pts, torch.zeros(3, dtype=torch.float64))
    assert float(s.exp()) == pytest.approx(2 * 0.001 * 0.5)


def test_sh_fit_cpu_reproduces_observations():
    from gsplat.cuda._torch_impl import _spherical_harmonics

    g = torch.Generator().manual_seed(0)
    dirs = torch.nn.functional.normalize(torch.randn(8, 2, 3, generator=g), dim=-1)
    obs = torch.rand(8, 2, 3, generator=g)
    H = core.fit_sh(dirs, obs)
    out = _spherical_harmonics(3, dirs, H[:, None].expand(-1, 2, -1, -1)) + 0.5
    assert torch.allclose(out, obs, atol=1e-5)


def test_sidecar_lookup(tmp_path):
    pts = np.random.default_rng(0).normal(size=(20, 3)).astype(np.float32)
    d = tmp_path / "ds" / "scene"
    d.mkdir(parents=True)
    np.savez(d / "edgs_init.npz", points_sha256=np.array(points_sha256(pts)))
    assert means_sha256(torch.from_numpy(pts)) == points_sha256(pts)
    assert find_sidecar(tmp_path, points_sha256(pts)) == d / "edgs_init.npz"
    with pytest.raises(FileNotFoundError):
        find_sidecar(tmp_path, "0" * 64)


def test_nearest_cameras_excludes_self():
    v = torch.eye(4).repeat(5, 1, 1)
    v[:, 0, 3] = torch.tensor([0.0, 1.0, 2.0, 3.0, 10.0])
    nn = core.nearest_cameras(v, 2)
    assert nn[0].tolist() == [1, 2]
    assert all(i not in row for i, row in enumerate(nn.tolist()))
    assert math.isfinite(float(nn.sum()))


def test_residual_sh_reference_exact_and_bounded():
    g = torch.Generator().manual_seed(3)
    dirs = torch.nn.functional.normalize(torch.randn(32, 2, 3, generator=g), dim=-1)
    dirs[:8, 1] = torch.nn.functional.normalize(dirs[:8, 0] + 1e-4, dim=-1)  # ~parallel
    obs = torch.rand(32, 2, 3, generator=g)
    shn = core.fit_sh_residual(dirs, obs)
    out = core.compose_sh_colors(dirs, core.rgb_to_sh(obs[:, 0])[:, None], shn)
    assert torch.allclose(out[:, 0], obs[:, 0], atol=1e-5)
    r = (obs[:, 1] - obs[:, 0]).abs()
    assert (shn.norm(dim=1) <= 2 * r + 1e-5).all()
    assert float(shn[:8].abs().max()) < 1e-2  # parallel views -> no higher bands
