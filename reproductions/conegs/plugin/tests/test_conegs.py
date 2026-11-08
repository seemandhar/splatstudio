import numpy as np
import torch

from ss_conegs import method, nobudget, probes
from ss_conegs.density import means_sha256, rays_to_cast
from ss_conegs.hashgrid import level_layout
from ss_conegs.ingp import RayCaster, contract_to_unisphere, gs2nerf_transform, per_level
from ss_conegs.preprocess import knn_log_scales, points_sha256


def test_cpu_probes():
    m = method()
    for fn in (probes.proposal_sampler, probes.median_depth, probes.cone_placement,
               probes.budget, probes.opacity_penalty):
        fn(m, "cpu")


def test_nobudget_variant():
    spec = next(c for c in nobudget().components if c.kind == "density")
    assert spec.config["budget"] == 0 and spec.config["growth"] == 0.02


def test_rays_formula():
    assert rays_to_cast(1_000_000, None, 0.02) == 201
    assert rays_to_cast(1_000_000, 300_000, 0.02) == 3601


def test_tcnn_layout():
    scales, res, offsets = level_layout(16, 19, 16, per_level(16, 4096, 16))
    assert res[0] == 16 and res[-1] == 4096
    assert offsets[1] == 16**3
    assert offsets[-1] - offsets[-2] == 1 << 19


def test_contraction_bounds():
    x = torch.randn(1000, 3) * 100
    y = contract_to_unisphere(x)
    assert (y >= 0).all() and (y <= 1).all()
    inner = torch.tensor([[0.1, -0.2, 0.3]])
    torch.testing.assert_close(contract_to_unisphere(inner), (inner + 2) / 4)


def test_normalisation_roundtrip():
    rng = np.random.default_rng(0)
    c2w = np.tile(np.eye(4), (10, 1, 1))
    c2w[:, :3, 3] = rng.normal(size=(10, 3)) * 3
    T, s = gs2nerf_transform(c2w)
    caster = RayCaster(T, s, "cpu")
    x = torch.randn(20, 3)
    xn = s * (x @ caster.T[:3, :3].T + caster.T[:3, 3])
    torch.testing.assert_close(caster.to_world(xn), x, atol=1e-5, rtol=1e-5)
    _, _, o = caster.camera(torch.eye(3), torch.linalg.inv(torch.tensor(c2w[0], dtype=torch.float32)))
    torch.testing.assert_close(caster.to_world(o[None])[0], torch.tensor(c2w[0, :3, 3], dtype=torch.float32), atol=1e-4, rtol=1e-4)


def test_knn_and_sha():
    pts = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [10, 10, 10]])
    ls = knn_log_scales(pts, chunk=2)
    expected = np.log(np.sqrt((1 + 4 + 9) / 3))
    assert abs(float(ls[0]) - expected) < 1e-5
    arr = pts.numpy()
    assert points_sha256(arr) == means_sha256(pts)


def test_opaque_last_sample_transmittance():
    # regression: exclusive cumsum via cumsum - x cancels against the opaque last sample
    from ss_conegs.ingp import OPAQUE_SIGMA, transmittance_from_density

    ts = torch.tensor([[0.0, 1.0, 2.0]])
    te = torch.tensor([[1.0, 2.0, 1e6]])
    sig = torch.tensor([[0.1, 0.1, OPAQUE_SIGMA]])
    trans, alphas = transmittance_from_density(ts, te, sig)
    torch.testing.assert_close(trans[0, -1], torch.exp(torch.tensor(-0.2)))
    assert alphas[0, -1] == 1.0
