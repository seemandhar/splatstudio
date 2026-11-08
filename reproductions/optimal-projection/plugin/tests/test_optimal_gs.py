import torch

from ss_optimal_gs import optimal_gs, optimal_gs_ewa
from ss_optimal_gs.probes import cov_gradient_probe, random_means, tangent_jacobian_probe
from ss_optimal_gs.projection import ewa_jacobian, tangent_jacobian_exact
from ss_optimal_gs.reference import dense_render, exact_ray_response, tile_mask


def test_jacobian_probes_pass_on_cpu():
    tangent_jacobian_probe(None, "cpu")
    out = cov_gradient_probe(None, "cpu")
    assert out["j12_x_rel_diff"] > 1e-3  # the official typo is replicated


def test_tangent_basis_is_orthonormal_and_orthogonal_to_ray():
    t = random_means(50)
    J = tangent_jacobian_exact(t, 1.0, 1.0) * t.norm(dim=-1)[:, None, None]
    assert torch.allclose(J @ J.transpose(1, 2), torch.eye(2, dtype=t.dtype).expand(50, 2, 2),
                          atol=1e-12)
    assert torch.allclose(J @ t[..., None], torch.zeros(50, 2, 1, dtype=t.dtype), atol=1e-12)


def test_dense_on_axis_tangent_equals_planar():
    """A tiny Gaussian at the principal point: tangent coords ~ pixel offsets."""
    K = torch.tensor([[600.0, 0, 32.0], [0, 600.0, 32.0], [0, 0, 1]], dtype=torch.float64)
    m = torch.tensor([[32.0, 32.0]], dtype=torch.float64)
    q = torch.tensor([[1 / 9.0, 0.0, 1 / 9.0]], dtype=torch.float64)
    c = torch.ones(1, 3, dtype=torch.float64)
    o = torch.tensor([0.8], dtype=torch.float64)
    mask = torch.ones(1, 64, 64, dtype=torch.bool)
    kw = {"focal_floor": 512.0}
    _, a_opt = dense_render(m, q, c, o, torch.ones(1), mask, K, 64, 64, mode="optimal", **kw)
    _, a_ewa = dense_render(m, q, c, o, torch.ones(1), mask, K, 64, 64, mode="ewa", **kw)
    assert (a_opt - a_ewa).abs().max() < 2e-3


def test_exact_ray_response_peak_at_mean():
    K = torch.tensor([[50.0, 0, 20.5], [0, 50.0, 20.5], [0, 0, 1]], dtype=torch.float64)
    mean = torch.tensor([0.0, 0.0, 3.0], dtype=torch.float64)
    r = exact_ray_response(mean, torch.eye(3, dtype=torch.float64) * 0.01, K, 40, 40)
    assert torch.isclose(r.max(), torch.tensor(1.0, dtype=torch.float64), atol=1e-9)
    assert int(r.argmax()) == 20 * 40 + 20


def test_ewa_jacobian_matches_3dgs():
    t = torch.tensor([[0.3, -0.2, 2.0]], dtype=torch.float64)
    J = ewa_jacobian(t, 500.0, 400.0, 10.0, 10.0)
    want = torch.tensor([[[250.0, 0, -500 * 0.3 / 4], [0, 200.0, 400 * 0.2 / 4]]],
                        dtype=torch.float64)
    assert torch.allclose(J, want)


def test_tile_mask_rect():
    m = torch.tensor([[20.0, 5.0]], dtype=torch.float64)
    radii = torch.tensor([[3, 3]], dtype=torch.int32)
    mask = tile_mask(m, radii, 48, 32)
    assert mask[0, :16, 16:32].all() and not mask[0, :, :16].any() and not mask[0, 16:].any()


def test_method_specs():
    for spec, mode in ((optimal_gs(), "optimal"), (optimal_gs_ewa(), "ewa")):
        assert spec.components[0].config["projection"] == mode
        assert spec.components[1].name == "adaptive-density"
        assert spec.components[2].name == "photometric"
