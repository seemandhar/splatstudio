import math

import torch

from ss_sugar import method, smoke
from ss_sugar.geometry import (
    bound_gaussians,
    edge_face_pairs,
    expon_lr,
    matrix_to_quaternion,
    normal_consistency,
    quat_to_rotmat,
    saturate_density,
    sdf_values,
)


def test_quaternion_roundtrip():
    q = torch.nn.functional.normalize(torch.randn(100, 4), dim=-1)
    q2 = matrix_to_quaternion(quat_to_rotmat(q))
    assert torch.allclose(quat_to_rotmat(q2), quat_to_rotmat(q), atol=1e-5)


def test_sdf_plane_distance():
    t = torch.linspace(-0.1, 0.1, 21)
    d = torch.exp(-0.5 * t**2 / 0.05**2)
    f = sdf_values(d, torch.full_like(t, 0.05))
    assert torch.allclose(f, t.abs(), atol=1e-5)


def test_saturate_density_gradient_kept():
    d = torch.tensor([0.5, 2.0], requires_grad=True)
    out = saturate_density(d)
    assert torch.allclose(out, torch.tensor([0.5, 1.0]))
    out.sum().backward()
    assert torch.allclose(d.grad, torch.tensor([1.0, 0.5]))


def test_bound_gaussian_axes_orthonormal():
    verts = torch.randn(30, 3)
    faces = torch.randint(30, (20, 3))
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    bary = torch.tensor([[1 / 3, 1 / 3, 1 / 3]])
    rot2d = torch.randn(len(faces), 2)
    _, quats, scales = bound_gaussians(verts, faces, bary, rot2d, torch.zeros(len(faces), 2), math.log(1e-6))
    R = quat_to_rotmat(quats)
    eye = torch.eye(3).expand_as(R)
    assert torch.allclose(R.transpose(1, 2) @ R, eye, atol=1e-4)
    assert torch.allclose(scales[:, 0], torch.full((len(faces),), 1e-6))


def test_normal_consistency_cube_zero_for_flat_pairs():
    verts = torch.tensor([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]])
    faces = torch.tensor([[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]])
    pairs = edge_face_pairs(faces, 6)
    assert len(pairs) == 3
    assert float(normal_consistency(verts, pairs)) < 1e-6


def test_lr_schedule():
    assert abs(expon_lr(0, 1.0, 0.01, 100) - 1.0) < 1e-12
    assert abs(expon_lr(50, 1.0, 0.01, 100) - 0.1) < 1e-12
    assert abs(expon_lr(200, 1.0, 0.01, 100) - 0.01) < 1e-12


def test_specs():
    for m in (method(), smoke()):
        names = [c.name for c in m.components]
        assert names[0] == "sugar-renderer"
        assert "sugar-sdf" in names and "photometric" in names
    d = {c.name: c.config for c in method().components}
    assert d["sugar-density"]["extract_at"] == 15000
    assert d["sugar-sdf"]["weight"] == 0.2 and abs(d["sugar-sdf"]["density_factor"] - 1 / 16) < 1e-12
