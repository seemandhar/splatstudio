"""C3 quaternion -> Euler replacement (official `gaussian_renderer.ToEulerAngles_FT`,
`scene/gaussian_model.build_rotation_from_euler`, `build_covariance_from_scaling_euler`).

Paper Eq. 4 / Eq. 5. The official covariance calls `build_rotation_from_euler(e[:, 2], e[:, 1],
e[:, 0])` (argument names roll, pitch, yaw) with e = [roll, pitch, yaw], which is the paper's Eq. 5
matrix Rz(yaw) Ry(pitch) Rx(roll); the argument order is kept. `matrix_to_quaternion` (PyTorch3D, wxyz) recovers a quaternion of the decoded
rotation for the native parameter layout (copied from the c3dgs plugin `vq.py`).
"""

import torch
import torch.nn.functional as F

SQRT_EPS = 1e-12


def to_euler(q):
    """q [N,4] wxyz (normalised by the caller as in ft_render) -> [roll, pitch, yaw]."""
    w, x, y, z = q.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    t = 2 * (w * y - x * z)
    # official: sqrt(1 + t), sqrt(1 - t); clamped at 1e-12 so a float-rounded negative argument
    # (|t| <= 1 mathematically) gives no NaN gradient (official: nan_to_num of the output only)
    pitch = 2 * torch.atan2(torch.sqrt((1 + t).clamp_min(SQRT_EPS)),
                            torch.sqrt((1 - t).clamp_min(SQRT_EPS))) - torch.pi / 2
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return torch.stack((roll, pitch, yaw), -1).nan_to_num(0.0)


def rotation_from_euler(roll, pitch, yaw):
    """Official build_rotation_from_euler (paper Eq. 5)."""
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    R = torch.stack(
        (
            cp * cr, -cy * sr + sy * sp * cr, sy * sr + cy * sp * cr,
            cp * sr, cy * cr + sy * sp * sr, -sy * cr + cy * sp * sr,
            -sp, sy * cp, cy * cp,
        ),
        -1,
    )
    return R.reshape(roll.shape + (3, 3))


def rotation_of(euler):
    """Rotation used by the official covariance: build_rotation_from_euler(e2, e1, e0)."""
    return rotation_from_euler(euler[:, 2], euler[:, 1], euler[:, 0])


def covariance_from_euler(scales, euler):
    """Official build_covariance_from_scaling_euler (full 3x3): (R S)(R S)^T."""
    L = rotation_of(euler) * scales[:, None, :]
    return L @ L.transpose(1, 2)


def quat_to_matrix(q):
    """3DGS build_rotation (wxyz, normalised inside)."""
    q = F.normalize(q, dim=-1)
    r, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
            2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
            2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(q.shape[:-1] + (3, 3))


def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive = x > 0
    ret[positive] = torch.sqrt(x[positive])
    return ret


def matrix_to_quaternion(matrix):
    """PyTorch3D matrix_to_quaternion (wxyz), as vendored by c3dgs utils/splats.py."""
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1)
    q_abs = _sqrt_positive_part(torch.stack(
        [1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22,
         1.0 - m00 - m11 + m22], dim=-1))
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    return quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,))
