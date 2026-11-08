"""Projection of the 3D Gabor frequencies to 2D splat frequencies (paper Eq. 9-10, supp. A).

Paper: f_proj = (J W)^{-T} f in ray space, then integrating along ray-space z gives
f2d_x = f_x - S02/S22 f_z, f2d_y = f_y - S12/S22 f_z with S = (Sigma')^{-1}. Writing the
z-integration as the Gaussian conditional mean shows this equals the coordinate-free
    f2d = Sigma2d^{-1} J2 Sigma f,     J2 = J W (2x3), Sigma2d = J2 Sigma J2^T,
independent of how the 3x3 ray-space Jacobian is completed (checked by the
`frequency-projection` probe against the explicit Eq. 9-10 and numerical integration).
The renderer uses gsplat's dilated conic (Sigma2d + 0.3 I)^{-1} (plan.md, ambiguity 7).
"""

import torch


def quat_scale_to_rs(quats, scales):
    """quats [N,4] wxyz (unnormalised), scales [N,3] (activated) -> M = R S [N,3,3]."""
    w, x, y, z = torch.unbind(torch.nn.functional.normalize(quats, dim=-1), dim=-1)
    R = torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ), dim=-1).reshape(*quats.shape[:-1], 3, 3)
    return R * scales[..., None, :]


def gsplat_jacobian(means_cam, K, width, height):
    """gsplat 1.5.3 persp_proj Jacobian J [N,2,3] (with its clamped tx/ty, Utils.cuh)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x, y, z = torch.unbind(means_cam, dim=-1)
    tan_fovx, tan_fovy = 0.5 * width / fx, 0.5 * height / fy
    lim_x_pos, lim_x_neg = (width - cx) / fx + 0.3 * tan_fovx, cx / fx + 0.3 * tan_fovx
    lim_y_pos, lim_y_neg = (height - cy) / fy + 0.3 * tan_fovy, cy / fy + 0.3 * tan_fovy
    rz = 1.0 / z
    tx = z * torch.minimum(lim_x_pos, torch.maximum(-lim_x_neg, x * rz))
    ty = z * torch.minimum(lim_y_pos, torch.maximum(-lim_y_neg, y * rz))
    zero = torch.zeros_like(z)
    return torch.stack((fx * rz, zero, -fx * tx * rz * rz,
                        zero, fy * rz, -fy * ty * rz * rz), dim=-1).reshape(-1, 2, 3)


def project_frequencies(means, quats, scales, freqs, viewmat, K, width, height, conics,
                        visible):
    """2D splat frequencies [N,F,2] in cycles/pixel.

    means [N,3], quats [N,4], scales [N,3] (activated), freqs [N,F,3] (world cycles/unit),
    viewmat [4,4] world-to-camera, conics [N,3] = inverse 2D covariance (a, b, c),
    visible [N] bool (non-visible rows use z = 1 so gradients stay finite; they are not drawn).
    """
    Rc, tc = viewmat[:3, :3], viewmat[:3, 3]
    means_cam = means @ Rc.T + tc
    z = torch.where(visible, means_cam[:, 2], torch.ones_like(means_cam[:, 2]))
    means_cam = torch.cat((means_cam[:, :2], z[:, None]), dim=-1)
    J2 = gsplat_jacobian(means_cam, K, width, height) @ Rc  # [N,2,3] world -> pixel
    M = quat_scale_to_rs(quats, scales)  # Sigma = M M^T
    sig_f = torch.einsum("nij,nkj,nfk->nfi", M, M, freqs)  # Sigma f  [N,F,3]
    a = torch.einsum("nij,nfj->nfi", J2, sig_f)  # J2 Sigma f [N,F,2]
    ca, cb, cc = conics[:, 0, None], conics[:, 1, None], conics[:, 2, None]
    return torch.stack((ca * a[..., 0] + cb * a[..., 1], cb * a[..., 0] + cc * a[..., 1]),
                       dim=-1)


def paper_frequency(mean_cam, cov_cam, f_cam, K, third_row):
    """Literal paper Eq. 9-10 for one Gaussian (float64 reference, no clamping/dilation).

    third_row: completion of the 3x3 ray-space Jacobian, e.g. (0,0,1) or the EWA t/|t|.
    """
    fx, fy = K[0, 0], K[1, 1]
    x, y, z = mean_cam
    J = torch.stack((torch.stack((fx / z, torch.zeros_like(z), -fx * x / z**2)),
                     torch.stack((torch.zeros_like(z), fy / z, -fy * y / z**2)),
                     third_row))
    f_proj = torch.linalg.inv(J).T @ f_cam  # Eq. 9 (W already applied: camera frame)
    S = torch.linalg.inv(J @ cov_cam @ J.T)
    return torch.stack((f_proj[0] - S[0, 2] / S[2, 2] * f_proj[2],
                        f_proj[1] - S[1, 2] / S[2, 2] * f_proj[2]))  # Eq. 10
