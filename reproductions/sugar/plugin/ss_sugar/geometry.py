"""Pure tensor functions of SuGaR (official code: github.com/Anttwo/SuGaR, read-only reference).

Line references are to `source/code/` of the reproduction workspace.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

C0 = 0.28209479177387814

# sugar_model.py:183-225: barycentric coordinates and in-triangle circle radius per count.
BARY = {
    1: ([[1 / 3, 1 / 3, 1 / 3]], 1.0 / 2.0 / math.sqrt(3.0)),
    3: ([[1 / 2, 1 / 4, 1 / 4], [1 / 4, 1 / 2, 1 / 4], [1 / 4, 1 / 4, 1 / 2]],
        1.0 / 2.0 / (math.sqrt(3.0) + 1.0)),
    4: ([[1 / 3, 1 / 3, 1 / 3], [2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]],
        1.0 / (4.0 * math.sqrt(3.0))),
    6: ([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3],
         [1 / 6, 5 / 12, 5 / 12], [5 / 12, 1 / 6, 5 / 12], [5 / 12, 5 / 12, 1 / 6]],
        1.0 / (4.0 + 2.0 * math.sqrt(3.0))),
}


def quat_to_rotmat(q):
    """wxyz quaternion (normalized here) -> rotation matrix whose columns are the local axes."""
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def _sqrt_positive(x):
    return torch.where(x > 0, torch.sqrt(x.clamp_min(1e-30)), torch.zeros_like(x))


def matrix_to_quaternion(m):
    """PyTorch3D's matrix_to_quaternion (wxyz), used by the official bound-Gaussian model."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    q_abs = _sqrt_positive(
        torch.stack(
            (1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22),
            dim=-1,
        )
    )
    cand = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), dim=-1),
        ),
        dim=-2,
    )
    cand = cand / (2.0 * q_abs[..., None].clamp_min(0.1))
    best = q_abs.argmax(dim=-1)
    return cand.gather(-2, best[..., None, None].expand(*best.shape, 1, 4))[..., 0, :]


def inverse_scaled_rotation(rot, scales):
    """sugar_model.py get_covariance(return_sqrt, inverse_scales): R diag(1/s)."""
    return rot * (1.0 / scales.clamp_min(1e-8))[..., None, :]


def smallest_axis(rot, scales):
    """sugar_model.py:961 get_smallest_axis: column of R with the smallest scale."""
    idx = scales.min(dim=-1).indices
    return rot.gather(2, idx[:, None, None].expand(-1, 3, 1))[..., 0]


def neighbour_density(x, idx, means, inv_sr, opac, factor):
    """sugar_model.py:1300-1308: rho * alpha_j * exp(-q_j/2) over the K neighbours idx [M,K]."""
    shift = x[:, None] - means[idx]
    warped = (inv_sr[idx].transpose(-1, -2) @ shift[..., None])[..., 0]
    q = warped.square().sum(dim=-1).clamp(0.0, 1e8)
    weights = factor * opac[idx] * torch.exp(-0.5 * q)
    return weights.sum(dim=-1), weights, warped


def saturate_density(d):
    """sugar_model.py:1310-1311: densities >= 1 are divided by their detached value."""
    mask = d >= 1.0
    return torch.where(mask, d / (d.detach() + 1e-12), d)


def sdf_values(d, beta, threshold=1.0, sqrt_floor=1e-12):
    """Eq. 7 in code form (sugar_model.py:1331-1335). The floor keeps the sqrt gradient finite."""
    d = d.clamp_min(1e-16)
    offset = math.sqrt(-2.0 * math.log(min(threshold, 1.0)))
    return beta * (torch.sqrt((-2.0 * torch.log(d)).clamp_min(sqrt_floor)) - offset)


def sample_in_gaussians(means, rot, scales, mask, n, factor, generator=None):
    """sugar_model.py:916: uniform choice among masked Gaussians, x = mu + R (f s * eps)."""
    valid = mask.nonzero()[:, 0]
    pick = valid[torch.randint(len(valid), (n,), device=means.device, generator=generator)]
    eps = torch.randn(n, 3, device=means.device, generator=generator)
    x = means[pick] + (rot[pick] @ (factor * scales[pick] * eps)[..., None])[..., 0]
    return x, pick


def world_to_camera(points, viewmat):
    return points @ viewmat[:3, :3].T + viewmat[:3, 3]


def lookup_depth(depth, points_cam, K, width, height):
    """sugar_model.py:1349 get_points_depth_in_depth_map: bilinear, border padding.

    gsplat pixel i covers [i, i+1) so u / W * 2 - 1 is the align_corners=False grid coordinate.
    """
    z = points_cam[:, 2].clamp_min(1e-8)
    u = K[0, 0] * points_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * points_cam[:, 1] / z + K[1, 2]
    grid = torch.stack((u / width * 2 - 1, v / height * 2 - 1), dim=-1).view(1, -1, 1, 2)
    return F.grid_sample(
        depth.view(1, 1, height, width), grid, mode="bilinear", padding_mode="border",
        align_corners=False,
    )[0, 0, :, 0]


def camera_center(viewmat):
    return -viewmat[:3, :3].T @ viewmat[:3, 3]


def std_along(rot, scales, direction):
    """sugar_model.py: || s * (R^T v) ||, the Gaussian's standard deviation along unit v."""
    return (scales * (rot.transpose(-1, -2) @ direction[..., None])[..., 0]).norm(dim=-1)


def knn(points, k, query=None, workers=16):
    """Exact K nearest centres (incl. the point itself), scipy cKDTree on CPU.

    A fixed worker count: the host is shared and all-core thread pools oversubscribe it."""
    from scipy.spatial import cKDTree

    base = points.detach().float().cpu().numpy()
    tree = cKDTree(base)
    q = base if query is None else query.detach().float().cpu().numpy()
    _, idx = tree.query(q, k=k, workers=workers)
    idx = np.asarray(idx).reshape(len(q), k)
    return torch.from_numpy(idx.astype(np.int64)).to(points.device)


def level_set_points(
    depth, alpha, rgb, camera, means, rot, scales, opac, n_points, level, *, knn_fn,
    n_in_range=21, range_size=3.0, k=16, alpha_min=0.5, chunk=2_000_000, generator=None,
):
    """compute_level_surface_points_from_camera_fast (sugar_model.py:1885) for one view.

    Depth map: gsplat expected depth (alpha > alpha_min); neighbours: K-NN of the back-projected
    point (the official `use_gaussian_depth=True` branch). Returns points, normals, colours.
    """
    h, w = depth.shape
    valid = (alpha > alpha_min) & torch.isfinite(depth) & (depth > 0)
    pix = valid.reshape(-1).nonzero()[:, 0]
    if len(pix) > n_points:
        pix = pix[torch.randperm(len(pix), device=pix.device, generator=generator)[:n_points]]
    py, px = pix // w, pix % w
    z = depth.reshape(-1)[pix]
    K = camera["K"]
    cam = torch.stack(
        ((px + 0.5 - K[0, 2]) / K[0, 0] * z, (py + 0.5 - K[1, 2]) / K[1, 1] * z, z), dim=-1
    )
    viewmat = camera["viewmat"]
    world = (cam - viewmat[:3, 3]) @ viewmat[:3, :3]
    center = camera_center(viewmat)
    closest = knn_fn(world, k)
    inv_sr = inverse_scaled_rotation(rot, scales)
    g0 = closest[:, 0]
    to_cam = F.normalize(center - means[g0], dim=-1)
    stds = std_along(rot[g0], scales[g0], to_cam)
    t = torch.linspace(-range_size, range_size, n_in_range, device=depth.device)[None] * stds[:, None]
    dirs = F.normalize(world - center, dim=-1)
    samples = (world[:, None] + t[..., None] * dirs[:, None]).reshape(-1, 3)
    sample_idx = closest[:, None].expand(-1, n_in_range, -1).reshape(-1, k)
    dens = torch.empty(len(samples), device=depth.device)
    for s in range(0, len(samples), chunk):
        d, _, _ = neighbour_density(samples[s:s + chunk], sample_idx[s:s + chunk], means, inv_sr,
                                    opac, 1.0)
        dens[s:s + chunk] = saturate_density(d)
    dens = dens.view(-1, n_in_range)
    under, above = dens < level, dens > level
    first = above.float().argmax(dim=-1)
    empty = ~under[:, 0] | (first == 0)
    keep = ~empty
    f = first[keep][:, None]
    dv, tv = dens[keep], t[keep]
    d1, d0 = dv.gather(1, f)[:, 0], dv.gather(1, f - 1)[:, 0]
    t1, t0 = tv.gather(1, f)[:, 0], tv.gather(1, f - 1)[:, 0]
    ts = (level - d0) / (d1 - d0) * (t1 - t0) + t0
    points = world[keep] + ts[:, None] * dirs[keep]
    # Normals (sugar_model.py:2088-2110): -normalize(sum_j w_j R_j S_j^-1 (S_j^-1 R_j^T shift)).
    idx = closest[keep]
    _, weights, warped = neighbour_density(points, idx, means, inv_sr, opac, 1.0)
    grad = (weights[..., None] * (inv_sr[idx] @ warped[..., None])[..., 0]).sum(dim=-2)
    normals = -F.normalize(grad, dim=-1)
    colors = rgb.reshape(-1, 3)[pix[keep]]
    return points, normals, colors


def bound_gaussians(verts, faces, bary, rot2d, plane_log_scales, log_thickness):
    """Supp. Sec. 1 / sugar_model.py:408-505: means, wxyz quats and scales of bound Gaussians.

    bary [n,3]; rot2d [F*n,2]; plane_log_scales [F*n,2]. Gaussian index = face * n + j.
    """
    n = bary.shape[0]
    fv = verts[faces]  # F,3,3
    means = (bary[None, :, :, None] * fv[:, None]).sum(dim=-2).reshape(-1, 3)
    r0 = F.normalize(torch.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0], dim=-1), dim=-1)
    b1 = F.normalize(fv[:, 0] - fv[:, 1], dim=-1)
    b2 = F.normalize(torch.cross(r0, b1, dim=-1), dim=-1)
    c = F.normalize(rot2d, dim=-1).view(len(faces), n, 2)
    r1 = c[..., 0:1] * b1[:, None] + c[..., 1:2] * b2[:, None]
    r2 = -c[..., 1:2] * b1[:, None] + c[..., 0:1] * b2[:, None]
    rot = torch.stack((r0[:, None].expand(-1, n, -1), r1, r2), dim=-1).reshape(-1, 3, 3)
    quats = F.normalize(matrix_to_quaternion(rot), dim=-1)
    scales = torch.cat(
        (torch.full_like(plane_log_scales[:, :1], log_thickness), plane_log_scales), dim=-1
    ).exp()
    return means, quats, scales


def initial_plane_log_scales(verts, faces, n):
    """sugar_model.py:330-333: min edge length * in-circle radius, per Gaussian (2 equal)."""
    fv = verts[faces]
    edges = (fv - fv[:, [1, 2, 0]]).norm(dim=-1).min(dim=-1).values * BARY[n][1]
    return edges.clamp_min(1e-7).log()[:, None, None].expand(-1, n, 2).reshape(-1, 2).clone()


def edge_face_pairs(faces, n_verts):
    """Face pairs sharing an edge, as (v0, v1, opposite_a, opposite_b) [P,4] (PyTorch3D order)."""
    e = torch.cat((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    opp = torch.cat((faces[:, 2], faces[:, 0], faces[:, 1]))
    e = e.sort(dim=1).values
    key = e[:, 0] * n_verts + e[:, 1]
    order = key.argsort()
    ks = key[order]
    _, counts = torch.unique_consecutive(ks, return_counts=True)
    rows = []
    for off in range(1, int(counts.max()) if len(counts) else 1):
        i = torch.arange(len(ks) - off, device=faces.device)
        same = ks[i] == ks[i + off]
        a, b = order[i[same]], order[i[same] + off]
        rows.append(torch.stack((e[a, 0], e[a, 1], opp[a], opp[b]), dim=-1))
    if not rows:
        return torch.zeros(0, 4, dtype=torch.long, device=faces.device)
    return torch.cat(rows)


def normal_consistency(verts, pairs):
    """pytorch3d.loss.mesh_normal_consistency for one mesh: mean of 1 - cos over face pairs."""
    if len(pairs) == 0:
        return verts.sum() * 0
    v0, v1, va, vb = (verts[pairs[:, i]] for i in range(4))
    n0 = torch.cross(v1 - v0, va - v0, dim=-1)
    n1 = -torch.cross(v1 - v0, vb - v0, dim=-1)
    return (1 - F.cosine_similarity(n0, n1, dim=1, eps=1e-6)).mean()


def expon_lr(step, lr_init, lr_final, max_steps):
    """3DGS get_expon_lr_func with lr_delay_steps = 0 (sugar_utils/general_utils.py)."""
    t = min(max(step / max_steps, 0.0), 1.0)
    return math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)
