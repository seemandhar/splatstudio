"""Independent float64 PyTorch half-Gaussian compositor (probe reference, not used in training).

Written from paper Eq. 7-10 and the conventions of the official kernel (see raster.py docstring),
with plain matrix algebra instead of the kernel's glm column-major code and hand-written
derivatives. Autograd provides the reference gradients. Brute force over all pixels; only for
tiny scenes.
"""

import math

import torch
import torch.nn.functional as F

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792,
         0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
         -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def quat_to_rot(q):
    q = F.normalize(q, dim=-1)
    r, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def sh_color(sh, dirs, degree):
    """Inria computeColorFromSH: sh [N,16,3], unit dirs [N,3]; +0.5, clamped at 0."""
    x, y, z = (dirs[:, i:i + 1] for i in range(3))
    res = SH_C0 * sh[:, 0]
    if degree > 0:
        res = res - SH_C1 * y * sh[:, 1] + SH_C1 * z * sh[:, 2] - SH_C1 * x * sh[:, 3]
    if degree > 1:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        res = (res + SH_C2[0] * xy * sh[:, 4] + SH_C2[1] * yz * sh[:, 5]
               + SH_C2[2] * (2 * zz - xx - yy) * sh[:, 6] + SH_C2[3] * xz * sh[:, 7]
               + SH_C2[4] * (xx - yy) * sh[:, 8])
    if degree > 2:
        res = (res + SH_C3[0] * y * (3 * xx - yy) * sh[:, 9] + SH_C3[1] * xy * z * sh[:, 10]
               + SH_C3[2] * y * (4 * zz - xx - yy) * sh[:, 11]
               + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
               + SH_C3[4] * x * (4 * zz - xx - yy) * sh[:, 13] + SH_C3[5] * z * (xx - yy) * sh[:, 14]
               + SH_C3[6] * x * (xx - 3 * yy) * sh[:, 15])
    return (res + 0.5).clamp_min(0)


def project(means, scales, quats, normal, camera):
    """Per-Gaussian 2D mean (pixels), 3x3 ray-space covariance (+0.3 dilation), camera normal,
    depth. Ray space: (u, v) = image pixels, w = fy * z_cam / t_z (official third Jacobian row)."""
    V = camera["viewmat"].to(means)
    K = camera["K"].to(means)
    W, H = int(camera["width"]), int(camera["height"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    R, tv = V[:3, :3], V[:3, 3]
    t = means @ R.T + tv
    tz = t[:, 2]
    xy = torch.stack([fx * t[:, 0] / tz + cx - 0.5, fy * t[:, 1] / tz + cy - 0.5], -1)
    limx, limy = 1.3 * W / (2 * fx), 1.3 * H / (2 * fy)
    tx = (t[:, 0] / tz).clamp(-limx, limx) * tz
    ty = (t[:, 1] / tz).clamp(-limy, limy) * tz
    zero = torch.zeros_like(tz)
    J = torch.stack([fx / tz, zero, -fx * tx / tz ** 2,
                     zero, fy / tz, -fy * ty / tz ** 2,
                     zero, zero, fy / tz], -1).reshape(-1, 3, 3)
    Rq = quat_to_rot(quats)
    M = Rq * scales[:, None, :]
    cov_w = M @ M.transpose(1, 2)
    cov = J @ R @ cov_w @ R.T @ J.transpose(1, 2)
    cov = cov + 0.3 * torch.eye(3, dtype=means.dtype, device=means.device)
    n_cam = normal @ R.T + tv  # official transformPoint4x3 on the normal (translation added)
    return xy, cov, n_cam, tz


def render(params, camera, sh_degree=0, background=(0.0, 0.0, 0.0), opacities2=None,
           power_cutoff=-4.8, kernel="half"):
    """Returns HWC RGB. `params` holds means, scales (log), quats, opacities, opacity2 (logits),
    sh0, shN, hgs_normal (any float dtype). `opacities2` [N,2] overrides the activated pair.
    kernel="full" is a plain 3DGS compositor (alpha = a1 * G, no erf, normal unused)."""
    if kernel not in ("half", "full"):
        raise ValueError("kernel must be 'half' or 'full'")
    dt = params["means"].dtype
    dev = params["means"].device
    means = params["means"]
    scales = params["scales"].exp()
    if opacities2 is None:
        opacities2 = torch.stack([params["opacities"], params["opacity2"]], -1).sigmoid()
    xy, cov, n_cam, depth = project(means, scales, params["quats"], params["hgs_normal"], camera)
    a, b, c = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    det = a * c - b * b
    con = torch.stack([c / det, -b / det, a / det], -1)
    s_xz, s_yz, s_zz = cov[:, 0, 2], cov[:, 1, 2], cov[:, 2, 2]
    # conditional distribution of w given (u, v): mean = beta . (u, v), var = s_zz - ...
    beta = torch.stack([s_xz * con[:, 0] + s_yz * con[:, 1], s_yz * con[:, 2] + s_xz * con[:, 1]],
                       -1)
    var = s_zz - (s_xz ** 2 * con[:, 0] + 2 * s_xz * s_yz * con[:, 1] + s_yz ** 2 * con[:, 2])
    inv_sqrt2sigma = 1 / (math.sqrt(2) * var.clamp_min(1e-8).sqrt())
    slope = n_cam[:, :2] / (n_cam[:, 2:3] + 1e-6)  # plane w = -(n_u u + n_v v) / n_w
    campos = torch.linalg.inv(camera["viewmat"].to(dt))[:3, 3]
    dirs = F.normalize(means - campos, dim=-1)
    sh = torch.cat([params["sh0"], params["shN"]], 1)
    colors = sh_color(sh, dirs, sh_degree)

    W, H = int(camera["width"]), int(camera["height"])
    py, px = torch.meshgrid(torch.arange(H, dtype=dt, device=dev),
                            torch.arange(W, dtype=dt, device=dev), indexing="ij")
    T = torch.ones(H, W, dtype=dt, device=dev)
    C = torch.zeros(H, W, 3, dtype=dt, device=dev)
    done = torch.zeros(H, W, dtype=torch.bool, device=dev)
    for i in torch.argsort(depth).tolist():
        if depth[i] <= 0.2:  # in_frustum near test
            continue
        dx, dy = xy[i, 0] - px, xy[i, 1] - py  # official d = mean - pixel
        power = -0.5 * (con[i, 0] * dx * dx + con[i, 2] * dy * dy) - con[i, 1] * dx * dy
        arg = ((slope[i, 0] + beta[i, 0]) * dx + (slope[i, 1] + beta[i, 1]) * dy) * inv_sqrt2sigma[i]
        e = torch.erf(arg)
        G = torch.exp(power)
        # Eq. 10: 0.5 * (a1 * G * (1 + erf) + a2 * G * (1 - erf))
        if kernel == "half":
            alpha = 0.5 * (opacities2[i, 0] * G * (1 + e) + opacities2[i, 1] * G * (1 - e))
        else:
            alpha = opacities2[i, 0] * G
        alpha = alpha.clamp_max(0.99)
        valid = (power <= 0) & (alpha >= 1 / 255) & ~done
        if power_cutoff is not None:
            valid &= power >= power_cutoff
        test_T = T * (1 - alpha)
        stop = valid & (test_T < 1e-4)
        done = done | stop
        use = valid & ~stop
        w = torch.where(use, alpha * T, torch.zeros_like(T))
        C = C + w[..., None] * colors[i]
        T = torch.where(use, test_T, T)
    bg = torch.as_tensor(background, dtype=dt, device=dev)
    return C + T[..., None] * bg
