"""Independent float64 PyTorch implementation of Analytic-Splatting shading (paper Eq. 7, 11-13)
used only by the probes. Written from the paper; conventions (Inria pixel centres, EWA Jacobian
with 1.3·tan(fov) clamp, no 0.3 px dilation, alpha clamp 0.99, 1/255 skip, T < 1e-4 stop)
follow the official forward.cu. Autograd provides the reference gradients.
"""

import math

import torch

C0 = 0.28209479177387814


def logistic_cdf(x):
    """Eq. 7: S(x) = 1 / (1 + exp(-1.6 x - 0.07 x^3))."""
    return torch.sigmoid(1.6 * x + 0.07 * x ** 3)


def logistic_cdf_grad(x):
    """Supp.: dS/dx = (1.6 + 0.21 x^2) S (1 - S)."""
    s = logistic_cdf(x)
    return (1.6 + 0.21 * x ** 2) * s * (1 - s)


def gaussian_cdf(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2.0)))


def window_integral_2d(d, cov2):
    """Eq. 13 for offsets d = u - mu [..., 2] and 2D covariances cov2 [..., 2, 2].

    Eigen-decomposition by torch.linalg.eigh (independent of the kernel's closed form).
    """
    lam, vec = torch.linalg.eigh(cov2)  # ascending: lam[..., 1] = lambda_1 (major)
    sig1 = lam[..., 1].sqrt()
    sig2 = lam[..., 0].abs().sqrt()
    u = (d * vec[..., :, 1]).sum(-1)
    v = (d * vec[..., :, 0]).sum(-1)
    int_u = sig1 * (logistic_cdf((u + 0.5) / sig1) - logistic_cdf((u - 0.5) / sig1))
    int_v = sig2 * (logistic_cdf((v + 0.5) / sig2) - logistic_cdf((v - 0.5) / sig2))
    return 2 * math.pi * int_u * int_v


def point_sample_2d(d, cov2):
    """3DGS centre sample exp(-0.5 d^T Sigma^-1 d) (for comparison only)."""
    inv = torch.linalg.inv(cov2)
    return torch.exp(-0.5 * torch.einsum("...i,...ij,...j->...", d, inv, d))


def quadrature_2d(d, cov2, n=64):
    """Brute-force midpoint quadrature of the Gaussian over the unit pixel square around d."""
    t = (torch.arange(n, dtype=torch.float64) + 0.5) / n - 0.5
    oy, ox = torch.meshgrid(t, t, indexing="ij")
    off = torch.stack([ox, oy], -1).reshape(-1, 2)  # [n^2, 2]
    dd = d[..., None, :] + off  # [..., n^2, 2]
    return point_sample_2d(dd, cov2[..., None, :, :]).mean(-1)


def rotmat(q):
    """Rotation of normalized wxyz quaternions via v' = v + 2w(u x v) + 2u x (u x v)."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, u = q[:, :1], q[:, 1:]
    cols = []
    for e in torch.eye(3, dtype=q.dtype):
        v = e.expand_as(u)
        t = 2 * torch.cross(u, v, dim=-1)
        cols.append(v + w * t + torch.cross(u, t, dim=-1))
    return torch.stack(cols, dim=-1)


def project(params, camera, offset2d=None):
    """Pixel-space means (Inria ndc2Pix convention), 2D covariances (no dilation), depths."""
    dd = torch.float64
    view = camera["viewmat"].to(dd)
    K = camera["K"].to(dd)
    W, H = int(camera["width"]), int(camera["height"])
    means = params["means"].to(dd)
    t = means @ view[:3, :3].T + view[:3, 3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    tanx, tany = W / (2 * fx), H / (2 * fy)
    tz = t[:, 2]
    tx = (t[:, 0] / tz).clamp(-1.3 * tanx, 1.3 * tanx) * tz
    ty = (t[:, 1] / tz).clamp(-1.3 * tany, 1.3 * tany) * tz
    zero = torch.zeros_like(tz)
    J = torch.stack([torch.stack([fx / tz, zero, -fx * tx / tz ** 2], -1),
                     torch.stack([zero, fy / tz, -fy * ty / tz ** 2], -1)], -2)
    R = rotmat(params["quats"].to(dd))
    s = params["scales"].to(dd).exp()
    cov3 = R @ torch.diag_embed(s ** 2) @ R.transpose(1, 2)
    Wr = view[:3, :3]
    cov2 = J @ Wr @ cov3 @ Wr.T @ J.transpose(1, 2)
    # ndc = 2 (f x/z + c) / W - 1; ndc2Pix: ((ndc + 1) W - 1) / 2 = f x/z + c - 0.5
    mu = torch.stack([fx * t[:, 0] / tz + cx - 0.5, fy * t[:, 1] / tz + cy - 0.5], -1)
    if offset2d is not None:
        mu = mu + offset2d
    return mu, cov2, tz


def composite(params, camera, background, offset2d=None):
    """Front-to-back alpha blending with the kernel's thresholds. Returns HWC RGB (float64).

    Colours from SH degree 0: max(C0 * sh0 + 0.5, 0). All Gaussians are assumed to cover the
    whole (single-tile) image, as the probes construct it.
    """
    mu, cov2, depth = project(params, camera, offset2d)
    W, H = int(camera["width"]), int(camera["height"])
    opac = params["opacities"].to(torch.float64).sigmoid()
    col = (C0 * params["sh0"].to(torch.float64)[:, 0] + 0.5).clamp_min(0)
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float64),
                            torch.arange(W, dtype=torch.float64), indexing="ij")
    pix = torch.stack([xs, ys], -1).reshape(-1, 2)
    T = torch.ones(len(pix), dtype=torch.float64)
    C = torch.zeros(len(pix), 3, dtype=torch.float64)
    alive = torch.ones(len(pix), dtype=torch.bool)
    for i in torch.argsort(depth).tolist():
        G = window_integral_2d(pix - mu[i], cov2[i].expand(len(pix), 2, 2))
        alpha = torch.clamp(opac[i] * G, max=0.99)
        use = alive & (alpha >= 1.0 / 255.0)
        test_T = T * (1 - alpha)
        stop = use & (test_T < 1e-4)
        alive = alive & ~stop
        use = use & ~stop
        a = torch.where(use, alpha, torch.zeros_like(alpha))
        C = C + (a * T)[:, None] * col[i]
        T = torch.where(use, test_T, T)
    C = C + T[:, None] * background.to(torch.float64)
    return C.reshape(H, W, 3)
