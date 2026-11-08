"""Independent dense float64 PyTorch renderer of AAA-Gaussians (CPU, differentiable).

Written from the paper (§3.1 "Evaluation in 3D", Eq. 7-12) and the Inria conventions, not by
translating the kernels: every pixel ray (through the pixel centre, cx = W/2) is intersected
with every Gaussian in 3D,

    rho^2_min = min_t |S_d^-1 R^T (o + t d - mu)|^2,     S_d^2 = S^2 + v I,
    v = max(filter_3d^2, k z_view^2 / f^2)               (Eq. 12, f = max(fx, fy)),
    alpha = min(0.99, sigmoid(o) * a(mu, s, R) * exp(-rho^2_min / 2)),

with the amplitude a = perpendicular scaling (Eq. 11, default), volume scaling (Eq. 8,
``scaling="volume"``) or 1 (``scaling="none"``). Fragments are sorted per pixel by the view
depth of the max-contribution point, alpha < 1/255 is skipped, compositing stops when
T (1 - alpha) < 1e-4, background is added with the final T.

Conventions copied from the kernel's *gradient* (so ``backward`` probes compare like with
like): v is a constant (no gradient through z_view), the min(0.99) clamp passes the gradient
straight through. Colours are SH degree 0 (C0 sh0 + 0.5, clamped at 0).
"""

import torch

C0 = 0.28209479177387814
DD = torch.float64
ALPHA_MIN = 1.0 / 255.0


def rotmat(quats):
    """Standard wxyz -> rotation matrix (columns = Gaussian axes in world), any norm."""
    q = torch.nn.functional.normalize(quats.to(DD), dim=-1)
    r, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def pixel_rays(camera):
    """World ray origin [3] and unit directions [H*W, 3] through pixel centres."""
    K = camera["K"].detach().cpu().to(DD)
    view = camera["viewmat"].detach().cpu().to(DD)
    W, H = int(camera["width"]), int(camera["height"])
    ys, xs = torch.meshgrid(torch.arange(H, dtype=DD), torch.arange(W, dtype=DD), indexing="ij")
    dc = torch.stack([(xs.reshape(-1) + 0.5 - K[0, 2]) / K[0, 0],
                      (ys.reshape(-1) + 0.5 - K[1, 2]) / K[1, 1],
                      torch.ones(H * W, dtype=DD)], -1)
    c2w = torch.linalg.inv(view)
    d = dc @ c2w[:3, :3].T
    return c2w[:3, 3], d / d.norm(dim=-1, keepdim=True)


def amplitude(scales, R, view_dir, v, scaling):
    s2 = scales * scales
    sd = s2 + v[:, None]
    if scaling == "none":
        return torch.ones_like(v)
    if scaling == "volume":  # Eq. (8)
        return torch.sqrt(s2.prod(-1) / sd.prod(-1))
    if scaling != "perpendicular":
        raise ValueError(scaling)
    dp = torch.einsum("nji,nj->ni", R, view_dir)  # d' = R^T d
    r = dp * dp

    def pair(a):
        return torch.stack([a[:, 1] * a[:, 2], a[:, 0] * a[:, 2], a[:, 0] * a[:, 1]], -1)

    return torch.sqrt((r * pair(s2)).sum(-1) / (r * pair(sd)).sum(-1))  # Eq. (11)


def leaves_from(params):
    return {k: params[k].detach().cpu().to(DD).clone().requires_grad_()
            for k in ("means", "scales", "quats", "opacities", "sh0")}


def composite(leaves, camera, background, kernel=0.3, filter_3d=None, scaling="perpendicular",
              znear=0.01, zfar=100.0):
    """Render [H, W, 3] float64 from float64 leaves (means, log scales, quats, opacity
    logits, sh0); differentiable w.r.t. all leaves."""
    W, H = int(camera["width"]), int(camera["height"])
    K = camera["K"].detach().cpu().to(DD)
    view = camera["viewmat"].detach().cpu().to(DD)
    mu, s = leaves["means"], leaves["scales"].exp()
    R = rotmat(leaves["quats"])
    o, rays = pixel_rays(camera)
    z_view = (mu.detach() @ view[:3, :3].T + view[:3, 3])[:, 2]
    v = kernel * (z_view / torch.maximum(K[0, 0], K[1, 1])) ** 2
    if filter_3d is not None:
        v = torch.maximum(torch.as_tensor(filter_3d, dtype=DD).reshape(-1) ** 2, v)
    v = v.detach()
    sd = torch.sqrt(s * s + v[:, None])
    rel = mu - o
    view_dir = rel / rel.norm(dim=-1, keepdim=True)
    opac = leaves["opacities"].sigmoid() * amplitude(s, R, view_dir, v, scaling)
    # Gaussian-space line: y(t) = A (o - mu) + t A d, A = S_d^-1 R^T
    A = R.transpose(1, 2) / sd[:, :, None]
    y0 = torch.einsum("nij,nj->ni", A, -rel)  # [N,3]
    u = torch.einsum("nij,pj->pni", A, rays)  # [P,N,3]
    uu = (u * u).sum(-1)
    yu = (u * y0[None]).sum(-1)
    t = -yu / uu
    rho2 = (y0 * y0).sum(-1)[None] - yu * yu / uu
    raw = opac[None] * torch.exp(-0.5 * rho2)
    alpha = raw - (raw - 0.99).clamp_min(0).detach()
    x = o + t.detach()[..., None] * rays[:, None, :]  # max-contribution points [P,N,3]
    depth = (x @ view[:3, :3].T + view[:3, 3])[..., 2]
    ndc = zfar * (depth - znear) / ((zfar - znear) * depth)
    usable = (opac.detach()[None] >= ALPHA_MIN) & (ndc >= -1) & (ndc <= 1) & (t.detach() > 0)
    col = (C0 * leaves["sh0"][:, 0] + 0.5).clamp_min(0)
    idx = torch.argsort(torch.where(usable, depth, torch.inf), dim=1, stable=True)
    P, N = alpha.shape
    T = torch.ones(P, dtype=DD)
    C = torch.zeros(P, 3, dtype=DD)
    alive = torch.ones(P, dtype=torch.bool)
    for k in range(N):
        g = idx[:, k]
        al = alpha.gather(1, g[:, None])[:, 0]
        ok = usable.gather(1, g[:, None])[:, 0]
        valid = alive & ok & (al.detach() >= ALPHA_MIN)
        test_T = T * (1 - al)
        stop = valid & (test_T.detach() < 1e-4)
        use = valid & ~stop
        alive = alive & ~stop
        C = C + torch.where(use[:, None], col[g] * (al * T)[:, None], 0)
        T = torch.where(use, test_T, T)
    C = C + T[:, None] * torch.as_tensor(background, dtype=DD).cpu()
    return C.reshape(H, W, 3)
