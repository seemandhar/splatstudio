"""Independent float64 PyTorch compositor for the StopThePop probes (CPU, differentiable).

Written from the paper (Eq. 1, App. A) and the Inria/StopThePop conventions, not from the
kernels: EWA projection with the 1.3·tan(fov) clamp and +0.3 dilation, alpha = min(0.99,
o·exp(power)), skip if power > 0 or alpha < 1/255, stop when T·(1-alpha) < 1e-4, background
added with the final T. Blend order is either per pixel by t_opt (Eq. 1 along the ray through
``pix * 2/W - 1`` in NDC, S^-1 = 1/max(1e-3, s)) or global by view-space z of the means (3DGS).
"""

import torch

C0 = 0.28209479177387814
DD = torch.float64


def rotmat(quats):
    """Rotation matrices by rotating the basis vectors: v' = v + 2w(u x v) + 2u x (u x v)."""
    q = torch.nn.functional.normalize(quats.to(DD), dim=-1)
    w, u = q[:, :1], q[:, 1:]
    cols = []
    for e in torch.eye(3, dtype=DD):
        v = e.expand_as(u)
        t = 2 * torch.cross(u, v, dim=-1)
        cols.append(v + w * t + torch.cross(u, t, dim=-1))
    return torch.stack(cols, dim=-1)


def camera_matrices(camera, znear=0.01, zfar=100.0):
    """Row-vector world->NDC matrix (Inria full_proj_transform) and camera centre, float64."""
    K = camera["K"].detach().cpu().to(DD)
    view = camera["viewmat"].detach().cpu().to(DD)
    W, H = int(camera["width"]), int(camera["height"])
    P = torch.zeros(4, 4, dtype=DD)
    P[0, 0], P[0, 2] = 2 * K[0, 0] / W, 2 * K[0, 2] / W - 1
    P[1, 1], P[1, 2] = 2 * K[1, 1] / H, 2 * K[1, 2] / H - 1
    P[3, 2], P[2, 2], P[2, 3] = 1.0, zfar / (zfar - znear), -(zfar * znear) / (zfar - znear)
    full = view.T @ P.T
    campos = torch.linalg.inv(view)[:3, 3]
    return full, campos


def pixel_rays(camera):
    """Unit world ray directions [H*W, 3] through integer pixel coordinates (kernel pix2world)."""
    W, H = int(camera["width"]), int(camera["height"])
    full, campos = camera_matrices(camera)
    inv = torch.linalg.inv(full)
    ys, xs = torch.meshgrid(torch.arange(H, dtype=DD), torch.arange(W, dtype=DD), indexing="ij")
    ndc = torch.stack([xs.reshape(-1) * 2 / W - 1, ys.reshape(-1) * 2 / H - 1,
                       torch.zeros(H * W, dtype=DD), torch.ones(H * W, dtype=DD)], -1)
    pw = ndc @ inv
    pw = pw[:, :3] / pw[:, 3:]
    d = pw - campos
    return d / d.norm(dim=-1, keepdim=True), campos


def t_opt(means, log_scales, quats, rays, campos):
    """Eq. (1): t = d^T S^-1 (mu - o) / d^T S^-1 d  -> [P, N] (no gradient needed: order only)."""
    R = rotmat(quats.detach())
    inv_s2 = 1.0 / log_scales.detach().to(DD).exp().clamp_min(1e-3) ** 2
    Sinv = R @ torch.diag_embed(inv_s2) @ R.transpose(1, 2)
    rel = means.detach().to(DD) - campos
    num = torch.einsum("pi,nij,nj->pn", rays, Sinv, rel)
    den = torch.einsum("pi,nij,pj->pn", rays, Sinv, rays).clamp_min(1e-5)
    return num / den


def project(means, log_scales, quats, camera):
    """Pixel centres xy [N,2] (as a function of a leaf ``ndc``), conics [N,2,2], view z [N]."""
    view = camera["viewmat"].detach().cpu().to(DD)
    K = camera["K"].detach().cpu().to(DD)
    W, H = int(camera["width"]), int(camera["height"])
    fx, fy = K[0, 0], K[1, 1]
    m = means.detach().to(DD)
    t = m @ view[:3, :3].T + view[:3, 3]
    tz = t[:, 2]
    tanx, tany = W / (2 * fx), H / (2 * fy)
    tx = (t[:, 0] / tz).clamp(-1.3 * tanx, 1.3 * tanx) * tz
    ty = (t[:, 1] / tz).clamp(-1.3 * tany, 1.3 * tany) * tz
    J = torch.zeros(len(t), 2, 3, dtype=DD)
    J[:, 0, 0], J[:, 0, 2] = fx / tz, -fx * tx / tz ** 2
    J[:, 1, 1], J[:, 1, 2] = fy / tz, -fy * ty / tz ** 2
    R = rotmat(quats.detach())
    sc = log_scales.detach().to(DD).exp()
    cov3 = R @ torch.diag_embed(sc ** 2) @ R.transpose(1, 2)
    Wr = view[:3, :3]
    cov2 = J @ Wr @ cov3 @ Wr.T @ J.transpose(1, 2) + 0.3 * torch.eye(2, dtype=DD)
    conic = torch.linalg.inv(cov2)
    ndc = torch.stack([fx * t[:, 0] / tz * 2 / W + (2 * K[0, 2] / W - 1),
                       fy * t[:, 1] / tz * 2 / H + (2 * K[1, 2] / H - 1)], -1)
    return ndc, conic, tz


def composite(params, camera, background, order="perpixel"):
    """Render [H,W,3] float64. Leaves (requires_grad) for gradient checks are returned:
    ``opac_logit`` [N], ``sh0`` [N,1,3], ``ndc`` [N,2] (Inria screen-space means)."""
    W, H = int(camera["width"]), int(camera["height"])
    means, ls, q = params["means"].cpu(), params["scales"].cpu(), params["quats"].cpu()
    ndc0, conic, tz = project(means, ls, q, camera)
    leaves = {"opac_logit": params["opacities"].detach().cpu().to(DD).clone().requires_grad_(),
              "sh0": params["sh0"].detach().cpu().to(DD).clone().requires_grad_(),
              "ndc": ndc0.clone().requires_grad_()}
    xy = ((leaves["ndc"] + 1) * torch.tensor([W, H], dtype=DD) - 1) * 0.5  # ndc2Pix
    opac = leaves["opac_logit"].sigmoid()
    col = (C0 * leaves["sh0"][:, 0] + 0.5).clamp_min(0)
    ys, xs = torch.meshgrid(torch.arange(H, dtype=DD), torch.arange(W, dtype=DD), indexing="ij")
    pix = torch.stack([xs.reshape(-1), ys.reshape(-1)], -1)  # [P,2]
    d = xy[None] - pix[:, None]  # [P,N,2]
    a, b, c = conic[:, 0, 0], conic[:, 0, 1], conic[:, 1, 1]
    power = -0.5 * (a * d[..., 0] ** 2 + c * d[..., 1] ** 2) - b * d[..., 0] * d[..., 1]
    raw = opac[None] * power.exp()
    alpha = raw - (raw - 0.99).clamp_min(0).detach()  # forward clamp, straight-through grad
    P, N = alpha.shape
    if order == "perpixel":
        rays, campos = pixel_rays(camera)
        idx = torch.argsort(t_opt(means, ls, q, rays, campos), dim=1, stable=True)
    elif order == "proj":  # ablation: projection of the mean onto each ray, d . (mu - o)
        rays, campos = pixel_rays(camera)
        idx = torch.argsort(rays @ (means.detach().to(DD) - campos).T, dim=1, stable=True)
    elif order == "global":
        idx = torch.argsort(tz, stable=True)[None].expand(P, N)
    else:
        raise ValueError(order)
    T = torch.ones(P, dtype=DD)
    C = torch.zeros(P, 3, dtype=DD)
    alive = torch.ones(P, dtype=torch.bool)
    for k in range(N):
        g = idx[:, k]
        al = alpha.gather(1, g[:, None])[:, 0]
        pw = power.detach().gather(1, g[:, None])[:, 0]
        valid = alive & (pw <= 0) & (al.detach() >= 1 / 255)
        test_T = T * (1 - al)
        stop = valid & (test_T.detach() < 1e-4)
        use = valid & ~stop
        alive = alive & ~stop
        C = C + torch.where(use[:, None], col[g] * (al * T)[:, None], 0)
        T = torch.where(use, test_T, T)
    C = C + T[:, None] * torch.as_tensor(background, dtype=DD).cpu()
    return C.reshape(H, W, 3), leaves
