"""Dense pure-torch reference of the BBSplat forward pass, written independently of the CUDA
code from the paper's equations (§3.2): planar billboard H = [R S | mu], ray-plane intersection
in (u, v), bilinear texture lookup on [-1, 1]^2 with torch.nn.functional.grid_sample
(align_corners=True, zero padding), colour = T^RGB + max(SH(d_centre) + 0.5, 0), alpha clamped
at 0.99, contributions below 1/255 skipped, front-to-back blending stopped when the
transmittance would fall below 1e-4. Billboards are blended in camera-depth order of their
centres (the probe scenes are built so that this equals the kernel's per-tile depth order).
"""

import torch
import torch.nn.functional as F

C0, C1 = 0.28209479177387814, 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792,
      0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def eval_sh(deg, sh, dirs):
    """Real SH of the 3DGS convention (utils/sh_utils.eval_sh); sh [N,16,3], dirs [N,3]."""
    x, y, z = F.normalize(dirs, dim=-1).unbind(-1)
    x, y, z = x[:, None], y[:, None], z[:, None]
    r = C0 * sh[:, 0]
    if deg > 0:
        r = r - C1 * y * sh[:, 1] + C1 * z * sh[:, 2] - C1 * x * sh[:, 3]
    if deg > 1:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        r = (r + C2[0] * xy * sh[:, 4] + C2[1] * yz * sh[:, 5]
             + C2[2] * (2 * zz - xx - yy) * sh[:, 6] + C2[3] * xz * sh[:, 7]
             + C2[4] * (xx - yy) * sh[:, 8])
    if deg > 2:
        r = (r + C3[0] * y * (3 * xx - yy) * sh[:, 9] + C3[1] * xy * z * sh[:, 10]
             + C3[2] * y * (4 * zz - xx - yy) * sh[:, 11]
             + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
             + C3[4] * x * (4 * zz - xx - yy) * sh[:, 13] + C3[5] * z * (xx - yy) * sh[:, 14]
             + C3[6] * x * (xx - 3 * yy) * sh[:, 15])
    return r


def quat_to_rotmat(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def render_reference(means, quats, scales2, sh, tex_alpha, tex_color, sh_degree, viewmat, fx,
                     fy, width, height, background):
    """Returns rgb [H,W,3], alpha [H,W], depth-sum [H,W] (sum z alpha T), impact [N]."""
    dev, dt = means.device, means.dtype
    R = quat_to_rotmat(quats)
    Rw, t = viewmat[:3, :3], viewmat[:3, 3]
    pc = means @ Rw.T + t  # camera-space centres
    tu = (R[:, :, 0] * scales2[:, :1]) @ Rw.T  # camera-space plane axes
    tv = (R[:, :, 1] * scales2[:, 1:2]) @ Rw.T
    ys, xs = torch.meshgrid(torch.arange(height, device=dev, dtype=dt) + 0.5,
                            torch.arange(width, device=dev, dtype=dt) + 0.5, indexing="ij")
    # camera ray direction for pixel centre (principal point W/2, H/2)
    d = torch.stack([(xs - width / 2) / fx, (ys - height / 2) / fy, torch.ones_like(xs)], -1)
    # solve pc + u tu + v tv = z d  for (u, v, z): [tu tv -d] (u v z)^T = -pc
    A = torch.stack([tu[:, None, None, :].expand(-1, height, width, -1),
                     tv[:, None, None, :].expand(-1, height, width, -1),
                     -d[None].expand(len(means), -1, -1, -1)], dim=-1)  # [N,H,W,3,3]
    rhs = -pc[:, None, None, :].expand(-1, height, width, -1)
    sol = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
    u, v, z = sol.unbind(-1)
    grid = torch.stack([u, v], dim=-1).to(tex_alpha.dtype)  # x <- u (columns), y <- v (rows)
    a = F.grid_sample(tex_alpha[:, None], grid, mode="bilinear", padding_mode="zeros",
                      align_corners=True)[:, 0]  # [N,H,W]
    ct = F.grid_sample(tex_color, grid, mode="bilinear", padding_mode="zeros",
                       align_corners=True)  # [N,3,H,W]
    campos = -Rw.T @ t
    dirs = means - campos
    base = eval_sh(sh_degree, sh, dirs) + 0.5
    base = torch.clamp_min(base, 0.0)
    col = ct.permute(0, 2, 3, 1) + base[:, None, None, :]  # [N,H,W,3]
    order = torch.argsort(pc[:, 2])
    T = torch.ones(height, width, device=dev, dtype=dt)
    done = torch.zeros(height, width, dtype=torch.bool, device=dev)
    rgb = torch.zeros(height, width, 3, device=dev, dtype=dt)
    dsum = torch.zeros(height, width, device=dev, dtype=dt)
    impact = torch.zeros(len(means), device=dev, dtype=dt)
    for i in order.tolist():
        if pc[i, 2] <= 0.2:
            continue
        alpha = torch.clamp(a[i], max=0.99)
        valid = (~done) & (alpha >= 1.0 / 255.0) & (z[i] >= 0.2)
        test_T = T * (1 - alpha)
        stop = valid & (test_T < 1e-4)
        done = done | stop
        use = valid & ~stop
        w = torch.where(use, alpha * T, torch.zeros_like(T))
        rgb = rgb + w[..., None] * col[i]
        dsum = dsum + w * z[i]
        impact[i] = impact[i] + w.sum().detach()
        T = torch.where(use, test_T, T)
    rgb = rgb + T[..., None] * background
    return rgb, 1 - T, dsum, impact
