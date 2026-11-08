"""Independent float64 PyTorch references for the optimal projection.

`official_jacobian` transcribes forward.cu `computeCov2D` (mu / theta / phi / atan2 form)
verbatim; `paper_jacobian` evaluates Q J_p from the paper's Eq. 12-14 and 18; `dense_render`
evaluates every Gaussian at every pixel with the official renderCUDA trig formulas and the
Inria blending rules (alpha clamp with straight-through gradient, skip power > 0 and
alpha < 1/255, stop when T (1 - alpha) < 1e-4), restricted to each Gaussian's gsplat tile
rectangle so it sees exactly the pairs the tiled kernel sees.
"""

import torch

ALPHA_THRESHOLD = 1.0 / 255.0


def official_jacobian(t, fx, fy):
    """forward.cu computeCov2D J (math rows u, v), [N,2,3]."""
    tx, ty, tz = t.unbind(-1)
    dis_inv = 1.0 / (torch.sqrt(tx * tx + ty * ty + tz * tz) + 1e-7)
    mx, my, mz = tx * dis_inv, ty * dis_inv, tz * dis_inv
    mut = mx * tx + my * ty + mz * tz
    inv = 1.0 / (mut * mut + 1e-7)
    theta = torch.atan2(-my, torch.sqrt(mx * mx + mz * mz))
    phi = torch.atan2(mx, mz)
    sp, cp, st, ct = torch.sin(phi), torch.cos(phi), torch.sin(theta), torch.cos(theta)
    ju = torch.stack((
        fx * ((mx * tz * sp + my * ty * cp + mz * tz * cp) * inv),
        fx * ((my * (-tx * cp + tz * sp)) * inv),
        fx * (-(mx * tx * sp + my * ty * sp + mz * tx * cp) * inv)), -1)
    jv = torch.stack((
        fy * ((-mx * ty * ct - mx * tz * st * cp + my * ty * sp * st + mz * tz * sp * st) * inv),
        fy * ((mx * tx * ct - my * tx * sp * st - my * tz * st * cp + mz * tz * ct) * inv),
        fy * ((mx * tx * st * cp + my * ty * st * cp - mz * tx * sp * st - mz * ty * ct) * inv)),
        -1)
    return torch.stack((ju, jv), -2)


def paper_jacobian(t, fx, fy):
    """diag(fx, fy) * rows 1-2 of Q J_p with J_p from Eq. 14 and Q from Eq. 18."""
    n = t.shape[0]
    xp = t / t.norm(dim=-1, keepdim=True)  # Eq. 12
    d = (xp * t).sum(-1)  # x_p^T mu'
    eye = torch.eye(3, dtype=t.dtype, device=t.device).expand(n, 3, 3)
    Jp = eye / d[:, None, None] - t[:, :, None] * xp[:, None, :] / (d * d)[:, None, None]
    mx, my, mz = t.unbind(-1)
    rho = torch.sqrt(mx * mx + mz * mz)
    r = t.norm(dim=-1)
    zero = torch.zeros_like(mx)
    Q = torch.stack((torch.stack((mz / rho, zero, -mx / rho), -1),
                     torch.stack((-mx * my / (rho * r), rho / r, -my * mz / (rho * r)), -1),
                     torch.stack((mx / r, my / r, mz / r), -1)), -2)
    QJ = (Q @ Jp)[:, :2]
    return QJ * torch.tensor([fx, fy], dtype=t.dtype, device=t.device)[None, :, None]


def _sphere(vx, vy):
    """official renderCUDA: (vx, vy, 1) -> theta, phi and the unit vector."""
    theta = torch.atan2(-vy, torch.sqrt(vx * vx + 1.0))
    phi = torch.atan2(vx, torch.ones_like(vx))
    st, ct, sp, cp = torch.sin(theta), torch.cos(theta), torch.sin(phi), torch.cos(phi)
    return st, ct, sp, cp, (ct * sp, -st, ct * cp)


def tile_mask(means2d, radii, width, height, tile_size=16):
    """[N,H,W] bool: pixel lies in a tile of the Gaussian's gsplat isect_tiles rectangle."""
    tw = -(-width // tile_size)
    th = -(-height // tile_size)
    r = radii.to(means2d.dtype)
    tx, ty = means2d[:, 0] / tile_size, means2d[:, 1] / tile_size
    rx, ry = r[:, 0] / tile_size, r[:, 1] / tile_size
    x0 = torch.floor(tx - rx).clamp(0, tw)
    x1 = torch.ceil(tx + rx).clamp(0, tw)
    y0 = torch.floor(ty - ry).clamp(0, th)
    y1 = torch.ceil(ty + ry).clamp(0, th)
    px = torch.div(torch.arange(width, device=means2d.device), tile_size, rounding_mode="floor")
    py = torch.div(torch.arange(height, device=means2d.device), tile_size, rounding_mode="floor")
    inx = (px[None] >= x0[:, None]) & (px[None] < x1[:, None])
    iny = (py[None] >= y0[:, None]) & (py[None] < y1[:, None])
    ok = (radii > 0).all(-1)
    return iny[:, :, None] & inx[:, None, :] & ok[:, None, None]


def dense_render(means2d, conics, colors, opacities, depths, mask, K, width, height,
                 focal_floor=512.0, mode="optimal", alpha_max=0.99):
    """Float64 dense render, (rgb [H,W,C], alpha [H,W,1]) without background."""
    order = torch.argsort(depths)
    m, q, c, o, mask = means2d[order], conics[order], colors[order], opacities[order], mask[order]
    fx, fy, cx, cy = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    fpx, fpy = max(focal_floor, fx), max(focal_floor, fy)
    ys, xs = torch.meshgrid(torch.arange(height, dtype=m.dtype, device=m.device) + 0.5,
                            torch.arange(width, dtype=m.dtype, device=m.device) + 0.5,
                            indexing="ij")
    if mode == "optimal":
        _, _, _, _, (tx, ty, tz) = _sphere((xs - cx) / fx, (ys - cy) / fy)  # [H,W]
        st, ct, sp, cp, (mx, my, mz) = _sphere((m[:, 0] - cx) / fx, (m[:, 1] - cy) / fy)
        st, ct, sp, cp, mx, my, mz = (v[:, None, None] for v in (st, ct, sp, cp, mx, my, mz))
        dot = mx * tx + my * ty + mz * tz  # [N,H,W]
        front = dot >= 1e-7
        dot_s = torch.where(front, dot, torch.ones_like(dot))
        u = fpx * (tx * cp - tz * sp) / dot_s
        v = fpy * (tx * sp * st + ty * ct + tz * st * cp) / dot_s
        dx, dy = -u, -v
    else:
        dx = m[:, 0, None, None] - xs
        dy = m[:, 1, None, None] - ys
        front = torch.ones_like(dx, dtype=torch.bool)
    a, b, cc = (q[:, i, None, None] for i in range(3))
    power = -0.5 * (a * dx * dx + cc * dy * dy) - b * dx * dy
    raw = o[:, None, None] * torch.exp(power)
    alpha = raw - (raw - raw.clamp(max=alpha_max)).detach()  # Inria: gradient through clamp
    valid = mask & front & (power <= 0) & (alpha.detach() >= ALPHA_THRESHOLD)
    alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
    T_after = torch.cumprod(1.0 - alpha, dim=0)
    keep = T_after.detach() >= 1e-4  # stop at the first T (1 - alpha) < 1e-4 (excluded)
    alpha = torch.where(keep, alpha, torch.zeros_like(alpha))
    T_after = torch.cumprod(1.0 - alpha, dim=0)
    T_before = torch.cat((torch.ones_like(T_after[:1]), T_after[:-1]), dim=0)
    w = alpha * T_before
    rgb = torch.einsum("nhw,nc->hwc", w, c)
    return rgb, (1.0 - T_after[-1])[..., None]


def exact_ray_response(mean_cam, cov_cam, K, width, height):
    """Max along each pixel ray of an (unnormalised) 3D Gaussian in camera space, [H,W]:
    exp(-1/2 (mu^T S mu - (t^T S mu)^2 / (t^T S t))), S = Sigma^-1 — the quantity both
    projections approximate."""
    fx, fy, cx, cy = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    ys, xs = torch.meshgrid(torch.arange(height, dtype=mean_cam.dtype) + 0.5,
                            torch.arange(width, dtype=mean_cam.dtype) + 0.5, indexing="ij")
    t = torch.stack(((xs - cx) / fx, (ys - cy) / fy, torch.ones_like(xs)), -1)
    S = torch.linalg.inv(cov_cam)
    Smu = S @ mean_cam
    tSt = torch.einsum("hwi,ij,hwj->hw", t, S, t)
    tSmu = t @ Smu
    return torch.exp(-0.5 * (mean_cam @ Smu - tSmu * tSmu / tSt))
