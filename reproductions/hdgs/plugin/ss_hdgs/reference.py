"""Dense, differentiable torch re-implementation of the two official HDGS rasterizers for small
images, written independently from the CUDA (used by the probes to validate forward and
backward). It emulates per pixel the exact traversal of renderCUDA: tile list sorted by the
ray-splat depth at the tile centre, k-buffer insertion (stage 1: alpha < 0.1 surfels go to the
buffer head/tail without sorting), blend-on-full, T < 1e-4 termination, 5-sample frustum G,
alpha = min(0.99, o G), 1/255 skip. Ordering decisions use detached values; colours, alphas and
texture lookups stay in the autograd graph.
"""

import math

import torch

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792,
         0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
         -0.4570457994644658, 1.445305721320277, -0.5900435899266435)
BLOCK = 8
TEX_RANGE = 4.5
NEAR = 0.2


def quat_to_rotmat(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)  # R[:, :, 0] = first tangent axis


def sh_rgb(sh, dirs, deg, dc=None):
    """3DGS SH -> RGB (+0.5, clamp >= 0); ``dc`` replaces the 0-th coefficient if given."""
    d = dirs / dirs.norm(dim=-1, keepdim=True)
    x, y, z = d[..., 0:1], d[..., 1:2], d[..., 2:3]
    res = SH_C0 * (sh[..., 0, :] if dc is None else dc)
    if deg > 0:
        res = res - SH_C1 * y * sh[..., 1, :] + SH_C1 * z * sh[..., 2, :] - SH_C1 * x * sh[..., 3, :]
    if deg > 1:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        res = (res + SH_C2[0] * xy * sh[..., 4, :] + SH_C2[1] * yz * sh[..., 5, :]
               + SH_C2[2] * (2 * zz - xx - yy) * sh[..., 6, :] + SH_C2[3] * xz * sh[..., 7, :]
               + SH_C2[4] * (xx - yy) * sh[..., 8, :])
        if deg > 2:
            res = (res + SH_C3[0] * y * (3 * xx - yy) * sh[..., 9, :]
                   + SH_C3[1] * xy * z * sh[..., 10, :]
                   + SH_C3[2] * y * (4 * zz - xx - yy) * sh[..., 11, :]
                   + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12, :]
                   + SH_C3[4] * x * (4 * zz - xx - yy) * sh[..., 13, :]
                   + SH_C3[5] * z * (xx - yy) * sh[..., 14, :]
                   + SH_C3[6] * x * (xx - 3 * yy) * sh[..., 15, :])
    return torch.clamp_min(res + 0.5, 0.0)


def transmats(means, scales, quats, settings):
    """T = S^T @ full_proj @ N, S = [(L0,0) (L1,0) (p,1)], N = ndc2pix; rows Tu,Tv,Tw are the
    columns of T (kernel's glm T[0..2])."""
    W, H = settings["width"], settings["height"]
    R = quat_to_rotmat(quats)
    L0 = R[:, :, 0] * scales[:, 0:1]
    L1 = R[:, :, 1] * scales[:, 1:2]
    n = len(means)
    z = torch.zeros(n, 1, dtype=means.dtype, device=means.device)
    o = torch.ones(n, 1, dtype=means.dtype, device=means.device)
    S = torch.stack([torch.cat([L0, z], 1), torch.cat([L1, z], 1), torch.cat([means, o], 1)], 2)
    full = settings["projmatrix"].to(means.dtype)
    N = torch.tensor([[W / 2, 0, 0], [0, H / 2, 0], [0, 0, 0], [(W - 1) / 2, (H - 1) / 2, 1]],
                     dtype=means.dtype, device=means.device)
    T = S.transpose(1, 2) @ full @ N  # [n,3,3]
    return T[:, :, 0], T[:, :, 1], T[:, :, 2], R


def _intersect(Tu, Tv, Tw, x, y):
    k = x * Tw - Tu
    l_ = y * Tw - Tv
    p = torch.linalg.cross(k, l_)
    return p[..., :2] / p[..., 2:3], p[..., 2]


def preprocess(Tu, Tv, Tw, means, settings):
    """in_frustum (view z > 0.2), compute_aabb (cutoff 3), radius, tile rectangle."""
    W, H = settings["width"], settings["height"]
    gx, gy = (W + BLOCK - 1) // BLOCK, (H + BLOCK - 1) // BLOCK
    view = settings["viewmatrix"].to(means.dtype)
    pv = torch.cat([means, torch.ones_like(means[:, :1])], 1) @ view
    out = []
    for i in range(len(means)):
        if float(pv[i, 2]) <= 0.2:
            out.append(None)
            continue
        t = torch.tensor([9.0, 9.0, -1.0], dtype=means.dtype)
        tw = Tw[i].detach().cpu()
        d = float((t * tw * tw).sum())
        if d == 0:
            out.append(None)
            continue
        f = t / d
        tu, tv = Tu[i].detach().cpu(), Tv[i].detach().cpu()
        px, py = float((f * tu * tw).sum()), float((f * tv * tw).sum())
        h0x = px * px - float((f * tu * tu).sum())
        h0y = py * py - float((f * tv * tv).sum())
        hx, hy = math.sqrt(max(1e-4, h0x)), math.sqrt(max(1e-4, h0y))
        radius = math.ceil(max(hx, hy, 3.0 * 0.707106))

        def tr(v):  # C float -> int truncation toward zero
            return int(v)

        rmin = (min(gx, max(0, tr((px - radius) / BLOCK))), min(gy, max(0, tr((py - radius) / BLOCK))))
        rmax = (min(gx, max(0, tr((px + radius + BLOCK - 1) / BLOCK))),
                min(gy, max(0, tr((py + radius + BLOCK - 1) / BLOCK))))
        if (rmax[0] - rmin[0]) * (rmax[1] - rmin[1]) == 0:
            out.append(None)
            continue
        out.append({"xy": (px, py), "rect": (rmin, rmax)})
    return out


def tex2d(texbuf, start, Wt, Ht, u, v):
    """tex.h bilinear tex2D (BILINEAR 1), u, v in [-1, 1] (unclamped weights, as official)."""
    u = (u + 1.0) * 0.5
    v = (v + 1.0) * 0.5
    x = u * (Wt - 1)
    y = v * (Ht - 1)
    xf, yf = float(x), float(y)
    x0 = min(max(int(xf), 0), Wt - 1)
    y0 = min(max(int(yf), 0), Ht - 1)
    x1 = min(max(int(xf) + 1, 0), Wt - 1)
    y1 = min(max(int(yf) + 1, 0), Ht - 1)

    def t(ix, iy):
        return texbuf[start + iy * Wt + ix]

    if x0 == x1 and y0 == y1:
        return t(x0, y0)
    if x0 == x1:
        a = y - y0
        return (1 - a) * t(x0, y0) + a * t(x0, y1)
    if y0 == y1:
        a = x - x0
        return (1 - a) * t(x0, y0) + a * t(x1, y0)
    s0, s1, t0, t1 = x - x0, x1 - x, y - y0, y1 - y
    return s1 * t1 * t(x0, y0) + s0 * t1 * t(x1, y0) + s1 * t0 * t(x0, y1) + s0 * t0 * t(x1, y1)


def render_reference(means, scales, quats, opac, sh, settings, variant="stage1", texture=None,
                     index=None, frustum=True, sort=True):
    """Image [3,H,W] (black background assumed by the callers). ``frustum=False`` / ``sort=False``
    are ablations (centre sample only / blend in tile order without the k-buffer)."""
    W, H = settings["width"], settings["height"]
    K = settings["sort_window"]
    deg = settings["sh_degree"]
    bg = settings["bg"].to(means.dtype)
    campos = settings["campos"].to(means.dtype)
    Tu, Tv, Tw, R = transmats(means, scales, quats, settings)
    pre = preprocess(Tu, Tv, Tw, means, settings)
    gx = (W + BLOCK - 1) // BLOCK
    centre_rgb = sh_rgb(sh, means - campos, deg) if variant == "stage1" else None
    # per-tile lists sorted by depth at the tile centre ((x+0.5)8, (y+0.5)8)
    tiles = {}
    for i, p in enumerate(pre):
        if p is None:
            continue
        (x0, y0), (x1, y1) = p["rect"]
        for ty in range(y0, y1):
            for tx in range(x0, x1):
                s, _ = _intersect(Tu[i].detach(), Tv[i].detach(), Tw[i].detach(),
                                  (tx + 0.5) * BLOCK, (ty + 0.5) * BLOCK)
                depth = float(s[0] * Tw[i, 0] + s[1] * Tw[i, 1] + Tw[i, 2])
                tiles.setdefault(ty * gx + tx, []).append((depth, i))
    for key in tiles:
        tiles[key].sort(key=lambda e: (e[0], e[1]))
    rows = []
    for py in range(H):
        row = []
        for px in range(W):
            lst = tiles.get((py // BLOCK) * gx + px // BLOCK, [])
            row.append(_pixel(px, py, lst, Tu, Tv, Tw, R, scales, means, opac, sh, pre, K, deg,
                              campos, centre_rgb, variant, texture, index, frustum, sort, bg))
        rows.append(torch.stack(row, -1))
    return torch.stack(rows, -2)  # [3,H,W]


def _pixel(px, py, lst, Tu, Tv, Tw, R, scales, means, opac, sh, pre, K, deg, campos,
           centre_rgb, variant, texture, index, frustum, sort, bg):
    T = torch.ones((), dtype=means.dtype)
    C = torch.zeros(3, dtype=means.dtype)
    done = False
    buf = [None] * K  # entries: dict(depth, alpha, id, s)
    depth_slot = [math.inf] * K
    head = 0
    num = 0

    def blend(entry):
        nonlocal T, C, done
        test_T = T * (1 - entry["alpha"])
        if float(test_T) < 0.0001:
            done = True
            return
        w = entry["alpha"] * T
        i = entry["id"]
        if variant == "stage1":
            col = centre_rgb[i]
        else:
            s = entry["s"]
            st, wt, ht = (int(v) for v in index[i])
            dc = tex2d(texture, st, wt, ht, s[0] / TEX_RANGE, s[1] / TEX_RANGE)
            pos = R[i, :, 0] * scales[i, 0] * s[0] + R[i, :, 1] * scales[i, 1] * s[1] + means[i]
            col = sh_rgb(sh[i], pos - campos, deg, dc=dc)
        C = C + col * w
        T = test_T

    def blend_one():
        nonlocal head, num
        if num == 0:
            return
        num -= 1
        e = buf[head]
        blend(e)
        if done:
            return
        depth_slot[head] = math.inf
        head = (head + 1) % K

    for _depth_key, i in lst:
        if done:
            break
        if sort and num == K:
            blend_one()
        if done:
            break
        s, pz = _intersect(Tu[i], Tv[i], Tw[i], float(px), float(py))
        if float(pz) == 0.0:
            continue
        rho3d = (s * s).sum()
        xy = pre[i]["xy"]
        rho2d = 2.0 * ((xy[0] - px) ** 2 + (xy[1] - py) ** 2)
        depth = (s[0] * Tw[i, 0] + s[1] * Tw[i, 1] + Tw[i, 2]) if float(rho3d) <= rho2d else Tw[i, 2]
        if float(depth) < NEAR:
            continue
        if frustum:
            G = torch.exp(-0.5 * rho3d)
            for dx, dy in ((-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)):
                sc, _ = _intersect(Tu[i], Tv[i], Tw[i], px + dx, py + dy)
                G = G + torch.exp(-0.5 * (sc * sc).sum())
            G = G / 5.0
        else:
            G = torch.exp(-0.5 * rho3d)
        alpha = torch.clamp(opac[i, 0] * G, max=0.99)
        if float(alpha) < 1.0 / 255.0:
            continue
        e = {"depth": float(depth), "alpha": alpha, "id": i, "s": s}
        if not sort:
            blend(e)
            continue
        if variant == "stage1" and float(alpha) < 0.1:
            if e["depth"] < depth_slot[head]:
                head = (head - 1 + K) % K
                buf[head], depth_slot[head] = e, e["depth"]
            else:
                tail = (head + num) % K
                buf[tail], depth_slot[tail] = e, e["depth"]
        else:
            d = e["depth"]  # the kernel swaps (depth, alpha, id, s) through all K slots
            for kid in range(K):
                b = (head + kid) % K
                if d < depth_slot[b]:
                    e, buf[b] = buf[b], e
                    d, depth_slot[b] = depth_slot[b], d
        num += 1
    if sort and not done:
        while num > 0 and not done:
            blend_one()
    return C + T * bg
