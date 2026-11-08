"""Inverse Gaussian splatting (paper §3.3, supp. Listing 1; official
gaussian_2d_renderer/projector.py GaussianProjector.project), vectorised on the GPU.

Per 2D kernel: back-project the covered pixels with the depth map, keep the single-linkage
(SLINK) cluster that contains the max-weight pixel (threshold 5 * pixel footprint at the centre
depth, x1.5 until a cluster of >= 3 points exists), weighted PCA -> rotation / variances,
position from the centre (or weighted-mean) depth, scale calibration k (Eq. 6), lobe
initialisation (d -> camera, theta_bar from the closest other training view, beta = 0).

Camera conventions: viewmat is world->camera (OpenCV: x right, y down, z forward), K pinhole.
Like the official code (fov from the focal length, centred principal point) the frustum test and
the pixel footprint use tan(fov/2) = W / (2 fx); back-projection uses the full K.
"""


import torch

ZNEAR, ZFAR = 0.01, 100.0  # 3DGS Camera defaults used by the projector
ELEMENT_BUDGET = 4e7  # B * P * P elements per dense clustering batch
DENSE_MAX_P = 2500  # larger rectangles use the sparse edge-list path


def backproject(rows, cols, depth, K, c2w):
    """Pixel (row, col) centres at view-space depth z -> world points [N,3] (float64)."""
    x = (cols.double() + 0.5 - K[0, 2].double()) / K[0, 0].double() * depth.double()
    y = (rows.double() + 0.5 - K[1, 2].double()) / K[1, 1].double() * depth.double()
    pv = torch.stack((x, y, depth.double()), -1)
    return pv @ c2w[:3, :3].double().T + c2w[:3, 3].double()


def connected_components(adj, valid):
    """Single-linkage clusters at a distance threshold == connected components of the graph
    `dist < threshold` (sklearn AgglomerativeClustering(linkage='single', distance_threshold)).
    adj [B,P,P] bool, valid [B,P] -> labels [B,P] (min point index of the component; P if invalid).
    Min-label propagation with pointer jumping."""
    B, P = valid.shape
    big = P
    ar = torch.arange(P, device=adj.device).expand(B, P)
    lab = torch.where(valid, ar, torch.full_like(ar, big))
    while True:
        cand = torch.where(adj, lab[:, None, :], torch.full_like(adj, big, dtype=lab.dtype)).amin(-1)
        new = torch.minimum(lab, cand)
        new = torch.where(valid, new, torch.full_like(new, big))
        for _ in range(4):  # pointer jumping: lab[i] <- lab[lab[i]]
            jumped = torch.gather(torch.cat((new, torch.full_like(new[:, :1], big)), 1), 1,
                                  new.clamp(max=big))
            new = torch.where(valid, torch.minimum(new, jumped), new)
        if torch.equal(new, lab):
            return lab
        lab = new


def components_edges(n, edges):
    """Connected components of an undirected edge list [E,2] over n nodes -> min-index labels."""
    lab = torch.arange(n, device=edges.device)
    if edges.numel() == 0:
        return lab
    a, b = edges[:, 0], edges[:, 1]
    while True:
        new = lab.clone()
        new.scatter_reduce_(0, a, lab[b], reduce="amin")
        new.scatter_reduce_(0, b, lab[a], reduce="amin")
        for _ in range(4):
            new = torch.minimum(new, new[new])
        if torch.equal(new, lab):
            return lab
        lab = new


def sparse_edges(points, thr, chunk=2048):
    """Pairs (i < j) with ||p_i - p_j|| < thr, chunked distance evaluation."""
    out = []
    # float64, centred, direct (no matmul) distances: edges match sklearn's float64 SLINK
    pf = points.double() - points.double().mean(0, keepdim=True)
    for s in range(0, len(pf), chunk):
        d = torch.cdist(pf[s : s + chunk], pf, compute_mode="donot_use_mm_for_euclid_dist")
        i, j = torch.nonzero(d < float(thr), as_tuple=True)
        i = i + s
        keep = i < j
        out.append(torch.stack((i[keep], j[keep]), 1))
    return torch.cat(out) if out else torch.zeros(0, 2, dtype=torch.long, device=points.device)


def cluster_select_sparse(points, weight, depth, thr):
    """cluster_select for one kernel given only its valid points [n,3] (n >= 3)."""
    n = len(points)
    pivot = int(weight.argmax())
    thr = float(thr)
    retries = 0
    while True:
        lab = components_edges(n, sparse_edges(points, thr))
        counts = torch.bincount(lab, minlength=n)
        if int(counts.max()) >= 3:
            break
        thr *= 1.5
        retries += 1
    target = lab[pivot]
    if counts[target] < 3:
        target = counts.argmax()
    selected = lab == target
    if bool(selected[pivot]):
        nd = depth[pivot].double()
    else:
        w = weight[selected].double()
        nd = (depth[selected].double() * w).sum() / w.sum()
    return selected, nd, retries


def adjacency(points, valid, thr):
    """points [B,P,3] float64, valid [B,P], thr [B] -> dist < thr between valid points."""
    # float64, per-kernel centred, direct (no matmul) distances: edges match sklearn's float64
    w = valid.double()[..., None]
    centre = (points.double() * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1)
    pc = torch.where(valid[..., None], points.double() - centre, torch.zeros_like(points.double()))
    d = torch.cdist(pc, pc, compute_mode="donot_use_mm_for_euclid_dist")
    adj = d < thr.double()[:, None, None]
    return adj & valid[:, :, None] & valid[:, None, :]


def cluster_select(points, weight, depth, valid, thr0):
    """Official clustering loop per kernel (projector.compute_rotation). Returns selected [B,P],
    new depth [B], and the number of threshold enlargements."""
    B, P = valid.shape
    neg = torch.full_like(weight, -1.0)
    pivot = torch.where(valid, weight, neg).argmax(1)  # pivot among valid points (first max)
    thr = thr0.clone()
    labels = torch.full((B, P), P, dtype=torch.long, device=points.device)
    todo = torch.ones(B, dtype=torch.bool, device=points.device)
    retries = torch.zeros(B, dtype=torch.long, device=points.device)
    while todo.any():
        ids = torch.nonzero(todo).squeeze(1)
        lab = connected_components(adjacency(points[ids], valid[ids], thr[ids]), valid[ids])
        counts = torch.zeros(len(ids), P + 1, dtype=torch.long, device=points.device)
        counts.scatter_add_(1, lab, torch.ones_like(lab))
        counts = counts[:, :P]
        ok = counts.max(1).values >= 3
        labels[ids[ok]] = lab[ok]
        thr[ids[~ok]] *= 1.5  # no valid cluster: increase the distance threshold
        retries[ids[~ok]] += 1
        todo[ids[ok]] = False
    counts = torch.zeros(B, P + 1, dtype=torch.long, device=points.device)
    counts.scatter_add_(1, labels, torch.ones_like(labels))
    counts = counts[:, :P]
    target = labels.gather(1, pivot[:, None]).squeeze(1)
    small = counts.gather(1, target[:, None]).squeeze(1) < 3
    target = torch.where(small, counts.argmax(1), target)
    selected = (labels == target[:, None]) & valid
    pivot_in = selected.gather(1, pivot[:, None]).squeeze(1)
    w = torch.where(selected, weight, torch.zeros_like(weight)).double()
    wmean = (depth.double() * w).sum(1) / w.sum(1)
    new_depth = torch.where(pivot_in, depth.gather(1, pivot[:, None]).squeeze(1).double(), wmean)
    return selected, new_depth, retries


def weighted_pca(points, weight, selected):
    """wpca.WPCA(n_components=3).fit(points, weights=weight expanded to 3 columns):
    mean = sum(w x)/sum(w); C = ((x-mean) w)^T ((x-mean) w) / (w^T w); eigh, descending.
    Returns rotation [B,3,3] (columns = principal axes) and variances [B,3]."""
    w = torch.where(selected, weight, torch.zeros_like(weight)).double()[..., None]  # [B,P,1]
    mean = (w * points).sum(1) / w.sum(1)
    X = (points - mean[:, None]) * w
    cov = X.transpose(1, 2) @ X / (w * w).sum(1)[..., None]
    evals, evecs = torch.linalg.eigh(cov)
    return evecs.flip(-1), evals.flip(-1)


def rotmat_to_quat(R):
    """Rotation matrices [N,3,3] -> unit quaternions [N,4] (wxyz), robust branch selection."""
    R = R.double()
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    q = torch.zeros(R.shape[0], 4, dtype=R.dtype, device=R.device)
    cands = torch.stack((tr, m00, m11, m22), 1).argmax(1)
    s = torch.sqrt(torch.clamp(1.0 + tr, min=1e-12)) * 2
    c0 = cands == 0
    q[c0] = torch.stack((0.25 * s, (R[:, 2, 1] - R[:, 1, 2]) / s, (R[:, 0, 2] - R[:, 2, 0]) / s,
                         (R[:, 1, 0] - R[:, 0, 1]) / s), 1)[c0]
    s = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-12)) * 2
    c1 = cands == 1
    q[c1] = torch.stack(((R[:, 2, 1] - R[:, 1, 2]) / s, 0.25 * s, (R[:, 0, 1] + R[:, 1, 0]) / s,
                         (R[:, 0, 2] + R[:, 2, 0]) / s), 1)[c1]
    s = torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=1e-12)) * 2
    c2 = cands == 2
    q[c2] = torch.stack(((R[:, 0, 2] - R[:, 2, 0]) / s, (R[:, 0, 1] + R[:, 1, 0]) / s, 0.25 * s,
                         (R[:, 1, 2] + R[:, 2, 1]) / s), 1)[c2]
    s = torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=1e-12)) * 2
    c3 = cands == 3
    q[c3] = torch.stack(((R[:, 1, 0] - R[:, 0, 1]) / s, (R[:, 0, 2] + R[:, 2, 0]) / s,
                         (R[:, 1, 2] + R[:, 2, 1]) / s, 0.25 * s), 1)[c3]
    return torch.nn.functional.normalize(q, dim=-1).float()


def compute_theta(pos, self_idx, viewmats, Ks, widths, heights):
    """projector.compute_theta: angle to the closest *other* training camera that sees the point
    (3DGS frustum test without far plane); pi/2 if none. pos [N,3] -> [N]."""
    pos = pos.double()
    R = viewmats[:, :3, :3].double()
    t = viewmats[:, :3, 3].double()
    pc = torch.einsum("mij,nj->nmi", R, pos) + t[None]  # [N,M,3]
    tanx = (widths.double() / (2 * Ks[:, 0, 0].double()))[None]
    tany = (heights.double() / (2 * Ks[:, 1, 1].double()))[None]
    z = pc[..., 2]
    vis = (z >= ZNEAR) & (pc[..., 0].abs() <= z * tanx) & (pc[..., 1].abs() <= z * tany)
    centers = torch.linalg.inv(viewmats.double())[:, :3, 3]  # [M,3]
    vd = centers[None] - pos[:, None]
    vd = vd / vd.norm(dim=-1, keepdim=True)
    dot = (vd * vd[:, self_idx : self_idx + 1]).sum(-1).clamp(0.0, 1.0)
    dot[:, self_idx] = -1
    dot[~vis] = -1
    return torch.acos(dot.max(1).values.clamp(0.0, 1.0))


def calibrate_scale(pos_view, Rk, var, cov_target, K, viewmat, W, H):
    """projector.compute_scale (Eq. 4-6): q = (J W R) diag(V) (J W R)^T, k = <C,q>/<q,q>,
    scale = sqrt(k V); J at the view position with tx/tz, ty/tz clamped to 1.3 tan(fov/2).
    Returns scales [N,3] and a mask of kernels where the official assertion would fail."""
    fx, fy = K[0, 0].double(), K[1, 1].double()
    tanx, tany = W / (2 * fx), H / (2 * fy)
    t = pos_view.double()
    tx = (t[:, 0] / t[:, 2]).clamp(-1.3 * tanx, 1.3 * tanx) * t[:, 2]
    ty = (t[:, 1] / t[:, 2]).clamp(-1.3 * tany, 1.3 * tany) * t[:, 2]
    J = torch.zeros(t.shape[0], 2, 3, dtype=torch.float64, device=t.device)
    J[:, 0, 0] = fx / t[:, 2]
    J[:, 0, 2] = -(fx * tx) / (t[:, 2] * t[:, 2])
    J[:, 1, 1] = fy / t[:, 2]
    J[:, 1, 2] = -(fy * ty) / (t[:, 2] * t[:, 2])
    M = J @ viewmat[:3, :3].double()[None] @ Rk.double()  # [N,2,3]
    q = M @ torch.diag_embed(var.double()) @ M.transpose(1, 2)  # [N,2,2]
    qtq = (q * q).sum((1, 2))
    c = cov_target.double()
    ctq = c[:, 0] * q[:, 0, 0] + 2 * c[:, 1] * q[:, 0, 1] + c[:, 2] * q[:, 1, 1]
    bad = ~((qtq > 0) & (ctq > 0)) | ~torch.isfinite(ctq / qtq)
    k = torch.where(bad, torch.full_like(qtq, 1e-12), ctq / qtq).clamp(min=1e-12)
    return torch.sqrt(k[:, None] * var.double()).float(), bad


@torch.no_grad()
def inverse_splat(xy, log_scale, rot, opac_logit, color, depth_map, cam_idx, cameras):
    """Back-project one view's 2D kernels. xy (row, col) [N,2]; depth_map [H,W] (median depth);
    cameras: list of dicts with K, viewmat, width, height (all training views).
    Returns dict of new 3D Gaussian attributes (+ stats)."""
    from .fit2d import H_VAR, cov2d, sample_depth

    cam = cameras[cam_idx]
    K, viewmat = cam["K"], cam["viewmat"]
    H, W = depth_map.shape
    dev = depth_map.device
    c2w = torch.linalg.inv(viewmat)
    center = c2w[:3, 3].double()
    N = xy.shape[0]
    opacity = torch.sigmoid(opac_logit)
    scale2 = log_scale.exp()
    # 1. centre depth (nearest sample) -> world position
    depth0 = sample_depth(depth_map, xy, mode="nearest").double()
    pos0 = backproject(xy[:, 0], xy[:, 1], depth0, K, c2w)
    # 2. covered-pixel rectangles (cov + 0.3 I, 3-sigma radius), weights, validity
    cov = cov2d(scale2, rot)
    cxx, cxy, cyy = cov[:, 0] + H_VAR, cov[:, 1], cov[:, 2] + H_VAR
    det = cxx * cyy - cxy * cxy
    conic = torch.stack((cyy / det, -cxy / det, cxx / det), 1)
    mid = 0.5 * (cxx + cyy)
    lam1 = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
    lam2 = mid - torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
    radius = torch.ceil(3.0 * torch.sqrt(torch.maximum(lam1, lam2)))
    lim = torch.tensor([H, W], device=dev, dtype=torch.float32)
    rmin = torch.floor(torch.clamp(xy - radius[:, None], min=0)).long()
    rmax = torch.ceil(torch.minimum(xy + radius[:, None] + 1, lim[None])).long()
    ext = (rmax - rmin).clamp(min=0)
    side = ext.max(1).values
    fx = K[0, 0].double()
    rotation = torch.eye(3, dtype=torch.float64, device=dev).repeat(N, 1, 1)
    variance = torch.ones(N, 3, dtype=torch.float64, device=dev)
    new_depth = torch.zeros(N, dtype=torch.float64, device=dev)
    retries_total = 0
    skipped = 0
    order = torch.argsort(side)
    s_sorted = side[order]
    start = 0
    while start < N:
        S = int(s_sorted[start])
        P = max(S * S, 1)
        B = max(1, int(ELEMENT_BUDGET // (P * P)))
        # all kernels in this batch have side <= S_end; pad to the largest
        end = min(N, start + B)
        S = int(s_sorted[end - 1])
        P = max(S * S, 1)
        while end - start > 1 and (end - start) * P * P > ELEMENT_BUDGET:
            end = start + max(1, (end - start) // 2)
            S = int(s_sorted[end - 1])
            P = max(S * S, 1)
        ids = order[start:end]
        start = end
        b = len(ids)
        ar = torch.arange(S, device=dev)
        rr = rmin[ids, 0][:, None, None] + ar[None, :, None]  # [b,S,1]
        cc = rmin[ids, 1][:, None, None] + ar[None, None, :]  # [b,1,S]
        inside = (rr < rmax[ids, 0][:, None, None]) & (cc < rmax[ids, 1][:, None, None])
        rr_c, cc_c = rr.clamp(max=H - 1).expand(b, S, S), cc.clamp(max=W - 1).expand(b, S, S)
        dep = depth_map[rr_c, cc_c]  # [b,S,S]
        dxp = xy[ids, 1][:, None, None] - cc.float()
        dyp = xy[ids, 0][:, None, None] - rr.float()
        cn = conic[ids]
        power = -0.5 * (cn[:, 0, None, None] * dxp * dxp + cn[:, 2, None, None] * dyp * dyp) \
            - cn[:, 1, None, None] * dxp * dyp
        weight = torch.clamp(torch.exp(power) * opacity[ids][:, None, None], max=0.99)
        weight = torch.where(inside, weight, torch.full_like(weight, -1.0)).reshape(b, -1)
        dep = dep.reshape(b, -1)
        valid = (weight >= 1.0 / 255.0) & (dep < ZFAR)
        pivot0 = weight.argmax(1)  # official: argmax over the whole rectangle
        wv = torch.where(valid, weight, torch.zeros_like(weight)).double()
        wmean0 = (dep.double() * wv).sum(1) / wv.sum(1)
        piv_valid = valid.gather(1, pivot0[:, None]).squeeze(1)
        ref = torch.where(piv_valid, depth0[ids], wmean0)
        thr = 5.0 * torch.clamp(ref, min=0.2) / fx  # 5 * tan(fovx/2) * max(0.2, d) / (W/2)
        enough = valid.sum(1) >= 3
        skipped += int((~enough).sum())
        if enough.any() and P > DENSE_MAX_P:
            for k in torch.nonzero(enough).squeeze(1).tolist():
                v = valid[k]
                pts = backproject(rr_c[k].reshape(-1)[v], cc_c[k].reshape(-1)[v], dep[k][v], K, c2w)
                selected, nd, retries = cluster_select_sparse(pts, weight[k][v], dep[k][v], thr[k])
                retries_total += retries
                Rk, var = weighted_pca(pts[None], weight[k][v][None], selected[None])
                rotation[ids[k]] = Rk[0]
                variance[ids[k]] = var[0].clamp(min=1e-7)
                new_depth[ids[k]] = nd
        elif enough.any():
            sel_ids = ids[enough]
            pts = backproject(rr_c[enough].reshape(-1), cc_c[enough].reshape(-1),
                              dep[enough].reshape(-1), K, c2w).reshape(int(enough.sum()), -1, 3)
            selected, nd, retries = cluster_select(pts, weight[enough], dep[enough],
                                                   valid[enough], thr[enough])
            retries_total += int(retries.sum())
            Rk, var = weighted_pca(pts, weight[enough], selected)
            rotation[sel_ids] = Rk
            variance[sel_ids] = var.clamp(min=1e-7)
            new_depth[sel_ids] = nd
    # 3. orient the normal (third axis) towards the camera, make det(R) = +1
    ndotv = (rotation[:, :, 2] * (center[None] - pos0)).sum(1)
    rotation[ndotv < 0, :, 2] *= -1
    rotation[torch.linalg.det(rotation) < 0, :, 1] *= -1
    quats = rotmat_to_quat(rotation)
    # 4. final position: kernels without a cluster (new depth 0) or too close go to zfar
    new_depth[new_depth < 0.2] = ZFAR
    pos = backproject(xy[:, 0], xy[:, 1], new_depth, K, c2w)
    pos_view = pos @ viewmat[:3, :3].double().T + viewmat[:3, 3].double()
    # 5. lobe initialisation
    vms = torch.stack([c["viewmat"] for c in cameras])
    Ks = torch.stack([c["K"] for c in cameras])
    widths = torch.tensor([float(c["width"]) for c in cameras], device=dev)
    heights = torch.tensor([float(c["height"]) for c in cameras], device=dev)
    theta = torch.cat([compute_theta(pos[i : i + 4096], cam_idx, vms, Ks, widths, heights)
                       for i in range(0, N, 4096)]) if N else torch.zeros(0, device=dev)
    lobe_dir = center[None] - pos
    lobe_dir = lobe_dir / lobe_dir.norm(dim=-1, keepdim=True)
    # 6. scale calibration against the raw (no low-pass) 2D covariance
    scales, bad = calibrate_scale(pos_view, rotation, variance, cov, K, viewmat, W, H)
    return {
        "means": pos.float(),
        "sh0": ((color - 0.5) / 0.28209479177387814)[:, None, :],
        "opacities": torch.logit(opacity, eps=1e-6),
        "lobe_dir": lobe_dir.float(),
        "theta": theta.float(),
        "scales": scales.clamp(min=1e-30).log(),
        "quats": quats,
        "stats": {"kernels": N, "skipped": skipped, "cluster_retries": retries_total,
                  "scale_fit_failures": int(bad.sum())},
    }
