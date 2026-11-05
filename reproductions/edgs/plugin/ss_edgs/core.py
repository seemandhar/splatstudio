"""Pure EDGS initialization math (CPU/GPU torch). Mirrors CompVis/EDGS `source/corr_init.py`.

Coordinates: NDC with pixel centres at (2i+1)/W - 1, i.e. RoMa's normalized grid and the 3DGS
rasterizer's ndc2Pix convention. Cameras are SplatStudio dicts (OpenCV `viewmat` W2C, pixel `K`).
"""

import math

import numpy as np
import torch

SH_C0 = 0.28209479177387814


def ndc_projection(K, viewmat, width, height):
    """3x4 matrix M with M @ [X,1] = z * (ndc_x, ndc_y, 1) using the true principal point.

    Row form of the paper's P columns (Sec. 3.3); 3DGS's full_proj_transform equals this for a
    centred principal point (its z/w differ only by the depth normalization, which cancels).
    """
    K = K.to(torch.float64)
    ndc = torch.tensor(
        [
            [2 * K[0, 0] / width, 0.0, 2 * K[0, 2] / width - 1, 0.0],
            [0.0, 2 * K[1, 1] / height, 2 * K[1, 2] / height - 1, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
        device=K.device,
    )
    return ndc @ viewmat.to(torch.float64).to(K.device)


def project_ndc(M, points):
    """points [N,3] -> NDC [N,2] (official divides by EPS + w; EPS=1e-4)."""
    h = torch.cat([points, torch.ones_like(points[:, :1])], dim=1).to(M.dtype) @ M.T
    return h[:, :2] / (1e-4 + h[:, 2:3])


def triangulate(M1, M2, uv1, uv2):
    """DLT least squares of Sec. 3.3: rows P_col0 - u P_col2, P_col1 - v P_col2 for both views.

    Returns points [N,3] and the L1 NDC reprojection errors in view 1 and view 2 (official code
    `errors_proj = |proj - target|.sum(1)`).
    """
    uv1 = uv1.to(M1.dtype)
    uv2 = uv2.to(M1.dtype)
    rows = torch.stack(
        [
            M1[0][None] - uv1[:, :1] * M1[2][None],
            M1[1][None] - uv1[:, 1:2] * M1[2][None],
            M2[0][None] - uv2[:, :1] * M2[2][None],
            M2[1][None] - uv2[:, 1:2] * M2[2][None],
        ],
        dim=1,
    )  # [N,4,4]
    A, b = rows[:, :, :3], -rows[:, :, 3:]
    x = torch.linalg.lstsq(A, b).solution[..., 0]  # argmin ||A x + b||^2 (Eq. 6)
    e1 = (project_ndc(M1, x) - uv1).abs().sum(1)
    e2 = (project_ndc(M2, x) - uv2).abs().sum(1)
    return x, e1, e2


def select_best(points, err1, err2):
    """Per sample, keep the neighbour with minimal max(eps_i, eps_j) (official select_best_keypoints).

    points [J,N,3], err* [J,N] -> (points [N,3], best error [N], neighbour index [N]).
    """
    err = torch.maximum(err1, err2)
    best, idx = err.min(dim=0)
    return points[idx, torch.arange(points.shape[1], device=points.device)], best, idx


def sampling_weights(certainty, threshold=0.05):
    """Official: certainty[certainty > sample_thresh] = 1, then multinomial on the raw weights."""
    w = certainty.clone()
    w[w > threshold] = 1
    return w


def pixel_centres_ndc(index, width, height):
    """Flat pixel index -> NDC (x, y) of the pixel centre."""
    row, col = index // width, index % width
    return torch.stack([(2 * col + 1) / width - 1, (2 * row + 1) / height - 1], dim=1)


def ndc_to_pixel(ndc, width, height):
    """NDC -> integer pixel (col, row), clipped (nearest pixel centre)."""
    col = torch.round((ndc[:, 0] + 1) * width / 2 - 0.5).long().clamp(0, width - 1)
    row = torch.round((ndc[:, 1] + 1) * height / 2 - 0.5).long().clamp(0, height - 1)
    return col, row


def camera_center(viewmat):
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    return -R.T @ t


def nearest_cameras(viewmats, k):
    """Indices of the k closest cameras by Frobenius distance of flattened W2C (self excluded)."""
    flat = viewmats.reshape(len(viewmats), -1).to(torch.float64)
    d = torch.cdist(flat, flat)
    d.fill_diagonal_(float("inf"))
    return d.topk(min(k, len(flat) - 1), largest=False, dim=1).indices


def select_reference_views(viewmats, k, seed=228):
    """K-means on flattened W2C; camera nearest to each centre (official select_cameras_kmeans)."""
    from scipy.cluster.vq import kmeans, vq
    from scipy.spatial.distance import cdist

    cams = viewmats.reshape(len(viewmats), -1).double().cpu().numpy()
    k = min(k, len(cams))
    centres, _ = kmeans(cams, k, seed=np.random.RandomState(seed))
    assign, _ = vq(cams, centres)
    selected = set()
    for c in range(len(centres)):
        members = np.where(assign == c)[0]
        if len(members):
            dist = cdist(centres[c : c + 1], cams[members])[0]
            selected.add(int(members[np.argmin(dist)]))
    return sorted(selected)


def sh_basis(dirs):
    """The 16 real SH bases gsplat's rasterizer uses (degree 3), dirs [N,3] (normalized here)."""
    from gsplat.cuda._torch_impl import _eval_sh_bases_fast

    return _eval_sh_bases_fast(16, torch.nn.functional.normalize(dirs, dim=-1))


RIDGE_LAMBDA = 1.0 / 16.0  # ridge gain sigma/(sigma^2+lam) <= 1/(2 sqrt(lam)) = 2


def fit_sh(dirs, colors, chunk=1 << 18):
    """Literal Sec. 3.5 reading kept for comparison only: H = Y^+ (O - 0.5), all 16 coefficients.

    Overwriting H[:,0] by the reference colour afterwards double counts colour (review-1).
    """
    out = []
    for s in range(0, dirs.shape[0], chunk):
        Y = sh_basis(dirs[s : s + chunk].reshape(-1, 3)).reshape(*dirs[s : s + chunk].shape[:2], 16)
        out.append(torch.linalg.pinv(Y) @ (colors[s : s + chunk] - 0.5))
    return torch.cat(out, dim=0)


def fit_sh_residual(dirs, colors, lam=RIDGE_LAMBDA, chunk=1 << 18):
    """Self-consistent SH init (deviation from the paper's unspecified composition).

    dirs/colors [N,n,3]; observation 0 is the reference view. DC = RGB2SH(reference colour)
    (supp.: "first component initialized directly from the reference-view color"). Bands 1-3 are
    fitted to the residual R_o = O_o - O_ref of the other views, constrained to vanish along the
    reference direction (P = I - y_ref y_ref^T/|y_ref|^2), with ridge lam:
        H = P Y_o^T (Y_o P Y_o^T + lam I)^-1 R_o.
    So the reference view is reproduced exactly at every SH degree, and ||H|| <= ||R_o||/(2 sqrt
    lam) = 2 ||R_o|| per channel (conditioning bound; nearly parallel views -> H -> 0).
    Returns shN [N,15,3].
    """
    out = []
    for s in range(0, dirs.shape[0], chunk):
        d, c = dirs[s : s + chunk], colors[s : s + chunk]
        n = d.shape[1]
        Y = sh_basis(d.reshape(-1, 3)).reshape(d.shape[0], n, 16)[..., 1:].double()  # [B,n,15]
        y0 = Y[:, 0]
        P = torch.eye(15, dtype=Y.dtype, device=Y.device) - y0[:, :, None] * y0[:, None, :] / (
            y0.square().sum(-1)[:, None, None]
        )
        Yo = Y[:, 1:] @ P  # rows projected off the reference direction  [B,n-1,15]
        R = (c[:, 1:] - c[:, :1]).double()  # residual after the DC term  [B,n-1,3]
        G = Yo @ Yo.transpose(1, 2) + lam * torch.eye(n - 1, dtype=Y.dtype, device=Y.device)
        out.append((Yo.transpose(1, 2) @ torch.linalg.solve(G, R)).float())
    return torch.cat(out, dim=0)


def compose_sh_colors(dirs, sh0, shN, degree=3):
    """Colour each observing direction sees from coefficients (gsplat torch SH + 0.5)."""
    from gsplat.cuda._torch_impl import _spherical_harmonics

    coeffs = torch.cat([sh0, shN], dim=1)[:, None].expand(-1, dirs.shape[1], -1, -1)
    return _spherical_harmonics(degree, dirs, coeffs) + 0.5


def rgb_to_sh(rgb):
    return (rgb - 0.5) / SH_C0


def log_isotropic_scale(points, centre, scaling_factor=0.001, post_scale=0.5):
    """Official: scale = dist(point, ref camera) * scaling_factor, then all scales x0.5."""
    dist = torch.linalg.vector_norm(points - centre[None], dim=1)
    return torch.log((dist * scaling_factor * post_scale).clamp_min(1e-12))


def lr_floored(base, step, floor_step=8000, total=30000):
    """3DGS expon LR evaluated at max(step, floor_step) (`train.max_lr: True`)."""
    t = min(max(step, floor_step), total) / total
    return base * math.exp(math.log(0.01) * t)
