"""Pure-torch pieces of Unified Loss-Aware Density Control (arXiv 2611.02044).

* `proxy_curvature`    : omega_c(x), App. A.5 (Charbonnier + diagonal SSIM proxy).
* `screen_variables`   : z = (u, q) as a function of phi = (p, log s), replicating gsplat 1.5.3
                         `fully_fused_projection` (classic, eps2d = 0.3, clamped perspective J).
* `joint_split_matrix` : per-view S_phi = J_z^T H_z J_z + sum_m g_zm d^2 z_m / d phi^2
                         (App. Eq. app_joint_splitting_exact) with the mixed p-s second-order term
                         dropped (Eq. app_Sps_impl).
* `normalized_eigmin`  : S~ = J^T sym(S) J, J = blockdiag(R diag(s), I/sqrt 2) (Eq. 6), eigmin (Eq. 7).
* `split_children`, `select_*`: Eqs. 8, 9, 13, 17.
"""

import math

import torch
import torch.nn.functional as F
from torch.func import jacfwd, jacrev, vmap

SQRT_HALF = 1.0 / math.sqrt(2.0)


# ----------------------------------------------------------------------------- App. A.5
def _window(device, dtype, size=11, sigma=1.5):
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def _conv(img_chw, window):
    k = window[None, None].expand(img_chw.shape[0], 1, *window.shape)
    return F.conv2d(img_chw[None], k, padding=window.shape[-1] // 2, groups=img_chw.shape[0])[0]


@torch.no_grad()
def proxy_curvature(rgb, target, ssim_weight=0.2, charb_eps=1e-3, gamma_ssim=1e-4,
                    eps_ssim=1e-8, c1=0.01**2, c2=0.03**2):
    """omega [H,W,3] (Eq. mixed_proxy_curvature). rgb/target are HWC."""
    r = rgb - target
    w_charb = charb_eps**2 / (r * r + charb_eps**2) ** 1.5  # Eq. charb_curvature_weight
    win = _window(rgb.device, rgb.dtype)
    t = target.permute(2, 0, 1)
    mu = _conv(t, win)
    nu = _conv(t * t, win) - mu * mu
    d_mu = 1.0 / (2 * mu * mu + c1 + eps_ssim)
    d_nu = 1.0 / (2 * nu + c2 + eps_ssim)
    win2 = win * win
    w_ssim = 2 * _conv(d_mu, win2) + 2 * (_conv(d_nu, win) - _conv(d_nu, win2))  # Eq. ssim_diag
    return (1 - ssim_weight) * w_charb + ssim_weight * gamma_ssim * w_ssim.permute(1, 2, 0)


# ----------------------------------------------------------------------------- App. A.4
def quat_to_rotmat(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(*q.shape[:-1], 3, 3)


def screen_variables(phi, quat, viewmat, K, width, height, eps2d=0.3):
    """z = [u_x, u_y, q_xx, q_xy, q_yy] for one Gaussian (vmappable), gsplat conventions."""
    p, logs = phi[:3], phi[3:]
    Rw, tw = viewmat[:3, :3], viewmat[:3, 3]
    pc = Rw @ p + tw
    tx, ty, tz = pc[0], pc[1], pc[2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    tan_x, tan_y = 0.5 * width / fx, 0.5 * height / fy
    lim_xp, lim_xn = (width - cx) / fx + 0.3 * tan_x, cx / fx + 0.3 * tan_x
    lim_yp, lim_yn = (height - cy) / fy + 0.3 * tan_y, cy / fy + 0.3 * tan_y
    x = tz * torch.clamp(tx / tz, -lim_xn, lim_xp)
    y = tz * torch.clamp(ty / tz, -lim_yn, lim_yp)
    zero = torch.zeros_like(tz)
    J = torch.stack([torch.stack([fx / tz, zero, -fx * x / (tz * tz)]),
                     torch.stack([zero, fy / tz, -fy * y / (tz * tz)])])
    M = quat_to_rotmat(quat) * torch.exp(logs)[None, :]
    cov_c = Rw @ (M @ M.T) @ Rw.T
    cov2 = J @ cov_c @ J.T
    a, b, d = cov2[0, 0] + eps2d, cov2[0, 1], cov2[1, 1] + eps2d
    det = a * d - b * b
    u = torch.stack([fx * tx / tz + cx, fy * ty / tz + cy])
    return torch.cat([u, torch.stack([d / det, -b / det, a / det])])


def unpack_sym5(h15):
    """[...,15] upper triangle (row-major) -> [...,5,5] symmetric."""
    iu = torch.triu_indices(5, 5, device=h15.device)
    H = h15.new_zeros(*h15.shape[:-1], 5, 5)
    H[..., iu[0], iu[1]] = h15
    H[..., iu[1], iu[0]] = h15
    return H


def joint_split_matrix(phi, quats, g_z, h15, viewmat, K, width, height, chunk=262144,
                       drop_mixed=True):
    """Per-view S_phi [M,6,6] for M Gaussians (float32), App. Eq. app_joint_splitting_exact."""
    def z_fn(ph, q):
        return screen_variables(ph, q, viewmat, K, width, height)

    def scalar(ph, q, g):
        return (g * z_fn(ph, q)).sum()

    jac = vmap(jacfwd(z_fn, argnums=0))
    hess = vmap(jacfwd(jacrev(scalar, argnums=0), argnums=0))
    out = []
    for s in range(0, len(phi), chunk):
        sl = slice(s, s + chunk)
        Jz = jac(phi[sl], quats[sl]).to(phi.dtype)  # [m,5,6]
        Hz = unpack_sym5(h15[sl])
        S = Jz.transpose(1, 2) @ Hz @ Jz
        H2 = hess(phi[sl], quats[sl], g_z[sl]).to(phi.dtype)  # sum_m g_m d^2 z_m / d phi^2
        if drop_mixed:  # Eq. app_Sps_impl: first-order pullback form for the p-s block
            H2[:, :3, 3:] = 0
            H2[:, 3:, :3] = 0
        out.append(S + H2)
    if not out:
        return phi.new_zeros(0, 6, 6)
    return torch.cat(out)


def normalization(scales_log, quats):
    """J = blockdiag(L, I/sqrt2), L = R diag(s) (Eq. 6) -> [N,6,6]."""
    n = len(scales_log)
    J = scales_log.new_zeros(n, 6, 6)
    J[:, :3, :3] = quat_to_rotmat(quats) * scales_log.exp()[:, None, :]
    J[:, 3:, 3:] = torch.eye(3, device=J.device, dtype=J.dtype) * SQRT_HALF
    return J


def normalized_eigmin(S_mean, scales_log, quats):
    """(lambda_min, v_min, J) of S~ = J^T sym(S) J (Eqs. 6-7)."""
    S = 0.5 * (S_mean + S_mean.transpose(1, 2))
    J = normalization(scales_log, quats)
    St = J.transpose(1, 2) @ S @ J
    St = 0.5 * (St + St.transpose(1, 2))
    if len(St) == 0:
        return St.new_zeros(0), St.new_zeros(0, 6), J
    lam, vec = torch.linalg.eigh(St.double())
    return lam[:, 0].to(St.dtype), vec[:, :, 0].to(St.dtype), J


def split_children(means, scales_log, v_min, J):
    """Eq. 9: [delta_p; delta_s~] = J v_min; children p +- delta_p, log s +- delta_s~."""
    d = (J @ v_min[:, :, None])[:, :, 0]
    dp, ds = d[:, :3], d[:, 3:]
    return (torch.cat([means + dp, means - dp]), torch.cat([scales_log + ds, scales_log - ds]))


def select_split(gpos_norm, gscale_norm, lam_min, tau_pos, tau_scale, tau_split,
                 uncertain=None):
    """Eq. 8 (optionally AND the SteepGS 'uncertain' screen-gradient filter)."""
    m = (gpos_norm <= tau_pos) & (gscale_norm <= tau_scale) & (lam_min < tau_split)
    if uncertain is not None:
        m = m & uncertain
    return m


def select_clone(g_dup, h_dup, exclude, tau_clone, eta_max):
    """Eq. 13: eta = -g/h; i not in split, g < 0, tau_clone < eta <= eta_max."""
    eta = torch.where(h_dup > 0, -g_dup / h_dup.clamp_min(1e-30), torch.zeros_like(g_dup))
    m = (~exclude) & (g_dup < 0) & (h_dup > 0) & (eta > tau_clone) & (eta <= eta_max)
    return m, eta


def select_decay(g_dec, h_dec, exclude, tau_decay, eta_max):
    """Eq. 17: eta = g/h; i not in split/clone, g > 0, tau_decay < eta <= eta_max."""
    eta = torch.where(h_dec > 0, g_dec / h_dec.clamp_min(1e-30), torch.zeros_like(g_dec))
    m = (~exclude) & (g_dec > 0) & (h_dec > 0) & (eta > tau_decay) & (eta <= eta_max)
    return m, eta
