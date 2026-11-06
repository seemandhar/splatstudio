"""Independent dense float64 reference of the Faster-GS / 3DGS image formation (paper §2 recap,
eqs. mean_projection, cov_projection, alpha_computation, alpha_blending, with the constants of
`rasterization_config.h`): every Gaussian is evaluated at every pixel centre, no tiles, no culling
other than the per-Gaussian validity tests. Differentiable with torch autograd, for probes only.
"""

import torch

C0 = 0.28209479177387814
C1 = 0.48860251190291987
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525251999, -1.0925484305920792,
      0.54627421529603959)
C3 = (-0.59004358992664352, 2.8906114426405538, -0.45704579946446572, 0.3731763325901154,
      -0.45704579946446572, 1.4453057213202769, -0.59004358992664352)
DILATION, ALPHA_MIN, T_MIN, DET_MIN = 0.3, 1.0 / 255.0, 1e-4, 1e-6


def sh_to_color(sh0, shN, dirs, degree):
    """Standard 3DGS SH evaluation (utils/sh_utils.eval_sh) + 0.5."""
    sh = torch.cat([sh0, shN], dim=1)
    result = C0 * sh[:, 0]
    if degree > 0:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        result = result - C1 * y * sh[:, 1] + C1 * z * sh[:, 2] - C1 * x * sh[:, 3]
        if degree > 1:
            xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
            result = (result + C2[0] * xy * sh[:, 4] + C2[1] * yz * sh[:, 5]
                      + C2[2] * (2 * zz - xx - yy) * sh[:, 6] + C2[3] * xz * sh[:, 7]
                      + C2[4] * (xx - yy) * sh[:, 8])
            if degree > 2:
                result = (result + C3[0] * y * (3 * xx - yy) * sh[:, 9]
                          + C3[1] * xy * z * sh[:, 10]
                          + C3[2] * y * (4 * zz - xx - yy) * sh[:, 11]
                          + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
                          + C3[4] * x * (4 * zz - xx - yy) * sh[:, 13]
                          + C3[5] * z * (xx - yy) * sh[:, 14]
                          + C3[6] * x * (xx - 3 * yy) * sh[:, 15])
    return result + 0.5


def project(params, camera, near=0.2, far=10000.0):
    """Per-Gaussian 2D mean, conic, opacity, validity (float64, raw parameters activated here)."""
    d = torch.float64
    means = params["means"].to(d)
    w2c = camera["viewmat"].to(d)
    K = camera["K"].to(d)
    W, H = float(camera["width"]), float(camera["height"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    cam = means @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    x, y = cam[:, 0] / z, cam[:, 1] / z
    q = params["quats"].to(d)
    norm_sq = q.square().sum(-1)
    r, qx, qy, qz = (q / norm_sq.sqrt()[:, None]).unbind(-1)
    R = torch.stack([
        1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - r * qz), 2 * (qx * qz + r * qy),
        2 * (qx * qy + r * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - r * qx),
        2 * (qx * qz - r * qy), 2 * (qy * qz + r * qx), 1 - 2 * (qx * qx + qy * qy)],
        -1).reshape(-1, 3, 3)
    S = torch.diag_embed(params["scales"].to(d).exp())
    cov3d = R @ S @ S @ R.transpose(1, 2)
    xc = x.clamp((-0.15 * W - cx) / fx, (1.15 * W - cx) / fx)
    yc = y.clamp((-0.15 * H - cy) / fy, (1.15 * H - cy) / fy)
    zeros = torch.zeros_like(z)
    J = torch.stack([torch.stack([fx / z, zeros, -fx / z * xc], -1),
                     torch.stack([zeros, fy / z, -fy / z * yc], -1)], 1)
    JW = J @ w2c[:3, :3]
    cov2d = JW @ cov3d @ JW.transpose(1, 2) + DILATION * torch.eye(2, dtype=d, device=z.device)
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    conic = torch.linalg.inv(cov2d)
    mean2d = torch.stack([x * fx + cx, y * fy + cy], -1)
    opacity = params["opacities"].to(d).sigmoid()
    valid = (z >= near) & (z <= far) & (opacity >= ALPHA_MIN) & (norm_sq >= 1e-8) & (det >= DET_MIN)
    return {"mean2d": mean2d, "conic": conic, "cov2d": cov2d, "opacity": opacity, "depth": z,
            "valid": valid, "cam_center": torch.linalg.inv(w2c)[:3, 3]}


def render(params, camera, sh_degree, background, near=0.2, far=10000.0, mean2d_hook=None):
    """HWC float64 image. `mean2d_hook(mean2d)` may replace mean2d (e.g. to retain its grad)."""
    d = torch.float64
    p = project(params, camera, near, far)
    mean2d = p["mean2d"] if mean2d_hook is None else mean2d_hook(p["mean2d"])
    means = params["means"].to(d)
    dirs = torch.nn.functional.normalize(means - p["cam_center"], dim=-1)
    color = sh_to_color(params["sh0"].to(d), params["shN"].to(d), dirs, sh_degree).clamp_min(0)
    W, H = int(camera["width"]), int(camera["height"])
    ys, xs = torch.meshgrid(torch.arange(H, dtype=d, device=means.device) + 0.5,
                            torch.arange(W, dtype=d, device=means.device) + 0.5, indexing="ij")
    pix = torch.stack([xs, ys], -1).reshape(-1, 2)
    order = torch.argsort(p["depth"].detach(), stable=True)
    order = order[p["valid"][order]]
    out = torch.zeros(pix.shape[0], 3, dtype=d, device=means.device)
    T = torch.ones(pix.shape[0], dtype=d, device=means.device)
    alive = torch.ones(pix.shape[0], dtype=torch.bool, device=means.device)
    for i in order.tolist():
        delta = mean2d[i] - pix
        c = p["conic"][i]
        power = -0.5 * (c[0, 0] * delta[:, 0] ** 2 + 2 * c[0, 1] * delta[:, 0] * delta[:, 1]
                        + c[1, 1] * delta[:, 1] ** 2)
        alpha = p["opacity"][i] * torch.exp(power.clamp_max(0))
        use = alive & (alpha >= ALPHA_MIN)
        alpha = torch.where(use, alpha, torch.zeros_like(alpha))
        out = out + (T * alpha)[:, None] * color[i]
        T = T * (1 - alpha)
        alive = alive & ~(use & (T < T_MIN))
    out = out + T[:, None] * background.to(d)
    return out.reshape(H, W, 3)
