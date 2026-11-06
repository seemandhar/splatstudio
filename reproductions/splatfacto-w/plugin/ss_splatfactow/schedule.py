"""Pure schedule / normalisation helpers (no GPU state), mirroring the official Splatfacto-W
Nerfstudio code. ``k`` is always the 0-based Nerfstudio step (= SplatStudio step - 1)."""

import math

import torch


def exp_decay(lr_init, lr_final, max_steps, k):
    """nerfstudio ExponentialDecayScheduler (warmup 0): log-linear to lr_final, then constant."""
    if lr_final is None:
        return lr_init
    t = min(max(k / max_steps, 0.0), 1.0)
    return math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)


# splatfactow_config.py (splatfacto-w): name -> (lr_init, lr_final or None, max_steps)
LR_TABLE = {
    "means": (1.6e-5, 1.6e-7, 30000),          # nerfstudio scene units
    "appearance_features": (0.02, 1e-3, 40000),
    "opacities": (0.03, None, 1),
    "scales": (0.005, None, 1),
    "quats": (0.001, None, 1),
    "field_background_encoder": (2e-3, 1e-4, 30000),
    "field_background_base": (2e-3, 2e-4, 30000),
    "field_background_rest": (2e-3 / 20, 2e-4 / 20, 30000),
    "appearance_model_encoder": (2e-3, 1e-4, 30000),
    "appearance_model_base": (2e-3, 1e-4, 30000),
    "appearance_model_rest": (2e-3 / 20, 1e-4 / 20, 30000),
    "appearance_embed": (0.02, 3e-4, 40000),
}


def lr_at(name, k):
    lr0, lr1, n = LR_TABLE[name]
    return exp_decay(lr0, lr1, n, k)


def downscale_factor(k, num_downscales=2, interval=3000, training=True):
    """splatfacto _get_downscale_factor."""
    if not training:
        return 1
    return 2 ** max(num_downscales - k // interval, 0)


def sh_degree(k, interval, max_degree):
    return min(k // interval, max_degree)


def rotation_matrix_between(a, b):
    """nerfstudio camera_utils.rotation_matrix_between (Rodrigues)."""
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.linalg.cross(a, b)
    eps = 1e-6
    if torch.sum(torch.abs(v)) < eps:
        x = torch.tensor([1.0, 0, 0], dtype=a.dtype) if abs(a[0]) < eps else torch.tensor(
            [0, 1.0, 0], dtype=a.dtype)
        v = torch.linalg.cross(a, x)
    v = v / torch.linalg.norm(v)
    skew = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=a.dtype)
    theta = torch.acos(torch.clip(torch.dot(a, b), -1, 1))
    return torch.eye(3, dtype=a.dtype) + torch.sin(theta) * skew + (1 - torch.cos(theta)) * (
        skew @ skew)


def nerfstudio_normalisation(viewmats, scale_factor=3.0):
    """NerfW dataparser: auto_orient_and_center_poses(method='up', center='poses'), then scale
    so that max |t| = 1, times scale_factor. Returns (R [3,3], centre [3], s) with
    x_ns = s * R (x - centre) and d_ns = R d."""
    c2w = torch.linalg.inv(torch.stack([v.detach().double().cpu() for v in viewmats]))
    origins = c2w[:, :3, 3]
    up_gl = -c2w[:, :3, 1]  # OpenGL camera y = -(OpenCV camera y)
    centre = origins.mean(0)
    up = up_gl.mean(0)
    up = up / torch.linalg.norm(up)
    R = rotation_matrix_between(up, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    t = (origins - centre) @ R.T
    s = scale_factor / float(t.abs().max())
    return R.float(), centre.float(), float(s)
