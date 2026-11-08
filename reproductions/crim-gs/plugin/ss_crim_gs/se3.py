"""Screw-axis rigid body motion of CRiM-GS (Sec. 4.2, Eq. 7, 9, 10; appendix derivation)."""

import torch


def skew(w):
    """[w] for w [..., 3] -> [..., 3, 3] (appendix Eq. skew)."""
    zero = torch.zeros_like(w[..., 0])
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    return torch.stack(
        (zero, -wz, wy, wz, zero, -wx, -wy, wx, zero), dim=-1
    ).reshape(*w.shape[:-1], 3, 3)


def normalize_axis(w, eps=1e-10):
    """Unit rotation axis: w / (||w|| + eps) (decoder D_r output, ||w_hat|| = 1;
    CoMoGaussian como_kernel.exp_map)."""
    return w / (torch.linalg.vector_norm(w, dim=-1, keepdim=True) + eps)


def rodrigues(w_hat, theta):
    """Eq. 9: e^{[w]theta} = I + sin(theta)[w] + (1 - cos(theta))[w]^2.  theta [..., 1]."""
    W = skew(w_hat)
    th = theta[..., None]
    eye = torch.eye(3, device=w_hat.device, dtype=w_hat.dtype).expand_as(W)
    return eye + torch.sin(th) * W + (1 - torch.cos(th)) * (W @ W)


def g_matrix(w_hat, theta):
    """Eq. 10: G(theta) = I theta + (1 - cos(theta))[w] + (theta - sin(theta))[w]^2."""
    W = skew(w_hat)
    th = theta[..., None]
    eye = torch.eye(3, device=w_hat.device, dtype=w_hat.dtype).expand_as(W)
    return eye * th + (1 - torch.cos(th)) * W + (th - torch.sin(th)) * (W @ W)


def screw_exp(w_hat, theta, v):
    """T_r = e^{[S]theta} = [[e^{[w]theta}, G(theta) v], [0, 1]]  ->  [..., 4, 4]."""
    R = rodrigues(w_hat, theta)
    t = (g_matrix(w_hat, theta) @ v[..., None])[..., 0]
    return homogeneous(R, t)


def homogeneous(R, t):
    """[[R, t], [0, 1]] for R [..., 3, 3], t [..., 3]."""
    top = torch.cat((R, t[..., None]), dim=-1)
    bottom = torch.zeros(*R.shape[:-2], 1, 4, device=R.device, dtype=R.dtype)
    bottom[..., 0, 3] = 1.0
    return torch.cat((top, bottom), dim=-2)
