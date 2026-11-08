"""MCMC densification of the official FasterGSVDA code (``Model.mcmc_densification``,
``post_optimizer_step``) with its CUDA kernels (``densification/include/kernels_mcmc.cuh``)
re-expressed in PyTorch (same formulas; torch RNG instead of Philox).

relocation (3DGS-MCMC Eq. 9):  o' = 1 - (1 - o)^(1/n),
  s' = o / (sum_{i<n} sum_{k<=i} binom(i,k) (-1)^k / sqrt(k+1) * o'^(k+1)) * s,  n clamped to [1, 50]
noise: mu += lr * sigmoid(-(100 o - 0.5)) * Sigma eps,  Sigma = R diag(exp(2 s)) R^T,  lr = 5e5 lr_means
"""

import math

import torch

MAX_SAMPLES = 50
_COEF = {}


def relocation_coefficients(device):
    if device not in _COEF:
        c = torch.zeros(MAX_SAMPLES, MAX_SAMPLES, dtype=torch.float64)
        for n in range(MAX_SAMPLES):
            binom, sign = 1.0, 1.0
            for k in range(n + 1):
                c[n, k] = binom * sign / math.sqrt(k + 1)
                binom *= (n - k) / (k + 1)
                sign = -sign
        _COEF[device] = c.float().to(device)
    return _COEF[device]


def relocation_adjustment(opacities, scales, counts):
    """``relocation_cu``: opacities [M], scales [M, 3] (activated), counts [M] (incl. original)."""
    n = counts.clamp(1, MAX_SAMPLES)
    new_opacity = 1.0 - torch.pow(1.0 - opacities, 1.0 / n.float())
    powers = new_opacity[:, None] ** torch.arange(
        1, MAX_SAMPLES + 1, device=opacities.device, dtype=torch.float32)  # o'^(k+1)
    per_n = powers @ relocation_coefficients(opacities.device).t()  # [M, n]: sum_k C[n,k] o'^(k+1)
    mask = torch.arange(MAX_SAMPLES, device=opacities.device)[None] < n[:, None]
    denominator = (per_n * mask).sum(dim=1)
    return new_opacity, (opacities / denominator)[:, None] * scales


def noise_rotation(quats):
    r, x, y, z = quats.unbind(-1)
    inv = 1.0 / (r * r + x * x + y * y + z * z)
    return torch.stack([
        1 - 2 * (y * y + z * z) * inv, 2 * (x * y - r * z) * inv, 2 * (x * z + r * y) * inv,
        2 * (x * y + r * z) * inv, 1 - 2 * (x * x + z * z) * inv, 2 * (y * z - r * x) * inv,
        2 * (x * z - r * y) * inv, 2 * (y * z + r * x) * inv, 1 - 2 * (x * x + y * y) * inv,
    ], -1).reshape(-1, 3, 3)


@torch.no_grad()
def noise_delta(scales, quats, opacities, lr, generator=None):
    """``add_noise_cu`` displacement for every Gaussian (zero for degenerate quaternions)."""
    valid = quats.square().sum(-1) >= 1e-8
    R = noise_rotation(torch.where(valid[:, None], quats, torch.ones_like(quats)))
    cov = (R * torch.exp(2.0 * scales)[:, None, :]) @ R.transpose(1, 2)
    eps = torch.randn(scales.shape, device=scales.device, generator=generator)
    o = torch.sigmoid(opacities)
    factor = lr / (1.0 + torch.exp(100.0 * o - 0.5))
    return (cov @ eps[..., None])[..., 0] * (factor * valid)[:, None]


@torch.no_grad()
def reset_rows(optimizers, names, indices):
    """``Optim.adam_utils.reset_state(indices=...)``: zero both Adam moments at these rows."""
    for name in names:
        opt = optimizers[name]
        p = opt.param_groups[0]["params"][0]
        state = opt.state.get(p)
        if state:
            for key in ("exp_avg", "exp_avg_sq"):
                state[key][indices] = 0


def _sample_adjust(params, opacities, sampled, min_opacity):
    _, inverse, counts = sampled.unique(sorted=False, return_inverse=True, return_counts=True)
    new_o, new_s = relocation_adjustment(opacities[sampled], params["scales"][sampled].exp(),
                                         counts[inverse] + 1)
    new_o = new_o.clamp(min_opacity, 1.0 - torch.finfo(torch.float32).eps).logit()
    return new_o, new_s.log()


@torch.no_grad()
def mcmc_step(params, optimizers, append, cap_max, min_opacity=0.005, generator=None):
    """One ``mcmc_densification`` call. ``append(new_rows)`` concatenates rows to every field with
    zero Adam moments (official ``extend_param_groups``). Returns event counters."""
    names = list(params)
    dead = params["opacities"] <= math.log(min_opacity / (1 - min_opacity))
    dead |= params["quats"].square().sum(dim=1) < 1e-8
    n_dead = int(dead.sum())
    if n_dead > 0:
        dead_idx = torch.where(dead)[0]
        alive_idx = torch.where(~dead)[0]
        opacities = torch.sigmoid(params["opacities"])
        sampled = alive_idx[torch.multinomial(opacities[alive_idx], n_dead, replacement=True,
                                              generator=generator)]
        new_o, new_s = _sample_adjust(params, opacities, sampled, min_opacity)
        params["opacities"].data[sampled] = new_o
        params["scales"].data[sampled] = new_s
        for name in names:
            if name in ("scales", "opacities"):
                continue
            params[name].data[dead_idx] = params[name].data[sampled]
        params["scales"].data[dead_idx] = new_s
        params["opacities"].data[dead_idx] = new_o
        reset_rows(optimizers, names, sampled)
    n = len(params["means"])
    n_add = max(0, min(cap_max, int(1.05 * n)) - n)
    if n_add > 0:
        opacities = torch.sigmoid(params["opacities"])
        sampled = torch.multinomial(opacities, n_add, replacement=True, generator=generator)
        new_o, new_s = _sample_adjust(params, opacities, sampled, min_opacity)
        params["opacities"].data[sampled] = new_o
        params["scales"].data[sampled] = new_s
        rows = {k: params[k].detach()[sampled] for k in names}
        rows["opacities"], rows["scales"] = new_o, new_s
        append(rows)
        reset_rows(optimizers, names, sampled)
    return {"relocated": n_dead, "added": n_add, "gaussians": len(params["means"])}
