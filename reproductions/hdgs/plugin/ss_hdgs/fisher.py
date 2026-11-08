"""Fisher-information pruning (paper §3.3, Eq. U_i, after PUP-3DGS). Not in the released code.

    U_i = || log diag( sum_phi  grad_{x_i,s_i} I  grad_{x_i,s_i} I^T ) ||

diag(J^T J)_j = sum_p (dI_p/dtheta_j)^2 needs per-pixel Jacobians, which a standard backward
does not provide. We use the unbiased Hutchinson/Rademacher estimator
E_r[(r^T J_j)^2] = sum_p J_pj^2 with r in {-1,+1}^{3HW}: per training view and probe, one
backward of (r * I).sum() w.r.t. the rasteriser inputs (means [N,3], surfel scales [N,2]).
Score: sum_j log(F_jj + eps) (the log-det of the diagonal, as PUP's log|F|; the literal L1 norm
of the logs would rank F_jj < 1 the wrong way round). The lowest ``ratio`` are pruned.
"""

import torch

from .raster import activated, raster_settings, rasterize


def fisher_diagonal(params, cameras, sort_window, sh_degree=3, probes=1, seed=0,
                    background=None):
    """Hutchinson estimate of diag F for (means, scales[:2]) summed over ``cameras``: [N,5]."""
    dev = params["means"].device
    bg = background if background is not None else torch.zeros(3, device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    with torch.no_grad():
        means, sh, opac, scales, rots = (t.detach() for t in activated(params))
    means = means.clone().requires_grad_(True)
    scales = scales.clone().requires_grad_(True)
    diag = torch.zeros(len(means), 5, device=dev, dtype=torch.float64)
    for cam in cameras:
        s = raster_settings(cam, bg, sh_degree, sort_window)
        for _ in range(probes):
            color, _r, _a, _m = rasterize(means, sh, opac, scales, rots, s)
            r = torch.randint(0, 2, color.shape, generator=gen, device=dev).float() * 2 - 1
            gm, gs = torch.autograd.grad((color * r).sum(), [means, scales])
            diag[:, :3] += gm.double().square()
            diag[:, 3:] += gs.double().square()
    return diag / probes


def sensitivity(diag, eps=1e-30):
    return torch.log(diag + eps).sum(dim=1)


def prune_mask(scores, ratio):
    """Boolean mask of the floor(ratio N) lowest scores."""
    n = int(len(scores) * ratio)
    mask = torch.zeros(len(scores), dtype=torch.bool, device=scores.device)
    if n > 0:
        mask[torch.argsort(scores)[:n]] = True
    return mask
