"""C1 universal Gaussian pruning (paper §3.3, Eq. 3; official `mesongs.py`: `cal_imp`,
`pre_volume`, `prune_mask`).

I_d(g) = sum over all training pixels of alpha_g T_g (official modified rasteriser
`renderCUDA_imp`: `imp[id] += alpha * T`). With gsplat it is the gradient of the summed render of
a 1-channel per-Gaussian colour: d(sum_p sum_i c_i alpha_i T_i) / d c_g = sum_p alpha_g T_g
(same alpha clamp 0.99, 1/255 skip and T < 1e-4 stop as 3DGS).
I_i = (V / V_ref)^beta, V = prod(exp(log scales)), V_ref = sort(V, descending)[int(0.9 N)]
(official `pre_volume`, no clipping although the paper text says "clip to [0, 1]").
"""

import torch
from gsplat import rasterization


def view_dependent(params, cameras, chunk=None):
    """I_d [N] summed over `cameras` (dicts with K, viewmat, width, height)."""
    means = params["means"].detach()
    quats = params["quats"].detach()
    scales = torch.exp(params["scales"].detach())
    opacities = torch.sigmoid(params["opacities"].detach())
    total = torch.zeros(len(means), device=means.device)
    for cam in cameras:
        colors = torch.zeros(len(means), 1, device=means.device, requires_grad=True)
        render, _, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
            viewmats=cam["viewmat"][None], Ks=cam["K"][None], width=cam["width"],
            height=cam["height"], sh_degree=None, packed=False,
        )
        (grad,) = torch.autograd.grad(render.sum(), colors)
        total += grad[:, 0]
    return total


def volume_term(log_scales, beta):
    volume = torch.prod(torch.exp(log_scales), dim=1)
    index = int(volume.shape[0] * 0.9)
    ref = torch.sort(volume, descending=True).values[index]
    return torch.pow(volume / ref, beta)


def importance(params, cameras, beta):
    with torch.enable_grad():
        i_d = view_dependent(params, cameras)
    return volume_term(params["scales"].detach(), beta) * i_d


def prune_mask(percent, imp):
    """Official prune_mask: True for Gaussians with imp <= the value at int(percent * (N - 1))."""
    sorted_imp = torch.sort(imp, dim=0).values
    value = sorted_imp[int(percent * (sorted_imp.shape[0] - 1))]
    return imp <= value
