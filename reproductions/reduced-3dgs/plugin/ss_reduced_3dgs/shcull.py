"""C2 — adaptive SH band culling (paper §4.2, Eq. 1-2).

`official_quirks=True` (default of `reduced-3dgs`) reproduces two behaviours of the official
code that produced the paper's numbers:
(1) gaussian_model.py:742 passes the raw *logit* `_opacity` to calculate_colours_variance, whose
    rasterizer uses it directly as the alpha multiplier (alpha = min(0.99, logit * g)): Gaussians
    with logit <= 0 never blend, get zero weight and, through NaN -> 0, pass both culling tests
    (-> 0 bands, DC = (0 - 0.5) / C0);
(2) reduced_3dgs.cu `auto mean_old = mean;` aliases the tensor that `mean.index_put_` then
    updates, so the variance accumulates T * (c - mean_new)^2 instead of West's
    T * (c - mean_old) * (c - mean_new).
`official_quirks=False` is the textbook Eq. 1 with activated opacities.

Official code: diff-gaussian-rasterization/reduced_3dgs.cu:calculateColourVariance,
reduced_3dgs/sh_culling.cu:computeColorFromSH and scene/gaussian_model.py:cull_sh_bands /
_low_variance_colour_culling / _low_distance_colour_culling.
"""

import math

import torch
from gsplat import rasterization, spherical_harmonics

from . import kernels
from .renderer import DEGREE_FIELD

SH_C0 = 0.28209479177387814


def camera_center(viewmat):
    return torch.linalg.inv(viewmat)[:3, 3]


@torch.no_grad()
def band_colours(params, campos, degree):
    """[N,4,3] clamped colours using bands <= q (q = 0..3); slots above a Gaussian's own degree
    stay 0, exactly as the official computeColorFromSH returns early (sh_culling.cu)."""
    coeffs = torch.cat((params["sh0"], params["shN"]), dim=1).detach()
    dirs = params["means"].detach() - campos[None]
    out = torch.zeros((len(coeffs), 4, 3), device=coeffs.device)
    for q in range(4):
        c = spherical_harmonics(q, dirs, coeffs[:, : (q + 1) ** 2].contiguous())
        out[:, q] = torch.where((degree >= q)[:, None], (c + 0.5).clamp_min(0.0), 0.0)
    return out


def accumulate_view(state, t_bar, colours, present, official_quirks):
    """One view of calculateColourVariance. state = dict(dist [N,3], w_sum [N,1], mean, var)."""
    state["w_sum"] += t_bar
    for q in range(3):
        d = (colours[:, 3] - colours[:, q]).square().sum(-1, keepdim=True).sqrt()
        state["dist"][:, q : q + 1] += t_bar * torch.nan_to_num(d, nan=0.0)
    colour = colours[:, 3]
    mean, var, p = state["mean"], state["var"], present
    mean_old = mean if official_quirks else mean.clone()  # quirk (2): aliased storage
    coef = torch.nan_to_num(t_bar / state["w_sum"], nan=0.0)
    mean[p] = mean_old[p] + coef[p] * (colour[p] - mean_old[p])
    var[p] = var[p] + t_bar[p] * (colour[p] - mean_old[p]) * (colour[p] - mean[p])


def _stats_info(renderer, params, cam, official_quirks):
    """gsplat meta for the statistics pass; with the quirk, opacity := raw logit (quirk (1))."""
    if not official_quirks:
        return renderer.render(params, cam, sh_degree=3)["info"], None, 0.999
    logit = params["opacities"].detach()
    _, _, info = rasterization(
        means=params["means"].detach(), quats=params["quats"].detach(),
        scales=params["scales"].detach().exp(), opacities=logit,
        colors=params["sh0"].detach()[:, 0].clamp(0, 1), viewmats=cam["viewmat"][None],
        Ks=cam["K"][None], width=cam["width"], height=cam["height"], packed=False,
        rasterize_mode="classic",
    )
    return info, logit, 0.99


@torch.no_grad()
def colour_statistics(renderer, params, cameras, official_quirks=True):
    """Official calculateColourVariance over all training views.

    Returns (distances [N,3] : sum_i Tbar_i ||c_3 - c_q|| / sum_i Tbar_i for q=0..2,
             variance [N,3] : transmittance-weighted variance of c_3,
             mean [N,3]     : transmittance-weighted mean of c_3).
    Tbar_i = (sum of T over the pixels where the Gaussian is blended) / max(#pixels, 1).
    """
    degree = params[DEGREE_FIELD].detach().round().long()
    n = len(degree)
    dev = degree.device
    state = {"dist": torch.zeros((n, 3), device=dev), "w_sum": torch.zeros((n, 1), device=dev),
             "mean": torch.zeros((n, 3), device=dev), "var": torch.zeros((n, 3), device=dev)}
    for cam in cameras:
        info, opacity, alpha_max = _stats_info(renderer, params, cam, official_quirks)
        radii = info["radii"][0]
        present = (radii > 0).all(dim=-1) if radii.dim() == 2 else radii > 0
        sum_t, touched = kernels.transmittance_stats(info, opacity, alpha_max)
        t_bar = (sum_t / touched.clamp_min(1).float())[:, None]  # [N,1]
        colours = band_colours(params, camera_center(cam["viewmat"].float()), degree)
        colours[~present] = 0
        accumulate_view(state, t_bar, colours, present, official_quirks)
    w = state["w_sum"]
    return state["dist"] / w, state["var"] / w, state["mean"]


@torch.no_grad()
def cull_sh_bands(renderer, params, cameras, std_threshold=0.04,
                  cdist_threshold=6 * math.sqrt(3) / 255, active_degree=3,
                  official_quirks=True):
    """Two-pass culling; edits params in place (official keeps the Adam moments). Returns info."""
    degree = params[DEGREE_FIELD]
    before = torch.bincount(degree.round().long(), minlength=4).tolist()
    # pass 1: low colour variance -> RGB only (_low_variance_colour_culling)
    _, variance, mean = colour_statistics(renderer, params, cameras, official_quirks)
    std = torch.nan_to_num(variance.sqrt(), nan=0.0).mean(dim=1)
    low = std < std_threshold
    params["sh0"].data[low] = ((mean[low] - 0.5) / SH_C0)[:, None, :]
    degree.data[low] = 0
    params["shN"].data[low] = 0
    # pass 2: small colour change when dropping bands (_low_distance_colour_culling);
    # official loop range(active_sh_degree - 1, 0, -1): q = 2, 1 (never 0)
    distances, _, _ = colour_statistics(renderer, params, cameras, official_quirks)
    distances = torch.nan_to_num(distances, nan=0.0)
    for q in range(active_degree - 1, 0, -1):
        m = distances[:, q] < cdist_threshold
        degree.data[m] = torch.minimum(degree.data[m], torch.full_like(degree.data[m], float(q)))
        params["shN"].data[m, (q + 1) ** 2 - 1 :] = 0
    after = torch.bincount(degree.round().long(), minlength=4).tolist()
    total = max(1, sum(after))
    return {"bands_before": before, "bands_after": after,
            "band_fractions": [round(c / total, 4) for c in after],
            "paper_band_fractions": [0.89, 0.001, 0.027, 0.082],
            "variance_culled": int(low.sum()), "official_quirks": bool(official_quirks)}
