"""Per-term gradient statistics behind FreGS' Fig. 2 claim (used by the probe and calibration).

The densification statistic is gsplat DefaultStrategy's per-Gaussian view-space gradient norm:
||dL/dmeans2d * (W/2, H/2)|| over visible Gaussians (radii > 0). Every loss term contributes
linearly to dL/drgb and dL/dmeans2d, so gradients are taken once per term and recombined with
arbitrary weights.
"""

import torch

from .frequency import frequency_components

TERMS = ("photo", "low_a", "low_p", "high_a", "high_p")


def term_gradients(out, target, t, cfg, photometric):
    """{term: (dterm/drgb [H,W,3], dterm/dmeans2d of visible Gaussians, gsplat-scaled [n,2])}."""
    rgb, means2d = out["rgb"], out["info"]["means2d"]
    h, w = rgb.shape[:2]
    visible = (out["info"]["radii"] > 0).all(dim=-1)[0]
    scale = torch.tensor([w / 2.0, h / 2.0], device=rgb.device)
    values = {"photo": photometric({}, {"rgb": rgb}, target)}
    for band, (amp, phase) in frequency_components(rgb, target, t, cfg).items():
        values[f"{band}_a"], values[f"{band}_p"] = amp, phase
    grads = {}
    for name, value in values.items():
        if value.requires_grad:
            g_rgb = torch.autograd.grad(value, rgb, retain_graph=True)[0]
        else:  # e.g. a band with no bins
            g_rgb = torch.zeros_like(rgb)
        g_m = torch.autograd.grad(rgb, means2d, grad_outputs=g_rgb, retain_graph=True)[0]
        grads[name] = (g_rgb.detach(), (g_m[0, visible] * scale).detach())
    return grads


def combined(grads, weights):
    """Weighted sum of per-term gradients; weights: {term: w} (missing terms ignored)."""
    g_rgb = sum(weights[k] * grads[k][0] for k in weights if k in grads)
    g_m = sum(weights[k] * grads[k][1] for k in weights if k in grads)
    return g_rgb, g_m


def pooled_ratio(views, weights):
    """(rgb ratio, means2d ratio) of pooled means, (photo + weighted L_f) / photo only."""
    num_rgb = den_rgb = num_m = den_m = 0.0
    for grads in views:
        g_rgb, g_m = combined(grads, weights)
        num_rgb += g_rgb.abs().sum().item()
        den_rgb += grads["photo"][0].abs().sum().item()
        num_m += g_m.norm(dim=-1).sum().item()
        den_m += grads["photo"][1].norm(dim=-1).sum().item()
    return num_rgb / den_rgb, num_m / den_m


def magnitudes(views):
    """Per-term mean/max |dL/drgb| and mean ||dL/dmeans2d|| (unit weight), pooled over views."""
    out = {}
    for name in views[0]:
        rgb = torch.cat([v[name][0].flatten() for v in views]).abs()
        m = torch.cat([v[name][1].norm(dim=-1) for v in views])
        out[name] = {
            "rgb_mean": rgb.mean().item(),
            "rgb_max": rgb.max().item(),
            "means2d_mean": m.mean().item() if len(m) else 0.0,
        }
    return out


def band_magnitudes(views):
    """M_b: pooled mean ||d(d_ba + d_bp)/dmeans2d|| at unit weight, per band present."""
    out = {}
    for band in ("low", "high"):
        if f"{band}_a" not in views[0]:
            continue
        num = sum((v[f"{band}_a"][1] + v[f"{band}_p"][1]).norm(dim=-1).sum().item() for v in views)
        count = sum(len(v[f"{band}_a"][1]) for v in views)
        out[band] = num / count
    return out


def equalized_w_h(w_l, bands):
    """Per-band unit-gradient equalization (review-2): w_h * M_high = w_l * M_low."""
    if not bands.get("high", 0) > 0:
        raise ValueError("High band has no screen-space gradient; w_h is unidentifiable")
    return w_l * bands["low"] / bands["high"]
