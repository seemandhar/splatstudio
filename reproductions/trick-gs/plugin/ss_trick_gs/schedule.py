"""Trick-GS schedules as pure functions of the 1-based training step.

No official Trick-GS code exists; the choices for what the paper leaves open are documented in
reproductions/trick-gs/paper.md §7 and in each function.
"""

import math


def round_half_up(x):
    """The paper's nearest-integer bracket of Eq. 6."""
    return int(math.floor(x + 0.5))


def ds_scale(step, start=0.125, until=19500, mode="linear"):
    """Resolution factor r(i)/r_e of progressive downsampling (paper Eq. 6, §4.2).

    linear (Eq. 6): r(i) = min(r_s + (r_e - r_s) i / tau, r_e) with r_s = r_e / 8.
    log ("logarithmic decay" of §4.2, not the default): r(i) = r_e * start^(1 - i/tau).
    """
    if step >= until or start >= 1.0:
        return 1.0
    t = max(step, 0) / until
    if mode == "linear":
        return min(start + (1.0 - start) * t, 1.0)
    if mode == "log":
        return min(start ** (1.0 - t), 1.0)
    raise ValueError(f"unknown ds mode {mode!r}")


def ds_size(step, height, width, start=0.125, until=19500, mode="linear"):
    """(h, w) of the training render: Eq. 6 on the width in pixels, height by the same ratio."""
    s = ds_scale(step, start, until, mode)
    if s >= 1.0:
        return int(height), int(width)
    w = min(max(round_half_up(width * s), 16), int(width))
    h = min(max(round_half_up(height * w / width), 16), int(height))
    return h, w


def blur_params(step, scale, radius=4, sigma=2.4, every=100, until=19500):
    """Progressive blur (paper §3.2.4a, §4.2): (kernel radius, sigma) or None.

    The 9x9 / sigma 2.4 kernel (EAGLES `blur_<radius>_<sigma>` convention, radius 4) is lowered
    every `every` iterations (piecewise constant on multiples of `every`, i.e. updated at the
    densification step itself, "before we densify") with factor b = 1 - i_q / until, and scaled by
    the current downsampling factor (both radius and sigma). No blur once the radius rounds to 0
    (Python `round`, as EAGLES utils/image_utils.blur_image).
    """
    if step >= until:
        return None
    iq = (step // every) * every
    b = max(0.0, 1.0 - iq / until) * float(scale)
    r = round(radius * b)  # Python round, as EAGLES blur_image (round(kernel_radius * ...))
    if r < 1:
        return None
    return r, sigma * b


def lowpass_value(height, width, n, lo=0.3, hi=300.0):
    """s = HW / (9 pi N), bounded below by 0.3 (paper §3.2.4c) and above by 300 (RAIN-GS code)."""
    return min(max(height * width / max(n, 1) / (9.0 * math.pi), lo), hi)


def lowpass_due(step, every=1000, until=10000):
    """RAIN-GS train.py: recompute at iteration 1 and every 1000 while < until."""
    return step == 1 or (step % every == 0 and step < until)


def sg_rate(round_index, first=0.6, decay=0.7):
    """Significance prune rate of round k: first * decay^k (LightGaussian prune_decay)."""
    return first * decay**round_index


def in_stats_window(step, cfg):
    """Steps whose screen gradients are accumulated for densification."""
    lo, hi = cfg["redensify"]
    return step < cfg["densify_until"] or lo <= step <= hi


def densify_due(step, cfg):
    """3DGS densify every 100 its in (500, 15000); late densification (DE) in (20000, 20500]."""
    if step % cfg["densify_every"]:
        return False
    lo, hi = cfg["redensify"]
    return cfg["densify_from"] < step < cfg["densify_until"] or (cfg["late_densify"] and lo < step <= hi)


def mask_prune_due(step, cfg):
    """Mask prune every `mask_prune_every` its once densification stopped (paper §4.2)."""
    return step >= cfg["densify_until"] and step % cfg["mask_prune_every"] == 0


def reset_due(step, cfg):
    """3DGS opacity reset every 3000 its while densifying (not in the late window)."""
    return step < cfg["densify_until"] and step % cfg["reset_every"] == 0
