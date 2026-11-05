"""DashGaussian resolution and primitive schedules (pure functions).

Numerics follow the official `utils/schedule_utils.py:TrainingScheduler`
(https://github.com/YouyuChen0207/DashGaussian, CC BY-NC-SA 4.0), rewritten statelessly.
"""

import math

import torch


def window_significance(spectrum, scale):
    """X(F_r): |DFT| mass inside the centred (int(h/r), int(w/r)) window (paper Eq. 2).

    `spectrum` is an fftshift-ed magnitude map [..., h, w]; channels/leading dims are summed.
    The window bounds reproduce `compute_win_significance` exactly (centre (h+1)//2, half size
    win//2 on each side).
    """
    h, w = spectrum.shape[-2:]
    c0, c1 = (h + 1) // 2, (w + 1) // 2
    wh, ww = int(h / scale), int(w / scale)
    return spectrum[..., c0 - wh // 2 : c0 + wh // 2, c1 - ww // 2 : c1 + ww // 2].sum().item()


def solve_scale(spectrum, target, iterations=64):
    """Bisection on 1/r in (0, 1) for X(F_r) = target (`scale_solver`)."""
    lo, hi = 0.0, 1.0
    mid = 0.5
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if window_significance(spectrum, 1 / mid) < target:
            lo = mid
        else:
            hi = mid
    return 1 / mid


def centred_magnitude(image_chw):
    """fftshift(|fft2(I)|) per channel, computed as in the official code."""
    f = torch.fft.fftshift(torch.fft.fft2(image_chw), dim=(-2, -1))
    return (f.real.square() + f.imag.square()).sqrt()


def build_schedule(
    images,
    increase_until,
    start_significance_factor=4.0,
    max_reso_scale=8.0,
    sample_num=32,
):
    """Frequency-guided resolution schedule (Sec. 3.2, supp. Eq. 6-7).

    images: iterable of [3, H, W] float tensors in [0, 1] (training views, training resolution).
    Returns {"scales": [r_0 .. r_{m}=1], "begins": [...], "max_scale": r_m}; level i interpolates
    from scales[i-1] at begins[i] to scales[i] at begins[i+1] (see `res_scale`).
    """
    if sample_num < 2:
        raise ValueError("reso_sample_num must be at least 2")
    scene = None
    count = 0
    r_max = float(max_reso_scale)
    for image in images:
        mag = centred_magnitude(image)
        scene = mag.clone() if scene is None else scene + mag
        count += 1
        e_total = mag.sum().item()
        # r_m: X(F_{r_m}) = X(F) / a, minimum over views, capped at max_reso_scale.
        r_max = min(r_max, solve_scale(mag, e_total / start_significance_factor))
    if count == 0:
        raise ValueError("The resolution schedule needs at least one training image")
    scene = scene / count
    e_total = scene.sum().item()
    e_min = window_significance(scene, r_max)
    log_total = math.log(e_total / e_min)
    scales, begins = [r_max], [0]
    raw = [e_min]
    for i in range(1, sample_num - 1):
        # Significances linearly sampled between X(F_{r_m}) and X(F).
        raw.append((e_total - e_min) * i / (sample_num - 1) + e_min)
        scales.append(solve_scale(scene, raw[-1]))
        # supp. Eq. 7: s_{r} = S * ln(X(F_r)/X(F_{r_m})) / ln(X(F)/X(F_{r_m})), for the previous level.
        begins.append(int(increase_until * math.log(raw[-2] / e_min) / log_total))
    scales.append(1.0)
    begins.append(int(increase_until * math.log(raw[-1] / e_min) / log_total))
    begins.append(int(increase_until))
    return {"scales": scales, "begins": begins, "max_scale": r_max}


def res_scale(schedule, iteration, increase_until):
    """`get_res_scale` for resolution_mode='freq' (stateless). Returns the (floored) factor r."""
    if iteration >= increase_until:
        return 1
    scales, begins = schedule["scales"], schedule["begins"]
    if iteration < begins[1]:
        return scales[0]
    # next_i = first index >= 2 with begins[next_i] > iteration (official while-loop).
    nxt = 2
    while iteration >= begins[nxt]:
        nxt += 1
    i = nxt - 1
    i_now, i_nxt = begins[i], begins[i + 1]
    s_lst, s_now = scales[i - 1], scales[i]
    # Inverse-square interpolation between neighbouring levels (Sec. 3.2), then floor (Sec. 4).
    inv = (iteration - i_now) / (i_nxt - i_now) * (1 / s_now**2 - 1 / s_lst**2) + 1 / s_lst**2
    return int((1 / inv) ** 0.5)


def lr_decay_from(schedule):
    """Position LR is held until the first level with r < 2 (supp. Sec. 7.2)."""
    for begin, scale in zip(schedule["begins"], schedule["scales"]):
        if scale < 2:
            return begin
    raise ValueError("Resolution schedule never reaches r < 2")


def position_lr(step, lr_init, lr_final, max_steps, decay_from):
    """`helper_dashgaussian`: constant until decay_from, then log-linear to lr_final."""
    t = min(max((step - decay_from) / (max_steps - decay_from), 0.0), 1.0)
    return math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)


def densify_rate(iteration, cur_n, cur_scale, init_n, max_n, interval, increase_until,
                 densify_until, max_rate):
    """Resolution-guided primitive target (Eq. 4, code `get_densify_rate`, densify_mode='freq')."""
    if interval + iteration < increase_until:
        target = int((max_n - init_n) / cur_scale ** (2 - iteration / densify_until)) + init_n
    else:
        target = max_n
    return min(max((target - cur_n) / cur_n, 0.0), max_rate)


def update_momentum(momentum, added, integrate_factor, step_cap):
    """Momentum-based budget (Eq. 5 as implemented: on M = P_fin - P_init)."""
    return max(momentum, int(integrate_factor * momentum + min(step_cap, added)))
