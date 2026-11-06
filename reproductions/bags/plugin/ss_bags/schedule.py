"""BAGS training schedule (official train.py of github.com/snldmt/BAGS @ed04144), pure functions.

Two clocks drive the official loop:
  ori_iter  the global step (runtime ``step``): resolution (1/4 -> 1/2 -> 1 at upsample_iter
            [3000, 6000]) and the multi-scale kernel head (K 5 / 9 / 17);
  iteration = ori_iter if ori_iter <= ms_steps else ori_iter - ms_steps ("lr reset"): xyz LR,
            densification / opacity reset / 3D-filter clocks, and the BPN start (iteration > 250).
"""

import math


def phase_iteration(step, ms_steps):
    """train.py: ``if ori_iter > opt.ms_steps: iteration = iteration - opt.ms_steps``."""
    return step if step <= ms_steps else step - ms_steps


def resolution_divisor(step, upsample_steps=(3000, 6000)):
    """Downscale factor 2^(s-1) of the training resolution at global step ``step``."""
    if step < upsample_steps[0]:
        return 4
    if step < upsample_steps[1]:
        return 2
    return 1


def kernel_level(step, upsample_steps=(3000, 6000)):
    """1 -> K3 (coarsest, 5x5, head1), 2 -> K2 (9x9, head2), 3 -> K1 (17x17, head3).
    kernelnet_multires.forward and the unfold choice in train.py both use ori_iter."""
    if step < upsample_steps[0]:
        return 1
    if step < upsample_steps[1]:
        return 2
    return 3


def bpn_active(step, ms_steps, bpn_start=250):
    """train.py: ``if iteration > 250`` (phase-local) -> BPN-blurred loss, else sharp 3DGS loss."""
    return phase_iteration(step, ms_steps) > bpn_start


def uses_single_scale_net(step, ms_steps, use_another_mlp):
    """train.py: ``(ori_iter > opt.ms_steps) and opt.use_another_mlp`` -> mlp_rgb_ss."""
    return bool(use_another_mlp) and step > ms_steps


def expon_lr(step, lr_init, lr_final, delay_steps=0, delay_mult=1.0, max_steps=30000):
    """utils/general_utils.get_expon_lr_func."""
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
        return 0.0
    if delay_steps > 0:
        delay_rate = delay_mult + (1 - delay_mult) * math.sin(
            0.5 * math.pi * min(max(step / delay_steps, 0), 1))
    else:
        delay_rate = 1.0
    t = min(max(step / max_steps, 0), 1)
    return delay_rate * math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)


def densify_thresholds(step, cfg):
    """(grad threshold, min opacity) of densify_and_prune at global step ``step``."""
    if step <= cfg["ms_steps"]:
        dgt = cfg["densify_grad_threshold"] if cfg["init_dgt"] < 0 else cfg["init_dgt"]
        opa = cfg["min_opacity"] if cfg["init_opacity"] < 0 else cfg["init_opacity"]
        return dgt, opa
    return cfg["densify_grad_threshold"], cfg["min_opacity"]


def step_actions(step, cfg):
    """What the official loop does after the backward of global step ``step``.

    Returns dict(stats, densify, reset, filter). The official loop saves the final model
    *before* the densification block of the last iteration, so nothing that changes the
    Gaussians is applied at step == total_steps.
    """
    it = phase_iteration(step, cfg["ms_steps"])
    last = step >= cfg["total_steps"]
    until = cfg["densify_until_iter"]
    dens_window = it < until
    densify = (dens_window and it > cfg["densify_from_iter"]
               and it % cfg["densification_interval"] == 0 and not last)
    reset = dens_window and it % cfg["opacity_reset_interval"] == 0 and not last
    phase_len = cfg["total_steps"] - cfg["ms_steps"]
    late_filter = (it % cfg["filter_interval"] == 0 and it > until
                   and it < phase_len - cfg["filter_interval"] and not last)
    return {"stats": dens_window, "densify": densify, "reset": reset,
            "filter": densify or late_filter, "iteration": it}
