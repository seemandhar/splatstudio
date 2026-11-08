"""DecompGS schedules as pure functions (arXiv 2507.20239, Sec. 3.3-3.4).

No official code exists; everything follows the paper text. See paper.md for ambiguities.
"""

import math

import torch
import torch.nn.functional as F


def downscaled_size(height, width, r):
    """Size of D_r(I): floor(H/r) x floor(W/r) (r = 1 -> unchanged)."""
    return int(height // r), int(width // r)


def downscale(image_chw, r):
    """Eq. 1: I_r = Bilinear(I, scale=1/r) with anti-aliasing. image: [C, H, W]."""
    if r == 1:
        return image_chw
    h, w = downscaled_size(image_chw.shape[-2], image_chw.shape[-1], r)
    return F.interpolate(image_chw[None], size=(h, w), mode="bilinear", align_corners=False,
                         antialias=True)[0]


def spectrum_energy(image_chw):
    """||E(I)||_1 with E(I) = sqrt(Re(F(I))^2 + Im(F(I))^2), 2D FFT per channel, summed."""
    f = torch.fft.fft2(image_chw.double())
    return float((f.real.square() + f.imag.square()).sqrt().sum())


def scale_energies(images, max_scale):
    """Eq. 2 summed over the training views: E_r = sum_views ||E(D_r(I))||_1 * r^2, r = 1..K."""
    energies = {r: 0.0 for r in range(1, int(max_scale) + 1)}
    count = 0
    for image in images:
        count += 1
        for r in energies:
            energies[r] += spectrum_energy(downscale(image, r)) * r * r
    if count == 0:
        raise ValueError("The C2F schedule needs at least one training image")
    return energies


def end_iterations(energies, densify_until):
    """Eq. 3: T_r = Round((E_r / E_1) * T_densify) for every r in A."""
    e1 = energies[1]
    return {r: int(round(e / e1 * densify_until)) for r, e in energies.items()}


def scale_at(ends, iteration):
    """Scale r trains from T_{r+1} (T_{K+1} = 0) to T_r: r(t) = max{r : T_r > t}, else 1."""
    active = [r for r, t_end in ends.items() if t_end > iteration]
    return max(active) if active else 1


def full_resolution_step(ends):
    """First iteration rendered at r = 1 (= T_2, the G2L phase boundary)."""
    return max(t for r, t in ends.items() if r >= 2) if len(ends) > 1 else 0


def position_lr(step, lr_init, lr_final, max_steps, hold_until):
    """Constant lr_init before hold_until, then log-linear to lr_final at max_steps."""
    t = min(max((step - hold_until) / max(max_steps - hold_until, 1), 0.0), 1.0)
    return math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)


def adaptive_threshold(opacities, ratio, upper):
    """Sec. 3.4: tau = min(alpha_sorted[floor(N p)], tau_u); alpha = activated opacities [N]."""
    n = opacities.numel()
    k = min(int(math.floor(n * ratio)), n - 1)
    tau_k = torch.kthvalue(opacities.flatten().float(), k + 1).values  # 0-based sorted[k]
    return min(float(tau_k), float(upper))
