"""Pure schedule helpers (CPU-testable)."""

import torch


def keep_counts(n0, ratio, cycles, split="compound"):
    """Number of Gaussians kept after each of `cycles` re-quantification cycles.

    compound: keep_c = round(n0 (1-P)^(c/C)), i.e. r_step = 1-(1-P)^(1/C) of the remaining
    (the authors' per-step formula, supp.); linear: keep_c = round(n0 (1 - P c/C)) (budget B/C
    per cycle analogue of Sec. 4.3). The last entry is round(n0 (1-P)) in both cases.
    """
    if not 0 < ratio < 1 or cycles < 1:
        raise ValueError("ratio must be in (0,1) and cycles >= 1")
    if split == "compound":
        out = [round(n0 * (1 - ratio) ** (c / cycles)) for c in range(1, cycles + 1)]
    elif split == "linear":
        out = [round(n0 * (1 - ratio * c / cycles)) for c in range(1, cycles + 1)]
    else:
        raise ValueError(f"unknown split {split!r}")
    out[-1] = round(n0 * (1 - ratio))
    return out


def lowest_mask(scores, n_remove):
    """Boolean mask of the `n_remove` lowest scores (stable: earlier index first on ties)."""
    mask = torch.zeros(len(scores), dtype=torch.bool, device=scores.device)
    if n_remove > 0:
        order = torch.sort(scores, stable=True).indices
        mask[order[:n_remove]] = True
    return mask


def pretrain_means_lr(lr0, step, base_steps, final_ratio=0.01):
    """Means LR for step+1 that a `--steps base_steps` run sets (runtime.py log-linear decay,
    = 3DGS get_expon_lr_func(lr0, 0.01 lr0, max_steps=base_steps)); clamped after base_steps,
    which is also the 3DGS value for continued (fine-tune) training."""
    return lr0 * final_ratio ** (min(step, base_steps) / base_steps)
