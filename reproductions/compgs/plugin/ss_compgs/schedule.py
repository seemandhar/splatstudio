"""CompGS schedule (pure functions; official train_kmeans.py loop, 1-based iteration `it`)."""

import math


def assign_freq(it, cfg):
    """train_kmeans.py:124-127: freq 100, and 5000 for it > total - 5000."""
    if it > cfg["total_steps"] - cfg["freeze_window"]:
        return cfg["final_freq"]
    return cfg["assign_freq"]


def vq_event(it, cfg):
    """What happens to the codebooks before rendering iteration `it`: 'assign', 'update' or None."""
    if not cfg["vq_start"] < it <= cfg["total_steps"]:
        return None
    return "assign" if it % assign_freq(it, cfg) == 1 else "update"


def prunes(step, cfg):
    """train_kmeans.py:203-208: 15000 < it <= max_prune_iter and it % 1000 == 0."""
    return (cfg["opacity_prune"] and cfg["prune_start"] < step <= cfg["prune_stop"]
            and step % cfg["prune_every"] == 0)


def reg_active(step, start, stop):
    """train_kmeans.py:150: lambda = 0 if it > max_prune_iter or it < 15000."""
    return start <= step <= stop


def inferred_run_length(lr0, lr_after_step1):
    """The runtime sets the means LR to lr0 * 0.01 ** (1 / steps) after step 1."""
    if not 0 < lr_after_step1 < lr0:
        return None
    return math.log(0.01) / math.log(lr_after_step1 / lr0)
