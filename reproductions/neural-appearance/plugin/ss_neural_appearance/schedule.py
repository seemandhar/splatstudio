"""Learning-rate policies of NeRFICG ``Optim/lr_utils.py`` (used by the official FasterGSVDA code)."""

import math


def lr_decay(iteration, lr_init, lr_final, max_steps, delay_steps=0, delay_mult=1.0):
    """``LRDecayPolicy``: optional reverse-cosine delay, then log-linear lr_init -> lr_final."""
    if iteration < 0 or (lr_init == 0.0 and lr_final == 0.0):
        return 0.0
    if delay_steps > 0 and iteration < delay_steps:
        delay_rate = delay_mult + (1 - delay_mult) * math.sin(
            0.5 * math.pi * min(max(iteration / delay_steps, 0.0), 1.0))
    else:
        delay_rate = 1.0
    t = min(max(iteration / max_steps, 0.0), 1.0)
    return float(delay_rate * math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t))


def cosine_decay(iteration, lr_init, start, end, final_factor=0.0, warmup_factor=1.0):
    """``CosineDecayPolicy`` (APPEARANCE_COSINE_DECAY): lr_warmup before ``start``, cosine to
    final_factor * lr_init at ``end``, zero afterwards."""
    lr_final = final_factor * lr_init
    if iteration < start:
        return warmup_factor * lr_init
    if iteration >= end:
        return 0.0
    t = (iteration - start) / (end - start)
    return lr_final + 0.5 * (lr_init - lr_final) * (1.0 + math.cos(math.pi * t))
