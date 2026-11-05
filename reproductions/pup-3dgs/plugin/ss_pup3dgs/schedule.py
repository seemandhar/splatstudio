"""Learning-rate schedules of the official PUP 3D-GS pipeline (pure functions, CPU-testable)."""

import math


def pretrain_means_lr(lr0, step, base_steps, final_ratio=0.01):
    """LR for step+1 that the native runtime would set when run with --steps base_steps.

    runtime.py: lr = initial_lr * 0.01 ** (step / steps) after each step; 3DGS's
    get_expon_lr_func(1.6e-4, 1.6e-6, max_steps=30000) is the same log-linear decay.
    """
    return lr0 * final_ratio ** (min(step, base_steps) / base_steps)


def finetune_lr_iter(step, round_start, lr_start):
    """Official `lr_iter` of our step: rounds restart at iteration lr_start (30000)."""
    return lr_start + (step - round_start)


def exp_decay_count(lr_iter, lr_start, every):
    """Number of ExponentialLR.step() calls made up to and including lr_iter.

    prune_finetune.py calls scheduler.step() whenever lr_iter % 400 == 0, for lr_iter starting
    at lr_start + 1.
    """
    return lr_iter // every - lr_start // every


def finetune_means_lr(lr0, lr_iter, max_steps, final_ratio=0.01, gamma=0.95, every=400):
    """xyz LR at lr_iter: get_expon_lr_func(lr0, lr0*final_ratio, max_steps=35000)(lr_iter).

    update_learning_rate() sets it each iteration, then the ExponentialLR step of that same
    iteration (if lr_iter % every == 0) multiplies it by gamma once.
    """
    t = min(max(lr_iter / max_steps, 0.0), 1.0)
    lr = math.exp(math.log(lr0) * (1 - t) + math.log(lr0 * final_ratio) * t)
    return lr * (gamma if lr_iter % every == 0 else 1.0)


def finetune_other_lr(base_lr, lr_iter, lr_start, gamma=0.95, every=400):
    """Non-xyz LRs: training_setup base LR times gamma per ExponentialLR step so far."""
    return base_lr * gamma ** exp_decay_count(lr_iter, lr_start, every)
