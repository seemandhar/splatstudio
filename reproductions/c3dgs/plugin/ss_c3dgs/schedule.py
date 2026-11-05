"""c3dgs run schedule (pure functions, CPU-testable).

`pretrain_means_lr` is copied from the pup-3dgs plugin (reproductions/pup-3dgs/plugin/
ss_pup3dgs/schedule.py) and `inferred_run_length` follows reduced-3dgs'
`ReducedDensity.check_run_length` (reproductions/reduced-3dgs/plugin/ss_reduced_3dgs/density.py).
"""

import math


def pretrain_means_lr(lr0, step, base_steps, final_ratio=0.01):
    """LR for step+1 that the native runtime would set when run with --steps base_steps.

    runtime.py: lr = initial_lr * 0.01 ** (step / steps) after each step; 3DGS's
    get_expon_lr_func(1.6e-4, 1.6e-6, max_steps=30000) is the same log-linear decay.
    """
    return lr0 * final_ratio ** (min(step, base_steps) / base_steps)


def inferred_run_length(lr0, lr_after_step1):
    """The runtime sets the means LR to lr0 * 0.01 ** (1 / steps) after step 1."""
    if not 0 < lr_after_step1 < lr0:
        return None
    return math.log(0.01) / math.log(lr_after_step1 / lr0)


def events(step, base_steps, total_steps):
    """c3dgs events after the runtime optimizer step of `step` (1-based).

    * base (step <= base_steps): 3DGS density control;
    * compress (step == base_steps): sensitivity -> prune -> colour VQ -> shape VQ;
    * bank_step (base_steps < step < total_steps): Adam step of the codebooks / log eta; the
      official fine-tune loop skips the optimizer at its last iteration (`iteration < max_iter`);
    * encode (step == total_steps): Morton sort, npz, decode into the base fields.
    """
    out = []
    if step <= base_steps:
        out.append("base")
    if step == base_steps:
        out.append("compress")
    if base_steps < step < total_steps:
        out.append("bank_step")
    if step == total_steps:
        out.append("encode")
    return out
