"""MesonGS run schedule (pure functions, CPU-testable).

`pretrain_means_lr` / `inferred_run_length` are copied from the c3dgs plugin
(reproductions/c3dgs/plugin/ss_c3dgs/schedule.py, itself from pup-3dgs and reduced-3dgs).
"""

import math


def pretrain_means_lr(lr0, step, base_steps, final_ratio=0.01):
    """Means LR for step+1 that the native runtime would set when run with --steps base_steps."""
    return lr0 * final_ratio ** (min(step, base_steps) / base_steps)


def inferred_run_length(lr0, lr_after_step1):
    """The runtime sets the means LR to lr0 * 0.01 ** (1 / steps) after step 1."""
    if not 0 < lr_after_step1 < lr0:
        return None
    return math.log(0.01) / math.log(lr_after_step1 / lr0)


def events(step, base_steps, total_steps):
    """Events after the runtime optimizer step of `step` (1-based).

    * base (step <= base_steps): 3DGS density control;
    * compress (step == base_steps): importance -> prune -> octree -> RAHT tables -> VQ;
    * bank_step (base_steps < step < total_steps): Adam step of the SH codebook (the official
      fine-tune loop takes no optimizer step at its last iteration);
    * freeze (step == total_steps - 1, or at compress when total = base + 1): runtime LRs set to 0
      so the runtime's optimizer step of the last iteration is a no-op, as in the official loop;
    * encode (step == total_steps): bitstream, decode into the base fields. With
      total_steps == base_steps (no fine-tune, paper "Our") compress and encode run together.
    """
    out = []
    if step <= base_steps:
        out.append("base")
    if step == base_steps:
        out.append("compress")
    if base_steps < step < total_steps:
        out.append("bank_step")
    if total_steps > base_steps and step == total_steps - 1 and step >= base_steps:
        out.append("freeze")
    if step == total_steps:
        out.append("encode")
    return out
