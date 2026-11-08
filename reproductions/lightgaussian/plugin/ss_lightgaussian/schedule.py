"""LightGaussian single-run schedule (pure functions, CPU-testable).

Global iteration numbering of the official scripts: 3DGS 1..30000 (`train.py`), co-adaptation
30001..35000 (`prune_finetune.py --iteration 35000 --position_lr_max_steps 35000`), SH
distillation 35001..40000 (`distill_train.py --iteration 40000 --position_lr_max_steps 40000`),
VQ (`vectree.py`) after 40000, VQ fine-tune 40001..45000 (paper §3.4; no released script).
`pretrain_means_lr` / `inferred_run_length` follow reproductions/c3dgs/plugin/ss_c3dgs/schedule.py.
"""

import math

STAGES = ("3dgs", "coadapt", "distill", "vqft", "done")


def stage(k, cfg):
    """Stage that optimizer step `k` (1-based) belongs to."""
    if k <= cfg["prune_step"]:
        return "3dgs"
    if k <= cfg["distill_step"]:
        return "coadapt"
    if k <= cfg["vq_step"]:
        return "distill"
    if k <= cfg["total_steps"]:
        return "vqft"
    return "done"


def events(step, cfg):
    """Events after the runtime optimizer step of `step`."""
    out = []
    if step == cfg["prune_step"]:
        out.append("prune")
    if step == cfg["distill_step"]:
        out.append("start_distill")
    if step == cfg["vq_step"]:
        out.append("vq")
        if cfg["vq_step"] < cfg["total_steps"]:
            out.append("start_vqft")
    if step == cfg["total_steps"]:
        out.append("encode")
    return out


def validate(cfg):
    p, d, v, t = cfg["prune_step"], cfg["distill_step"], cfg["vq_step"], cfg["total_steps"]
    if not 0 < p < d < v <= t:
        raise ValueError("lightgaussian needs 0 < prune_step < distill_step < vq_step <= "
                         "total_steps")


def pretrain_means_lr(lr0, step, base_steps, final_ratio=0.01):
    """Means LR for step+1 that the runtime sets in a `--steps base_steps` run."""
    return lr0 * final_ratio ** (min(step, base_steps) / base_steps)


def expon_lr(lr0, k, max_steps, final_ratio=0.01):
    """3DGS get_expon_lr_func(lr0, final_ratio lr0, delay 0, max_steps)(k)."""
    t = min(max(k / max_steps, 0.0), 1.0)
    return math.exp(math.log(lr0) * (1 - t) + math.log(lr0 * final_ratio) * t)


def means_lr(lr0, k, cfg):
    """Means LR used at optimizer step k."""
    s = stage(k, cfg)
    if s == "3dgs":
        return pretrain_means_lr(lr0, k - 1, cfg["prune_step"])
    end = {"coadapt": cfg["distill_step"], "distill": cfg["vq_step"]}.get(s, cfg["total_steps"])
    return expon_lr(lr0, k, end)


def other_lr_factor(k, cfg):
    """ExponentialLR factor of the non-xyz groups at step k.

    coadapt: gamma 0.95 stepped when k % 400 == 0 (prune_finetune.py), distill: gamma 0.90 when
    k % 500 == 0 (distill_train.py), vqft: as coadapt (prune_finetune-style fine-tune; choice).
    The scheduler step happens at the start of the iteration, so step k already uses it.
    """
    s = stage(k, cfg)
    if s == "coadapt":
        start, gamma, every = cfg["prune_step"], cfg["coadapt_gamma"], cfg["coadapt_every"]
    elif s == "distill":
        start, gamma, every = cfg["distill_step"], cfg["distill_gamma"], cfg["distill_every"]
    elif s == "vqft":
        start, gamma, every = cfg["vq_step"], cfg["vqft_gamma"], cfg["vqft_every"]
    else:
        return 1.0
    return gamma ** (k // every - start // every)


def weight_decay(k, cfg):
    """Decoupled (AdamW) weight decay of step k: 0 in co-adaptation (Adam param groups restored
    from the 3DGS checkpoint), torch AdamW default 0.01 after a fresh `training_setup`."""
    return {"coadapt": cfg["coadapt_weight_decay"], "distill": cfg["distill_weight_decay"],
            "vqft": cfg["vqft_weight_decay"]}.get(stage(k, cfg), 0.0)


def pseudo_view(k, cfg):
    """distill_train.py: `if args.augmented_view and iteration % 3` -> pseudo view."""
    return stage(k, cfg) == "distill" and cfg["pseudo_views"] and k % cfg["pseudo_every"] != 0


def inferred_run_length(lr0, lr_after_step1):
    """The runtime sets the means LR to lr0 * 0.01 ** (1 / steps) after step 1."""
    if not 0 < lr_after_step1 < lr0:
        return None
    return math.log(0.01) / math.log(lr_after_step1 / lr0)
