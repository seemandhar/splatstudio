# splatstudio-absgs

Native SplatStudio port of **AbsGS: Recovering Fine Details for 3D Gaussian Splatting**
(Ye et al., ACM MM 2024, https://arxiv.org/abs/2404.10484; official code
https://github.com/TY424/AbsGS, densification logic followed from `scene/gaussian_model.py`).

## Mechanism
- C1 homodirectional gradient: per-pixel |dL_j/dmu_x|, |dL_j/dmu_y| summed, then L2 norm,
  averaged over visible views (gsplat `absgrad=True` = official `backward.cu:549-550`).
- C2 decoupled densification: clone if signed ||g|| >= 0.0002 and max scale <= tau_S*extent;
  split if ||g_hat|| >= 0.0004 (absgs) / 0.0008 (absgs-0008) and max scale > tau_S*extent;
  tau_S = 0.001. Clones are never split in the same round. Pruning/opacity reset = 3DGS.
- Official periodic opacity reduction (`use_reduce=True`, every 500 steps before 15k: opacity ->
  min(o, 0.8), Adam moments zeroed; `train.py:132`, `gaussian_model.py:260`) is ON in `absgs`:
  it is the released default, `full_eval.py` leaves it on, and the README's "not used in our
  paper" list does not include it. The paper text never mentions it, so it is unknown whether
  the published numbers used it; `absgs-noreduce` turns it off.

Built-in `gsplat-absgrad` is **not** reused for density: it applies |grad| to both clone and
split with one threshold 0.0008 and tau_S 0.01. Only the gsplat renderer (absgrad) is reused.

## Methods
`absgs` (official defaults, paper main row), `absgs-0008` (variant), `absgs-noreduce`
(no opacity reduction), `absgs-noreset` (no opacity reset).

## Config (density component; defaults = paper)
clone_grad2d 0.0002, split_grad2d 0.0004, grow_scale3d 0.001, prune_opa 0.005,
prune_scale3d 0.1, refine 500..15000 every 100, reset_every 3000, opacity_reset true,
reduce_every 500 (official `use_reduce`/`opacity_reduce_interval`; 0 disables).

## Fidelity gaps
Runtime protocol (white bg, sampling with replacement, SSIM valid conv, LPIPS package);
the extras the official README lists as "not used in our paper" (initial SfM err/track filtering +
init_prune, weight-based pruning) are off.
Review-1 minors (documented, no action):
1. SfM filtering: released `dataset_readers.py:162-170` filters (err>2, track<3) whenever
   `points3D.ply` is absent, so it is on for fresh data in the release; we follow the README
   ("not used") and keep all SfM points (garden init 138,766 = unfiltered count).
   Weight pruning and init_prune are off by default in the release too.
2. "Clones never split in the same round" (zero padding, official `padded_grad`) is not probed:
   clone and split sets are disjoint by scale, so removing the padding only raises a shape error.
3. The runtime densifies after `optimizer.step()`; official densifies before it. Shared with the
   baseline, so it does not affect the AbsGS delta.
4. White-background compositing adds an alpha term to dL/dmu2D that enters both accumulators;
   official uses black. Shared protocol gap.

## Commands
`tools/ssenv ss-verify absgs --device cuda`; `python3 tools/bench.py ...` with method `absgs`.
