# FisherRF native plan

Method names: `fisherrf` (sequential EIG, 4 init → 20 views, Blender Table 2),
`fisherrf-random` (official random selector, same schedule), `fisherrf-batch` /
`fisherrf-batch-random` (4 views / 300 epochs, Table 3), `fisherrf-v10` / `fisherrf-v10-random`
(2 init → 10 views, Table 4). Module `ss_fisherrf`.

## Why the active loop lives inside components
`ss-train` samples a random frame from *all* training views every step and does not expose
images to components. FisherRF needs (a) training only on the selected subset, (b) a restore of
the initial Gaussians + fresh optimizers after each acquisition, (c) a schedule in *effective*
iterations. Implementation inside the three components (shared `ActiveController`):
- **renderer** (`fisherrf-renderer`, gsplat rasterization, same capabilities as `gsplat`):
  identifies the frame among the training cameras (viewmat match). Non-selected training frames
  (and frames after the budget is spent) are *skipped*: no rasterization, outputs carry
  `fisherrf_skip=True` and a dummy leaf so the loss is `0*dummy` → no parameter gets a gradient
  → Adam skips every parameter (grads are None after `zero_grad(set_to_none=True)`).
  On a selected frame the effective iteration `it` advances; if `it ∈ load_its` the
  controller first runs the selector on the current model (Fisher kernel, all training
  cameras = candidate pool; test/validation cameras are never visible to the controller), adds
  the views, restores the initial params/optimizer state/density state. It sets the xyz LR
  `lr0·0.01^(cur/30000)` (official expon_lr, cur = it − base_iter), SH degree (+1 when
  it % 5000 == 0, 0 after restore). Frames that are not training cameras (validation/test
  evaluation) render normally with the runtime's degree; if they arrive before the effective
  budget is spent the controller raises (run fails loudly instead of under-training).
- **density** (`fisherrf-density`): wraps `splatstudio.backends.gsplat.Density`
  (DefaultStrategy + 3DGS opacity reset) driven with `cur_iter`; ignores skipped steps;
  extra opacity reset at cur_iter == 500 (official white-background rule); densify until
  10000; registers the training cameras and the optimizer dict with the controller.
- **loss** (`fisherrf-photometric`): official L1 + 0.2 D-SSIM; `0*dummy` on skipped steps.
- Global `ss-train --steps` must exceed the expected loop count (~218k for seq1 with a
  100-view pool): use `--steps 230000` (effective iterations stay 30000 = paper).

## Contributions → code → probes
| C | code | probe (fails if removed) |
|---|---|---|
| C1 Fisher statistic rendering | `ss_fisherrf/csrc` = official `diff/` (modified Inria rasterizer, fused squared-gradient backward) JIT-built; `fisher.py` wrapper (activated inputs, autograd to raw params like the official) | `fisher-kernel` (CUDA): kernel diag H == Σ_pixels (∂(R+G+B)_p/∂θ)² from per-pixel autograd of gsplat (independent rasterizer) on a small scene; differs from the plain-gradient square (Σ_p ∂/∂θ)² |
| C2 EIG selection | `selection.py`: `fisher_scores`, greedy batch (Alg. 1), `random_select` | `eig-selection` (CPU): scores vs independent numpy; greedy == brute force sequential argmax; random == `random.Random(0)` shuffle; fisher pick ≠ random pick on a designed case |
| C3 active loop | `active.py` `ActiveController`, `schedule()` = VNSeqMInplace, farthest-point init | `active-schedule` (CPU): load_its seq1/seq4/v10 = official; init views; `active-loop` (CUDA): mini loop on a synthetic scene — only selected views get gradients, params restored to init at each acquisition, pool excludes held-out cameras, budget reached |

## Fields
None (only base 3DGS fields).

## Unchanged from 3DGS / gsplat baseline
gsplat rasterizer for training, Adam LRs, DefaultStrategy clone/split/prune thresholds.

## Fidelity gaps
1. View sampling: random with replacement among the selected set (official: stack without
   replacement, stale for one epoch after a selection). Small.
2. gsplat vs Inria rasterizer for training (Fisher uses the official kernel); SSIM valid conv;
   LPIPS-VGG lpips package vs lpipsPyTorch. Small (calibrated elsewhere ~0.1 dB).
3. 3DGS 20-px screen-size pruning not applied (gsplat default, same as shared baseline).
4. Training RNG differs from official → different selected views from the same algorithm.
5. Mip-NeRF 360 setting not benchmarked (paper text 30k vs script 20k; black background in the
   official code, native runtime is white-background only).
