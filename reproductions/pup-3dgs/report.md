# PUP 3D-GS (plan #36): reproduction report

**Outcome: matched.** With 90% of Gaussians pruned, all 3 scenes are within ±0.5 dB of the paper (−0.42 / −0.33 / −0.14 dB). The pruning loss also matches the paper's (−0.62/−1.00/−0.99 vs −0.37/−1.05/−1.11 dB).
Single seed (0). Only 3 scenes, so this is not a dataset average.

- Paper: PUP 3D-GS, CVPR 2025, [arXiv 2406.10219](https://arxiv.org/abs/2406.10219) (v3). Official code: [gaussian-splatting-pup](https://github.com/j-alex-hanson/gaussian-splatting-pup), with its Fisher kernel from `rasterization_and_pup_fisher` @ eaef2fa.
- Plugin: `plugin/`. `pup-3dgs` prunes 80% then 50% (90% total). `pup-3dgs-v1` prunes 66% then 66% (88.44%) and has not been benchmarked.
- Runs: `runs/pup-3dgs-mipnerf360/pup-3dgs/mipnerf360-{bicycle,garden,counter}/` (40k steps). Pre-prune reference: `runs/baseline3dgs-mipnerf360-30000/gsplat/`.

## Pipeline (one native 40k-step run)
| steps | phase |
|---|---|
| 1–30000 | Unchanged native 3DGS (gsplat DefaultStrategy plus the opacity-reset fix). The density component sets the means LR to the 30k schedule `lr0·0.01^(s/30000)`, so this phase uses the same recipe, seed and view sequence as `baseline3dgs-*-30000`. On a 2k-step smoke test the training logs agree: loss 0.09862 vs 0.09863 and 1.17M vs 1.15M Gaussians at step 2000. |
| after 30000 | Compute the Fisher over all training views with the official CUDA kernel. It uses the mean and log-scale blocks (6×6) on a 1/4-resolution render, i.e. 4×4 patches. Score = log\|F\|. Prune the lowest 80% using the official percentile rule. Reset Adam to a fresh state. |
| 30001–35000 | Refine with the official fine-tune LRs: xyz uses `expon(1.6e-4·s→1.6e-6·s, max 35000)` at lr_iter 30001…35000, other fields are multiplied by 0.95 every 400 iterations, no densification. |
| after 35000 | Prune 50% the same way. Reset Adam. |
| 35001–40000 | Refine again with the schedule restarted at lr_iter 30001 (the official `--first_iter 30000`). |

## Results (Mip-NeRF 360 test split, llffhold8, images_4 outdoor / images_2 indoor)
| scene | ours PSNR | paper PSNR | gap | ours SSIM | paper SSIM | ours LPIPS | paper LPIPS | Gaussians: base → final | train min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 24.30 | 24.72 | −0.42 | 0.709 | 0.7270 | 0.317 | 0.2965 | 6.59M → 0.652M | 74 |
| garden | 25.90 | 26.23 | −0.33 | 0.800 | 0.8189 | 0.228 | 0.1997 | 4.76M → 0.478M | 128 |
| counter | 27.86 | 28.00 | −0.14 | 0.881 | 0.8917 | 0.316 | 0.2297 | 1.22M → 0.121M | 108 |

- Final counts are exactly 10% of the in-run 30k count.
- Paper values come from supplementary Tables psnr/ssim/lpips_scenes in v3. The v3 SSIM table is identical to v1's.
- LPIPS is the VGG model from the `lpips` package. Our baseline LPIPS is offset from the paper's, so compare LPIPS as deltas only (next table).

## Pruning loss (pruned − 3DGS) vs paper (PUP − its own 3DGS)
| scene | our 3DGS (baseline3dgs) | ours pruned | Δ ours | paper 3DGS | paper PUP | Δ paper | ΔSSIM ours / paper | ΔLPIPS ours / paper |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 24.92 | 24.30 | −0.62 | 25.09 | 24.72 | −0.37 | −0.049 / −0.020 | +0.074 / +0.052 |
| garden | 26.89 | 25.90 | −1.00 | 27.28 | 26.23 | −1.05 | −0.060 / −0.037 | +0.097 / +0.077 |
| counter | 28.85 | 27.86 | −0.99 | 29.11 | 28.00 | −1.11 | −0.027 / −0.022 | +0.055 / +0.046 |
| mean | 26.89 | 26.02 | −0.87 | 27.16 | 26.32 | −0.84 | | |

Our absolute gap is mostly inherited from the lower base model (−0.17 / −0.39 / −0.26 dB vs the authors' 3DGS). SSIM and LPIPS degrade somewhat more than in the paper.

## Prune events (from the training logs)
| scene | prune @30000: before → after (zero-Fisher, s) | prune @35000: before → after (zero-Fisher, s) |
|---|---|---|
| bicycle | 6,521,575 → 1,304,315 (1,016,971; 16.7 s) | 1,304,315 → 652,157 (2,438; 7.5 s) |
| garden | 4,780,521 → 956,103 (503,993; 15.5 s) | 956,103 → 478,051 (1,106; 13.6 s) |
| counter | 1,212,796 → 242,559 (137,676; 34.2 s) | 242,559 → 121,279 (41; 21.6 s) |

- The in-run 30k counts differ from the baseline run's checkpoint by ≤1.5% because CUDA atomics are nondeterministic.
- Zero-Fisher Gaussians have no gradient in any training view; they get score −inf and are pruned first.
- No PSNR trajectory is available: the runtime evaluates only at the end, and `training.jsonl` logs only the single-view training loss every 500 steps.
- That loss rises after each prune and then settles. Bicycle: 0.047 @29.5k → 0.104 @31k → 0.076 @35k → 0.079 @40k. Garden: 0.037 → 0.084 @30.5k → 0.042 @35k → 0.089 @36k → 0.054 @40k. Counter: 0.027 → 0.095 @30.5k → 0.034 @35k → 0.039 @40k.

## Sanity check: Fisher vs opacity ranking
`tools/fisher_on_checkpoint.py` was run on the bicycle `baseline3dgs` 30k checkpoint (6.59M Gaussians, 169 views; result in `evidence/fisher_bicycle_baseline30k.json`).
- Pruning 80% in one shot without refinement: Fisher ranking gives **19.16 dB**, opacity ranking gives **15.91 dB** (from 24.92 unpruned).
- The Fisher pass took 43 s with 9.3 GB peak memory on a shared GPU.
- For context, the paper's 9-scene mean after the 80% prune without refinement is 21.00 dB (Table 1).

## Probes and tests (`evidence/verify.json`, `evidence/test.json`)
- `fisher-oracle` (CUDA): compares the vendored kernel with per-pixel autograd through gsplat. Setup: 24 Gaussians, 3 low-res views, off-centre principal point.
  - Relative error 4.9e-4; log-det correlation 0.99999994.
  - It would catch a Fisher taken w.r.t. linear scales (error 0.65) or at full resolution (error 0.96).
- `prune-schedule` (CUDA): two prune rounds on a 400-Gaussian scene.
  - Survivors are the highest float64 `slogdet` scores; counts follow the official rule (400→80→40).
  - Unseen Gaussians are pruned and Adam is reset.
  - LRs match an independent replay of `get_expon_lr_func` + torch `ExponentialLR`, including the boundaries at lr_iter 30399/30400/30401/34800/35000. The pretrain LR follows the 30k schedule.
- Other checks:
  - `ss-inspect`: no issues for either method.
  - Unit tests: 6 passed (LR functions, percentile rule with ties, 21→6×6 expansion, camera conversion vs pinhole).
  - Lint: ruff clean.
  - `ss-verify`: `declared_checks_passed`.
  - Fix made during testing: the oracle originally used one view, where the Fisher is rank ≤5 and its log-det is noise. It now uses 3 views.

## Fidelity gaps
1. **Base 3DGS is not the authors' models.** Ours is the native gsplat recipe: white background, random views with replacement, valid-convolution SSIM, gsplat DefaultStrategy with the reset fix. It scores 0.17–0.39 dB below the authors' pretrained 3DGS on these scenes. This cannot be closed without their checkpoints.
2. **Fisher rasterizer.** The official 3DGS-rasterizer kernel runs on a gsplat-trained model. Differences: alpha clamp 0.99 vs 0.999, near plane 0.2 vs 0.01, white background (our training background) vs black officially. The kernel's carry-over of colour gradients across channels is kept as in the official code. Agreement with gsplat autograd is 5e-4, so the effect on ranking should be negligible.
3. **Refine updates.** Each round has 5000 effective updates vs the official 4999, because the official prune iteration discards its gradient.
4. **Pre-prune reference.** It comes from the separate same-recipe `baseline3dgs` run, not the same run. Only the training-loss logs are compared, not bitwise identity.
5. **Not measured.** FPS and model size (the paper reports RTX A4000 FPS).
6. **Untested settings.** Only 3 of 13 scenes and one seed. The `pup-3dgs-v1` variant (88.44%) has not been benchmarked.

## Review items
- On bicycle the pruning loss is 0.25 dB larger than the paper's, and SSIM and LPIPS degrade more on all three scenes. A likely cause is that our base model has more weakly constrained background Gaussians: 15% of bicycle has zero Fisher. Not investigated further, and no tuning was done.
- The paper's ±0.5 dB match is partly helped by the pruning loss on garden and counter being slightly smaller than the paper's.

## Reproduce
```bash
tools/ssenv pip install -e reproductions/pup-3dgs/plugin
tools/ssenv ss-verify pup-3dgs --device cuda --output reproductions/pup-3dgs/evidence/verify.json
python3 tools/bench.py launch --exp pup-3dgs-mipnerf360 --methods pup-3dgs --dataset mipnerf360 --scenes bicycle garden counter --steps 40000 --recipe 3dgs
# one scene by hand:
tools/ssenv ss-train pup-3dgs --data ~/datasets/mipnerf360/native/bicycle --output runs/x --steps 40000 \
    --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000 --device cuda:0
tools/ssenv ss-eval --run runs/x --split test --device cuda:0
# Fisher sanity check on a trained checkpoint:
tools/ssenv python reproductions/pup-3dgs/tools/fisher_on_checkpoint.py \
    --run runs/baseline3dgs-mipnerf360-30000/gsplat/mipnerf360-bicycle --ratio 0.8 --output out.json
```
