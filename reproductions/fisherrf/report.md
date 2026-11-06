# FisherRF (plan #92) — build report

**Status: built and verified, ready for benchmark (no PSNR comparison yet).**

- Paper: arXiv 2311.17874 (ECCV 2024). Official code github.com/JiangWenPL/FisherRF @ b747328
  (`main` is the Gaussian-splatting code; there is no separate GS branch). Plugin
  `plugin/` (`ss_fisherrf`); methods `fisherrf`, `fisherrf-random` (+ `-batch`, `-v10` variants).
- C1 Fisher statistic rendering: the official modified Inria rasterizer (`diff/`, fused
  squared-gradient backward) is vendored and JIT-built (custom CUDA). Probe: kernel diag H
  matches Σ_pixels (∂(R+G+B)/∂θ)² from per-pixel autograd of gsplat (rel err 1.7e-3 means,
  ≤3e-7 others); the plain (unsquared) gradient differs by 99.8%.
- C2 EIG selection: Σ H_c/(H_train+1e-6), greedy batch (Alg. 1), official random selector.
- C3 active loop inside the native runtime: training view 0 + 3 farthest cameras, acquisitions
  at it = 100·Σ|S| (400 … 18400), restore of the initial Gaussians + fresh optimizers after each
  (official code; the paper text says only "reset the opacity"), per-round LR/densification,
  SH +1 every 5000, budget 20. Non-selected frames from `ss-train` are skipped without a
  gradient or Adam step. Candidate pool = the 100 training views; test views never visible.

## Benchmark choice
No per-scene numbers anywhere. Cheapest complete average: **Blender, 8 scenes, sequential
4→20 views (Table 2)**: FisherRF 29.525 / random 28.732 (gain +0.79 dB). Mip-NeRF 360 (Table 5)
is 9 scenes at width 1600, the official scripts use 20000 iterations (the paper says 30000) and a black
background (the native runtime is white only), so it is not used.
New dataset `blender-fisherrf` (registered in tools/datasets.json; data on /bigdata, built by
`tools/prepare_blender_full.py`): official split, compositing, random init (np seed 0).

## Smoke test (lego, scaled schedule 4→8 views, 2000 effective its, 40k loop steps)
| selector | PSNR | SSIM | LPIPS | Gaussians | s |
|---|---|---|---|---|---|
| fisher | 22.33 | 0.833 | 0.168 | 42k | 815 |
| random | 22.82 | 0.847 | 0.159 | 49k | 616 |
Both runs complete and are finite. Fisher scored lower than random on this tiny run. One scene with
8 views says nothing about the paper's 20-view gain, but it should be checked in the benchmark.

## Fidelity gaps
1. Views sampled with replacement within the selected set (official: per-epoch stack).
2. gsplat trains the model (only the Fisher scoring uses the official kernel); SSIM uses valid convolution; LPIPS comes from the `lpips` package.
3. No 3DGS 20-px screen-size pruning (gsplat default, same as the shared baselines).
4. The RNG differs from the official code, so the selected views differ even though the algorithm is the same.
5. `--steps 230000` loop iterations are needed to get 30000 effective iterations. This is a
   wall-time cost of about 15 ms per skipped step on the shared GPU, not a change to the method.

## Review-1 resolution (orchestrator)
Approved, with 0 critical, 0 major and 5 minor issues. Every builder claim was checked against the official code:
- the kernel is identical to the official one;
- the Fisher statistic backprops rendered colour with ones;
- the parameters exclude rotation;
- λ = 1e-6 is added to H_train only;
- the seq1 schedule and the checkpoint restore follow the code;
- a 230k-step budget is enough (simulated 218.0k ± 1.3k, maximum 222.7k).
- **Smoke caveat:** the lego smoke result (Fisher < random) is uninformative. All of its acquisitions happen before densification, when the scores are dominated by Gaussians the training views never see.
- **Minor, not fixed:**
  - The M=1 scoring path is inline and not covered by a probe; the code itself is correct.
  - The -v10 variants need about 336k steps.
  - Densification runs after the Adam step.
  - The official viewpoint stack is stale after an acquisition.
- **Benchmark:** fisherrf-blender, fisherrf and fisherrf-random on all 8 Blender scenes at 230k loop steps. Compare with Table 2 (29.525 / 28.732, gain +0.79).

## Benchmark result (fisherrf-blender, sequential 4→20 views, 30k training iterations; all 8 Blender scenes) — MATCHED
| selector | ours (8-scene mean) PSNR / SSIM / LPIPS | paper Table 2 | Δ |
|---|---|---|---|
| Fisher (fisherrf) | **29.48** / 0.943 / 0.058 | 29.525 | −0.05 |
| random (fisherrf-random) | **28.71** / 0.938 / 0.062 | 28.732 | −0.02 |
| gain from active selection | **+0.77** | +0.79 | |

Per scene (Fisher / random):

| scene | Fisher | random |
|---|---|---|
| chair | 32.57 | 31.72 |
| drums | 23.19 | 23.33 |
| ficus | 30.27 | 30.63 |
| hotdog | 33.99 | 31.91 |
| lego | 30.66 | 29.69 |
| materials | 26.47 | 24.49 |
| mic | 31.71 | 31.35 |
| ship | 26.98 | 26.54 |

- Both complete averages and the active-selection gain are reproduced within 0.05 dB.
- The earlier 2k smoke result (Fisher < random) was not representative, as the review predicted.
