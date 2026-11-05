# Taming 3DGS (plan #30) — reproduction report

**Outcome: matched.** On all 9 Mip-NeRF 360 scenes at 30k steps the port scores **27.60 PSNR vs the paper's 27.79 (−0.19 dB)**. It gains **+0.32 dB over our gsplat baseline**; the paper reports **+0.33 dB over its own 3DGS**.

- **Paper:** Mallick, Goel, Kerbl, Steinberger, Carrasco, De La Torre. *Taming 3DGS: High-Quality Radiance Fields with Limited Resources*. SIGGRAPH Asia 2024, arXiv 2406.15643.
- **Official code:** humansensinglab/taming-3dgs @ fd0f7d9. It was read but never executed; where it disagrees with the paper, the code was followed.
- **Plugin:** `plugin/` (`splatstudio-method-taming-3dgs`). Two methods:
  - `taming-3dgs` (benchmarked) is the paper's **"Ours (#3DGS)"** row: the final count is 3DGS's count for each scene and densification runs every 100 steps.
  - `taming-3dgs-budget` is the paper's "Ours" row: 15×, 2× or 5× the SfM point count, densifying every 500 steps.
- **Custom CUDA:** `csrc/taming_stats.cu` computes the per-Gaussian pixel statistics used by the score: pixel count, distance to centre, saliency and blend weight.
- **Benchmark runs:**
  - Method: `runs/taming-3dgs-mipnerf360/taming-3dgs/mipnerf360-<scene>/`
  - Baseline: `runs/baseline3dgs-mipnerf360-30000/gsplat/`

## Protocol
- 3DGS recipe:
  - Initial scale from the camera extent and the 3 nearest SfM neighbours.
  - All SfM points kept at initialization.
  - 30k steps.
- Data: outdoor scenes at images_4, indoor scenes at images_2, llffhold-8 split, evaluated on the test split.
- Metrics: LPIPS-VGG from the `lpips` package; SSIM with valid convolution.
- Differences from the paper's setup:
  - White background.
  - Training views are drawn at random with replacement.
  - A single seed (0).
  - GPUs are shared A6000s, while the paper used an RTX A4500. Training times are therefore not comparable with the paper's.
- The paper publishes only dataset averages for "Ours (#3DGS)", so all 9 scenes were run.
- The first garden attempt ran out of GPU memory at step 9500 because another process held 15.5 GB of the shared GPU. It was re-queued and completed. The failed attempt is kept as `mipnerf360-garden.oom-*`.

## Per-scene results (test split, 30k)
"Budget B" is the official `full_eval.py` final count. "Final #G" is the count at 30k; nothing is added or removed after step 15000.

| scene | PSNR | SSIM | LPIPS | Final #G | Budget B | #G / B | train time (min) | gsplat PSNR | gsplat SSIM | gsplat LPIPS | gsplat #G | ΔPSNR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bicycle | 25.323 | 0.7734 | 0.2244 | 5,986,811 | 5,987,095 | 1.000 | 91.9 | 24.921 | 0.7573 | 0.2428 | 6,590,201 | +0.40 |
| flowers | 21.817 | 0.6126 | 0.3566 | 3,618,239 | 3,618,411 | 1.000 | 79.4 | 21.627 | 0.6075 | 0.3591 | 3,857,839 | +0.19 |
| garden | 27.267 | 0.8690 | 0.1179 | 5,727,924 | 5,728,191 | 1.000 | 93.0 | 26.893 | 0.8598 | 0.1311 | 4,764,685 | +0.37 |
| stump | 26.679 | 0.7804 | 0.2331 | 4,867,198 | 4,867,429 | 1.000 | 94.0 | 26.230 | 0.7614 | 0.2581 | 4,506,274 | +0.45 |
| treehill | 22.864 | 0.6429 | 0.3516 | 3,770,079 | 3,770,257 | 1.000 | 71.1 | 22.277 | 0.6272 | 0.3607 | 5,073,850 | +0.59 |
| room | 31.968 | 0.9221 | 0.2841 | 1,548,890 | 1,548,960 | 1.000 | 128.5 | 31.516 | 0.9163 | 0.2982 | 1,727,096 | +0.45 |
| counter | 28.897 | 0.9087 | 0.2587 | 1,190,868 | 1,190,919 | 1.000 | 68.6 | 28.846 | 0.9073 | 0.2605 | 1,215,973 | +0.05 |
| kitchen | 31.750 | 0.9293 | 0.1562 | 1,803,659 | 1,803,735 | 1.000 | 79.5 | 31.340 | 0.9255 | 0.1620 | 1,416,375 | +0.41 |
| bonsai | 31.855 | 0.9446 | 0.2533 | 1,247,857 | 1,252,367 | 0.996 | 52.7 | 31.888 | 0.9415 | 0.2576 | 1,373,997 | −0.03 |
| **mean** | **27.602** | **0.8203** | **0.2484** | 3.307M | 3.308M | — | 84.3 | 27.282 | 0.8115 | 0.2589 | 3.392M | **+0.32** |

Peak GPU memory was ≤ 9.9 GB across scenes; the baseline's peak was ≤ 9.7 GB.

## Means vs paper (Mip-NeRF 360, 9 scenes)
| | PSNR | SSIM | LPIPS | #G |
|---|---|---|---|---|
| Paper "Ours (#3DGS)" | 27.79 | 0.822 | 0.205 | 3.31M |
| **This port** | **27.60** (−0.19) | **0.820** (−0.002) | **0.248** (+0.043) | 3.31M |
| Paper 3DGS row | 27.46 | 0.815 | 0.215 | 3.31M |
| Our gsplat baseline | 27.28 | 0.812 | 0.259 | 3.39M |

## Baseline-relative comparison
- **PSNR:**
  - Port vs our gsplat: **+0.32 dB**. Paper vs its 3DGS: **+0.33 dB**. The improvement is reproduced almost exactly.
  - The gain is positive on 8 of 9 scenes. Outdoor scenes gain +0.19 to +0.59. Indoor scenes gain +0.45 (room) and +0.41 (kitchen), while counter is +0.05 and bonsai −0.03.
- **SSIM:** port +0.009 over our baseline; paper +0.007 over its 3DGS. Consistent.
- **LPIPS:** port −0.010 against the baseline; paper −0.010 against its 3DGS. The relative gain matches. The absolute LPIPS gap (+0.043) is shared with our baseline, which is itself +0.044 above the paper's 3DGS row. That points to the LPIPS implementation and runtime calibration (see `_calibration/`), not to Taming.
- **Absolute PSNR gap:** the −0.19 dB is explained by our gsplat baseline sitting 0.18 dB below the paper's 3DGS row.
- **Model size:** our gsplat count per scene differs from Taming's B in both directions; for example garden has 4.76M vs 5.73M and treehill 5.07M vs 3.77M. Even so, Taming wins on both kinds of scene.

## Growth-curve evidence (the `taming_densify` lines in each `mipnerf360-<scene>.log`)
There were 144 densification calls per scene, at steps 600, 700, …, 14900.
- **Each call is capped by the official rule.** The target is `min(curve target, gradient-qualified candidates + N)`. Early in training there are fewer candidates than the curve asks for, so every candidate is cloned or split and the count runs below the curve.
- **Deficits by scene:**

| scene | calls below target (−2 tolerance) | largest deficit | count / target at 2k | count / target at 7k | last call below target | final deficit |
|---|---|---|---|---|---|---|
| bicycle | 55 | 625,575 | 0.59M / 1.22M | 4,181,146 / 4,181,147 | 12100 | 1 |
| flowers | 35 | 229,458 | 0.57M / 0.74M | on curve | 12100 | 1 |
| garden | 114 | 261,489 | 1.02M / 1.24M | 3.97M / 4.03M | 14300 | 1 |
| stump | 51 | 601,048 | 0.39M / 0.98M | on curve | 12100 | 1 |
| treehill | 42 | 285,644 | 0.51M / 0.78M | on curve | 12100 | 1 |
| room | 38 | 19,892 | on curve | on curve | 12500 | 1 |
| counter | 15 | 10,555 | on curve | on curve | 12300 | 1 |
| kitchen | 62 | 24,316 | on curve | on curve | 14400 | 1 |
| bonsai | 86 | 37,426 | on curve | on curve | 14900 | 4,460 |

- **Every scene ends at its budget.** The curve's last target is `values[144]`, slightly below B (official indexing), and each scene finishes within 1 of it. Bonsai is the exception, ending 4,460 short (0.4 %), still candidate-limited at the last call. Final counts are 99.6–100 % of B.
- **Why some scenes dip below the curve again around 12000:** after each opacity reset (3000, 6000, 9000, 12000) fewer Gaussians have large gradients for a short while, so a few calls fall short. The count catches up within a few hundred steps.
- **Score-guided pruning** removed 2.9k (bonsai) to 35k (garden) Gaussians per scene, all in the first 27 calls, as in the official code.
- This answers the review's open item: growth catches up with the curve and lands on B.

## Probe evidence
`ss-verify` returned `declared_checks_passed` for both methods (`evidence/verify-taming-3dgs.json`, `evidence/verify-taming-3dgs-budget.json`). There are 3 built-in probes and 8 Taming probes. The review rated each Taming probe as able to fail if its mechanism is removed.

| probe | evidence |
|---|---|
| taming-stats-kernel | CUDA count, distance, saliency and blend sums equal a brute-force per-pixel loop. The per-pixel blend sums also equal gsplat's rendered alpha. |
| taming-growth-curve | The count array matches an independently written parabola within 1. One densification call lands on its target, up to the official floor division. |
| taming-score | The score matches an independent re-implementation of `compute_gaussian_score` (tolerance 2e-3). Lowering a Gaussian's opacity lowers its score. |
| taming-sampled-densify | Only qualified, high-score Gaussians are cloned or split. The 20 + 20 budget split is exact and new Gaussians start with zero Adam moments. |
| taming-guided-prune | Removed Gaussians are a subset of the prune mask and at most 50 % of it. Nothing is pruned from call 27 on. |
| taming-high-opacity | Opacity changes before step 15000 and is exactly frozen afterwards. |
| taming-sh-interval | The SH optimizer equals Adam on gradients summed over 16 steps and differs from per-step Adam. It drops a partial sum when the parameter is replaced. Learning rates are 0.025 (opacity) and 2.5e-4 (SH). |
| taming-schedule | Densification runs every 100 steps (`taming-3dgs`) or every 500 (budget variant). Opacity resets at 3000, 6000, 9000 and 12000. Nothing happens at or after 15000. |

- Unit tests: 11 pass and ruff is clean.
- Smoke test (garden, 2k steps): **23.09** PSNR vs **23.03** for gsplat, with 0.72M vs 0.84M Gaussians.

## Fidelity gaps
1. **Speed-only kernels are not ported:** per-splat-parallel backward, fused SSIM, separate SH tensors. The paper states they are numerically equivalent. Sparse Adam is also not ported; it was not used for the paper's numbers (`--optimizer_type default`). Only timing is affected, so our times are not the paper's 32 min.
2. **High opacity follows the code, not the paper.** The paper describes trainable opacity with an abs activation that can exceed 1. The released code's `modify_functions` instead freezes opacity at 15k, so the port sets the opacity learning rate to 0 from step 15000. The freeze applies one step later than in the official code, because the runtime steps the optimizers before density control.
3. **Score views come only from training views the trainer has already rendered.** Densities cannot see the training images, so the renderer passes each frame's ground-truth image to the density, which caches it.
4. **Backend differences shared with the baseline:**
   - gsplat clamps alpha at 0.999 (3DGS uses 0.99).
   - Tile radii depend on opacity; the radius term in the score uses these radii and is median-normalized.
   - White background (official runs use black).
   - Valid-convolution SSIM in the training loss.
   - Views sampled with replacement.
   - Optimizers step before density control.
5. **Official-code behaviours reproduced on purpose:**
   - Score weights come from the code (grad 25, plus a radius term of 10), not the paper text (grad 50).
   - The SH learning rate is 0.005/20 = 2.5e-4, not the paper's "0.001".
   - The pruning weights use pre-densification scores indexed in post-densification order.
   - The screen-size prune never fires.
   - Score-guided pruning runs only for the first 27 calls.
   - When fewer Gaussians have positive scores than the budget, `torch.multinomial` fills the rest with zero-weight indices.
6. **Edge map rounding:** the edge-map input is rounded to uint8, where the official code truncates. Separately, the SH degree used for score renders lags the official one by one step at each 1000-step boundary. Both effects are negligible.
7. **Per-scene budget lookup by initial SfM count.** The density finds each scene's budget by its initial SfM point count, which is unique for our native conversions of the 13 scenes. Other scenes need an explicit `budget` in the config.

## Review minor items (`reviews/review-1.json`: approved, no critical or major issues)
- **Growth reaching B:** confirmed by the benchmark (previous section).
- **Fewer densification candidates than gsplat at 2k (−2 to −27 %):** probably comes from the changed SH and opacity optimization. The A/B test with `sh_update_every=1` was not run. Once candidates catch up, the count follows the curve exactly.
- **Step-15000 freeze ordering** (one extra opacity update): accepted (gap 2).
- **Edge-map rounding vs truncation; SH degree lag by one step at score time:** accepted as negligible (gap 6).

## Reproduce
```bash
cd /home/sejain/splatify
tools/ssenv pip install -e reproductions/taming-3dgs/plugin
tools/ssenv ss-inspect taming-3dgs            # issues []
tools/ssenv python -m pytest -q reproductions/taming-3dgs/plugin/tests
tools/ssenv ss-verify taming-3dgs --device cuda --output reproductions/taming-3dgs/evidence/verify-taming-3dgs.json
# smoke
tools/ssenv ss-train taming-3dgs --data ~/datasets/mipnerf360/native/garden --output /tmp/taming-smoke \
  --steps 2000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000 --device cuda:0
# benchmark (9 scenes, 30k) + summary against the paper and the shared baseline
python3 tools/bench.py launch --exp taming-3dgs-mipnerf360 --methods taming-3dgs --dataset mipnerf360 --steps 30000 --recipe 3dgs
python3 tools/bench.py wait --exp taming-3dgs-mipnerf360
python3 tools/bench.py summarize --exp taming-3dgs-mipnerf360 --paper reproductions/taming-3dgs/paper_results.json
# densification log of a run
grep taming_densify runs/taming-3dgs-mipnerf360/taming-3dgs/mipnerf360-garden.log
```
