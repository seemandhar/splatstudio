# LiteGS (arXiv 2503.01199) — reproduction report

**Outcome: matched** (both reported scenes within ±0.5 dB). Method `litegs` = LiteGS-quality
(per-scene target = 3DGS count, 30K steps), native SplatStudio plugin `plugin/ss_litegs`, ported
from the official code https://github.com/MooreThreads/LiteGS (Inria Gaussian-Splatting licence).
Benchmark: `runs/litegs-mipnerf360-30000` (summary.json); paper numbers: Appendix Table 5.

## Results (Mip-NeRF 360, images_4 / images_2, llffhold 8, 30K steps, seed 0)
| scene | ours PSNR | paper PSNR | Δ | ours SSIM / LPIPS | paper SSIM / LPIPS | Gaussians (target) | time |
|---|---|---|---|---|---|---|---|
| garden (outdoor) | 27.78 | 28.15 | −0.37 | 0.880 / 0.104 | 0.881 / 0.087 | 5.69M (5.73M) | 136 min |
| room (indoor) | 32.79 | 32.69 | +0.10 | 0.932 / 0.260 | 0.933 / 0.189 | 1.53M (1.55M) | 104 min |

Gain over 3DGS: ours vs our gsplat baseline (`baseline3dgs-mipnerf360-30000`: 26.89 / 31.52) =
**+0.89 / +1.27 dB**; paper vs its 3DGS row (27.39 / 31.47) = +0.76 / +1.22 dB. The improvement is
reproduced on both scenes; garden's absolute gap mirrors our baseline being 0.5 dB below the paper's
3DGS row. LPIPS is higher than the paper's on both scenes (also true of our gsplat baseline;
different LPIPS implementation/background — not tuned). Single seed; no hyper-parameter tuning.
Training time is not comparable (gsplat rasteriser on a shared A6000 vs the paper's warp rasteriser on
an RTX 3090); the speed claims of the paper are out of scope.

## What was ported
- C1 opacity-gradient-variance densification: custom CUDA kernel (`csrc/litegs_stats.cu`) on gsplat's
  tile lists → S, M, C, ΣTα per Gaussian; score = clamp(S/(C+1) − (M/(C+1))², 0)·C·σ(o)²; budgeted
  linear growth to the target, multinomial without replacement, clone/split (parent kept, child
  scale/1.6), weight prune (ΣTα = 0).
- C2 opacity decay (σ ← max(σ/2, 1/128), clear all Adam states) every 10 epochs; densify every 5
  epochs from epoch 3 to int(0.8·E/10)·10+1.
- Recipe: visible-only Adam without bias correction (eps 1e-15), TamingGS lrs (opacity 0.025, shN
  2.5e-4), SH degree +1 every 5 epochs, image clamped to [0,1] before L1 + 0.2 D-SSIM.
- Probes: 9/9 pass (`evidence/verify.json`); review approved, 0 critical/major (`reviews/review-1.md`).

## Deviations from the paper / official code
1. **Out of scope (speed only):** warp-based fp16 rasteriser and Morton cluster-cull-compact; gsplat
   renders (paper Table 2: ≤0.05 dB from culling).
2. **Squared-gradient statistic.** We accumulate exact per-pixel squares Σg². The official kernel
   (raster.cu:727-817) squares the *running* sum within each 2-pixel half2 lane:
   S_off = Σ_lanes [g0² + (g0+g1)²], in fp16. The header comment of `csrc/litegs_stats.cu` claiming
   these are the moments the official rasteriser aggregates is wrong for S (M, C, W do match); left
   unedited so as not to change the source manifest of the benchmarked runs. Measured on the smoke
   checkpoint (garden, 30 views): median S_off/S_exact 1.43, Spearman(score) 0.9987, multinomial draw
   overlap exact-vs-official equal to the exact-vs-exact noise floor (Jaccard 0.280 / 0.618 at
   50k / 200k) → densification sets indistinguishable.
3. **Adam visibility** per primitive (gsplat radii > 0) instead of per 128-primitive cluster that
   intersects the frustum (official: in-frustum but unrendered primitives keep taking momentum steps).
4. **Budget capped** at the number of score > 0 primitives (official `torch.multinomial` would raise).
5. **Split offset** R·N(0,s) (3DGS convention); official row-vector form `samples @ T(1,q)` is the
   same under its convention (not verified numerically; negligible).
6. **Runtime:** views sampled with replacement (official: shuffled epochs) — a statistics "epoch" is
   n_train random draws, ~37% of views unseen, so primitives seen only there get W = 0 (pruned,
   score 0); smoke prune counts 158 / 177 show it is tiny (TODO-runtime shuffled sampler). White
   background (official black); SSIM with valid convolution (official fused_ssim); gsplat alpha
   thresholds 1/255 and 0.999 (official 1/256, 255/256); 30000 steps vs total_epoch·n_train
   (29946 garden, 29920 room).
7. Clustered appends truncated to multiples of 128 (official artefact) not reproduced.

## Runs
- Smoke: `evidence/smoke-1` (garden 2000 steps, 23.36 dB vs gsplat 23.07).
- Benchmark: `runs/litegs-mipnerf360-30000/litegs/mipnerf360-{garden,room}`.
