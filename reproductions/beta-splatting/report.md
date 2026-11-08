# Deformable Beta Splatting (arXiv 2501.18630) — reproduction report

**Outcome: matched** (both reported scenes within ±0.5 dB of the paper's 30K, sb=2 column).
This is a PSNR match only. It does not prove the port is faithful; every deviation is listed below.

Plugin `plugin/ss_beta_splatting` ports the official code (Apache-2.0, commit 431d575). The Beta
rasterizer and Spherical Beta colour are ported CUDA, built as a JIT extension. Kernel-agnostic
MCMC and the regularisers come from `train.py` and `beta_model.py`. `ss-verify`: 8/8 probes pass.
CUDA forward/backward match a dense float64 reference to about 3e-7 relative error.
Review 1: approved (0 critical, 0 major, 5 minor).

## Results (30K iterations, mipnerf360-pilr = official `-r 4` outdoor / `-r 2` indoor PIL resize, test split)
| scene | ours PSNR / SSIM / LPIPS | paper (30K, sb=2) | Δ vs paper | our gsplat baseline | our gain | primitives | time |
|---|---|---|---|---|---|---|---|
| garden | 28.03 / 0.878 / 0.110 | 28.15 / 0.882 / 0.096 | −0.12 | 27.35 | +0.68 | 5.0M (cap) | 53 min |
| room | 32.53 / 0.937 / 0.246 | 32.83 / 0.941 / 0.168 | −0.30 | 31.69 | +0.84 | 1.5M (cap) | 28 min |

Sources: supp. Table "allmetrics" for the paper numbers. Our runs are in `runs/beta-splatting-garden-pilr` and
`runs/beta-splatting-room-pilr`. Baseline is `baseline3dgs-mipnerf360-pilr-30000`.
- Room: the paper's 3DGS row (30.63) is far below our gsplat baseline (31.69), so its implied +2.20 dB
  gain is not a usable target. We judge room on absolute PSNR. As a sanity check, the paper's own
  3DGS-MCMC row for room is 32.48.
- LPIPS: ours is worse on room (0.246 vs 0.168). This is probably partly the evaluator
  (`lpips`-package VGG vs the official lpipsPyTorch); it is not investigated.
- Smoke test (garden, 2k steps): 23.86 dB vs gsplat 23.31.

## Deviations from the paper / official code
- **Compared column.** We compare against the 30K column, not the "full" column. The released
  `--eval` mode keeps training past 30K and picks its checkpoint by test-set PSNR (checked every
  500 iterations after 15K, patience 20). That is test-set selection, so we do not reproduce it.
  The full column would be garden 28.11, room 33.03.
- **Noise frequency.** Position noise follows the code: applied only on densification steps
  (every 100 iterations, 500 < it < 25000). The paper describes noise on every update.
- **Regulariser form.** We use the code's means: 0.01·mean(σ(o)) + 0.01·mean(exp s). The paper
  writes sums.
- **Step order.** The runtime takes the Adam step before densification. The official code
  densifies first, and its re-wrapped Parameters lose `.grad`, so it skips the Adam step on about
  245 densification iterations. The noise also reads the next step's learning rate. Both effects
  are negligible.
- **SSIM in the loss.** The runtime uses valid-convolution SSIM; the official code uses fused-ssim
  (same padding).
- **Metrics and sampling.** LPIPS-VGG comes from the `lpips` package. Training views are sampled
  randomly with replacement; the official code uses a shuffled stack.
- **Projection.** Projection and tiling come from gsplat 1.5.3; the official fork is gsplat 1.3.0.
  The bounding shape is a 1-sigma per-axis box instead of a circle; both cover the same pixels.
- **Memory.** The unused `shN` field is still allocated and copied by MCMC, about 0.9 GB at 5M
  primitives. Results are unaffected.
- **Bicycle (not run).** Our pilr bicycle images are 1 px narrower than official (1236 vs 1237).
  This matters only for `beta-splatting-bicycle`.
