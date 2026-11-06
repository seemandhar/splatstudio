# BAGS reproduction report (plan #89) — built, awaiting benchmark

**Paper**: BAGS: Blur Agnostic Gaussian Splatting through Multi-Scale Kernel Modeling (ECCV 2024,
arXiv 2403.04926). **Official code**: github.com/snldmt/BAGS @ed04144 (read, ported; not run).
**Plugin**: `plugin/` (`ss_bags`), methods `bags-<scene>` (20 official scene scripts), `bags`,
`bags-smoke`. No custom CUDA.

## Contributions → components
| contribution | implementation | probes |
|---|---|---|
| Blur Proposal Network (per-view, per-pixel dense K×K kernels + quality mask, MMF input) | `bank.py` (BPN ported verbatim), `blur.spatial_conv` (official Unfold form), `losses.py` (C_out blend, mask / TV terms) | spatial-conv-reference (vs float64 shift-and-add loop, values + grads, impulse orientation, gradcheck), bpn-architecture, objective-terms |
| Multi-scale BPN + coarse-to-fine | `renderer.py` (1/4 → 1/2 → full at 3000/6000, scaled intrinsics), `bank.net_for` (K 5/9/17 heads), `schedule.py` (two-phase clock) | multiscale-schedule (CUDA), phase-schedule (vs independent transcription of the official loop) |
| Mip-Splatting base + official densification | `density.py`, `filters.py` | density-recipe, official-config, state-transitions |

## Protocol
- Data: `deblurnerf-motion` (ball) and `deblurnerf-real` (caps): original Deblur-NeRF Drive
  folder with its `sparse/0` COLMAP points, which is the folder BAGS's README links. The
  Deblurring-3DGS re-release (`deblur3dgs-*`, re-run COLMAP, denser points) is **not** what BAGS
  used. Resolution 600×400 (official `-r 4`), split sorted idx % llffhold == 0 → test (sharp),
  holds equal the official scripts (ball 7, caps 5). No defocus / low-res variant sets are used
  for Table 1. Mixed-resolution Mip-NeRF 360 and the drone set are out of scope.
- Scenes: paper reports per-scene numbers → ball (motion, 27.68 / 0.799 / 0.150) and caps
  (defocus, 24.16 / 0.725 / 0.159). Iterations per official script: 49000 / 12000.
- Metrics: `python -m ss_bags.evaluate` = official `metrics.py -t deblur-nerf` (8-bit renders,
  PSNR, skimage 0.19 SSIM 7×7 on images mapped to [−1,1] with data_range 2, LPIPS-Alex on
  [−1,1]); ss-eval numbers reported alongside. (review-1 major: the first evaluator computed SSIM on
  [0,1] with data_range 1, which overstates SSIM by 0.04–0.10 because the luminance term is not
  shift-invariant; fixed, unit test `test_official_ssim_convention`.)
- Controls: `bags-ball-nobpn` / `bags-caps-nobpn` = the official scene config with the BPN never
  activated (bpn_start = total_steps) = Mip-Splatting under the BAGS schedule — the paper's
  comparison row (ball 23.22, caps 21.00) and the BPN ablation; plus fresh gsplat controls.

## Verification
- ss-inspect: no issues; pytest 12 passed; ruff clean.
- ss-verify `bags-ball`, `bags-caps`, `bags-ball-nobpn`, `bags-caps-nobpn`, `bags-smoke`,
  `bags-cake`: declared_checks_passed (8/8). pytest 16 passed.
- Smoke, ball, 2000 steps (shared GPU), sharp test renders:

| run | PSNR | SSIM | LPIPS-VGG | official PSNR / SSIM / LPIPS-Alex | Gaussians | s |
|---|---|---|---|---|---|---|
| bags-smoke (all phases, compressed) | 24.95 | 0.714 | 0.445 | 24.94 / 0.677 / 0.378 | 16251 | 236 |
| bags-ball (first 2000 steps, 1/4 res only) | 24.25 | 0.665 | 0.486 | – | 5106 | 153 |
| gsplat baseline | 23.58 | 0.658 | 0.487 | 23.58 / 0.622 / 0.393 | 87365 | 87 |

All BAGS loss terms appear in training.jsonl with plausible magnitudes; peak 3.6 GB at full
resolution with 17×17 kernels.

## Fidelity gaps
1. Coarse-scale targets are antialiased-bicubic resizes of the stored 600×400 image (official:
   PIL resize of the 2400×1600 JPG straight to 150×100 / 300×200). Small.
2. Runtime Adam step precedes densification (official skips the Gaussian Adam update on
   densify / reset steps, ~1 % of steps). Negligible.
3. SH degree raised one step later (runtime). Negligible.
4. skimage not installed: its SSIM is re-implemented (valid-region 7×7 uniform, sample covariance,
   on [−1,1] with data_range 2 as skimage 0.19 does in the official metrics.py), checked against an
   independent window loop in the unit tests.
5. Code overrides paper: per-scene iterations / ms_steps / flags from scripts/*.sh, not the
   paper's "3000 steps per coarse scale, 40K overall". README of the release warns it may not
   exactly match the manuscript numbers.
6. Screen-size prune is not ported because the official one never fires (max_radii2D is zeroed
   in densification_postfix before the prune); world-size prune only with min_opacity 0.005
   (generic `bags` only), as in the official code.
7. No shared gsplat baseline on deblurnerf-*; paper's comparison row is Mip-Splatting.
8. Random init of the BPN / seeds differ from official; no checkpoint selection (final state).
9. The runtime applies a Gaussian Adam update at the final step; the official code saves the
   model before it and skips `optimizer.step()` at the last iteration. Negligible.
10. gsplat depth-sorts Gaussians by camera z; the official Mip-Splatting rasterizer sorts by
   Euclidean distance (`depths = dist`). Only the blending order of overlapping off-axis
   Gaussians can differ. Negligible. (The rendered depth D is the Euclidean distance as officially.)
11. The caps gsplat run from the DOF-GS build (40k steps, white background, 15.61 dB) is an
   outlier and is not used. A fresh gsplat control is run with the same settings as ball.

## Result: caps (bags-caps, 12k) — within tolerance
Official Deblur-NeRF metrics: **24.40 / SSIM 0.739 / LPIPS-Alex 0.148**, vs paper BAGS 24.16 / 0.725 / 0.159 (Δ +0.24 dB).
Runtime metrics: 24.40 / 0.852 / 0.207 (VGG).
Still pending: ball, the -nobpn controls (paper Mip-Splatting caps 21.00, ball 23.22) and the gsplat controls.

## Result: ball (bags-ball, 49k) — within tolerance
Official metrics: **28.00 / SSIM 0.812 / LPIPS-Alex 0.163**, vs paper 27.68 / 0.799 / 0.150 (Δ +0.32 dB).

## Verdict: MATCHED (2/2 reported scenes within ±0.5 dB)
| scene | ours | paper | Δ |
|---|---|---|---|
| ball (motion) | 28.00 | 27.68 | +0.32 |
| caps (defocus) | 24.40 | 24.16 | +0.24 |

Still pending: the -nobpn controls (paper Mip-Splatting ball 23.22, caps 21.00) and the gsplat controls, which give the gain comparison.
