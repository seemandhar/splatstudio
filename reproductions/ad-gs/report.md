# AD-GS — reproduction report

**Outcome: not-matched.** LLFF 3-view, 10k steps, 3 seeds (iteration 1): PSNR is 0.37–0.77 dB below the paper. horns 19.03 vs 19.79 (−0.76), fortress 23.10 vs 23.87 (−0.77), room 22.02 vs 22.39 (−0.37). Only room is within ±0.5 dB. SSIM is close; the LPIPS gap is mostly a metric convention (see gap analysis).

- Paper: arXiv 2509.11003 (SIGGRAPH Asia 2025). Official code: github.com/gurutvapatle/AD-GS @a8d12fc (read, not run).
- Plugin: `plugin/` (`ad-gs`, module `ss_ad_gs`), no custom CUDA.
- Contributions → components:
  - Alternating densification → `adgs-density`.
  - Phase-gated losses (regularisers only in low-densification steps) → shared schedule.
  - Pseudo-view consistency between two co-trained fields → `adgs-cotrain`; the density component owns the second field.
  - Edge-aware depth smoothness + depth range → `adgs-depth`.
  - Random training background, black test background → `adgs-renderer`.
- Verification: iteration 0 `evidence/verify.json` 7/7 probes; iteration 1 `evidence/verify-i1.json` 8/8 (`declared_checks_passed`); 7 unit tests pass.

## Protocol (new few-view LLFF setup)
- Data: `nerf_llff_data` from the Hugging Face mirror `YouLiXiya/nerf` (Google Drive refused the download; byte-identity not verified).
- `tools/prepare_llff_fewview.py` builds `~/datasets/llff/fewview3/<scene>/`:
  - Split (FSGS/CoR-GS, as the official reader): test = every 8th sorted image; train = 3 views by `round(np.linspace)` over the rest; val = unused images, never used for selection.
  - Images: 1008×756, PIL bicubic from full resolution.
  - Cameras: pinhole, k1 dropped, centred principal point (as the official reader/projection).
  - Init: COLMAP 3.9.1 with CUDA (conda env `colmap`), exact `tools/colmap_llff.py` commands on the 3 training views only (triangulation with known poses, then MVS fusion). Points: horns 17,962, fortress 11,113, room 19,139.
  - Check: our first horns training view's bound equals the constant hard-coded in official `pose_utils.py`.
- Training: 10k steps, `--recipe 3dgs`, all init points kept. Eval: test split, black background, PSNR/SSIM (runtime)/LPIPS-VGG.
- Dataset key `llff-fewview3` (registered in tools/datasets.json).

## Results (test, PSNR / SSIM / LPIPS)
| scene | iter 0 (seed 0) | iter 1 seeds 0/1/2 | iter 1 mean | paper AD-GS | Δ PSNR | our gsplat 10k | paper 3DGS |
|---|---|---|---|---|---|---|---|
| horns | 18.89 / .693 / .330 | 19.20 / 18.93 / 18.96 | 19.03 (sd .15) / .694 / .330 | 19.79 / .714 / .274 | −0.76 | 15.83 / .511 / .441 | 17.59 |
| fortress | 23.29 / .748 / .269 | 23.21 / 22.79 / 23.31 | 23.10 (sd .28) / .740 / .273 | 23.87 / .772 / .207 | −0.77 | 18.28 / .623 / .351 | 21.25 |
| room | 21.80 / .874 / .233 | 22.19 / 21.75 / 22.12 | 22.02 / .875 / .233 | 22.39 / .879 / .155 | −0.37 | 12.86 / .633 / .535 | 21.61 |

- Gaussians at the end: ours 145k / 99k / 54k. The paper's model sizes at 248 B/Gaussian give ~123–129k / 83–87k / 48–50k, so we keep 8–17% more. The high/low cycles show in the counts (horns peaks at 218k at step 3600).
- Gain over our gsplat is larger than the paper's gain over its 3DGS (horns +3.2 vs +2.2 dB), but this is confounded: gsplat is evaluated on white, AD-GS on black, and gsplat room (12.86) is hurt by that; the paper's 3DGS step count is unknown. Absolute PSNR against the paper is the primary result.

## Iteration log
- Iteration 0: as in `plan.md`.
- Iteration 1 (code-grounded):
  - pseudo-view poses use the official `poses_bounds.npy` near/far;
  - official SH schedule (degree up every 500 steps);
  - net effect: horns +0.14, fortress −0.19, room +0.22 dB, within about one seed sd.

## Gap analysis (logs and code only, no new runs)
1. LPIPS: official `metrics.py` passes [0,1] images to `lpipsPyTorch` with no rescale to [−1,1]; our runtime rescales, as the `lpips` reference does. This lowers the paper's LPIPS and explains most of our 0.05–0.08 LPIPS gap. It does not affect PSNR.
2. View sampling: the runtime samples with replacement; the official code pops a shuffled stack (each of the 3 views once every 3 steps). Per-view counts in a 100-step phase window vary by about ±5 around 33. Minor; fixing it needs a trainer change.
3. Step-then-densify ordering, valid-convolution SSIM (borders excluded from the SSIM term), gsplat alpha clamp 0.999 vs 3DGS 0.99: each known to be small; none plausibly worth 0.4 dB alone.
4. Initialisation: COLMAP version and stochastic MVS differ from the authors' runs, and the paper gives no point counts. Few-view results are very sensitive to this, but there is no reference to fix against.
5. Checked and matching the official code:
   - alternating schedule and per-phase thresholds;
   - FSGS split rule;
   - opacity reset steps;
   - learning rates and means-LR decay (identical at 10k);
   - scene extent;
   - pseudo-view weight and detach;
   - depth weights and code form;
   - shared random training background, black at eval;
   - second field initialised identically;
   - accumulated depth;
   - evaluation of the first field only.

   No per-scene settings exist in the run scripts.
6. Paper vs code (followed the code):
   - the pseudo-view depth term (ω2 = 0.05) is commented out in the official code;
   - edge-aware weights use means, not the paper's sums;
   - co-pruning is off, as in the official run scripts.
7. Seeds: the paper reports single runs (seed 42). Our seed sd (0.15–0.28) is below the horns and fortress gaps, so those misses are real.

No remaining code-grounded deviation is likely to recover ~0.4 dB, so no further iteration is proposed.

## Remaining fidelity gaps
- view sampling with replacement;
- optimizer step before densification;
- valid-convolution SSIM;
- gsplat alpha clamp 0.999;
- LPIPS input-range convention (evaluation);
- COLMAP version / MVS randomness in the init;
- different RNG streams.
