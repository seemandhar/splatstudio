# Scaffold-GS (plan #17, arXiv 2312.00109): reproduction report

**Outcome: matched.** Both benchmarked scenes are within ±0.5 dB of the paper's per-scene PSNR.

- **Method:** `scaffold-gs`, a native SplatStudio plugin in `plugin/ss_scaffold_gs`.
- **Code origin:** the Scaffold-GS backbone was taken from the HGS plugin (`reproductions/hard-gs`, control method
  `scaffold-gs-hgsbase`), which was written against the official code github.com/city-super/Scaffold-GS @59c833b.
  All HGS code was removed. This plugin was not reviewed separately. Its 2k smoke test matches hgsbase: 23.85 vs
  23.88 PSNR, 252k vs 253k anchors.
- **Benchmark:** experiment `runs/scaffold-gs-1600`. Settings: 30k steps, `--recipe 3dgs`, mipnerf360-1600, voxel 0.001.
- **Paper source:** per-scene numbers from supp. tables `tab:psnr_mip`, `tab:ssim_mip` and `tab:LPIPS_mip`.

## Results (test split, llffhold8, PIL width 1600)
| scene | run | PSNR | SSIM | LPIPS | anchors | train min |
|---|---|---|---|---|---|---|
| garden | **scaffold-gs (ours)** | **27.20** | 0.849 | 0.155 | 751k | 37 |
| garden | paper Scaffold-GS | 27.17 | 0.842 | 0.146 | – | – |
| garden | gsplat baseline (ours, w1600) / paper 3DGS row | 26.93 / 27.41 | | | | |
| counter | **scaffold-gs (ours)** | **29.60** | 0.920 | 0.243 | 285k | 49 |
| counter | paper Scaffold-GS | 29.34 | 0.914 | 0.191 | – | – |
| counter | gsplat baseline (ours, w1600) / paper 3DGS row | 29.04 / 28.70 | | | | |

How the results compare with the paper:
- **PSNR vs paper:** garden +0.03 dB, counter +0.26 dB.
- **Gain over our gsplat baseline at the same resolution:** +0.27 dB on garden, +0.56 dB on counter.
- **The paper's own gain cannot be compared directly.** Its 3DGS row is copied from the 3DGS paper, which used
  images_4/_2, a different resolution from Scaffold's 1600-wide runs. Taken at face value it is −0.24 dB on garden
  and +0.64 dB on counter.
- **SSIM:** matches the paper or is slightly higher.
- **LPIPS:** higher than the paper's, most clearly on counter (0.243 vs 0.191). This is the same LPIPS offset seen
  in the hard-gs control and appears to come from our shared LPIPS implementation/protocol, not from Scaffold-GS.

## Deviations and fidelity gaps
1. **Added opacity cap (not in the official code):** neural opacity is capped at 0.999 with a straight-through
   estimator.
   - **Why:** gsplat's backward pass drops gradients where opacity·G > 0.999 (see TODO-runtime). The Scaffold
     rasterizer fork instead clamps alpha at 0.99 in the forward pass and always back-propagates.
   - **Off switch:** renderer config `opacity_cap: null`.
2. **Code vs paper, code used:**
   - Volume loss weight: 0.01 in train.py, while the paper text says λ_vol = 0.001.
   - Pruning: an anchor is pruned when its accumulated opacity is below 0.005 × its visit count (official args),
     while the paper says 0.5.
   - Growth threshold: τ_g = 2e-4 in NDC (densify_grad_threshold), while the paper says 64ε.
   - The feature bank and distance inputs described in the paper are disabled in the released scripts, as is the
     appearance embedding (appearance_dim 0).
3. **Resolution:** the official `train.sh` loads full-res `images/` with `--resolution -1`, which PIL-resizes them to
   width 1600. We matched this with `mipnerf360-1600`. The paper only says preprocessing "adhered to 3D-GS", so this
   protocol comes from the code. The hgsbase control ran on pilr, so its runs were not reused.
4. **Rasterizer:** gsplat (alpha clamp 0.999, eps2d 0.3) instead of the Scaffold fork.
5. **Training-loop differences:**
   - white background instead of black;
   - views sampled at random with replacement instead of a shuffled stack;
   - the SSIM loss uses a valid convolution;
   - the runtime steps Adam before growing anchors;
   - LR schedules are fixed at 30k steps.
6. **Weight transport:** the MLP weights reach ss-eval inside the per-anchor field `sgs_blob`.
7. **Coverage:** only 2 of the paper's 7 Mip-NeRF 360 scenes were run (batch rule), and T&T/DB were not run (the
   `scaffold-gs-tandt` and `-db` presets exist). No averages are claimed.

## Evidence
- `evidence/verify.json`: 8/8 probes pass.
- `evidence/test.json`: unit tests, ruff and smoke results.
- `evidence/smoke-1`: the 2k-step smoke run.
