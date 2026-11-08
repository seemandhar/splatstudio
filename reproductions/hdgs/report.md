# HDGS (arXiv 2412.01823) — outcome: **matched**

Native plugin `plugin/` (methods `hdgs` outdoor tex_res 200, `hdgs-indoor` tex_res 500) built on the
official code github.com/TimSong412/HDGS (main b13a359 = stage 1, texture 01c6fd4 = stage 2), with
both official CUDA rasterizers vendored (only change: SORT_WINDOW as a build flag, k=16).
Protocol: Mip-NeRF 360 images_4 (garden) / images_2 (counter), llffhold 8, black background,
60000 steps = 40000 stage 1 (2DGS + per-ray k-sort + frustum sampling) → Fisher prune 10 % →
20000 texture steps (geometry frozen). `--recipe 3dgs`, single seed 0.

## Results (test split)
| scene | method | PSNR | paper | Δ | SSIM (paper) | LPIPS (paper) | surfels | texels | peak mem | train time |
|---|---|---|---|---|---|---|---|---|---|---|
| garden | hdgs | **26.89** | 26.98 | −0.09 | 0.857 (0.857) | 0.129 (0.115) | 3.71 M | 60.6 M | 9.36 GB | 320 min |
| counter | hdgs-indoor | **29.00** | 28.94 | +0.06 | 0.909 (0.909) | 0.236 (0.188) | 0.69 M | 144.6 M | 9.64 GB | 298 min |

Both scenes are within ±0.5 dB of the paper's per-scene PSNR (supp. Table 360_full).
For reference: our gsplat baseline (`baseline3dgs-mipnerf360-30000`) gets garden 26.89 and counter 28.85.
The paper's 3DGS row is 26.63 / 29.07 and its 2DGS row 26.91 / 28.45. So:
- HDGS − our baseline: garden ±0.00, counter +0.15.
- Paper HDGS − paper 3DGS: garden +0.35, counter −0.13.

Relative gains therefore differ in sign per scene, but the absolute numbers match. Our LPIPS is
higher than the paper's (+0.014 garden, +0.048 counter), consistent with the runs-wide
LPIPS-package offset noted in `_calibration`.
Texels were counted from `checkpoint.pt["density_state"]["texture"]`. The peak memory figure is
torch max allocated.

## Evidence
- `ss-verify hdgs`: all 6 probes pass. Both kernels match an independent dense torch emulation to ~1e-7 in the forward pass and in every gradient, including when the k=4 buffer overflows. The Fisher Hutchinson estimate is within 5 % of the exact Jacobian sum.
- Smoke (garden, 2k steps): 23.11 vs gsplat 23.02.
- Stage-switch smoke: a fresh-process eval reloads the texture exactly.
- Review: `reviews/review-1.md` approved the plugin with 0 critical, 0 major and 6 minor issues.

## Deviations / fidelity gaps
1. Fisher pruning is not in the released code. We use a Hutchinson estimate of diag F with 1 Rademacher probe per training view. The score is Σ_j log F_jj, which equals the log-det of the diagonal as in PUP-3DGS; the paper's literal "L1 norm of log diag" inverts the ranking for F<1. Reviewer minor-1/2: about 20 % noise per entry at the 10 % boundary. The paper's ablation gives −0.06 dB without pruning.
2. Stage-2 SH-rest delay: we use lr 0 for 1001 iterations, while the official code accumulates 1001 un-zeroed gradients and then takes one step (minor-3).
3. Zero-size textures (aspect > 500:1 after the 500 cap) are clamped to 1 texel. The official code can read a neighbour's texel (minor-4).
4. One-step offsets: the SH degree rises at step 1001 instead of 1000, and the official code skips the optimizer step at iteration 40000. Separately, the official code skips the optimizer step at densification iterations (minor-5).
5. Runtime differences: views are sampled with replacement (official uses a shuffled stack) and SSIM uses a valid convolution.
6. Stage 2 runs at the stage-1 resolution, as in the official script; the paper text says "on-demand higher resolution".
7. Code is preferred over paper text: indoor tex_res 500 (paper 400), and textures are initialised to the surfel DC (paper says background colour).
8. The texture is persisted through `density_state`, and `ss-eval` locates it via `--run` in argv, because the runtime has no global learned fields.
9. Only 2 scenes were run with one seed each; no NeRF-Synthetic or multi-resolution (anti-aliasing) evaluation was done.
