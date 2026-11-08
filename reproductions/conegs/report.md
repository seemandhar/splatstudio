# ConeGS reproduction report

**Outcome: matched (iter 1).** Both reported scenes are within ±0.5 dB of the paper. Garden only
matches after a post-hoc change (iteration 1, black background).

The paper is ConeGS (Baranowski et al., 3DV 2026, arXiv 2511.06810). This is a native
SplatStudio port of the official code, github.com/baranowskibrt/conegs @ f86df2f. The target is
the paper's fixed 1M-primitive budget (appendix 1M table), Mip-NeRF 360 images_4 (outdoor) and
images_2 (indoor), llffhold-8, 30k steps, seed 0.

| iter | method | scene | ours PSNR | paper ConeGS-1M | gap | Gaussians | run |
|---|---|---|---|---|---|---|---|
| 0 | conegs (white bg) | garden | 26.35 | 27.42 | −1.07 | 1.0M | runs/conegs-mipnerf360-conegs |
| 0 | conegs (white bg) | room | 32.15 | 32.30 | −0.15 | 1.0M | runs/conegs-mipnerf360-conegs |
| 1 | conegs-blackbg (post-hoc) | garden | **27.46** | 27.42 | **+0.04** | 1.0M | runs/conegs-blackbg-mipnerf360-conegs |

Only the missed scene was re-run (batch rule). Room's reported number is the iteration-0 white
background run; `conegs-blackbg` has not been run on room.

Training time:
- garden iteration 1: 22 min of 3DGS training;
- garden iNGP preprocessing (part of the method's time): 395 s iNGP plus 97 s for init points
  and kNN, on a shared A6000 (paper: about 3.1 min of iNGP on an A100); see
  `~/datasets/conegs-init/mipnerf360/garden/conegs_init.json`;
- room preprocessing: recorded in that scene's `conegs_init.json`.

## Iteration 1 (post-hoc)
I switched the render, the error map and the evaluation composite from the runtime's white
background to black. This matches the official config (`white_background: false`). The change
was chosen after seeing the garden test result, so it is labelled post-hoc. The supporting
diagnosis used train views only (`gap-analysis-1.md`).

## Gap analysis summary
- **Runtime offset.** Our gsplat baseline on garden scores 26.89, against 27.41 for Inria 3DGS
  (−0.52, calibration report). Room has a +0.89 offset, because the Inria room number 30.63 is
  low. So room's iteration-0 match is partly offset luck: against our own unbudgeted baseline
  it is +0.63, while the paper reports +1.67 against Inria 3DGS.
- **Transparency and recycling (outdoor-specific).** With a white background, 15.5% of garden
  train pixels stay partly transparent (alpha < 0.99), against 2.8% in room. The white
  background fills bright sky for free, so fewer Gaussians fade and get pruned and refilled by
  cone densification.
  - Garden recycled about 0.6× the per-100-step prune/refill counts in the paper's own garden-1M
    curve (parsed from `gaussians_number_analysis.pgf`).
  - In a 7k diagnostic, black background raised recycling to 9.4k / 14.4k / 19.0k at 2k / 4k / 6k
    steps (white: 5.6k / 11.3k / 14.5k; paper: about 10k / 18k / 22k).
  - Train-view PSNR at 7k rose by +0.40 dB.
- **Ruled out:**
  - iNGP quality: train-view PSNR 25.8; depth within 1.3% median of the baseline; 0.8% floaters.
  - Init distribution: 46% of init points in the contracted background, against 57% of SfM points.
  - Opacity penalty, SH schedule, learning rates and means-LR extent: all code-exact.
  - Opacity saturation: 0.02% of Gaussians above 0.999.

## Remaining deviations from the official code
1. The iNGP uses fp32 PyTorch plus our own hash-grid CUDA kernel and a PyTorch port of the
   nerfacc proposal estimator. The official code uses tiny-cuda-nn (fp16 FullyFusedMLP) and
   nerfacc, so initialisation and precision differ.
2. Rasterizer is gsplat classic; the official code uses Inria diff-gaussian-rasterization
   (dr_aa branch, no antialiasing).
3. Training views are sampled at random with replacement, not from the shuffled per-epoch stack.
4. SSIM uses valid convolution.
5. Init 3-NN scales are exact; the official code uses the approximate simple-knn `distCUDA2`.
6. The paper text and the code disagree in three places; I used the code each time:
   - growth: 0.02 in the code, Eq. 12 says 0.2;
   - opacity penalty: signed mean of the logits, the paper writes an L1 norm;
   - footprint size: `densification_gaussian_size: 1.0`, the paper writes λ_scale = 2 × radius.
7. The paper's figure accumulates about 267 rays per step on garden; the code formula gives 201
   (unexplained).
8. Room uses the iteration-0 white-background method. Its iteration-1 result is unmeasured.
9. Single seed. Tanks & Temples and Deep Blending were not benchmarked: the paper uses the
   original releases, not the 3DGS `tandt_db` copy.

## Evidence
- `paper.md`, `paper_results.json` (all four budgets), `plan.md`
- `gap-analysis-1.md`
- `evidence/verify*.json`, `evidence/test.json`
- 7k diagnostic runs: `evidence/diag-{white,blackbg}-7k`
- Plugin: `plugin/` (`conegs`, `conegs-blackbg`, `conegs-nobudget`)
- Preprocessing: `ss-conegs-init`, output under `~/datasets/conegs-init/mipnerf360/`
