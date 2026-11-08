# DWTGS review 1 (splat-reviewer, adversarial)

**Approved: no.** Critical 1, major 2, minor 4.

The paper is arXiv 2507.15690 v3 and has no official code. The review read `source/latex/main.tex`, the figures, `paper.md`, `plan.md` and `plugin/ss_dwtgs/*`, plus the reused `ss_dropgaussian` renderer and density.

## Q1. Haar normalisation: the averaging (LL = block mean) reading is the more literal one

Evidence in the paper:
1. **§IV-B, after Eq. 5**: λ_LL "is easier to control … as the LL subband is in the **same space as the original image, only a coarser version**". An orthonormal LL_n is 2^n × the image, with range [0, 2^n]. That is not the same value space, and one shared λ across levels is only "easy to control" if every level has the same scale.
2. **Fig. 3 (`fig_vis_subbands_multi.png`)**: X¹_LL and X²_LL are shown with the same brightness as X and are not saturated. Orthonormal X²_LL would be 4× brighter and close to white wherever the image is bright, unless it was rescaled for display.
3. **Fig. 2 (`fig_arch_dwtgs.png`) notes**: "All subbands upsampled …" and "**Contrast of HH subband is enhanced**". Only HH is said to be rescaled. X^{1}_LL, X^{2}_LL and the GT versions are shown at normal image brightness next to X.
4. No library is named anywhere: no pywt, pytorch_wavelets or 'db1'. The only statement is "we used the Haar wavelet [wavelet-book]" (§V-A). The library-default argument (pywt/pytorch_wavelets are orthonormal) is therefore unsupported by the text. It is an assumption.

Effective weights. Take a per-pixel residual and compare each term's gradient with the photometric L1 (0.8·sign/HW):
- Orthonormal: the level-n gradient is λ·2^n·sign/HW, so the total is 0.5·2 + 0.5·4 = **3.0/HW**. That is **3.75×** the L1 part of the photometric loss, and it is coherent over 2×2 and 4×4 blocks, so it lands on the screen-space mean gradients.
- Averaging: each level contributes λ·sign/HW, so the total is **1.0/HW**. That is 1.25× the photometric L1, which is still substantial.
- HF (Eq. 6): orthonormal HH = (a−b−c+d)/2 versus /4 averaging, a 2× difference.

Measured in the logs: at step 1 the orthonormal LF term is 0.73 against photometric 0.385 (`evidence/smoke-dwtgs.log`).

## 2k horns check (seed 0, same protocol as the evidence smokes)

| run | PSNR | SSIM | LPIPS | #G @2k |
|---|---|---|---|---|
| dropgaussian (evidence) | 18.88 | 0.653 | 0.336 | 59k |
| dwtgs, orthonormal (evidence) | 18.04 | 0.595 | 0.368 | 152k |
| dwtgs-lf, orthonormal (evidence) | 18.11 | 0.604 | 0.358 | 151k |
| **dwtgs, average** (this review) | **18.44** | 0.630 | 0.341 | 86k |
| **dwtgs-lf, average** (this review) | 18.20 | 0.625 | 0.348 | 86k |

The averaging variant recovers +0.40 dB and removes about 45% of the extra Gaussians. It is still −0.44 dB below DropGaussian at 2k. That number comes from one scene, one seed, and 2k of a 10k schedule in which the DropGaussian drop rate is still small, so it does not settle the question. The 10k benchmark has to decide.

Scripts: `/tmp/claude-1013/-home-sejain/6ad4135e-056e-4bd4-9844-6b8764bea041/scratchpad/dw/run.py`. It overrides `haar_norm` in the spec config; the plugin was not edited.

## Issues

### Critical
**C1. The default Haar normalisation contradicts the paper's own description of LL.**
- Location: `plugin/ss_dwtgs/wavelet.py:16,21`; `losses.py:63,73` (`haar_norm: "orthonormal"`); `probes.py:81,120` (the probes pin "orthonormal" as "the paper value").
- Evidence: §IV-B "same space as the original image, only a coarser version"; Fig. 3 and Fig. 2 show unscaled LL with only HH contrast-enhanced; no library is named. Orthonormal gives a 3× effective LF weight (3.75× the photometric L1), against 1× for averaging. The red-flag 2k smoke shows 2.5× the Gaussians and −0.84 dB. Averaging gives +0.40 dB and 86k Gaussians.
- Fix: make `average` the default for both LF and HF, and update the probes to pin it. Keep the orthonormal variant as a labelled ablation or fidelity-gap row, not the headline.

### Major
**M1. The novel-view generator contradicts the paper.**
- Location: `plan.md` gap 2 and `paper.md` §7.3 say "paper silent". `pseudo.py` / `losses.py:122-159` use the FSGS `generate_random_poses_llff`, which samples a spiral/ellipse.
- Evidence: Fig. 2, note 3: "Novel view sampling is performed using **spherical interpolation between training views**, only at every K iterations to save computation."
- Fix: sample a pair of training cameras and a t ∈ U(0,1). Slerp the rotations, lerp the camera centres, and use the intrinsics of the training view. CoR-GS/DNGaussian-style interpolation is the closest published recipe. Correct `paper.md` and `plan.md`, and update the pseudo-poses probe so it checks that each pose lies on a geodesic between two training views.

**M2. The probes cement the questionable choice instead of detecting a mechanism error.**
- Location: `probes.py:40-71` (`haar_dwt_probe`) asserts energy preservation, "orthonormal"; `probes.py:81,120` reject any `haar_norm` other than orthonormal.
- Problem: a fidelity probe should check that the config matches the paper. Here it enforces an assumption the paper text contradicts.
- Fix: check the subbands against an independent loop for the configured normalisation. For averaging, check that LL equals 2×2 average pooling and the perfect-reconstruction identity with ×4. Pin the paper-literal default.

### Minor
- **m1. The LF and HF terms are added to the loss, not substituted into it.** This is consistent with the paper: Fig. 2 shows L_3DGS alongside L_LF and L_HF, and the commented Eq. in main.tex:233-236 has the same additive form. The photometric loss stays, which is correct.
- **m2. HF schedule** (`losses.py:101`, `hf_active`): it is active at step%10==0 and step ≤ 5000, so the first use is at step 10. This matches §V-A, "every 10 iterations, and stop after 5K". Weight 1 is unwritten but the natural reading of Eq. 6. HF uses the same normalisation as LF (the `haar_norm` key), so switching to averaging halves the HF term automatically. That is consistent: the paper uses one DWT.
- **m3. Dropout on the novel render**: gamma 0 through a private `Renderer` (`losses.py:162-164`). The paper is silent. The choice is reasonable and stays a recorded gap. Note that the private renderer increments its own `step` counter under autograd. That is harmless because rate=0 and it is a separate instance from the training renderer that the density cross-checks.
- **m4. Densification interaction.** The LF gradient feeds the training render's means2d, so it is in the densify statistic. That is inherent: the paper adds LF to the same loss before `backward()` in DropGaussian/3DGS code. The HF novel render is correctly excluded from the statistics. The paper reports no Gaussian counts, so neither the 2.5× (orthonormal) nor the 1.45× (averaging) growth can be checked against it. Report #G in `report.md`. requires_grad pitfall: the density is the unchanged `DropGaussianDensity` (`density.py:202-217` only adds attributes), which has no Parameter rebuild, so the pitfall does not apply. The averaging and orthonormal runs trained normally, and #G grew.

## Equation coverage
| eq | status | code |
|---|---|---|
| (1) L_3DGS | exact (reused) | ss_dropgaussian.losses:create_photometric |
| (2) DWT | deviates (normalisation, C1) | wavelet.py:haar_dwt |
| (5) L_LF | exact form; effective weight deviates (C1) | losses.py:lf_loss |
| (6) L_HF | exact form; novel-view source deviates (M1) | losses.py:hf_loss, create_hf |
| §V-A schedule (λ 0.5, N 2, every 10, ≤5k) | exact | losses.py:63-74,101 |

## Launch (after the C1 and M1 fixes)
```
python3 tools/bench.py launch --exp dwtgs-i1-llff-r8 --methods dwtgs dwtgs-lf dropgaussian \
  --dataset llff-fewview3-r8 --steps 10000 --recipe 3dgs --per-gpu 2
```
Include `dropgaussian` for a paired comparison: the paper's gain is +0.39 dB over its own DropGaussian re-run. If M1 cannot be fixed first, run it anyway with averaging Haar and list FSGS poses as a gap.
