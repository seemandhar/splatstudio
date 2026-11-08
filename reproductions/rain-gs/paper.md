# RAIN-GS — Relaxing Accurate Initialization Constraint for 3D Gaussian Splatting

arXiv 2403.09413 (v3, NeurIPS-2024 template, "updated" version with ABE-Split).
Official code: https://github.com/cvlab-kaist/RAIN-GS (MIT, KU-CVLAB) — cloned to `source/code/`
(read-only). Paper LaTeX in `source/latex/`. The headline runs are `train.py --ours_new
--train_from random` (README); code defaults override the paper where they disagree.

## 1. Method summary
- 3DGS trained from a *random* point cloud of only N=10 points (sparse-large-variance init).
- The 2D low-pass (dilation) constant `s` of the EWA projection is recomputed every 1000 steps
  as `min(max(HW/(9πN), 0.3), 300)` — huge splats while N is small, 0.3 once N is large.
- ABE-Split: during the first 10k steps every split also emits a third Gaussian placed at
  0.3·extent·μ (pushed away from the origin) with the parent's unchanged scale.
- A 10k-step warm-up: means LR held at its initial value, densification window extended by
  10k (until 25k), SH degree raised from step 5000, split scale divisor 1.6 → 1.4.

Contributions
- C1 SLV random initialization (§4.1, supp. A.1.1, Alg. 1).
- C2 Progressive Gaussian low-pass filter control (§4.2, supp. A.1.2, Alg. 2 l.8).
- C3 ABE-Split (§4.3, supp. A.1.3, Alg. 2 l.21–22).
- C4 Warm-up schedule (§5.2 impl. details, supp. A.1): fixed means LR for 10k, densify until
  25k, SH from 5000, divide ratio 1.4.

## 2. Equations
- Eq. (2) dilated splat: $G_i'(x)=e^{-\frac12(x-\mu_i')^T(\Sigma_i'+sI)^{-1}(x-\mu_i')}$, 3DGS s=0.3.
- Progressive control (§4.2 / Alg. 2 l.8): $s = \min(\max(HW/(9\pi N), 0.3), 300)$,
  recomputed every 1000 steps. Code `train.py:107-114`: at `iteration == 1 or (iteration %
  1000 == 0 and iteration < densify_until_iter)`, H,W of the sampled training view, N current
  #Gaussians; otherwise the last value is kept (→ 0.3 at the end). Test rendering uses 0.3.
- SLV (Alg. 1; code `scene/dataset_readers.py:187-208`, `--ours_new` branch):
  `cam_pos` = COLMAP `tvec` of **all** images (world→camera translation, not centres);
  `lo = min(cam_pos)`, `hi = max(cam_pos)` (scalars over all coordinates);
  `xyz = U[0,1)^3 · (hi-lo)·3 − (1.5(hi-lo) − (lo+hi)/2)` i.e. a cube of side 3(hi−lo) centred at
  (lo+hi)/2 on every axis. N=10. Colours: `shs ~ U[0,1)^3` stored via SH2RGB → sh0 = shs (up to
  uint8 quantisation). Scales: log sqrt(mean of 3-NN squared distances) (distCUDA2), isotropic;
  quats identity; opacity 0.1 (code; Alg. 1 says InverseSigmoid(1) — code wins).
- ABE-Split (Alg. 2 l.21–22 says $\mu'=\lambda_s s(\mu-B_0)+B_0$; code
  `gaussian_model.py:densify_and_split` uses $B_0=0$): `new_xyz = xyz * 0.3 * scene_extent`, scale,
  rotation, colour and opacity copied unchanged, applied to the split set (grad ≥ τ and
  max scale > 0.01·extent) when `iteration <= warmup_iter (10000)`; then the regular 2-way split
  with scale/(0.7·2)=scale/1.4 and the parent removed.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | code arguments `iterations` |
| N init | 10 | train.py:`args.num_gaussians = 10` |
| warmup_iter | 10000 | train.py `--ours_new` |
| densify_until | 15000+10000 = 25000 | train.py:29 |
| densify_from / interval | 500 / 100 (`iteration > 500`) | arguments |
| grad threshold | 0.0002 | arguments |
| percent_dense | 0.01 | arguments |
| prune opacity / big ws | 0.005 / 0.1·extent after step 3000 | gaussian_model / train.py |
| opacity reset | every 3000 while < 25000, value min(α,0.01) | train.py |
| divide_ratio | 0.7 (scale / 1.4) | train.py:26 |
| ABE factor | 0.3·scene_extent | gaussian_model.py |
| low-pass | min(max(HW/(9πN),0.3),300), every 1000 | train.py |
| means LR | 1.6e-4·extent held for i<10000, then expon(i−10000) to 1.6e-6·extent over max_steps 30000 (never reached: ends at ×0.0464) | train.py:79, gaussian_model |
| SH | +1 every 1000 starting at iteration 5000 (deg 3 at 7000) | train.py:84 |
| other LRs | 3DGS defaults (f_dc 2.5e-3, f_rest /20, opacity 0.05, scale 5e-3, rot 1e-3) | arguments |
| loss | 0.8 L1 + 0.2 D-SSIM | train.py |
| scene_extent | getNerfppNorm radius (1.1·max centre distance) | scene/__init__.py |

## 4. Schedules
step 1: s computed; 500–25000: densify every 100; ≤10000: ABE-Split; 3000k: opacity resets
until 24000; 5000/6000/7000: SH up; 10000: means LR starts decaying; s recomputed every 1000
until 24000.

## 5. Builds on
Official 3DGS code (Inria rasterizer with a `low_pass` argument). Everything else unchanged.

## 6. Protocol
M360 (images_4 outdoor / images_2 indoor), T&T, DB; llffhold 8; 30k iterations; black
background; LPIPS-VGG. Headline per-scene rows (Table 1 / Table 2 "Ours, Random" and Table 3
"Ours ✗ SfM") use **random** init N=10. Table 3 also lists "cluster" and "<10%" SfM-subset
variants (not targeted here). The 3DGS(SfM) rows equal 3DGS's published numbers.

## 7. Ambiguities
- B_0: paper = bbox centre, code = world origin → code used.
- Initial opacity: Alg. 1 says 1, code 0.1 → code.
- Cube from `tvec` (not camera centres) and scalar min/max — code quirk, reproduced.
- Paper says "optimization phase reduced to 5000 steps": consistent with densify until 25k.
- README warns "minor bugs including rendering code"; the published numbers come from this code.
