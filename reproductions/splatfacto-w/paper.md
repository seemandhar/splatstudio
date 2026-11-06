# Splatfacto-W (arXiv 2407.12306) — implementation spec

Sources: `source/latex/sec/*.tex` (paper), `source/code/splatfactow/*.py` (official Nerfstudio
plugin, Apache-2.0, commit cloned 2026-11-05). **Code overrides paper where they disagree; both recorded.**

## 1. Summary
- Built on Nerfstudio *splatfacto* (gsplat). Gaussian colour = SH coefficients predicted by an MLP
  from a per-Gaussian feature f_i and a per-image embedding ℓ_j (no view direction in the MLP).
- Transient handling: RobustNeRF-like per-pixel residual mask, mask percentage interpolated from
  the running min/max of the image L1, upper 40 % of the image never masked, 5×5 box smoothing.
- Background: SH (degree 4) environment at infinity, coefficients predicted by an MLP from ℓ_j,
  composited behind the Gaussians; alpha loss pushes accumulation to 0 where the background
  already explains the pixel.

Contributions
- C1 Latent appearance modelling (§4.1): b_i = MLP_θ(ℓ_j, f_i), c_i = colour from SH(b_i, d_i).
- C2 Transient handling with robust mask (§4.2).
- C3 Background model + alpha loss (§4.3).
- Protocol (§5.2 + code): NeRF-W right-half evaluation.

## 2. Equations
C1: `\mathbf{b}_i = \text{MLP}_{\theta}(\ell_j, f_i)`,
`c_i = \text{Sigmoid}(\sum_{\ell}\sum_m b_{i,\ell}^m Y_\ell^m(\mathbf{d}_i))`.
**Code**: no sigmoid — the coefficients go to gsplat `rasterization(colors=sh, sh_degree=k)`,
i.e. `c_i = max(SH(b_i, d_i) + 0.5, 0)` with d_i = μ_i − camera centre. MLP (nerfstudio `MLP`,
torch): Linear(48+72→256) ReLU Linear(256→256) ReLU, heads `sh_base` Linear(256→3) (default init)
and `sh_rest` Linear(256→45) (zero init). SH degree = min(step // 2000, 3).

C2 (paper):
`k = \frac{L1_{cur} - L1_{min}}{L1_{max}-L1_{min}} (Per_{max}-Per_{min}) + Per_{min}`,
`\mathcal{T}_\epsilon` = (1−k) quantile of per-pixel residuals,
`\tilde\omega = (\epsilon \le \mathcal{T}_\epsilon) \lor U`, U = rows `r_y ≤ 0.4H`,
`\mathcal{W} = (\tilde\omega * \mathcal{B}_{5×5}) ≥ 0.4`.
**Code** (`robust_mask`): errors = |gt − pred| [H,W,3]; rows `[: int(0.4 H)]` set to 0 (→ always
inliers); L1 = errors.mean() (incl. zeroed rows); max_loss ← L1 if L1 > max or step % 6000 == 0;
min_loss ← min; k = (L1 − min)/(max − min + 1e-6)·(0.4 − 0) + 0; per-pixel error = mean over
RGB; thr = torch.quantile(per_pixel, 1 − k); inl = per_pixel ≤ thr; nb = box5(inl, 'same') > 0.4;
mask = (nb + inl > 1e-3) (inlier if itself inlier OR smoothed > 0.4). Active from step 6000.
Loss: L1 = mean(|gt−pred|·mask); SSIM on gt·mask, pred·mask (pytorch_msssim SSIM, 11×11 σ 1.5
valid); main = 0.8 L1 + 0.2 (1 − SSIM). (In code the in-place zeroing makes the *logged* L1 skip the
upper rows, but its gradient equals mean(|gt−pred|·mask) with the upper rows inliers.)

C3: `\mathbf{b} = \text{MLP}(\ell_j)`, `C_{bg}(r) = \text{Sigmoid}(\sum b_\ell^m Y_\ell^m(d_{ray}(r)))`,
`C_{final} = C + (1-\alpha) C_{bg}`. **Code**: no sigmoid and no +0.5: `C_bg = SH(b, d_ray)` (gsplat
`spherical_harmonics`), final rgb clamped to [0,1]. BG MLP: Linear(48→128) ReLU Linear(128→128)
ReLU, base Linear(128→3), rest Linear(128→72) zero-init; degree min(step // 1000, 4).
Alpha loss: `L_α = λ Σ_{r∈p_i} α(r)`, `p_i = {M' > 0.6}`, `M = 1[|GT − BG| < thr]`, `M' = M * B_3×3`.
**Code**: M = mean_c|gt − bg| < 0.003 (gt after robust masking), box3 'same', L_α = 0.15 ·
mean(α[M' > 0.6]) (mean, not sum; 0 if empty).

## 3. Hyper-parameters (code `splatfactow_config.py` / `SplatfactoWModelConfig`)
| name | value | source |
|---|---|---|
| iterations | 55000 (paper: 65000) | config `max_num_iterations` / paper §5.1 |
| appearance embed dim | 48 | paper §5.1, code |
| Gaussian appearance features | 72, init 0 | paper, code |
| app MLP | 3 layers width 256 (2 hidden + heads) | paper, code |
| bg MLP | 3 layers width 128 | paper, code |
| sh_degree / interval | 3 / 2000 | code |
| bg_sh_degree / interval | 4 / 1000 (= sh_interval/2) | code |
| robust mask % | (0.0, 0.40), start 6000, reset max every 6000 | code |
| never_mask_upper | 0.4 | paper, code |
| alpha loss λ / thr / box | 0.15 / 0.003 / 3, select > 0.6 | code |
| ssim_lambda | 0.2 | code |
| resolution schedule | 1/4 res, ×2 every 3000 steps (full at 6000) | splatfacto default |
| warmup / refine_every | 1000 / 100 | code |
| reset_alpha_every | 25 refinements (2500 steps), reset value 0.2 | code |
| densify_grad_thresh | 0.0008 (absgrad, ×0.5·max(W,H)) | code |
| densify_size_thresh | 0.01 (scene units) | code |
| cull_alpha / cull_scale | 0.1 / 0.5 | code |
| cull / split screen size | 0.15 / 0.05, stop at 15000 | code |
| stop_split_at | 20000, culling continues afterwards | code |
| rasterize_mode | antialiased | code |
| init | SfM points, scale log(mean 3-NN dist), random quats, opacity 0.1 | code |
| background (no bg model) | random per step; eval (0.149, 0.1647, 0.2157) | code |
| LRs (Adam eps 1e-15) | means 1.6e-5→1.6e-7 (30k); features 0.02→1e-3 (40k); opac 0.03; scales 0.005; quats 0.001; embed 0.02→3e-4 (40k); app enc 2e-3→1e-4, base 2e-3→1e-4, rest 1e-4→5e-6 (30k); bg enc 2e-3→1e-4, base 2e-3→2e-4, rest 1e-4→1e-5 (30k) | config |
| scene normalisation | auto_orient 'up', center 'poses', scale to ±1 then ×3 | `nerfw_dataparser.py` |

Exponential schedules are nerfstudio `ExponentialDecayScheduler` (log-linear to lr_final at
max_steps, constant afterwards; lr at step k uses f(k), 0-based steps).

## 4. Schedules
Downscale 4/2/1 at steps [0,3000)/[3000,6000)/≥6000; SH +1 per 2000 (bg per 1000); robust mask
≥6000; densify at step%100==0 for 1000<step<20000 and step%2500 > N_train+100; opacity reset at
step%2500 == 100 (<20000); cull only after 20000; too-big culls after 2500.

## 5. Builds on
Splatfacto (Nerfstudio 3DGS) with gsplat; L1 + 0.2 D-SSIM kept; SH colours replaced by C1.

## 6. Protocol
Photo Tourism (NeRF-W splits, tsv files with NaN rows dropped), Brandenburg Gate, Sacre Coeur,
Trevi; full resolution `dense/images`. Metrics on the right half (columns ≥ W//2) of the NeRF-W test
images: PSNR (torchmetrics), SSIM (pytorch_msssim), LPIPS (torchmetrics, AlexNet).
**Paper**: "optimize an embedding on the left half of each test image". **Code**
(`splatfactow_datamanager.next_train`): the train split is *all* images; test images are trained
jointly with every parameter, with their right half masked (gt, pred × mask); evaluation renders
the test view with its own trained embedding (cam_idx) and scores the right half.
Variants: Splatfacto-W-A (appearance only), Splatfacto-W-T (appearance + robust mask).

## 7. Ambiguities
- Iterations 65000 (paper) vs 55000 (code) → code.
- Sigmoid colour/background (paper) vs +0.5/clamp and raw SH (code) → code.
- W = box ≥ 0.4 (paper) vs inlier OR box > 0.4 (code) → code.
- L_α sum (paper) vs mean (code) → code; threshold unspecified in paper → 0.003.
- Test embedding: left-half optimisation of embedding only (paper) vs joint training on left halves
  (code) → code (headline); embedding-only test-time optimisation provided as a secondary protocol.
- Whether Table 1 used the -W-A/-T variants with the random background (code default "random"
  when the bg model is off) → code.
