# DOF-GS — implementation spec

Wang, Chakravarthula, Chen. *DOF-GS: Adjustable Depth-of-Field 3D Gaussian Splatting for
Post-Capture Refocusing, Defocus Rendering and Blur Removal*, CVPR 2025, arXiv 2405.17351.
Official code: github.com/rongduo/DOFGS_implementation (cloned to `source/code`, HEAD of
default branch, depth 1). Code is 3DGS + Mip-Splatting (3D filter, 2D kernel 0.1) + a DOF
rasterizer (`submodules/diff-gaussian-rasterization-dof`, Inria/MPII Gaussian-Splatting licence).
README: "original checkpoints were lost ... reproduced models may slightly differ from the paper".

## 1. Summary and contributions
- Thin-lens camera per training view m with learnable focal distance f_m and aperture Q_m;
  every projected 2D Gaussian is convolved with an isotropic Gaussian whose variance follows
  the circle-of-confusion (CoC) radius at the Gaussian's camera depth (Sec. 3.1.2, 3.2).
- The rasterizer also renders a depth map and a CoC map (alpha-blended CoC radii) (Sec. 3.2).
- All-in-Focus (AiF) image = render with Q = 0 (pinhole). Test views are evaluated AiF.
- CoC-guided detail enhancement (Sec. 3.3): a 4-conv "In-Focus Localization Network" (ILN)
  predicts a per-pixel mask from (defocused RGB, depth, CoC map, PE(view), PE(pixel));
  composite AiF/defocused images are supervised; Pearson mask loss vs CoC map + entropy reg.
- Training tricks (supp. B.2): 5k warm-up with L_rec only, 60k uniform extra points at 2k.

Contributions:
- C1 Finite-aperture camera + differentiable CoC convolution in the rasterizer (Sec. 3.1.2, 3.2,
  supp. A; CUDA).
- C2 Per-view learnable focus/aperture with heuristic initialization (Eq. 7 obj., supp. B.1).
- C3 CoC/depth map rendering + ILN in-focus mask with L_mk and L_reg (Sec. 3.3).
- C4 L_detail: composite AiF / defocused supervision (Sec. 3.3), two-stage schedule (supp. B.2).
- C5 Densification extras used by the code: extra uniform points at 2k, "compactness"
  densification every 1k (code only, not in paper), Mip-Splatting 3D filter.

## 2. Equations
Eq. (coc):  $R^{(coc)}=\frac{1}{2}FA\frac{|z_o-f|}{z_o(f-F)}$, simplified (Eq. coc_final)
$R^{(coc)} = \frac{1}{2} Q |\frac{1}{z_o} - \frac{1}{f}|$, Q = FA.
Blur kernel: $\Sigma^{(coc)}_k = a I$, $a=\frac{1}{2\ln 4}(R^{(coc)}_k)^2$ (least-squares fit of a
Gaussian to the uniform CoC disc). $\Sigma''_k = \Sigma'_k + aI$ (supp. Eq. sigma_coc).
Code (`forward.cu` preprocessCUDA): `coc_radius = 0.5*K[v]*|1/F[v] - 1/max(z_view,1e-7)|`,
`cov.xx += 0.36067376*coc^2`, `cov.yy += ...` where cov already contains the Mip 2D kernel
(+0.1). Opacity is multiplied by the Mip 2D-filter factor
sqrt(det(Σ'+aI)/det(Σ'+aI+0.1 I)) only — **no energy normalisation for the CoC blur** (the
blurred Gaussian keeps its peak opacity, apart from the Mip 2D factor). Radii from Σ'+aI+kI.
z for the CoC is the camera z of the centre; uniform depth over the footprint assumed.
Gradients (supp. A, `backward.cu`): da/dQ = c R |1/z-1/f|, da/df = ±c R Q / f^2; gradient from
Σ^(coc) to z is disabled (`coc2z_coeff = 0`).
Depth map: $\tilde{\tilde D}(x)=\sum T'_i\alpha'_i z_i$ (code: **ray distance |p_view|**, not z).
CoC map: $M^{(coc)}(x)=\sum T'_i\alpha'_i R^{(coc)}_i$.
Objective stage 1 (Eq. obj_final): $\min_{S,\{f_m,Q_m\}} \sum_m L_{rec}(\tilde{\tilde I}_m, I_m)$,
$L_{rec}=(1-\lambda)\|\cdot\|_1+\lambda L_{dssim}$, λ_dssim = 0.2 (code: (1-0.2)L1 + 0.2(1-SSIM)).
ILN: $M^*_m=ILN(\tilde{\tilde I}_m,\tilde{\tilde D}_m,M^{(coc)}_m,PE(m),PE(x))$.
$L_{mk}=1-\mathrm{Pearson}(1-M^*_m, M^{(coc)}_m)$; $L_{detail}=L_{rec}(M^*\tilde{\tilde I}^*+(1-M^*)\tilde{\tilde I}, I)$;
$L_{reg}=-M^*\log M^*$; total (Eq. obj_full) $L_{rec}$(?) $+L_{detail}+\lambda_{mk}L_{mk}+\lambda_{reg}L_{reg}$
(supp. B.2 says L_rec is deactivated after warm-up).
**Code (training.py:189-226)** — the mask `m` is the *out-of-focus* mask (paper M* = 1-m):
```
norm_depth = minmax(depth_defoc); norm_coc = minmax(coc)
m = ILN(view, PE, cat[rgb_defoc, norm_depth].detach(), norm_coc.detach())   # sigmoid
t = minmax(log1p(coc.detach()))
mask_loss = λm * (1 - pearson(m, t))   (+ eps 1e-8 floors on the variances)
reg       = -λm*0.1 * mean(m * log(m + 1e-6))
L1 = Lrec((1-m)*aif.detach() + m*rgb_defoc, gt)        # backward separately
L2 = Lrec((1-m)*aif + m*rgb_defoc.detach(), gt)
loss = L1 + λ_infocus*L2 + mask_loss + reg                (step > 5000)
loss = Lrec(rgb_defoc, gt)                                  (step <= 5000)
```
AiF render: F = [1.0], K = [0.0] → CoC = 0 exactly (pinhole).
Parameterisation: f = clamp(Fs,0.01,1000)*10, Q = clamp(Ks,0.01,200)*100+100.
Init (gaussian_model.setup_defocus_params, uses **world z** of SfM points + nerf++ translate):
z_med = 1/q_0.5(1/z_w) + translate_z; Fs = z_med/10;
z_max = 1/q_0.05(1/z_w), z_min = 1/q_0.95(1/z_w); Kv = 15/(|1/z_min-1/z_max|+1e-6);
Ks = 10*(Kv-100)/1000 (so Q_init = Kv). τ = coc_max = 15.
ILN (neural_module.MaskNet): conv_rgb = Conv(5→48,3x3,p1)-ReLU-InstanceNorm-Conv(48→16,3x3);
input2 = [Embedding(N_view,32) broadcast, PE(16ch)] (48 ch); feat = Conv1x1(64→20,no bias)-ReLU;
mask = sigmoid(Conv1x1(20→1,no bias)). PE: get_2d_emb(H,W,16) sinusoidal, freq 1/10000^(k/8).

## 3. Hyper-parameters (code `config_files/config_real_*.yaml` + `arguments/__init__.py`)
| name | value | source |
|---|---|---|
| iterations | 40000 | paper §4.1, config |
| position lr | 5e-4 → 5e-6 (×extent), expon, max_steps 80000 (!) | config (paper: decays over 40k) |
| scaling lr | 0.01 | config/supp |
| feature/opacity/rotation lr | 0.0025 / 0.05 / 0.001; SH rest /20 | 3DGS defaults |
| fdist lr / kapert lr | 0.05 / 0.01 (on Fs / Ks reparam) | config / args |
| mask net lr | 5e-4 | config |
| F,K join optimizer | step 100 (grads accumulate before) | iter_defocus_start |
| F,K frozen | step 15000 | iter_stop_cam_gd |
| infocus start | 5000 | config / supp B.2 |
| λ_dssim | 0.2 | paper |
| lambda_infocus | 1.0 | config |
| lambda_mask_loss (λ_mk) | real, per scene: 0.002 for cake and daisy, 0.001 for the other 7; λ_reg = 0.1 λ_mk (hard-coded) | config_real_<scene>.yaml, training.py:212 (paper: 0.001 / 0.0001) |
| band_limit (Mip 3D filter in render + filter-aware reset) | real, per scene: true for cake, daisy, cisco, sausage, seal, tools; false for caps, cupcake, cups (plain reset min(σ(o), 0.01)) | config_real_<scene>.yaml, training.py:286-289, renderer/__init__.py:63-82 |
| densify from/until/interval | 500 / 25000 / 100 | args / config |
| grad threshold | 0.0002 | args |
| percent_dense | 0.03 | config |
| min_opacity (prune) | 0.005; no size/world prune (use_size_prune false) | config |
| opacity reset | every 3000 (<25000), filter-aware, 0.01 | args |
| compactness densify | every 1000 (<25000), K=3, shrink 1.5 | config (not in paper) |
| extra points | step 2000, M=30000 → 60000 samples, bbox ±25%, dist_thres 15, k=4 | config |
| Mip 3D filter | coef 0.2, near 0.2, margin 0.15; recomputed after densify, every 100 after 25k | code |
| Mip 2D kernel | 0.1 | args |
| coc_max τ | 15 | supp B.1, config |
| SH | degree 3, +1 every 1000 | 3DGS |
| background | black | args (white_background False) |
| resolution | real: `images/` downscaled ×4 by PIL (600×400) | config |
| holdout | per scene (cake 9, caps 5, cisco 7, cupcake 6, cups 8, daisy 8, sausage 6, seal 6, tools 7) | scene_info.yaml |
| seed | 3467 | training.py |

## 4. Schedules
1–99: F/K receive gradients but are not optimised (grads accumulate). 100: F/K join Adam.
≤5000: L_rec on the defocused render. >5000: L1+L2+mask terms (above). 2000: extra points.
Every 1000 (<25000): compactness densification. 15000: F/K frozen. 25000: densification ends.

## 5. Builds on
3DGS (Inria) with Mip-Splatting 2D kernel, plus the 3D filter and filter-aware opacity reset when
band_limit is true (per scene; plain 3DGS reset otherwise). Kept:
SH schedule, clone/split criterion, Adam eps 1e-15, black background.

## 6. Protocol
Deblur-NeRF defocus datasets (Ma et al. 2022). Real: 9 scenes (cake, caps, cisco, cupcake,
cups, daisy, sausage, seal, tools) at 600×400, COLMAP poses from the dataset; synthetic: 5 scenes
(cozyroom, factory, pool, tanabata, trolley(=wine in code)) with COLMAP poses from blurry images.
Train on blurry views, test = every `hold`-th sorted view (sharp reference), rendered AiF.
Metrics: PSNR/SSIM via skimage on [-1,1] (PSNR identical to [0,1]); no LPIPS in the tables.
Official val scripts evaluate per-scene checkpoints at different iterations (val_real.sh: cake
15000, caps 15000, tools 40000; test-set selection on re-trained models). The paper states 40k,
so the protocol is the final 40k state; a 15k evaluation may be reported only as a labelled
diagnostic, never as the result.

## 7. Ambiguities
- Paper says L_rec is deactivated after 5k; code keeps a defocused-branch composite loss (L1
  above) — follow the code.
- Paper λ_mk = 0.001, λ_reg = 1e-4. The code's real configs agree for 7 of 9 scenes; cake and
  daisy use 0.002 / 2e-4. band_limit is also per scene (false for caps, cupcake, cups). Follow
  the per-scene code configs (method variants `dofgs`, `dofgs-bl-m001`, `dofgs-nobl`).
- Paper says 60k extra points; code samples 2×30000 = 60000 candidates, but the nearest-SfM-
  neighbour distance filter (< 15) rejects most of them: on cake only 539 of 60000 survive,
  because the SfM bbox is large (≈[-100,110]×[-127,29]×[15,291]) and is expanded by ±25 %.
  The port reproduces this (official behaviour).
- "Compactness" densification exists only in code. Its scale arguments are raw log-scales
  divided by 1.5 (official quirk, ported verbatim); the shrink itself is not written back.
- Depth map channel is ray distance (|p_view|), not camera z.
- Official LR schedule max_steps 80000 → final xyz lr at 40k is 5e-5·extent, not 5e-6.
