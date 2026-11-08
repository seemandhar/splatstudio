# WE-GS — implementation spec

Wang, Wang, Qi. *WE-GS: An In-the-wild Efficient 3D Gaussian Representation for Unconstrained
Photo Collections*, arXiv 2406.02407 (v1, LaTeX source in `source/latex/`).
Project page https://yuzewang1998.github.io/we-gs.github.io/ (no code link).

**Code status.** No WE-GS training code is public (re-checked 2026-11-07: project page, author's
GitHub `yuzewang1998`, GitHub search). The same authors' later repo
`yuzewang1998/takinglangsplatw` (TVCG 2026, Inria 3DGS licence, cloned read-only into
`source/code_langsplatw/`) ships the **WE-GS inference model** used to render their released
WE-GS checkpoints (`in_the_wild_renderer/`: `WEGSRenderer`, `appearance_encoder/ae_cnn.py`,
`transient_encoder/te_unet.py`, `itw_gaussian_renderer/__init__.py`, `arguments/__init__.py`
`itw_*Params`). Its README: "The WE-GS training code is not included in this repository yet".
That model differs from the paper text in several places (see §7); it is used here only for
values the paper omits (lrs, MLP width, output activation, PE frequencies, encoder layer shapes,
input resize).

## 1. Method summary
- 3DGS whose per-Gaussian SH is made image-dependent by a **residual SH transfer MLP**
  `Δsh_jk = G_φ(γ(sh_j), γ(μ_j), l_k)`, `sh'_jk = sh_j + Δsh_jk` (§4.1).
- The per-image appearance embedding `l_k` (dim 16) is **encoded from the image** by a CNN with
  a CBAM-style channel attention + U-Net spatial attention; the spatial attention map `M_k`
  (1/4 resolution) doubles as the **static-content mask**, `1 - M_k` = transient mask (§4.2).
- Training loss masks occluders with `M_k`; regularisers: mask size (`L_regM`), SH residual
  magnitude (`L_regSH`), and an opacity penalty on Gaussians projecting into transient regions
  (`L_regTS`) (§4.3).
- Test images are rendered feed-forward: `l_k` and `M_k` are encoded from the test image itself
  (no test-time optimisation; §5.3 "NeRF-W and K-Planes ... requiring optimization of appearance
  embedding for each test image, potentially biasing the comparison").

Contributions:
- **C1** residual-based SH transfer module + `L_regSH` (§4.1, Eq. 9-10, 13).
- **C2** simultaneous transient-mask prediction and appearance encoding (spatial attention
  module: Enc¹, CAM, U-Net, Enc²) (§4.2, Eq. 14-18).
- **C3** masked photometric loss + `L_regM` (§4.3, Eq. 19-20).
- **C4** transient-Gaussian opacity regulariser `L_regTS` (§4.3, Eq. 21).

## 2. Equations (verbatim from `section/3method.tex`)
Residual SH transfer:
```
\Delta sh_{jk} = \mathcal{G}_\phi(\gamma(sh_j), \gamma(\mu_j), l_k)          (9)
sh_{jk}^{\prime} = sh_j + \Delta sh_{jk}                                      (10)
\mathcal{L}_{regSH} = \sum_{j \in N, k \in M} \Vert \Delta sh_{jk} \Vert     (13)
```
γ = NeRF positional encoding. `G_φ` "is implemented as a four-layer MLP".

Encoder (I_k ∈ R^{3×W×H}):
```
F_k = Enc^1_{\theta1}(\mathcal{I}_k)                                        (14)  F_k ∈ R^{C×W'×H'}
F_{k}^{c} = Sig(MLP_{\theta2}(AvgPool(F_k)) + MLP_{\theta2}(MaxPool(F_k)))  (15)
F^{\prime}_k = F_{k}^{c} \otimes F_k                                        (16)
M_k = {UNet}_{\theta3}(F^{\prime}_k)                                         (17)  M_k ∈ R^{1×W'×H'}
F^{\prime\prime}_k = M_k \otimes F^{\prime}_k
l_k = Enc^2_{\theta4}(F^{\prime\prime}_k + F_k)                              (18)
```
transient mask = `R(1 - M_k)` (R = resize to W×H).

Loss:
```
\mathcal{L}_{c} = \lambda_1 \mathcal{L}_1( M_k \otimes I^{r}, M_k \otimes I^{gt}) +
                  \lambda_{ssim} \mathcal{L}_{SSIM}( M_k \otimes I^{r}, M_k \otimes I^{gt})   (19)
\mathcal{L}_{regM} = \| 1- M_k \|^{2}                                        (20)
\mathcal{L}_{regTS} = \sum_{i \in N} \alpha_i \mathbb{I}(M(\Pi(\mu_i))< \epsilon)            (21)
\mathcal{L} = \mathcal{L}_{c} +\lambda_1 \mathcal{L}_{regM} +\lambda_2 \mathcal{L}_{regSH} +\lambda_3 \mathcal{L}_{regTS}  (22)
```

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 150 000 | §5.1 |
| optimiser | Adam | §5.1 |
| λ_L1 / λ_SSIM (in L_c) | 0.8 / 0.2 | §5.1 |
| λ1 (L_regM) | 1e-5 | §5.1 (symbol λ1 reused) |
| λ2 (L_regSH) | 0.1 | §5.1 |
| λ3 (L_regTS) | 1e-5 | §5.1 |
| ε (L_regTS threshold) | **not given** | assumed 0.5 |
| appearance dim | 16 | §5.1, Tab. 3 rows 1-2 (8, 32 worse) |
| W', H' | W/4, H/4 | §5.1 |
| Enc¹, Enc² | 3 conv layers each | §5.1 |
| MLP_θ2 (CAM) | 2 FC + ReLU | §5.1 |
| U-Net | 2 conv + 2 transposed conv | §5.1 |
| G_φ | 4-layer MLP (ReLU) | §4.1 + commented §5.1 |
| SH "degree" L | 4 in paper notation (sh ∈ R^{L²}) = 16 coeffs = 3DGS degree 3 | Tab. 3 rows 3-4 (L=3: 45.9 MB, L=5: 87.8 MB vs 66.4 MB) |
| G_φ width | 256 | code `ae_cnn.py TransNet` |
| G_φ output | tanh | code `TransNet.forward` |
| PE of sh / of μ | 0 freqs (identity) / 4 freqs (1,2,4,8) | code `itw_AEParams app_pe_freq=0, pe_freq_xyz=4` |
| encoder input | image resized to 256×256 | code `ae_cnn.encode`, `te_unet.ConvMaskNet` |
| Enc¹ layers | ReflPad3-Conv7(3→128)-ReLU, ReflPad1-Conv4s2(128→256)-ReLU, ReflPad1-Conv4s2(256→512)-ReLU (= 1/4 res) | code `ae_cnn.Encoder` first 3 convs |
| Enc² layers | ReflPad1-Conv4s2(512→512)-ReLU, ReflPad1-Conv4s2(512→512)-ReLU, AvgPool, Conv1×1(512→16) | code `ae_cnn.Encoder` last 3 convs |
| lr G_φ / encoder / mask net | 1e-4 / 1e-4 / 1e-5, Adam eps 1e-15 | code `itw_AEParams`, `itw_TEParams` |
| Gaussian lrs | 3DGS (means 1.6e-4→1.6e-6 over 30 000, SH 2.5e-3 /20 for rest, opacity 0.05, scale 5e-3, rot 1e-3) | code `arguments/__init__.py OptimizationParams` (3DGS defaults) |
| densification | 3DGS (every 100 from 500 to 15 000, grad 2e-4, reset 3000, prune α<0.005) | §5.1 "following similar settings as in 3DGS" |
| background | black | code `itw_white_background=False` |

## 4. Schedules
- 3DGS: SH degree +1 every 1000 steps; means lr exp decay to 1% over `position_lr_max_steps`
  (30 000 in the code's argument defaults, not changed for 150k); densification until 15k.
- No schedule is given for the WE-GS losses (all active from step 1).

## 5. Builds on
Vanilla 3DGS (Inria code; the authors' repo is a fork). Everything else of 3DGS is unchanged.

## 6. Protocol
- Photo Tourism: Brandenburg Gate, Sacre Coeur, Trevi Fountain; NeRF-W train/test split (.tsv).
- Resolution: a commented-out line in §5.1 says "with images downsampled twice during both training
  and evaluation stages, following the practices of previous works [NeRF-W, Ha-NeRF, CR-NeRF,
  GS-W, SWAG]" → **downscale 2** (Ha-NeRF `img_downscale=2`, LANCZOS).
- Test appearance: encoded feed-forward from the full test image; metrics on the full test image
  (Ha-NeRF / CR-NeRF / GS-W protocol). The paper notes NeRF-W / K-Planes rows use test-time
  embedding optimisation.
- Metrics PSNR / SSIM / LPIPS (net not stated; Ha-NeRF/CR-NeRF/GS-W use LPIPS-Alex).
- Hardware: one RTX 4090; training time 1.8 h average (Tab. 4); storage 66.4 MB average.

## 7. Ambiguities and paper/code conflicts
1. **L_regM reduction.** `‖1-M_k‖²` with λ=1e-5. With a mean reduction, λ=1e-5 cannot stop the
   trivial `M→0` solution of Eq. 19 (L1 gradient on M ≈ 0.8|e| per pixel ≫ 2e-5). Chosen: squared
   Frobenius norm (sum) over the W'×H' mask (literal reading). Equilibrium per mask pixel:
   `1 - M ≈ 0.8|e|·(HW/W'H')/(HW)/(2e-5) = 4e4|e|/(W'H')` (L1 part only).
2. **L_regSH reduction.** Literal sum over N Gaussians with λ=0.1 would dominate L_c by orders of
   magnitude; chosen: mean over Gaussians of the L2 norm of the 48-dim Δsh (per image k).
3. **L_regTS**: ε not given (0.5 assumed); α_i = activated opacity; only Gaussians visible in the
   view (radius > 0); M sampled at the projected centre; indicator has no gradient.
4. **γ(sh_j)**: paper says PE; the released model uses 0 frequencies (identity). Code followed.
5. **G_φ depth**: paper 4 layers; released model 8 layers with skips (256 wide, tanh). Paper
   depth followed, code width/activation used.
6. **Encoder**: the released model feeds the mask as a 4th input channel of a 6-conv encoder and
   predicts the mask with a separate ULite U-Net (TVCG variant). The paper's CBAM-style shared
   attention design (Eq. 14-18) is implemented; the code's 6-conv encoder is split 3+3 into
   Enc¹/Enc² (exactly the paper's layer counts).
7. **Appearance dim / SH degree**: released model uses `appearance_dim=32`, `itw_sh_degree=5`;
   paper uses 16 and L=4 (3DGS degree 3). Paper followed.
8. `sh_j` input to G_φ not detached (paper's gradient Eq. 12 includes the extra term).
9. Mask resize R: bilinear (not stated).
10. 150k steps vs 3DGS densification ending at 15k: assumed unchanged.

## 8. Post-smoke / review-1 decisions (supersede §3/§7 where they differ)
- **L_regM**: literal Eq. 20 — squared Frobenius sum over M_k at W'×H' = W/4×H/4 of the training
  image, λ1 = 1e-5 (`wegs-mask-reg.resolution: quarter`).
- **DEVIATION (paper silent): M is detached inside the D-SSIM term of Eq. 19** (value unchanged;
  the L1 term and L_regM still train M). Cause: D-SSIM(M⊗I^r, M⊗I^gt) → 0 as M → 0, which gains
  ≈0.1 per step vs L_regM's maximum cost 1e-5·W'H' (0.04–0.11), so the literal form collapses to
  M ≡ 0 (evidence/smoke-1; reviews/review-1.md table, also with L_regTS off or bias-init M≈0.95).
  `ssim_detach_mask: false` restores the literal (collapsing) form. The intermediate default
  "full" (sum over R(1−M) at W×H, evidence/smoke-2) left the mask inert (mean 0.97) and is an
  ablation only.
- **Encoder input**: the training image itself (resized only to the nearest multiple of 16 so the
  U-Net strides close), so W' = W/4, H' = H/4 per §4.2/§5.1 — not the TVCG code's 256² resize
  (`bank_input_size: 256` keeps that as an ablation).
- **ε (L_regTS) = 0.1**, chosen a priori from training statistics only: the sigmoid-initialised
  mask is ≈0.47 at step 1, so ε = 0.5 marks every visible Gaussian transient at init (L_regTS
  0.088 at step 1 in smoke-1/2; review-1: 100k → 15k Gaussians by step 600). ε = 0.1 only flags
  pixels the mask has learned to call transient. Logged as `wegs_diagnostics` lines
  (transient_gaussians, static_mask_mean every 100 steps; also checkpoint
  `density_state.counters.history`).
- Bank init uses a forked RNG (respects `--seed`).
