# DET-GS — implementation spec (arXiv 2508.04099, Huang, Xu, Perry, UTS)

Sources: LaTeX `source/latex/bare_jrnl_new_sample4.tex`, PDF `source/paper.pdf`.
**Official code: none** ("link will be made available after publication"; GitHub/HF/arXiv checked
2026-11-07). The depth-supervision part is, equation for equation, DNGaussian's Global-Local Depth
Normalization (CVPR'24); its official code (`reproductions/dngaussian/source/code/`) is used to
resolve numerics DET-GS leaves unspecified (δ, error tolerance, hard-depth opacity ω, patchify).

## 1. Summary and contributions
- 3DGS (official PyTorch framework, unchanged densification) + three extra losses, 30k iterations.
- C1 **Hierarchical Geometric Depth Supervision** (§III-B, Eq. 4–7): "hard" depth rendered with all
  opacities set to ω, split into non-overlapping random-size patches, compared to Depth Anything
  V2-Large depth after patch-local normalisation (patch mean/std) and image-level normalisation
  (patch mean, image std); L2 with an error-tolerance margin; every 5 iterations.
- C2 **Edge-Aware Depth Regularization** (§III-C, Eq. 8–11): Canny(20,200) mask of the GT image;
  on non-edge pixels, rendered depth is pulled to its masked cross-shaped 3×3 neighbourhood mean.
- C3 **RGB-guided edge-preserving TV** (§III-D, Eq. 12–15): hinge TV (margin τ_smooth) on the
  rendered image, only where the GT gradient magnitude < τ_edge.

## 2. Equations (verbatim)
Eq. 3 (soft depth): $\mathcal{D}(x_i) = \sum_{n} ||\mu_n - o||_2 \tilde\alpha_n \prod_{j<n}(1-\tilde\alpha_j)$.
Eq. 4 (hard depth): $\mathcal{D}(x_i) = \sum_{n} \omega \exp(-\iota(n-1)) \mathcal{G}^{2D}_n(x_i) ||\mu_n - o||_2,\ \iota=-\ln(1-\omega)$
  (= alpha compositing with every opacity = ω; DNGaussian `render_for_depth`, value=0.95).
Eq. 5: $\mathcal{D}_{patch}(x) = \frac{\mathcal{D}(x) - \mu_P}{\sigma_P + \delta}$ (patch mean/std).
Eq. 6: $\mathcal{D}_{image}(x) = \frac{\mathcal{D}(x) - \mu_P}{\sigma_I}$ (patch mean, image std).
Eq. 7: $\mathcal{L}_{depth} = \gamma\|\mathcal{D}_{patch} - \tilde{\mathcal{D}}_{patch}\|_2^2 + \eta\|\mathcal{D}_{image} - \tilde{\mathcal{D}}_{image}\|_2^2$, plus "error tolerance mechanism" (§IV-B).
Eq. 8–9: $\mathcal{E}=\mathrm{Canny}(\mathcal{I})$, $m=1$ where $\mathcal{E}=0$.
Eq. 10: $\bar{\mathcal{D}}(x_i) = \frac{\sum_{\mathcal{N}(x_i)} \mathcal{D}(x_j)m(x_j)}{\sum_{\mathcal{N}(x_i)} m(x_j)+\epsilon}$, ε=1e-8, $\mathcal{N}$ = centre + 4-neighbours.
Eq. 11: $\mathcal{L}_{edge} = \frac{1}{P}\sum_{x_i} m(x_i)|\mathcal{D}(x_i)-\bar{\mathcal{D}}(x_i)|^2$, P = #valid pixels.
Eq. 13: forward differences $\nabla_h, \nabla_v$ of $\mathcal{I}_{pred}$, $\mathcal{I}_{gt}$.
Eq. 14: $M_{h/v} = \mathbb{I}(|\nabla_{h/v}\mathcal{I}_{gt}| < \tau_{edge})$.
Eq. 15: $\mathcal{L}_{tv} = \frac{1}{|\Omega|}\sum_x M_h\max(|\nabla_h\mathcal{I}_{pred}|-\tau_{smooth},0) + M_v\max(|\nabla_v\mathcal{I}_{pred}|-\tau_{smooth},0)$.
Eq. 16: $\mathcal{L}_{color} = \mathcal{L}_1 + \lambda\mathcal{L}_{D-SSIM}$ ("following 3DGS").
Eq. 17: $\mathcal{L} = \mathcal{L}_{color} + \mathcal{L}_{depth} + \beta\mathcal{L}_{edge} + \phi\mathcal{L}_{tv}$.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| γ (patch-local depth) | 0.1 | §IV-B |
| η (image-level depth) | 1 | §IV-B |
| β (edge) | 0.1 | §IV-B |
| ϕ (TV) | 0.8 | §IV-B |
| depth-loss period | every 5 iterations | §IV-B |
| patch size | uniform integer in [5, 20], resampled | §IV-E "Patch size selection" |
| Canny thresholds | 20 / 200 | §III-C |
| ε (edge) | 1e-8 | §III-C |
| neighbourhood | cross 3×3 (centre + 4) | §III-C |
| τ_edge, τ_smooth | 1e-2, 1e-4 | §III-D |
| depth prior | Depth Anything V2-Large (vitl) | §IV-E |
| iterations | 30k | §IV-B |
| λ (D-SSIM) | **not given**; "following 3DGS" → 3DGS 0.8·L1 + 0.2·D-SSIM | assumption |
| δ | **not given** → DNGaussian `1e-2·std(whole map)` | DNG code utils/loss_utils.py:21 |
| error tolerance | 0.2 — **not given** (§IV-B only says an error-tolerance mechanism is added to the depth L2). DNGaussian, whose loss this is, uses 0.2 as its code default (arguments/__init__.py:92), 0.01 for DTU / LLFF-MVS-init, 2.5e-4 for 3-view LLFF (scripts/run_llff.sh). **Chosen: 0.2**, the code default (not tuned for a dataset). 2.5e-4 would make the margin a no-op (review-1: about 1% of the normalised entries excluded); 0.2 drops about 95% of the image-term entries at p=10 on a converged garden model. Not tuned on test images. | DNG arguments/__init__.py:92 |
| ω (hard-depth opacity) | **not given** → DNGaussian 0.95 | DNG gaussian_renderer:128 |

## 4. Schedules
Depth loss on iterations ≡ 0 mod 5 (start/stop not stated → whole run). Edge and TV every iteration
(not stated otherwise). 3DGS densification/LR/SH schedules unchanged.

## 5. Builds on
Official 3DGS (densify 500–15k, opacity reset 3k, SfM init). No other change stated.

## 6. Protocol
Full-view (NOT sparse-view, despite the abstract): Mip-NeRF 360 9 scenes (images_4 outdoor,
images_2 indoor), T&T 2, DB 2, NeRF/NSVF synthetic; 3DGS split (llffhold8 standard); 30k iterations;
PSNR/SSIM/LPIPS (net unspecified). Per-scene PSNR/SSIM/LPIPS for Mip-NeRF 360 in the appendix
(garden 28.09, counter 30.07, bicycle 25.95, room 32.09); T&T and DB averages only.
Paper's 3DGS row (27.79 avg) is higher than the 3DGS paper (27.21) → gains: garden +0.36, counter +0.95.

## 7. Ambiguities
1. Monocular depth sign/convention: DA-V2 outputs affine-invariant *disparity*. Patch
   normalisation is affine-invariant but not inversion-invariant; DNGaussian uses `255 − disparity`
   (a sign flip). We use target = −disparity (identical after normalisation).
2. Depth in Eq. 3/4 is distance ||μ−o||; gsplat renders camera-z. Kept z (sub-percent difference
   near the image centre, up to ~1/cos(FOV/2) at corners).
3. Which depth L_edge uses: soft depth of Eq. 3 (the "rendered depth"); Eq. 3 is not alpha-
   normalised → D = Σ w z (gsplat "D" = ED·alpha).
4. TV normalisation over channels: mean over pixels and RGB channels.
5. Whether L_depth gradients go to opacities: hard depth uses constant ω → only means/scales/quats.
6. Ablation table rows are labelled "w/" but the text calls them removals.
7. δ, λ, ω, error tolerance, depth-loss start/stop not stated (see table).
8. Supervision space (depth vs disparity): not stated. Default: z vs −disparity (DNGaussian-consistent).
   Option `space: disparity` gives 1/z vs disparity, the current official-3DGS style. On converged gsplat garden/counter,
   corr(z, −disp)=0.44 vs corr(1/z, disp)=0.28 (review-1), so the data do not favour either.
9. Optimiser step: DNGaussian back-propagates the hard-depth loss with its own optimizer.step(). Here it is summed into
   the shared step (Eq. 17 writes one total loss; DET-GS does not say which). Small difference.
10. TV normalisation: divided by H·W·3 (channels averaged). Summing the channels would give an effective ϕ of 2.4.
