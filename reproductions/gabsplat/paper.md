# 3DGabSplat: 3D Gabor Splatting for Frequency-adaptive Radiance Field Rendering

Zhou, Huang, Dai, Zou, Zheng, Kan, Li, Xiong — ACM MM 2025, arXiv 2508.05343.
Sources: `source/latex/3DGabSplat-final.tex` (main), `source/latex/Supplementary.tex` (supp.).
**Official code: none.** `github.com/Zhjy1/3DGabSplat` (re-checked 2026-11-07) contains only a
README ("The code will comming soon!"). Everything below comes from the paper text.

## 1. Method summary
- Each primitive = 3D Gaussian envelope × [Gaussian (DC) term + F=2 cosine-modulated 3D Gabor
  terms] with per-primitive learnable 3D frequency vectors f_{k,i} ∈ R³ and weights ω_{k,i}
  (sigmoid-activated).
- Projection: the 3D frequency is transformed with (JW)^{-T}; integrating the ray-space 3D Gabor
  along z gives a 2D Gaussian × cos with a 2D frequency corrected by the z-frequency
  (closed form, supp. A.2). Splatted with the 3DGS tile rasterizer, Gaussian kernel replaced by
  the 2D Gabor primitive.
- Frequency-adaptive optimisation: children created by densification get their frequencies and
  weights reset to small values; all Gabor weights are reset every 3000 iterations together with
  the opacity reset.
- Everything else = 3DGS (L1 + 0.2 D-SSIM, 30k iters, densify until 15k).

Contributions
- C1 3D Gabor-based primitive (Sec. 3.2.1, Eq. 7).
- C2 3D Gabor splatting: frequency projection + 3D→2D integration + rasterizer (Sec. 3.3, Eq. 9-13; supp. A).
- C3 Frequency-adaptive optimisation: child reset at densification + periodic weight reset (Sec. 3.4.2; supp. B).
- (Recipe) Initial values / LRs / opacity threshold (Sec. 4.1, supp. B).

## 2. Equations
Eq. 7 (primitive):
$$\{Gabor\}_k(\mathbf{x})=\mathcal{G}_k(\mathbf{x})\Big[\big(1-\sum_{i=1}^F \omega_{k,i}\big) + \sum_{i=1}^F \omega_{k,i}\cos\big(2\pi \boldsymbol{f}_{k,i}^\top(\mathbf{x}-\boldsymbol{\mu}_k)\big)\Big]$$
Eq. 9 (frequency projection): $\boldsymbol{f}_{k,i}^{proj} = {(\mathbf{JW})^{-1}}^\top \boldsymbol{f}_{k,i}$.
Eq. 10 (3D→2D, S = entries of the ray-space $\Sigma'^{-1}$):
$$f^{2D}_x = f^{proj}_x - \tfrac{S_{02}}{S_{22}} f^{proj}_z,\qquad f^{2D}_y = f^{proj}_y - \tfrac{S_{12}}{S_{22}} f^{proj}_z$$
Eq. 12 (2D primitive):
$$\{Gabor\}_k^{2D}(\mathbf{x})=\mathcal{G}_k^{2D}(\mathbf{x})\Big[(1-\sum_i\omega_{k,i})+\sum_i\omega_{k,i}\cos(2\pi {\boldsymbol{f}_{k,i}^{2D}}^\top(\mathbf{x}-\boldsymbol{\mu}_k^{2D}))\Big]$$
Eq. 13 (blending): $\mathbf{C}(\mathbf{x})=\sum_i\mathbf{c}_i\alpha_i\{Gabor\}_i^{2D}(\mathbf{x})\prod_{j<i}(1-\alpha_j\{Gabor\}_j^{2D}(\mathbf{x}))$.
Eq. 14 (loss): $\mathcal{L}=(1-\lambda)\mathcal{L}_1+\lambda\mathcal{L}_{D-SSIM}$, λ = 0.2.

Supp. A.2 derivation: ∫ g dz = C1 C2 G^{2D}(x,y) cos(2π f̂·(x,y)) with C1 = (2π S22)^{-1/2},
C2 = exp(±(2π f_z)²/(2 S22)) (sign typo in the paper; correct is exp(-2π² f_z²/S22)); the final
rendered primitive (Eq. 12) **drops C1 C2** — no amplitude attenuation by f_z.

Equivalent coordinate-free form (used in the port, derived here): with J2 = J W (2×3, the
first two rows), Σ2D = J2 Σ J2ᵀ and the Gaussian conditional mean,
f^{2D} = Σ2D^{-1} J2 Σ f. This equals Eq. 9-10 for any invertible completion of the 3×3 Jacobian
(the third row only reparametrises the integration variable); a probe checks the equality.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| F (Gabor kernels / primitive) | 2 | Sec. 4.1 |
| init frequency (all comps) | **0.05 (Mip-NeRF 360)**; 1e-5 T&T; 0.001 others | supp. B (main text: 0.001 "default") |
| init weight ω (post-sigmoid) | 0.01 (Mip360, others); 0.02 T&T | supp. B / Sec. 4.1 |
| child reset at densification | f = 0.001, ω = 0.01 | supp. B |
| periodic weight reset | ω = 0.01 every 3000 iters with opacity reset | Sec. 3.4.2, supp. B |
| lr frequency | **0.02** (supp. B) — main text says 0.01 | supp. B vs Sec. 4.1 |
| lr weight | **0.01** (supp. B; 0.005 T&T/DB) — main text says 0.02 | supp. B vs Sec. 4.1 |
| lr opacity | 0.025 Mip360; 0.01 T&T; 0.02 DB | Sec. 4.1, supp. B |
| opacity prune threshold | 0.01 (3DGS: 0.005) | supp. B |
| opacity reset value | 0.02 (3DGS: 0.01) | supp. B ("reset the opacity to 0.02") |
| λ D-SSIM | 0.2 | Eq. 14 |
| iterations / densify stop | 30k / 15k | supp. B |
| everything else | 3DGS defaults | supp. B |

## 4. Schedules
Densify every 100 iters from 500 to 15k (3DGS); opacity + Gabor-weight reset every 3000 iters
(< 15k); SH degree +1 every 1000; means lr exp decay (3DGS).

## 5. Builds on
Vanilla 3DGS (Inria). Unchanged: SfM init, SH colour, adaptive density control (thresholds as
3DGS except opacity prune/reset), loss, LR schedule of means. Variants +Scaffold-GS,
+Mip-Splatting+AbsGS, +MCMC are not reproduced here.

## 6. Protocol
Mip-NeRF 360 (9 scenes; outdoor images_4, indoor images_2, official downsampled images),
T&T (truck, train), DB (drjohnson, playroom), NeRF-Synthetic. 30k iterations; llffhold 8
(3DGS convention, not stated explicitly); metrics PSNR/SSIM/LPIPS (net unstated; 3DGS uses VGG).
Baseline row "3DGS*" = authors' reproduction (Mip360 avg 27.41). RTX 3090.

## 7. Ambiguities (choices in the port)
1. Main text vs supp. conflict on init frequency (0.001 vs 0.05 for Mip360) and on the
   frequency/weight LRs (0.01/0.02 vs 0.02/0.01). Port: supplementary values (dataset-specific,
   more detailed); both recorded.
2. Identical initialisation of the F=2 kernels (same f vector, same ω) makes the two kernels
   receive identical gradients → they stay identical forever (F=2 ≡ F=1 with weight 2ω). Paper
   describes constant init; port follows literally and records the gap.
3. Frequency direction at init: "set to 0.05" — port sets all three components to 0.05.
4. Weight reset: "reset to 0.01" — port sets ω = 0.01 exactly (not min(ω, 0.01)) and zeroes the
   Adam moments, like 3DGS's opacity reset does for opacity.
5. Clone vs split children: port resets the appended clone (original keeps its frequency) and
   both split children (the parent is removed), following 3DGS's densification bookkeeping.
6. Negative modulation: 1 - Σω + Σω cos can be < 0 when Σω > 0.5; the 3DGS rasterizer skips
   alpha < 1/255, so negative contributions are dropped (port keeps that rule).
7. Low-pass dilation: 3DGS adds 0.3 px² to Σ2D. The port uses the dilated conic in
   f^{2D} = conic · (J2 Σ f) (Gabor convolved with the dilation filter has exactly this centre
   frequency); the paper's Eq. 10 uses the undilated Σ'^{-1}. Identical for footprints ≫ 0.3 px².
8. C1·C2 amplitude factor dropped (Eq. 12 literal).
