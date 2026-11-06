# Faster-GS: Analyzing and Improving Gaussian Splatting Optimization — implementation spec

Hahlbohm, Franke, Eisemann, Magnor. CVPR 2026, arXiv 2602.09999.
Project: https://fhahlbohm.github.io/faster-gaussian-splatting/ ·
Code: https://github.com/nerficg-project/faster-gaussian-splatting (Apache-2.0).
Sources read: LaTeX e-print (`source/latex/`), PDF, official code `main` @ 090e8ad
(`source/code/`), `FasterGSTestbed` @ ffa262a (`source/code-testbed/`, ablation toggles),
NeRFICG framework helpers (`source/nerficg/`: `Optim/adam_utils.py`, `lr_utils.py`,
`CudaUtils/MortonEncoding`).

## 1. Method summary
- Faster-GS is **3DGS optimization, unchanged in algorithm** (same ADC, losses, schedules,
  hyper-parameters), re-implemented for speed and VRAM. The quality claim is "same as 3DGS";
  the main claim is 3.1–4.6× faster training (RTX 4090), up to 5.2× vs 3DGS.
- Consolidates prior speed-ups (survey §3.2) into one CUDA rasterizer and adds own ones (§3.3).
- Only the rasterizer, optimizer step, and parameter memory layout change.

Contributions / components of the **main configuration** ("Ours"/"Full", Tab. 1–2):
- **C1 Basis implementation** (§3.1, supp. "Testbed Basis"): rewrite of the 3DGS rasterizer
  using intrinsics directly (arbitrary principal point), front-to-back alpha-blending gradients
  (no 0.99 alpha clamp, no "skip-if-T-would-drop" early stopping), Gaussians not rendered if
  ‖q‖²<1e-8 or |Σ2D|<1e-6, degenerate-quaternion pruning during densification, persistent
  densification-info buffer, optimizer step every iteration (fix of the 29855-step bug),
  no colour clamp during training.
- **C2 Surveyed improvements consolidated** (§3.2, all from prior work, integrated):
  (a) tight rectangular opacity-aware AABBs (StopThePop/Speedy-Splat/AdR-Gaussian),
  (b) load-balanced exact tile-based culling (StopThePop), (c) separate depth / tile sorting
  (Splatshop), (d) per-Gaussian backward (Taming-3DGS) **with the own shared-memory
  improvement** (supp. "Per-Gaussian Backward"), (e) fused activations + separate SH buffers.
- **C3 Own fused Adam** (§3.3 "Parameter Updates"): CUDA Adam kernel matching
  `torch.optim.Adam` (fmaf, fast math).
- **C4 Locality-preserving densification** (§3.3): Morton (z-order) sort of all Gaussian
  parameters + optimizer state every 5000 iterations while densifying (≤15000).
- Optional, NOT in the headline row: **fused backward+optimizer** (§3.3, `FasterGSFused` branch,
  Tab. 3 "+ fused updates"), skip-invisible updates (Taming), SH-degree-0; 4D extension (§3.4).
- Optional borrowed features in the `main` branch, all **off by default** and not Faster-GS
  contributions: Mip-Splatting 3D filter / 2D mip filter, 3DGS-MCMC, Speedy-Splat pruning,
  AbsGS/GOF gradient, PPISP, random background, random init with carving.
- **Borrowed and always on in the `main` CUDA kernels**: warp-level culling in the blend kernel
  (Yang et al., "Warp-Level Culling…", HPG 2026; 16×12 tiles split into 8×4 sub-tiles, each warp
  skips Gaussians whose screen bounds miss its sub-tile). Lossless by construction (bounds are the
  opacity-aware α≥1/255 extents); it is a speed feature only and is labelled as borrowed.

## 2. Equations (paper numbering is \cref labels; 3DGS recap, §2)
- Projection (eq:mean_projection): $\mu_{2D} = (f_x\hat\mu_x/\hat\mu_z + c_x,\; f_y\hat\mu_y/\hat\mu_z + c_y)$.
- (eq:cov_projection): $\Sigma_{2D} = J W_{1:3,1:3}\Sigma W_{1:3,1:3}^\top J^\top$,
  $J=\begin{pmatrix} f_x/\hat\mu_z & 0 & -f_x\hat\mu_x/\hat\mu_z^2\\ 0 & f_y/\hat\mu_z & -f_y\hat\mu_y/\hat\mu_z^2\end{pmatrix}$;
  code clamps $x/z, y/z$ to the 1.3× frustum in $J$ and adds dilation 0.3 to the diagonal.
- (eq:alpha_computation): $\alpha = o\cdot e^{-\frac12 (x-\mu_{2D})^\top\Sigma_{2D}^{-1}(x-\mu_{2D})}$, skip if $\alpha<1/255$.
- (eq:alpha_blending): $C=\sum_i \alpha_i T_i c_i + T_N c_{bg}$, $T_i=\prod_{j<i}(1-\alpha_j)$;
  code stops after the fragment that brings $T<10^{-4}$; colours $c=\max(\mathrm{SH}(d)+0.5, 0)$.
- Opacity-aware AABB half-extents (§3.2 "Splat Bounding"):
  $r_x=\sqrt{2\ln(255\,o)\,\Sigma_{2D,11}}-0.5$, $r_y=\sqrt{2\ln(255\,o)\,\Sigma_{2D,22}}-0.5$ (code).
- Densification statistic (code, kernels_backward.cuh:227): count += 1 for every Gaussian that
  touches ≥1 tile; grad += ‖0.5·(W·∂L/∂μx, H·∂L/∂μy)‖ (NDC-scaled, as 3DGS).
- Adam (adam.cu): $m=\beta_1 m+(1-\beta_1)g$, $v=\beta_2 v+(1-\beta_2)g^2$,
  $p \mathrel{-}= \frac{lr}{1-\beta_1^t}\, m/(\sqrt{v}/\sqrt{1-\beta_2^t}+\epsilon)$.
- Morton code (NeRFICG `morton_encode`): normalise means by the bbox min and the largest bbox
  side, saturate to [0,1], quantise to 21 bits (×(2²¹−1), truncation), interleave x|y<<1|z<<2;
  order = argsort(code).

## 3. Hyper-parameters (all = 3DGS, supp. §A; code `Trainer.py` defaults)
| name | value | source |
|---|---|---|
| iterations | 30000 | Trainer.py |
| densify start / end / interval | 600 / 14900 / 100 (iterations before which ADC runs) | Trainer.py (comment: equals official 3DGS behaviour) |
| grad threshold | 2e-4 | supp. §A, Trainer.py |
| percent dense | 0.01 | Trainer.py |
| prune min opacity | 0.005 (logit compare, strict <) | Trainer.py `adaptive_density_control(…, 0.005, …)` |
| prune large | max scale > 0.1·extent, only when iteration > 3000 | Model.py |
| screen-size prune | none | Model.py (3DGS max_radii2D prune dropped) |
| opacity reset | every 3000 in [3000, 14900], clamp to 0.01, Adam moments zeroed | Trainer.py, adam_utils |
| extra reset at 500 | only with non-black background | Trainer.py |
| Morton ordering | every 5000 iterations, end 15000 (iterations 0, 5000, 10000, 15000) | Trainer.py |
| SH degree | 3, +1 every 1000 iterations | Trainer.py |
| loss | 0.8 L1 + 0.2 D-SSIM (fused-ssim, same padding) | Loss.py, supp. §A |
| Adam | β=(0.9,0.999), ε=1e-15 | Model.py, supp. §A |
| lr means | 1.6e-4·extent → 1.6e-6·extent, log-linear over 30000 | Model.py |
| lr sh0 / shN | 2.5e-3 / 1.25e-4 | Model.py |
| lr opacity | **0.025** (new 3DGS default; paper unifies it for all methods) | supp. §A |
| lr scale / rot | 5e-3 / 1e-3 | Model.py |
| init | SfM points, opacity 0.1, scale = sqrt(mean sq. 3-NN dist), SH0 from colour | Model.py |
| extent | 1.1 · max camera distance from the camera centroid | Trainer.py |
| near / far | 0.2 / 10000 | config yaml |
| background | black | config yaml |
| after training | prune opacity < 1/255 and degenerate quats; Morton sort | Model.training_cleanup |
| rasterizer constants | dilation 0.3, α_min 1/255, T_min 1e-4, det_min 1e-6, tiles 16×12 | rasterization_config.h |

## 4. Schedules
SH +1 at 1000/2000/3000; ADC every 100 in [600, 14900]; reset 3000/6000/9000/12000 (ADC first
at the same iteration, then Morton, then reset — callback priorities 100/99/90); Morton at
0/5000/10000/15000; densification info re-zeroed after each ADC; means LR log-linear decay.

## 5. Builds on
3DGS (Kerbl et al. 2023) with the current official hyper-parameters. Everything in 3DGS training
is kept except: no colour clamp in training, optimizer step every iteration, no screen-size prune.

## 6. Protocol
Mip-NeRF 360 (9 scenes, pre-downscaled images_4 / images_2), Tanks&Temples (truck, train),
Deep Blending (drjohnson, playroom); every 8th image test; 30k iterations; black background;
fused-ssim in training for all methods; LPIPS on [−1,1] (net not stated; NeRFICG uses VGG by
default — assumption); image metrics averaged over **5 runs**; RTX 4090; training images in VRAM.
Only **dataset averages** are reported (no per-scene quality tables; supp. Tab. 5 is per-scene
inference FPS only).

## 7. Ambiguities
- LPIPS network not stated (assume VGG, as the 3DGS protocol).
- Main-branch kernels include warp-level culling (HPG 2026) and 16×12 tiles; the paper's testbed
  may have used 16×16 tiles and no sub-tile culling. Both are lossless speed features.
- Paper results average 5 runs; bonsai alone varies up to 0.5 dB between runs.
- Fused backward+optimizer is a separate branch (optional row in Tab. 3), not the headline.
