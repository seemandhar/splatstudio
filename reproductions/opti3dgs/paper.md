# Opti3DGS — paper spec (arXiv 2503.14475, Farooq, Guillemaut, Hilton, Volino; Univ. of Surrey)

Sources: `source/latex/*.tex` (arXiv e-print), `source/paper.pdf`. **No official code** (the paper
says "code ... will be made publicly available"; GitHub/web search found none; the follow-up
AutoOpti3DGS repo `source/followup_autoopti3dgs` contains only a README/LICENSE).

## 1. Method summary
- Training images are low-pass filtered (mean blur, stride 1, same-size padding) before the 3DGS
  loss; the blur kernel shrinks with a **step schedule** and filtering stops at 12k of 30k iters.
- Coarse images → ADC creates fewer, larger Gaussians early; detail added later (Eq. 3-5).
- Renderer, ADC logic and loss form are unchanged 3DGS; only the target image changes.
- On plain 3DGS ("3DGS+Ours*") three 3DGS hyper-parameters are also changed (§4 Impl. details).

Contributions
- C1 Image frequency modulation: mean low-pass filter of training images (§3.3, Eq. 2).
- C2 Frequency training schedule: step decay of kernel size 15→9→5→3→1, off after 12k (§3.4, Fig. 3, §4).
- C3 Progressive growing pipeline G_n = 3DGS(G_{n-1}, I_n), n=1..5 (§3.5, Eq. 3-5) — this is
  just continuous training across schedule levels (no restart, no LoD export needed for metrics).
- (Setting, not a contribution) 3DGS+Ours*: scaling_lr=0.01, densification_interval=500,
  densify_grad_threshold=0.0001 (§4 Implementation details).

## 2. Equations
Eq. 2: `G[i,j] = \sum_{u=-k}^{k}\sum_{v=-k}^{k} H[u,v] F[i-u, j-v]`, H = uniform kernel
(mean filter, H = 1/K^2 over a KxK window), F input image, stride 1, padding keeps size.
Eq. 3 `G_0 = SfM(I)`; Eq. 4 `G_n = 3DGS(G_{n-1}, I_n)`, `I_n = B(I, K_n, P_n)`; Eq. 5 `G_5 = 3DGS(G_4, I)`.
Loss: unchanged 3DGS `(1-λ) L1 + λ D-SSIM`, λ=0.2, against I_n.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| filter | mean (box) | §3.3, §4, ablation §4.4 |
| initial kernel | 15 | §4 Impl. details |
| schedule | step: K=15 (it ≤1000), 9 (≤4000), 5 (≤8000), 3 (≤12000), 1 after | read off Fig. 3 (purple "Step" curve) |
| stop filtering | 12000 | §4, Fig. 3 caption |
| iterations | 30000 | §4 Metrics |
| scaling_lr (3DGS+Ours*) | 0.01 (3DGS default 0.005) | §4 |
| densification_interval (3DGS+Ours*) | 500 (default 100) | §4 |
| densify_grad_threshold (3DGS+Ours*) | 0.0001 (default 0.0002) | §4 |
| everything else | 3DGS defaults | §4 |

## 4. Schedules
Kernel size per iteration as above (5 levels: 4 filtered + original). 3DGS densification
500→15000, opacity reset every 3000, position-LR decay — unchanged.

## 5. Builds on
3DGS (Inria code); also applied to Compact-3DGS, Reduced-3DGS, Mini-Splatting, Taming3DGS,
EAGLES (Table 1). We reproduce the 3DGS rows only.

## 6. Protocol
Mip-NeRF 360 (9 scenes; images_4 outdoor / images_2 indoor), T&T (truck, train), DB
(drjohnson, playroom), full-res T&T/DB, llffhold 8, 30k iters, H200. Metrics PSNR/SSIM/LPIPS
(net unstated). **Only dataset averages are reported** (13 scenes); no per-scene table.

## 7. Ambiguities
- Exact step boundaries: read from Fig. 3 (drops at ~1000, 4000, 8000, 12000); kernel per level 15/9/5/3.
- Padding mode of the mean filter: unspecified ("adjust the padding to ensure spatial dims don't
  change"). Zero padding would darken borders; we use reflect (= OpenCV `cv2.blur` default
  BORDER_REFLECT_101) as the most likely implementation; configurable.
- Whether the filter is applied to the SSIM term too: the images I_n replace I, so yes (whole loss).
- 3DGS+Ours* is the headline (−58..66 % #G but −0.44/−0.54 dB on Mip/T&T); Table 2 "Ours-Filtering"
  (filter alone, default 3DGS) is the quality-preserving variant. Both are reproduced as methods.
- Whether densification_interval=500 also shifts the opacity-reset or densify_from: assumed not.
