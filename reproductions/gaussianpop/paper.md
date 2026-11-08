# GaussianPOP — implementation spec

arXiv 2602.06830 (Lee, Kim, Sasse, Borges, Sánchez, Ryu, Schierl, Hellge; Fraunhofer HHI / SKKU).
Sources: `source/latex/sec/{3_proposed,4_results,suppl}.tex`. **No official code**: project page
fraunhoferhhi.github.io/GaussianPOP says "Coming Soon"; github.com/fraunhoferhhi/GaussianPOP
contains only README.md (re-checked 2026-11-07).

## 1. Summary
- Criterion-based pruning: score each Gaussian by the analytic squared colour error its removal
  would cause, accumulated over all pixels of all training views.
- The error is computed in one forward pass ("render once, compute locally"): per pixel the
  prefix colour P_k and transmittance T_{k+1} are cached; the background behind k, b_{k+1}, is
  recovered from the final colour.
- Prune the lowest-error P% globally; fine-tune.
- Two scenarios: on-training (prune at 15k/20k of 30k or 20k/25k of 40k, Mini-Splatting
  densification / GaussianSpa config) and post-training (pre-trained 30k 3DGS, 8 iterative
  re-quantification cycles, optional 5k fine-tune).

Contributions:
- C1 (§3.2) per-Gaussian removal error ΔSE_k = ||T_k α_k (c_k − b_{k+1})||² (Eq. 4).
- C2 (§3.3, Alg. 1) single-pass computation via cached prefix colour / transmittance, ε=1e-9,
  atomic accumulation into a global Δ_SE buffer over all training views.
- C3 (§3.4, §4.3) simplification framework: global sort, prune lowest P%, iterative
  re-quantification over C cycles (post-training, C=8), fine-tune 5k iterations.

## 2. Equations
Eq. 1: $C_{render} = \sum_{k=1}^{N} T_{k}\alpha_{k}c_{k}$, $T_k=\prod_{j<k}(1-\alpha_j)$.
Eq. 2: $C_{render} = P_{k-1} + T_{k}\alpha_{k}c_{k} + T_{k}(1-\alpha_{k})b_{k+1}$.
Eq. 3: $C_{render}^{\prime} = P_{k-1} + T_{k}b_{k+1}$.
Eq. 4: $\|C_{render}-C_{render}^{\prime}\|^2=\|T_k \alpha_k(c_k-b_{k+1})\|^2$.
Supp.: $b_{k+1} = \sum_{j>k} \frac{T_j}{T_{k+1}}\alpha_j c_j = (C_{render} - P_k)/T_{k+1}$,
$P_k=\sum_{i\le k}T_i\alpha_i c_i$. Alg. 1: `b = (C_render − P_k)/(T_{k+1}+ε)`, ε = 1e-9,
`Δ_SE[ID] += ||(T_k α_k)(c_k − b)||²`.
c_k = "base color" of the k-th Gaussian (view-dependent SH colour of the rendered view; the
3DGS rasterizer's per-Gaussian colour), α_k pixel-specific opacity.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| ε | 1e-9 | §3.3 |
| N_max contributing Gaussians per pixel | 64 (128 no difference) | Supp. Impl. details |
| post-training base | 30k-iteration pre-trained 3DGS | §4.1 |
| post-training cycles C | 8 | §4.1, Tab. 8 caption |
| post-training prune ratio P | 0.7 / 0.8 / 0.9 (Tab. 8); 0.8 for Tab. 3, Fig. 7 | Supp. Tab. 8 |
| post-training fine-tune | 5000 iterations | §4.1 |
| on-training prune steps | [15k,20k] (30k), [20k,25k] (40k) | §4.1 |
| on-training P | 0.75 M360 outdoor, {0.65, 0.7} others | Supp. |
| quantification views | all training views | §3.4 |

## 4. Schedules
Post-training: 0–30k vanilla 3DGS; at 30k: C cycles of (quantify over all train views →
remove a fraction); then 5k fine-tune (30k→35k). Per-cycle amount: for a ratio target P the
paper does not state the split; Tab. 4 splits an error budget B as B/C per cycle; the authors'
(commented-out) supp. text for iterative schedules uses $r_{step}=1-(1-P)^{1/N}$.

## 5. Builds on
Vanilla 3DGS (Inria). Post-training uses an unmodified 3DGS model; fine-tune is ordinary 3DGS
training without densification (LR schedule unspecified).

## 6. Protocol
Mip-NeRF 360 (pre-downsampled images_4 outdoor / images_2 indoor — matches our data), T&T,
Deep Blending; llffhold8 (3DGS standard). Metrics PSNR/SSIM/LPIPS (net unstated; 3DGS uses VGG).
GPU A100. Background not stated (3DGS default black).

## 7. Ambiguities and choices
1. Per-cycle prune split for ratio P with C=8: chosen compound $r=1-(1-P)^{1/C}$ of the
   remaining (keep_c = round(N0 (1−P)^{c/C})), final count exact = (1−P)N0 (Tab. 8 counts are
   exactly (1−P)·N0, e.g. bicycle 0.973M = 0.2·4.87M). Linear split is a config option.
2. N_max=64: interpreted as only the first 64 contributing Gaussians per pixel receive error;
   C_render is the full blend.
3. C_render with background: our models are trained over white; we include T_final·bg in
   C_render (physically the background behind k). Identical for black bg.
4. Fine-tune LR: unspecified. Chosen: continue 3DGS (means LR held at its final value
   lr0·0.01 = 3DGS get_expon_lr_func clamped past max_steps; other LRs constant), fresh Adam
   (as when loading a pre-trained .ply), no densification, SH degree 3.
5. Whether quantification uses all pixels at full training resolution: assumed yes.
6. Post-training per-scene 3DGS reference row is not given in Tab. 8; Tab. 7's 3DGS row
   (bicycle 25.13, counter 29.11) is the closest stated reference; Tab. 5b quotes T&T
   pre-trained 23.62 dB/1.812M.
7. Scout's numbers (bicycle 25.08, bonsai 32.05, counter 29.01) are from Tab. 7
   (on-training, 40k, Mini-Splatting densification) — not post-training.
