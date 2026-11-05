# Revising Densification in Gaussian Splatting — implementation spec

Rota Bulò, Porzi, Kontschieder (Meta Reality Labs Zurich), ECCV 2024, arXiv 2404.06109.
Source: arXiv LaTeX `source/latex/main.tex` (main paper + supplementary in one file).
**Official code: none released** (papers_100.json `code_url` empty; no repo linked in the paper).
Everything below is re-derived from the paper text.

## 1. Method summary
- Replace the 3DGS gradient-magnitude densification score τ_k with an **error score** E_k: a
  per-pixel auxiliary error (SSIM-based) is distributed onto primitives with their
  alpha-compositing weights ω_k(u), summed over pixels, and max-pooled over the views seen
  between two ADC runs. Primitives with E_k > 0.1 are candidates for clone/split.
- **Growth control**: a global maximum number of primitives (set to the baseline's final count)
  and a per-run budget of 5 % of the current number of primitives; when more candidates qualify,
  keep the highest-E_k ones.
- **Opacity correction on clone**: cloned primitive and its source get α̂ = 1 − √(1 − α);
  split keeps the opacity.
- **Alternative to opacity reset**: no hard reset; after each densification run every opacity is
  decreased by 0.001; plus a residual-transmittance regulariser (weight 0.1) on the mean of
  (1 − accumulated alpha).
- ADC runs for 27k of 30k iterations (90 %) instead of stopping at 15k.

Contributions:
- **C1** Error-based densification score (Sec. 3.2 "Error-based densification").
- **C2** Opacity correction after cloning (Sec. 3.3).
- **C3** Primitive growth control: global cap + per-run fraction, top-score selection (Sec. 3.4).
- **C4** Alternative to opacity reset: per-run opacity decrement + residual-transmittance loss
  (Sec. 3.5).
- **C5** (protocol) ADC active until 27k iterations (Sec. 4.2).

## 2. Equations (verbatim from main.tex)
Alpha-compositing weights (Sec. 2):
```latex
\omega^\pi_k(\vct u)\coloneqq\alpha_k\mathcal G^\pi_k(\vct u)\prod_{j=1}^{k-1}\left(1-\alpha_j\mathcal G^\pi_j(\vct u)\right).
```
Per-primitive, per-view error (Sec. 3.2, unnumbered):
```latex
E^\pi_k\coloneqq \sum_{\vct{u}\in\text{Pix}} \mathcal E_\pi(\vct u)w^\pi_k(\vct u)
```
Densification score, max over views π∈Π seen between two ADC runs:
```latex
E_k\coloneqq \max_{\pi\in\Pi} E^\pi_k.
```
Implementation via auxiliary loss with zero-initialised, never-updated scalar e_k:
```latex
L^\mathtt{aux}_\pi\coloneqq \sum_{\vct u\in\text{Pix}} \cancel\nabla[\mathcal E_\pi(\vct u)]\,\mathcal R[\pi, \Phi_\mathtt{ERR}](\vct u),\qquad
\frac{\partial L^\mathtt{aux}_\pi}{\partial e_k}=\sum_{\vct{u}} \mathcal E_\pi(\vct u)\omega^\pi_k(\vct u)=E^\pi_k
```
Opacity correction (Sec. 3.3): solve (1−α) = (1−α̂)² →
```latex
\hat\alpha\coloneqq1-\sqrt{1-\alpha}
```
and for every pixel `1-α_k G ≥ (1-α̂_k G)^2 > (1-α_k G)^2`.
Residual transmittance regulariser (Sec. 3.5, prose): `λ_T · mean_u T_K(u)`, λ_T = 0.1, where
T_K(u) = Π_k (1 − α_k G_k(u)) = 1 − accumulated alpha.

Variables: 𝓔_π(u) per-pixel auxiliary error ("e.g. Structural Similarity"; the main results use
SSIM, supp. Table "Ours-ℓ1" uses ℓ1); w/ω the compositing weights; Pix = image pixels.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| guiding error 𝓔 | SSIM-based (main results); ℓ1 variant in supp. | Sec. 3.2, supp. Sec. C |
| error threshold | E_k > 0.1 | Sec. 4.2 |
| per-run growth budget | 5 % of current #primitives | Sec. 4.2 |
| global cap | #primitives of the corresponding baseline (median of 5 runs) | Sec. 4.2 |
| opacity decrement | 0.001 per densification run | Sec. 3.5 |
| residual-transmittance weight | 0.1 | Sec. 3.5 |
| ADC stop | 27k iterations (90 %) | Sec. 4.2 |
| iterations | 30k | Sec. 4.2 (3DGS settings) |
| densify start / interval | 500 / 100 (3DGS default; "other hyper-parameters default") | Sec. 4.2 |
| clone/split size threshold | percent_dense 0.01 · extent (3DGS default) | Sec. 3.1, 4.2 |
| split: 2 offspring, scale /1.6, opacity kept | 3DGS default; opacity kept per Sec. 3.3 | Sec. 3.1, 3.3 |
| prune opacity | 0.005 | Sec. 3.1 |
| LRs, SH schedule, loss L1 + 0.2 D-SSIM | 3DGS defaults | Sec. 4.2 |

## 4. Schedules
- Error statistics collected every training step until 27k; ADC run every 100 steps after 500
  (3DGS convention: 600, 700, …, 26 900).
- Opacity decrement after each ADC run (so also stops at 27k).
- No opacity reset.
- Residual-transmittance loss: whole training (not stated otherwise).
- SH degree +1 every 1000 steps (3DGS).

## 5. Builds on
Their own re-implementation of 3DGS (and Mip-Splatting); "Ours, 3DGS" is the target. All 3DGS
training settings (LRs, iterations, resolution, loss) kept; only ADC changed + the transmittance
loss.

## 6. Protocol
Mip-NeRF 360 (9 scenes, 3DGS resolution: images_4 outdoor / images_2 indoor), Tanks&Temples
(truck, train), Deep Blending (drjohnson, playroom); every 8th image test; PSNR/SSIM/LPIPS
("correct" LPIPS, i.e. higher than the 3DGS-code values); averages over 5 runs. Per-scene
numbers only for Mip-NeRF 360 (supp. Table "scene-res"); T&T and DB averages only.
Primitive cap = baseline count → identical model sizes.

## 7. Ambiguities and choices
1. **Error map**: "SSIM" without a formula. Choice: 𝓔(u) = 1 − SSIM(u), SSIM map with the 3DGS
   11×11 σ=1.5 Gaussian window, same padding, averaged over RGB, computed on the rendered (white
   background composited) image vs GT. (1−SSIM)/2 would halve all scores; the 0.1 threshold
   is rarely binding because the 5 % budget binds first (verified in smoke logs).
2. **Order of prune/grow within an ADC run, budget base**: not stated. Choice: prune
   (opacity < 0.005, world-size > 0.1·extent after 3000 steps as in 3DGS), then grow with budget
   min(⌊0.05·N⌋, N_max − N) counted in *net new* primitives (one clone or one split = +1), then
   decrement. Fig. 3: "pruned primitives are immediately replaced by newly spawned ones".
3. **Opacity correction applies to both the source and the clone** ("reducing the opacity of the
   primitives after cloning"); split keeps opacity (explicit, Sec. 3.3 last paragraph — unlike
   gsplat's `revised_opacity` flag which applies to split).
4. **Decrement space**: post-sigmoid opacity, clamped to ≥ 1e-6 before logit.
5. **Big-Gaussian pruning**: 3DGS world-size prune (0.1·extent) kept after the first 3000 steps;
   the Inria screen-size prune is a no-op in official code (TODO-runtime.md) and is not used.
6. **Global cap value**: the paper uses its own baseline's count; we use our gsplat baseline's
   final count per scene (`baseline3dgs-*-30000`).
7. LPIPS / background / clamp numerics: paper's re-implementation unknown (presumably black
   background, Inria rasterizer alpha clamp 0.99).
