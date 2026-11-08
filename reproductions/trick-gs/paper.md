# Trick-GS — A Balanced Bag of Tricks for Efficient Gaussian Splatting

arXiv 2501.14534 (ICASSP 2025, Samsung R&D UK; Armagan et al.). LaTeX in `source/latex/`, PDF in
`source/paper.pdf`. **No official code** (no link in paper / project page; GitHub API search for
"trick-gs" and "trick gaussian splatting" returns nothing relevant, 2026-11-07). The method is an
assembly of published tricks, so where the paper is silent we follow the official code of the
paper each trick is taken from (cited per item; their native ports in `reproductions/` are reused).

## 1. Method summary
- 3DGS backbone (unchanged representation, 59 params/Gaussian, L1 + 0.2 D-SSIM).
- Training-time tricks: progressive blur + progressive resolution of the training images,
  progressive Gaussian low-pass scale with ABE split and a 20 % lowest-error SfM init,
  learned Gaussian volume masks (prune), learned per-band SH masks, significance-score pruning,
  batched SH updates + separable SSIM, late re-densification.
- Post-processing: all parameters stored in FP16 after training.
- Headline rows **"Trick-GS"** (Tables 1–2) use **all eight tricks**: BL, DS, SG, GM, AT, SHM, DE,
  SC (Table 3 caption: "Our full model Trick-GS uses all the tricks while Trick-GS-small uses all
  but DE and SC"). We implement that combination.

Contributions (paper §3.2 "Tricks"; abbreviations of Table 3):
- C1 **GM** Pruning with volume masking (§3.2.1, from Compact-3DGS [Lee et al. 2024]).
- C2 **SG** Pruning with significance of Gaussians (§3.2.2, Eq. 4, from LightGaussian).
- C3 **SHM** SH-band masking (§3.2.3, Eq. 5, from RDO-Gaussian [Wang et al. 2024]).
- C4 **BL** Progressive training by blurring (§3.2.4a, from EAGLES).
- C5 **DS** Progressive training by resolution (§3.2.4b, Eq. 6, from EAGLES / Spec-Gaussian).
- C6 **SC** Progressive training by scales of Gaussians (§3.2.4c, RAIN-GS low-pass + ABE split,
  20 % lowest-error SfM init).
- C7 **AT** Accelerated training (§3.2.5, from Taming 3DGS: SH rest updated every 16 its,
  separable/fused SSIM).
- C8 **DE** Late densification 20K–20.5K (§4.2 implementation details).
- (post) FP16 storage of all parameters after training (§4.1).

## 2. Equations (verbatim from `methodology.tex`)
- Eq. 3 (volume mask): `M_n = \operatorname*{sg}(\mathds{1}(\sigma(m_n) > \epsilon) - \sigma(m_n)) + \sigma(m_n)`;
  applied to opacity α and (activated, non-negative) scale s. Pruned at densification steps and
  every k_m its after densification. L_m: "see [Compact-3DGS]" → `L_m = \frac{1}{N}\sum_n \sigma(m_n)`.
- Eq. 4 (significance): `GS_j = \sum_{i=1}^{MHW} \mathbbm{1}(G(X_j), r_i)\cdot\alpha_j\cdot\gamma(\Sigma_j)`;
  volume normalised by the volume of the 90 % largest (sorted) Gaussian. Pruning a preset
  percentile, k_sg times with a decay factor.
- Eq. 5 (SH mask): `M_{sh_i}^l = sg(\mathbbm{1}(\sigma(m^l_{sh_i}) > \epsilon_{sh}) - \sigma(m^l_{sh_i})) + \sigma(m^l_{sh_i})`,
  `\hat c_i^l = M^l_{sh_i} c_i^l`, `c_i^l \in R^{(2l+1)\times 3}`; "masking loss for each degree
  weighted by its number of coefficients" → RDO Eq. 8:
  `L_sh = \frac1N\sum_i\sum_{l=1}^3 \frac{2l+1}{15}\sigma(m^l_{sh_i})`.
- Eq. 6 (progressive resolution): `res(i) = \min(\lfloor res_s + (res_e - res_s)\cdot i/\tau_{res}\rceil, res_e)`.
- Low-pass: `\Sigma' \to \Sigma' + sI`, `s = HW / (9\pi N)`, lower bound 0.3 (RAIN-GS).
- Eq. 7 (loss): `L = (1-\lambda_{SSIM})L_1 + \lambda_{SSIM}L_{D-SSIM} + \lambda_m L_m + \lambda_{sh} L_{sh}`.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | §4.1 |
| λ_SSIM | 0.2 | 3DGS |
| DS start | 1/8 of full resolution, back to full at τ_res = 19500 | §4.2 |
| BL kernel | 9×9 (radius 4), σ = 2.4, lowered every k_b = 100 its until 19500; lowered *before* densify; when DS active the kernel is scaled by the current DS scale | §4.2 |
| SC (ABE split + progressive low-pass) | first 10K its | §4.2 |
| low-pass s | max(HW/(9πN), 0.3) (RAIN-GS: also ≤ 300, recomputed at it 1 and every 1000) | §3.2.4c; RAIN-GS code |
| SfM init | 20 % lowest-reprojection-error points | §3.2.4c |
| densification | standard 3DGS every 100 its until 15K (from 500, grad 2e-4, opacity reset every 3000) | §4.2, 3DGS |
| DE | densification re-enabled every 100 its between 20K and 20.5K | §4.2 |
| mask LRs | Gaussian 0.5, SH 0.05 | §4.2 |
| λ_m = λ_sh = ε_m | 0.05 | §4.2 |
| ε_sh | 0.1 | §4.2 |
| mask prune | every densification step; every 500 its after densification stops | §4.2 |
| SG | 6 rounds until ≈22K, first rate 60 %, decay 0.7 (rate_k = 0.6·0.7^k) | §4.2 |
| SG round steps | 5000, 10000, 15000, 20000, 22000, 23000 | read off Fig. 3 (#G of bicycle: drops of 58/42/31/21/15/10 % at these steps) |
| AT | SH bands ≥1 Adam step every 16 its; separable SSIM | §3.2.5, Taming 3DGS |
| storage | FP16 for every parameter after training | §4.1 |

## 4. Schedules
1: init (20 % SfM), low-pass s; 500–15000 densify/100 (+mask prune); ≤10000 ABE split, s every
1000; 3000/6000/9000/12000 opacity reset; 5000/10000/15000/20000/22000/23000 significance prune;
≤19500 blur/downsample; 15000–30000 mask prune every 500; 20000–20500 re-densify every 100;
30000 FP16.

## 5. Builds on
3DGS (Inria code); RAIN-GS densification/low-pass, LightGaussian significance, Compact-3DGS
mask, RDO-Gaussian SH mask, EAGLES progressive transforms, Taming 3DGS SH batching. 3DGS LRs,
means LR decay, SH degree schedule (+1 every 1000), split/clone rules are kept.

## 6. Protocol
M360 (9 scenes; 3DGS images_4/images_2), T&T (truck, train), DB (drjohnson, playroom); every
8th image test; 30K its; PSNR/SSIM/LPIPS (3DGS protocol, LPIPS net not stated → VGG as 3DGS).
Only dataset **averages** are reported (Tables 1–2); per-scene only bicycle ablation (Table 3).
All Trick-GS and "retrained 3DGS" numbers are the authors' runs on an RTX 3090.

## 7. Ambiguities (and choices)
1. DS: Eq. 6 is linear in resolution, text says "logarithmic decay" → Eq. 6 (linear in pixels;
   `res_mode="log"` available but not default).
2. BL decay law not given ("decay factor") → kernel radius/σ scale with b = 1 − i_q/τ (i_q =
   iteration floored to k_b=100), times the current DS scale; no blur once the radius rounds to 0.
   EAGLES-style torchvision Gaussian kernel with reflect padding; applied to the target only.
3. Target downsampling filter: EAGLES `F.interpolate(bilinear, align_corners=False)` (no antialias).
4. Significance: Eq. 4 copied from LightGaussian; implemented as the LightGaussian official code
   (`imp = Σ α·T` over hit pixels, `(V/V_90)^0.1`), computed over all training views at the
   current training resolution/low-pass; prune `v ≤ sorted(v)[int(p(N−1))]`.
5. SG steps: "6 times until 22K" vs Fig. 3 drops at 5K/10K/15K/20K/22K/23K → Fig. 3 steps.
6. Mask inits not given: Gaussian mask logit 1.0 (Compact-3DGS), SH mask logit 0.0 (RDO).
   SH-mask start not given: masked-SH counts in Fig. 3 appear only after ≈3500, and a band masked
   before its coefficients are trained can never recover (c=0 ⇒ zero mask gradient) → SH masking
   from step 3500.
7. L_m over all Gaussians (Compact-3DGS), L_sh mean over all Gaussians (RDO Eq. 8).
8. ABE split: RAIN-GS code (third Gaussian at 0.3·extent·μ, B0 = origin); split divisor stays 3DGS
   1.6 (RAIN-GS's 1.4 belongs to its own warm-up recipe, not cited by Trick-GS). Means LR: 3DGS.
9. Low-pass H, W: the size of the (downsampled) training render.
10. Re-densification: grad statistics restart at 20000, densify at 20100…20500; no opacity reset.
11. FP16: params rounded to half precision at the last step (evaluation uses the FP16 model).
