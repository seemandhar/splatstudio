# Turbo-GS: Accelerating 3D Gaussian Fitting for High-Quality Radiance Fields

Lu, Dhiman, Srinath, Arslan, Xing, Xiangli, Babu, Sridhar — CVPR 2026, arXiv 2412.13547.
Sources: `source/latex/` (arXiv e-print: `sec/4_method.tex`, `sec/5_experiments.tex`,
`sec/X_suppl.tex`), `source/paper.pdf`.

**Official code**: https://github.com/inspirelt/Turbo-GS, cloned into `source/code/` at commit
`aeddd12` ("init"). It holds **only a README and a teaser image**, with no code and no license.
Every implementation detail below therefore comes from the paper. The paper's own Algorithm 1
(supp.) is the most detailed source for the budget scheduler.

## 1. Method summary
- The method is built on 3DGS ("Turbo-3DGS") or on Scaffold-GS ("Turbo-Scaffold-GS"). It uses
  Taming-3DGS's optimized CUDA backward for speed only (§5.1).
- **C1 Dilated rendering (§4.1)**: each training step renders a chessboard-subsampled image with
  dilation d=2 and a random offset (o_x,o_y) ∈ {0,1}². The 2D covariance is computed with the
  original focal length, low-pass filtered in high-res pixel units, and its radius is divided by
  d. Alpha-blending happens at the high-res pixel positions p = d·p_ds + offset. The loss
  supervises only the sampled pixels.
- **C2 Convergence-aware budget control (§4.2, supp. Alg. 1)**: the Gaussian count is capped by
  a power-law budget B(t), going from N (initial) to M (manual maximum). The exponent α and the
  final budget M_adaptive adapt to the log-log slope of the EMA-smoothed training loss.
- **C3 Position-appearance densification (§4.3)**: Gaussians qualify for densification if the
  positional 2D gradient exceeds τ_pos, or (with probability 20 %) if the colour gradient exceeds
  τ_color = 0.01·τ_pos. They are then cloned or split as in the baseline.
- C4 (optional, †-rows only) sensitivity-based pruning at 60 % (Speedy-Splat). **Not part of
  the plain Turbo-3DGS row.**
- README TL;DR also mentions "better initialization, and batched training". Neither is described
  in the paper. "+Batch" appears only as a commented-out ablation row; its 4K value (26.80)
  equals the Turbo-3DGS 4K-table row. This is unspecified, so it is recorded as a gap.

## 2. Equations (verbatim)
Dilated rendering:
- NDC → downsampled plane: `p_x^{ds} = W^{ds} p_x^{NDC},  p_y^{ds} = H^{ds} p_y^{NDC}`
- radius: `r^{ds} = r/d` (r = 3σ radius of the low-pass-filtered high-res 2D covariance)
- pixel mapping: `p_x = d p_x^{ds} + \frac12 (d-1)`, `p_y = d p_y^{ds} + \frac12 (d-1)`
  (the offsets (o_x,o_y) select which chessboard sub-lattice is used)
- training: d = 2, O = {(i,j)}, i,j∈{0,1}, one random offset per iteration.
- Validation in the paper: merging the four offset renders reproduces the full render at
  PSNR > 70 dB.

Budget (Eq. 5, 6):
- `\alpha = \alpha_\text{base}+\lambda\cdot\tanh(\epsilon)`, with `\epsilon = \alpha_{recent} - \alpha_{history}`, λ = 0.5
- `B(t)=N + \frac{t^{\alpha} - 1}{100^\alpha - 1} (M - N)`
- Supp. Alg. 1: EMA α_ema = 0.1. If len(loss_history) ≤ warmup → α = 1.0. Otherwise:
  - fit `log(ema) ~ α_base·log(steps)`; α_base ← moving average of recent α_base;
  - `rate = -dlog(ema)/dlog(steps)` in the recent window;
  - rate > 0.05 → `M_ad = min(1.1 M_ad, 1.5 M)`; rate < −0.05 → `M_ad = max(0.9 M_ad, 0.5 M)`;
  - `deviation = -rate - α_base`; `α = α_base + 0.5 tanh(deviation)`; clip α to [0.1, 2.0];
  - return `N + (t^α − 1)/(100^α − 1)·(M_ad − N)`.
- Warm-up: 100 steps ("record the loss at each iteration after 100 warm-up steps").

Densification (§4.3): threshold the position gradient with τ_position and the appearance gradient
with τ_color = 0.01 τ_position. The colour branch is active with probability 20 %. "follow
baseline's design to add points" (3DGS: clone/split).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| dilation d | 2 | §4.1 |
| offsets | {0,1}², uniform random per step | §4.1 |
| iterations (3DGS-based) | 30 000 | §5.1 |
| λ | 0.5 | §4.2, supp. Tab. sensitivity |
| α_ema | 0.1 | Alg. 1 |
| warm-up | 100 steps; α = 1 before | §4.2 / Alg. 1 |
| rate thresholds / factors | ±0.05, ×1.1 / ×0.9, range [0.5 M, 1.5 M] | Alg. 1 |
| α clip | [0.1, 2.0] | Alg. 1 |
| τ_position | 0.0002 (3DGS default, implied "baseline's design") | assumption |
| τ_color | 0.01 τ_position = 2e-6 | §4.3 |
| colour-branch probability | 0.2 | §4.3 |
| M | "manually set". Peak #G for Turbo-3DGS (DB 2.82 M, M360 3.64 M) matches Taming-3DGS "big" budgets = 3DGS counts (DB mean 2.80 M) → use Taming `full_eval.py` big_budgets | inferred |
| densification schedule (3DGS-based) | not stated → 3DGS (500–15000, every 100, opacity reset every 3000) | assumption; §5.3 says the 3DGS reset is kept |
| densification schedule (Scaffold-based) | first 3k steps, every 20 after 300 warm-up, 10k steps | §5.1 |
| sensitivity pruning | 60 % (only for † rows) | §4.3 |
| window k, α_base moving-average length, fit period | unspecified | ambiguity |

## 4. Schedules
- Loss recording starts after step 100. Budget fitting happens periodically (at densification
  events).
- Densification and opacity reset follow 3DGS (assumed).
- SH degree +1 every 1000 steps (3DGS).
- Dilated rendering is used for every training step. Evaluation renders at full resolution.

## 5. Builds on
3DGS (the Turbo-3DGS row). Unchanged: initialization (SfM), L1 + 0.2 D-SSIM, Adam LRs, the
position-LR exponential decay, clone/split operators, opacity pruning at 0.005, world-size prune,
and opacity reset. The Taming backward is a speed-only kernel.

## 6. Protocol
- 1K setting: "follow the evaluation strategy of 3DGS" on Mip-NeRF 360 (images_4 outdoor,
  images_2 indoor), Deep Blending, and Tanks & Temples (supp.). llffhold 8.
- Main Table 2 gives averages for Turbo-3DGS: M360 27.39 / 0.814 / 0.205, DB 29.21 / 0.905 / 0.240.
  The paper's 3DGS row is M360 27.45, DB 29.75. GPU A100. LPIPS net not stated.
- Supp. per-scene tables (PSNR/SSIM/LPIPS): the "Turbo-Scaffold-GS" and "Turbo-3DGS" PSNR rows
  are **identical**. Their means (DB (29.85+31.12)/2 = 30.49, M360 9-scene mean 27.43) equal the
  **Turbo-Scaffold-GS** averages of Table 2 (30.48 / 27.43), not the Turbo-3DGS averages
  (29.21 / 27.39). The per-scene numbers are therefore Turbo-Scaffold-GS (anchor/MLP, 10k
  steps). The Turbo-3DGS row is a copy-paste duplicate. **Turbo-3DGS has averages only.**
- 4K setting (Table 1, ablations): MipNeRF-360 4K and EyefulTower. Not reproduced here, since
  we have no 4K data in the native layout.

## 7. Ambiguities and choices
1. **t in B(t)** (revised after review-1 M1): Alg. 1 lists "steps: Total steps" as an input, so
   t = 100·step/total_steps (total 30k), clamped to [1, 100]. Densification stops at 15k, so
   t ≤ 50 and the final budget is B(50) ≈ 0.75–0.8·M_ad ≈ 1.1–1.2·M. That matches the paper's
   peak/M ratios (DB 2.82/2.80, M360 3.64/3.31). The first reading (t over the densify window
   500–15000) let B reach 1.5·M; the cap never bound in smoke-1 (selected = candidates at
   every event). It is kept as config `t_mode: "densify_window"`.
2. **Sign of the deviation**: Eq. 5 (ε = α_recent − α_hist added to α) and Alg. 1
   (deviation = −rate − α_base, which reads as slope_recent − slope_hist when α_base is the fitted
   *slope*) disagree in sign. The prose says "if the convergence speed is lower than the
   expectation, slow down densification". With normalized (t^α−1)/(100^α−1), a larger α means
   slower growth. That behaviour needs α to increase when the recent decay rate falls below the
   historical one, so we use deviation = α_hist − α_recent (Alg. 1 with the slope reading) and
   α_base = mean of historical α_hist (positive exponents).
3. **Colour gradient**: the attribute is not named. Taken as ‖∂L/∂sh0‖ (the DC colour
   coefficient, Inria `features_dc`), averaged over the steps where the Gaussian is visible,
   like the position gradient.
4. **Budget enforcement when candidates exceed B(t) − n**: unspecified. We keep the top
   candidates by max(g_pos/τ_pos, g_col/τ_col).
5. **Window k / α moving average / fit period**: unspecified. We use window = last 500 recorded
   steps, the mean of all historical α_hist fits, and refitting at each densification event.
6. Image size of the dilated render for odd W/H: floor(W/d), floor(H/d).
8. **m1 pixel offset**: Eq. 3 is written p = d·p_ds + ½(d−1), i.e. the block centre with no
   offset. We use p = d·p_ds + o (o ∈ {0,1}², the chessboard sub-lattice). This is the only
   reading consistent with the 4-offset merge validation (> 70 dB) and the dilation figure.
9. **m2 α sign/clip**: under the literal Alg. 1, α_base is the fitted (negative) slope. Then
   α = α_base + 0.5 tanh(·) would clip to 0.1 almost always. We use positive exponents
   (α ≈ 0.1–0.3 observed). Both readings give a front-loaded curve, and the deviation term is
   the same quantity (slope_recent − slope_hist = α_hist − α_recent).
10. **m3 colour branch**: one Bernoulli(0.2) draw per densification event (per event vs per
    Gaussian is unstated). With seed 0 the first firing is event 26 (step 3100); 25/145 events fire.
11. **m4 colour attribute**: the gradient is taken w.r.t. SH-space sh0. A gradient w.r.t. RGB
    would be 1/C0 ≈ 3.5× smaller, which changes what τ_col = 2e-6 selects. The paper does not
    name the attribute.
7. Batched training and "better initialization" (README) are undocumented and not ported.
