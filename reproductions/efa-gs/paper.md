# EFA-GS — Low-Frequency First: Eliminating Floating Artifacts in 3D Gaussian Splatting

Wang, Zhou, Li, Quan, Qin. Cyberworlds 2025 (Best Paper Honorable Mention). arXiv 2508.02493.
Official code: https://github.com/jcwang-gh/EFA-GS (commit `57f330f`, cloned to `source/code/`,
Inria Gaussian-Splatting license: non-commercial research use; this port cites it).
The repo has two variants: `3DGS/` (EFA-GS(3DGS)) and `Mip-Splatting/` (EFA-GS(Mip)). **The paper's
default and every per-scene table use EFA-GS(Mip)** (§5.1 "The default setting is EFA-GS(Mip)
using Mip-splatting as baseline"; Table resofmip avg 27.94 = mean of the per-scene row).

## 1. Method summary
- Diagnoses floaters as *under-optimised* Gaussians that shrank before learning low frequencies.
- Replaces every `r`-th (r = 2) 3DGS densification step with the **LFCF** step: Gaussians whose
  (accumulated) view-space gradient is above threshold and has *increased* since the previous LFCF
  step are **enlarged**; those whose gradient decreased are **shrunk and split**.
- Depth-based strategy: enlarging factor interpolated between c_max=1.5 and c_min=1 by the
  normalised log sampling interval (deeper → smaller factor).
- Scale-based strategy: volume-preserving per-axis factors (grow shortest axis, shrink others).
- Appendix C strategies: (1) LFCF only every r densifications, (2) factor annealing to 1 by
  15k, (3) probabilistic splitting (deeper → more likely to split).
- Built on Mip-Splatting (3D filter + 2D mip filter) **including the GOF-style absolute-gradient
  densification** that the later official Mip-Splatting repo uses (see §5).

Contributions:
- C1 LFCF algorithm (§4.3, Alg. 1): gradient-ascent test → expand / shrink+split.
- C2 Depth-based strategy (§4.3, Eq. c(i)).
- C3 Scale-based strategy (§4.3, ∏c_i = 1).
- C4 Strategies 1–3 (App. C: LFCF interval r, annealing c = exp(k x^n), split probability η = 1−θ).

## 2. Equations (paper) and the code that implements them
Sampling rate (Mip-Splatting Eq. 8, paper Eq. 1–2):
`\nu_i \propto \max_k (1_k(p_i) f_k / d_k)`, `T = 1/\nu`.
Code `compute_3D_interval`: T_i = min visible camera-z / max focal_x (same visibility rule as
`compute_3D_filter`: z > 0.2, projection inside the image ±15 %; unseen → max visible depth).

Depth-based factor (paper): `c(i) = \theta(i) c_{max} + (1-\theta(i)) c_{min}`.
Code (`densify_lff`): θ = minmax-normalised **log** T over all Gaussians;
`c = θ·c_min + (1−θ)·c_max` — i.e. deep Gaussians (large T, θ→1) get c_min, matching the text
("deeper Gaussians ... lower enlarging factors"); the paper's formula has the labels swapped. Code wins.

Annealing (App. C Str. 2): paper `c = \exp(k x^n)`, k = ln c(i), x = (it−500)/(15000−500).
Code (`calculate_training_percent_powered`): `log c = ln c(i) · [ (1−x)^pow (1−lb) + lb ]`,
pow = 1, lb = ln(c_last)/ln(c_max) = 0 (c_last = 1) → **linear decay of log c from ln c(i) at
500 to 0 at 15000**. (Paper's x^n would *grow*; the code decays — code wins.)

Scale-based strategy (code, diffscale=True), per Gaussian with log-scales sorted:
- enlarge: Δlog s = log c · (+1 smallest axis, −1/3 middle, −2/3 largest) (Σ = 0, volume kept).
- shrink (split candidates): Δlog s = −½(log c + ln 2) · (1 largest, ½ middle, 0 smallest).
Without diffscale: enlarge Δ = log c on all axes; shrink Δ = −½(log c + ln 2) on all axes.

LFCF selection (code `densify_lff`, with Mip GOF-densify abs gradients):
- g = mean ‖∂L/∂μ2D‖ (NDC) since the last LFCF step; g_abs = mean Σ_pix(|∂L/∂x|+|∂L/∂y|) (NDC)
  since the last LFCF step (`lff_*` accumulators, 200 steps).
- Q = quantile(g_abs, 1 − mean(g ≥ τ)), τ = 2e-4. selected = (g ≥ τ) ∨ (g_abs ≥ Q).
- P = selected at the previous LFCF step (all True initially; Gaussians created since are True).
- enlarged = selected ∧ ¬P; for selected ∧ P: decent = g ≤ |g_prev − tol| ∨ g_abs ≤ |g_abs_prev − tol|
  (tol = 1e-5); enlarged |= ¬decent; shrink = decent.
- Scale edits are **in-place on the log-scale parameter; Adam moments untouched** (`set_attributes`).
- Split probability: thr = θ(ub − lb) + lb, ub = 1, lb = 1 − (it−500)/14500 (train.py overrides the
  CLI value); a shrunk Gaussian is split iff U[0,1] ≤ thr. Shrunk-but-not-split Gaussians stay shrunk.
- Split (N = 1): parent kept; one child at μ + R·(s ⊙ ε) with **the parent's (already shrunk) scales**
  and the same opacity/SH/rotation; NaN stds → 1e-3.
- Then: g_prev ← current g (children 0), P ← selected (children True), lff accumulators reset.
- Then the standard prune: σ(o) < 0.005 (unfiltered opacity), world-size > 0.1·extent after 3000;
  screen-size prune is a no-op (max_radii2D zeroed in the postfix); NaN Gaussians pruned.

Non-LFCF densification steps (odd hundreds) = Mip-Splatting GOF densification: clone if
(g ≥ τ ∨ g_abs ≥ Q) ∧ max s ≤ 0.01·extent, split (N = 2, s/1.6) if ... ∧ max s > 0.01·extent, using the
100-step accumulators; new Gaussians get P = True and zero LFCF stats; same prune.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| c_max (`init_scaling_multiplier_max`) | 1.5 | paper §5.1; `scripts/run_mipnerf360.py` (train.py default 2.0) |
| c_min (`init_scaling_multiplier_min`) | 1.0 | paper; script |
| c_last (`last_scaling_multiplier_max`) | 1.0 | train.py default |
| pow (n) | 1.0 | train.py default |
| r (`interval_times`) | 2 (LFCF every 200 steps) | paper App. C; script |
| diffscale | True | script |
| tolerance | 1e-5 | train.py default |
| splitting_ub | 1.0 | train.py default |
| splitting_lb | 1 − (it−500)/14500 (overrides CLI) | train.py |
| split_multiplier | 2.0 (ln 2 in shrink) | gaussian_model.py |
| N (LFCF split children) | 1 | densify_and_prune_lff default |
| τ grad threshold | 2e-4 | arguments |
| densify 500 < it < 15000, every 100 | | arguments/train.py |
| opacity reset every 3000 (filter-aware), min opacity 0.005, percent_dense 0.01 | | Mip-Splatting |
| kernel_size (2D mip filter) | 0.1 | script |
| 3D filter coef 0.2, recompute after each densify and every 100 steps 15000 < it < 29900 | | train.py |
| LRs: 3DGS defaults (pos 1.6e-4→1.6e-6, f 2.5e-3, opacity 0.05, scale 5e-3, rot 1e-3) | | arguments |
| iterations | 30000 | arguments |
| background | black (Mip), white (3DGS variant) | scripts |

## 4. Schedules
- 500 < it < 15000, it % 100 == 0: densify. it % 200 == 0 → LFCF; else GOF-densify (clone+split).
  First LFCF at 600, last at 14800. Annealed log c → 0 at 15000.
- Opacity reset (filter-aware) at 3000, 6000, 9000, 12000.
- 3D filter recomputed after every densify and every 100 steps in (15000, 29900).

## 5. What it builds on
Mip-Splatting at the official repo's **later** (post-paper, GOF) state: densification uses both the
signed view-space gradient and the absolute gradient (rasterizer writes Σ|∂L/∂x|+|∂L/∂y| into
grad[:, 2]), with the quantile threshold Q. Our `reproductions/mip-splatting` plugin follows the
paper-era commit 1ecf7d5 (signed gradient only), so it is *not* the exact EFA base; the plugin here
re-implements the GOF densification and exposes a control `efa-gs-base` (LFCF off) = the EFA code
path with interval_times = ∞ ("Mip-splatting (i=∞)" in Tab. speed).
Unchanged from 3DGS: L1 + 0.2 D-SSIM, Adam, SH schedule, LR schedule.

## 6. Protocol
Mip-NeRF 360, 9 scenes, `-r 4` outdoor / `-r 2` indoor from full-res images (PIL resize, like
Mip-Splatting's scripts) → our `mipnerf360-pilr` dataset; `--eval` llffhold 8; 30k; LPIPS-VGG
(lpipsPyTorch), black background. A100 GPU. T&T: 6 *advanced* scenes (not available). RWLQ: own data.

## 7. Ambiguities / notes
- Paper formula for c(i) and the annealing x^n disagree with code; code followed (above).
- README reports V100 vs A100 differences (±0.05 dB avg) from strict gradient comparisons; the
  tolerance version (1e-5, what the code does now) gives 27.90 (V100) vs paper 27.94.
- The paper's own Mip-Splatting row (27.92 avg; garden 28.04) is higher than the Mip-Splatting paper
  (27.79 avg; garden 27.76) — consistent with the GOF densification in the base code.
- On Mip-NeRF 360 the EFA gain over Mip-Splatting is tiny (+0.02 avg; per scene −0.10…+0.22) —
  within seed noise; matching absolute PSNR is the realistic target.
