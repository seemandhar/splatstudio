# Plan — Denoising-GS (plan #115), native SplatStudio plugin

Method `denoising-gs`, module `ss_denoising_gs`. Controls: `denoising-gs-mcmc` (all four
contributions off = the paper's 3DGS-MCMC backbone at the same per-scene cap), and
`denoising-gs-quick` (compressed prune/SCR schedule: prune at 1500, SCR at 1600..2000; smoke only).

## Components
| component | kind | factory | notes |
|---|---|---|---|
| gsplat | renderer | built-in `create_renderer` | unchanged 3DGS rasteriser |
| denoising-density | density | `ss_denoising_gs.density:create_density` | MCMC relocation/growth/noise + C1 + C2 + C3 + C4 |
| photometric | loss | built-in | L1 + 0.2 D-SSIM |
| mcmc-reg | loss | `ss_denoising_gs.losses:create_reg` | MCMC λ_o = λ_s = 0.01 L1 on activated opacity / scale, all steps (MCMC code; same formula as reproductions/confidence-densify) |

Density `requires = set()` (MCMC needs no screen gradients). No new fields.

## Mechanism → code (density.after_backward; the runtime has already done the Adam step)
1. Gradients ∇M_t, ∇(log S)_t are captured with one-shot tensor hooks registered in
   `before_backward` (the runtime zeroes `.grad` before `after_backward`).
2. C1 (Eq. msgld): default (review-1) `means -= α λ_t m̂/(√v̂+ε)` (Adam-normalised, v̂ from the means
   Adam); option `raw`: `means -= α λ_t m`. m holds grads ≤ t−1, then `m = β1 m + (1−β1) ∇M_t`.
   λ_t = current means LR (3DGS decay); m is per-Gaussian state kept aligned through
   relocate / add / remove / split (zeroed for relocated and new Gaussians).
3. C2 (Eqs. m3–m5): R from normalised quats, g^G = Rᵀ∇M; j_M = argmax|g^G|, j_S = argmax|∇S|;
   if equal: Δμ_j = −|∇S_j| sign(g^G_j) (default `descent`, Fig. 2; `literal` = +sign as printed);
   `means += β2 R Δμ`; option `dmu_magnitude: adam_step` uses the applied scale step.
4. MCMC (gsplat ops.relocate / sample_add, refine 500–25000 every 100, +5 % up to cap, min opacity
   0.005) and the SGLD noise √(2λτ)ξ = gsplat `inject_noise_to_position(scaler = λ_t · 5e5)`.
5. C3 at `prune_step = total_steps − 5000` (35k → 30k): reload the training images (pattern from
   reproductions/dashgaussian), per view L_v = ||I_v − Î_v||² (sum), autograd ∇ wrt (means, log
   scales), u_i = Σ_v ||∇θ_i L_v||² (= trace F_i = Σσ_j(F_i) for PSD F_i); remove the 10 % lowest u
   (ops.remove).
6. C4 at prune_step + 500·k, k = 1..5: d_i via scipy cKDTree (k=3, excluding self); top 0.05 %
   largest d are split in two with gsplat `compute_relocation` (Eqs. p6/p7, N = 2): source gets the
   new opacity/scale, an identical copy is appended (optimizer moments zero for the copy).
7. Cap: `cap_max: null` → per-scene count of the paper's own MCMC row (Table 2) looked up from the
   `--data` directory name; explicit integer overrides.

## Probes
| probe | contribution | fails if |
|---|---|---|
| momentum-drift | C1 | drift ≠ −αλ m with m an independent EMA over 3 steps (lagged one step); α=0 ablation passes → fail |
| spatial-denoise | C2 | Δμ ≠ independent numpy quaternion→R loop; analytic aligned/misaligned R=I cases |
| fisher-prune (CUDA) | C3 | u ≠ Σ svdvals of an explicit 6×6 Σ_v g gᵀ from per-view autograd; pruned set ≠ lowest-10 %; optimizer misaligned |
| scr-densify (CUDA) | C4 | d ≠ brute-force cdist; selected ≠ sparsest; copy opacity/scale ≠ hand-written Eqs. p6/p7 (N=2) |
| schedule | C3/C4 | prune at total−5000, SCR five times every 500 after; MCMC refine window 500–25000 |

## Unchanged from 3DGS / MCMC
Rasteriser, SH schedule, LRs, L1+0.2 D-SSIM, 3DGS init scales (recipe 3dgs), MCMC init opacity 0.5.

## Fidelity gaps
1. Appendix (all implementation details) missing: prune/SCR timing, Fisher view set, τ — guessed.
   Expected effect small (paper gain over MCMC is only +0.02…+0.06 dB per scene).
2. "Denser and noisier input SfM point clouds on Mip-NeRF 360" — unspecified; we use standard SfM.
3. ∇S read as log-scale gradient; Δμ sign taken from Fig. 2/text (descent), not Eq. m4 as printed.
4. Runtime: white background, random-with-replacement views, gsplat vs Inria rasteriser,
   gsplat MCMC vs official 3dgs-mcmc (minor: noise LR uses post-decay λ_t).
5. MCMC caps: from the paper's MCMC row (M360 only); T&T/DB caps are not in the paper.
6. Momentum: literal raw drift is a no-op (~1e-5 of the Adam step); default is our Adam-normalised
   interpretation (needed for Table 3(a)'s α sensitivity). Interpretation, not stated in the paper.
7. Paper MCMC row is 30k steps vs Denoising-GS 35k; our control runs 35k with the same cap.
8. Prune/SCR schedule tied to total_steps (35000); ss-train --steps must match (asserted).
