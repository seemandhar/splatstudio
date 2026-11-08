# Turbo-GS (Turbo-3DGS row) — native SplatStudio plan

Method `turbo-gs`, module `ss_turbo_gs`. The target is the Turbo-3DGS 1K row of Table 2. The
supp. per-scene table is Turbo-Scaffold-GS (see paper.md §6), so we compare the **Deep Blending
2-scene average 29.21** (paper 3DGS: 29.75, so Δ = −0.54 dB).

## Components
| component | kind | factory | requires | provides | config |
|---|---|---|---|---|---|
| `turbo-dilated-renderer` | renderer | `ss_turbo_gs.renderer:create_renderer` | state.gaussian3d, camera.pinhole | render.rgb, render.depth.expected_z, gradient.screen.signed, render.dilated_subset, render.color_coefficients | dilation 2, eps2d 0.3, seed 0 |
| `photometric` | loss | `ss_turbo_gs.losses:create_photometric` | render.rgb, render.dilated_subset | training.loss_value | ssim_weight 0.2 |
| `turbo-density` | density | `ss_turbo_gs.density:create_density` | gradient.screen.signed, render.color_coefficients, training.loss_value | — | see DEFAULTS |

No new FieldSpecs. Density state, kept aligned through gsplat ops: `grad2d`, `count`, `colgrad` [N].

## Mapping
- **C1 dilated rendering** (`renderer.py`). When grad is enabled (training), draw an offset
  (o_x,o_y) ∈ {0,1}² from a private RNG and render floor(W/d)×floor(H/d) with
  `K' = [fx/d, 0, (cx−o_x−½)/d+½; 0, fy/d, (cy−o_y−½)/d+½]` and `eps2d = 0.3/d²`.
  - Ds pixel j has its centre at high-res d·j+o_x+½, so the projected mean, the covariance
    (original focal length, filter 0.3 in high-res px²) and the radius (= r/d) are exactly §4.1.
    gsplat's tile grid is used in ds space, as in the paper.
  - Evaluation (no grad) renders the standard full image with eps2d = 0.3.
  - The loss supervises `image[o_y::d, o_x::d]`.
- **C2 budget** (`budget.py`, a pure CPU class). Losses are recorded after the 100-step warm-up
  with EMA 0.1. At each densification event:
  - fit α_hist = −slope of log ema vs log step (all records), and α_base = mean of all α_hist;
  - rate = −slope over the last `window` (500) records;
  - M_ad rule ±0.05 → ×1.1/×0.9 within [0.5M, 1.5M];
  - α = clip(α_base + λ·tanh(α_base − rate), 0.1, 2), λ = 0.5;
  - B(t) = N + (t^α − 1)/(100^α − 1)(M_ad − N), with t = clamp(100·step/total_steps, 1, 100)
    (total_steps = 30000; Alg. 1 input "steps"; review-1 M1). The alternative densify-window
    mapping is config `t_mode: "densify_window"`. Each refine row logs peak #G and
    selected vs candidates.
  - Before warm-up, α = 1.
  - M = Taming big budgets keyed by the SfM count, or config `budget`.
- **C3 densification** (`density.py`). 3DGS schedule (refine 500–15000 / 100, reset 3000).
  - Candidates are g_pos > 2e-4, OR (Bernoulli 0.2 per event) g_col > 2e-6, where
    g_col = mean over visible steps of ‖∂L/∂sh0‖.
  - If there are more than B(t) − n candidates, keep the top by max(g_pos/τ_pos, g_col/τ_col).
  - Clone small, split large (gsplat ops); then the gsplat 3DGS prune.
- Unchanged 3DGS: init, LRs, means-LR decay, SH schedule, L1 + 0.2 D-SSIM (valid SSIM), opacity
  reset to 0.01.

## Probes
| probe | measures | fails if |
|---|---|---|
| `turbo-dilated-merge` (cuda) | 4 interleaved dilated renders vs the full render (PSNR); naive variant (eps2d 0.3) worse; train size H/2×W/2; eval full size; all 4 offsets drawn | dilation removed or wrong K/eps |
| `turbo-dilated-target` | loss target = image[oy::2, ox::2]; value = 0.8 L1 + 0.2 D-SSIM; loss value exported | target misaligned |
| `turbo-budget-schedule` | B endpoints N/M_ad, α = 1 linear before warm-up, α ≈ exponent on a synthetic power law, plateau → α up and M_ad down, M_ad caps | schedule removed/changed |
| `turbo-color-densify` | colour-only qualifiers densify only when the branch fires; selection capped at B − n with top scores | colour branch or cap removed |
| `turbo-refine-allowance` | a real refine() event densifies exactly B(t) − n with t = 100·step/total; uncapped ablation densifies all | cap/scheduler wiring removed |
| `turbo-color-grad` (cuda) | accumulated colgrad = independent autograd ‖∂L/∂sh0‖ | wrong attribute/normalization |
| built-ins | state transitions, photometric gradient, renderer gradient | — |

## Interpretations (review-1 m1–m4, details in paper.md §7)
- m1: dilation offset p = d·p_ds + o (chessboard sub-lattice), not Eq. 3's block centre.
- m2: positive decay exponents for α (the literal negative slope would clip α at 0.1).
- m3: one colour-branch Bernoulli(0.2) draw per densification event.
- m4: colour gradient w.r.t. SH sh0, not RGB (factor 1/C0 ≈ 3.5).

## Fidelity gaps
1. No official code. The t-normalization, deviation sign, colour attribute, top-k cap, window
   and fit period are our reading (paper.md §7). Effect: unknown, probably small on PSNR.
2. 3DGS-variant densification schedule unstated (3DGS defaults assumed).
3. Batched training / "better initialization" (README only) are not ported. Effect: unknown.
4. Taming CUDA backward (speed only) is not ported. gsplat backend shared with the baseline
   (white bg, valid SSIM, alpha 0.999, random-with-replacement views).
5. The dilated render uses the gsplat tile grid in ds space with exact sub-pixel intrinsics,
   so it is numerically closer to the full render than the paper's kernel (>70 dB claimed).
6. Sensitivity pruning (†) is off, as in the benchmarked row.
7. Only DB average (2 scenes) is compared. M360 9-scene average (27.39) is available if wanted.
