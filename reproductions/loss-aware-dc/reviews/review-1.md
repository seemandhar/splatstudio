# Review 1 — loss-aware-dc (`ss_loss_aware_dc`, method `loss-aware-dc`)

Paper: arXiv 2611.02044, "Unified Loss-Aware Density Control for 3DGS". There is no official code, so I checked against `source/latex/sections/4_method.tex` and `7_appendix.tex`.

**Verdict: approved (0 critical, 0 major, 6 minor).** I found no correctness bug in the CUDA kernel, the equations, or the density ops. The fidelity risk sits in two places the paper leaves underspecified (sections A and B below). Both must be stated plainly in `report.md`.

## Checks performed
- I read all of `plugin/`, plan.md, paper.md §7, and the paper method and appendix sections line by line.
- I ran a CPU check of `_densify` on the probe toy scene, using an Adam that had already stepped. After densify:
  - every field is still a trainable leaf (`requires_grad=True`, `is_leaf`);
  - each optimizer's `param_groups[0]['params'][0] is p`;
  - `exp_avg.shape == p.shape`;
  - survivors keep their moments and new rows get zeros.
- The ResGS `p.clone()`-under-no_grad pitfall from TODO-runtime.md is **not present**. `append_gaussians` (density.py:61-71) builds from `p.detach()` with `requires_grad=p.requires_grad`, and pruning uses gsplat `ops.remove`.

## Equation coverage
| Paper | Code | Status |
|---|---|---|
| Eq. joint_split_matrix / app_joint_splitting_exact: S = E[Jzᵀ H_z Jz + Σ g_zm ∇²z_m] | stats.cu:124-143 (g_z, H_z) and stats.cu:269-339 (hyper-dual pullback) | exact |
| app_Sps_impl: drop Σ g_q ∇²_{p s̃} q | stats.cu:311-312, surrogates.py:118-120 | exact. In the p–s block, u has no s-dependence, so only the q term is dropped |
| Eq. 6: J = blockdiag(R diag(s), I/√2), S̃ = Jᵀ sym(S̄) J | surrogates.py:126-145 | exact (R diag(s) is formed by column scaling) |
| Eq. 7: eigmin | surrogates.py:144 (`vec[:,:,0]` is the column eigenvector) | exact |
| Eq. 8: split set | surrogates.py:157-164, density.py:185-197 | deviates: SteepGS "uncertain" AND added (see A) |
| Eq. 9: children p ± L v_p, s̃ ± v_s/√2, o/2, parent removed | surrogates.py:148-154, density.py:225-238 | exact (no 1/1.6 shrink, literal) |
| Eq. 11: Δ = Tα(c − b), Δ_dup = (1 − α)Δ | stats.cu:111-123 | exact. T_i b_{i+1} = (C − Σ_{j≤i})/(1 − α), and white bg is inside C = render + (1 − alpha) |
| Eq. 12 / 16: g = Σ ∂L/∂I·Δ, h = Σ ω Δ² | stats.cu:115-123; dL/dI = retained `rgb.grad` of the full L1 + D-SSIM loss | exact. The views-summed ratio equals the views-averaged ratio |
| Eq. 13: η = −g/h, ∉ split, g < 0, τ_c < η ≤ 1e-2, child opacity η·o | surrogates.py:167-171, density.py:217-222 | exact |
| Eq. 17: η = g/h, ∉ split ∪ clone, g > 0, τ_d < η ≤ 1e-2, o ← (1 − η)o | surrogates.py:174-178, density.py:212-215 | exact |
| App. A.5: Charbonnier ρ'', diagonal SSIM proxy, mix with γ = 1e-4 | surrogates.py:33-48 | exact (ε_ssim = 1e-8 is unspecified in the paper) |
| App. A.1: τ values; densify for 500 < s < 15000 with s % 100 == 0; 3DGS prune 0.005; no reset | density.py:27-56, 129-131 | exact |

**CUDA kernel vs gsplat 1.5.3 forward.** These all match `rasterize_to_pixels_3dgs_fwd`:
- pixel centres at +0.5;
- σ = ½(a dx² + c dy²) + b dx dy;
- α = min(.999, oe^{−σ});
- skip when σ < 0 or α < 1/255;
- exclusive stop at T ≤ 1e-4;
- tile_id, the batch loop and the `__syncthreads_count` break.

The rest of the kernel and the projection are also correct:
- ∇σ and ∇²σ with respect to z = (u, q_xx, q_xy, q_yy) check out by hand.
- α derivatives are zeroed when o·G > 0.999, as in gsplat's backward.
- The hyper-dual `inv` and `exp` second derivatives are correct.
- `screen_z` reproduces gsplat's clamped persp_proj and the eps2d = 0.3 blur.
- `post_sh_colors` matches gsplat's `clamp_min(SH + 0.5, 0)` with the same campos.

## A. Builder question (a): AND-ing the screen-gradient filter into Eq. 8
**Ruling:** keep it as the default, labelled as the primary fidelity gap.

Eq. 8 (4_method.tex:169-180) lists only three tests: ‖ḡ_pos‖ ≤ τ_pos, ‖ḡ_scale‖ ≤ τ_scale, and λ_min < τ_split. The prose just above says the paper splits Gaussians that are "nearly stationary, **has unresolved view-space variation**, and exhibits sufficiently negative curvature". That is three properties, while the equation uses two tests for stationarity alone. The equation is missing the middle property.

The best reading of that phrase is SteepGS's "uncertain" test: mean per-view ‖∇2D‖ is large while the view-averaged 3D gradient is small, i.e. the per-view gradients cancel. That is exactly what the code computes (g2d = mean of per-view norms; g_pos = norm of the mean).

The smoke logs show the literal Eq. 8 cannot be what was run:
- At step 2000, `uncertain` = 47,844, `candidates` = 47,040 and `split` = `neg_curv` = 41,588.
- So 98% of uncertain Gaussians pass the τ_pos/τ_scale = 1e-4 stationarity test, and ~88% of those have λ < −1e-8.
- Without the filter, nearly every visible Gaussian would split every 100 steps (exponential growth). That cannot give the paper's 1.9M average on Mip-NeRF 360.

**Consequence to state in report.md:** in this port, *which* Gaussians split is decided almost entirely by the borrowed 3DGS/SteepGS 2e-4 threshold. The paper's loss-aware criteria mostly set *how* they split (direction and placement), not *whether*.

**Action:** before the benchmark, log histograms of ‖ḡ_pos‖, ‖ḡ_scl‖ and λ_min for one interval, and run a 1–2k-step `require_uncertain=false` diagnostic, so the claim rests on data.

## B. Builder question (b): clones killed by the same-interval prune
**Ruling:** the reading is correct, so keep the default paper-literal.

The text supports this order and these thresholds:
- Alg. 1 (7_appendix.tex:282-288) applies Split → Clone → Decay and then "Remove low-opacity Gaussians".
- App. A.1 says the prune threshold is "kept the same as in the original 3DGS", i.e. 0.005.
- Eq. 13 caps η at 1e-2 as a *selection* condition, not a clip, and makes the child's opacity η·o.

So a clone survives only if η·o ≥ 0.005, which requires parent o > 0.5 (and η close to 1e-2). In addition, gsplat and Inria skip α < 1/255, so a surviving clone renders only where o_clone·G ≥ 0.0039. The observed clone 39.6k / pruned 41.2k at step 2000 is the paper-literal behaviour. It is not a bug.

**Action:** report the surviving-clone count per interval (split out from the prune count). Treat `exempt_new_clones_from_prune=true` as a labelled "diagnostic, not paper" variant, run only if the gap analysis needs it.

## Minor issues
1. **density.py:191-192 and :225-226** — J (Eq. 6) and the children are built from the *current* post-Adam-step scales and quats, while S was accumulated from the pre-step snapshots. This is a negligible one-step mismatch. Either use them consistently or note it.
2. **density.py:212-215** — decay is done in place and keeps the opacity's Adam moments. The paper is silent on this (paper.md §7.7 lists it). The rows in Eq. 17 are tiny ((1 − η) ≥ 0.99), so it is fine. Keep it in report.md.
3. **density.py:241-242 (prune)** — only the world-size prune is applied after step 3000; the Inria screen-size prune is not. That is correct per the TODO-runtime finding that the Inria screen-size prune is a no-op. Say so in report.md rather than "3DGS thresholds" alone.
4. **stats.cu:113** — `Tb = (Ct − a_new)/(1 − α)` amplifies float32 round-off in C_total by up to 1000× at α = 0.999. The effect on Δ is still about 1e-4 relative, so this is a note only. A probe scene with near-saturated α would document the effect.
5. **Probes** — `lad-decisions` would not catch a frozen field or a misaligned Adam state after `_densify`. Add asserts that every param `requires_grad and is_leaf` and that `opt.state[p]['exp_avg'].shape == p.shape`. I verified both by hand this round. The other probes are mechanism-level (independent float64 compositor, brute-force Hessian, ablation flags) and would fail if their mechanism were removed.
6. **Population trajectory** — splits run at ~40k per interval and are not slowing at 2k steps (n = 679k). Track N(t) to 15k against the paper's 1.9M average on Mip-NeRF 360 (Tab. 1); 3DGS reaches 3.1M. If N far exceeds 3.1M, the cause is the uncertain filter (A), not a bug.

## Fidelity gaps confirmed (already in plan.md / paper.md §7; all must appear in report.md)
- Uncertain filter added to Eq. 8 (A).
- The same-interval prune removes most clones (B).
- White background vs black.
- ε_ssim is unspecified.
- Rasterizer clamp is 0.999 (gsplat) vs 0.99 (Inria).
- Densify runs after the optimizer step (stats use pre-step parameters).
- SSIM uses valid convolution.
- View sampling is with replacement.
