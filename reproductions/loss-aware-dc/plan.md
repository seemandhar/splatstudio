# Plan — loss-aware-dc (method `loss-aware-dc`, module `ss_loss_aware_dc`)

## Components
| kind | name | factory | requires / provides | config (paper value) |
|---|---|---|---|---|
| renderer | `lossaware-gsplat` | `ss_loss_aware_dc.renderer:create_renderer` | provides render.rgb, render.depth.expected_z, gradient.screen.signed, `surrogate.inputs` | none. Identical gsplat 1.5.3 `rasterization` call to the built-in renderer (white bg). Also stashes the detached post-SH colours [N,3] used by the stats kernel |
| density | `lossaware-density` | `ss_loss_aware_dc.density:create_density` | requires gradient.screen.signed, `surrogate.inputs`, `surrogate.target` | tau_pos=tau_scale=1e-4, tau_split=-1e-8, tau_clone=1e-6, tau_decay=1e-4, eta_max=1e-2, charb_eps=1e-3, gamma_ssim=1e-4, ssim_weight=0.2, eps_ssim=1e-8, densify_from=500, every=100, until=15000, prune_opa=0.005, big_world=0.1 after 3000, require_uncertain=true (grad2d>=2e-4), ablation flags split/clone/decay/joint |
| loss | `photometric` | `ss_loss_aware_dc.losses:create_photometric` | requires render.rgb; provides `surrogate.target` | ssim_weight 0.2. It is the built-in L1+0.2 D-SSIM, unchanged numerically, and also stores the detached target in outputs so the density can build ω |

No new fields.

## Per-step flow
1. **before_backward.**
   - Retain the grads of `means2d` and `rgb`.
   - Snapshot the pre-step means, scales and quats.
   - Register hooks on means/scales to capture dL/dp and dL/d log s (the runtime zeroes grads
     before after_backward).
2. **after_backward** (step < 15000).
   - ω from rgb and target (App. A.5).
   - CUDA kernel `surrogate_stats` (custom, `csrc/stats.cu`, JIT).
     - It replays gsplat's forward tile traversal exactly: same pixel centres, α = min(.999, oG),
       1/255 skip and T ≤ 1e-4 stop.
     - It uses T_i·b_{i+1} = (C_total − Σ_{j≤i} c_j α_j T_j)/(1−α_i) and Δ_c = α(T c − T b).
     - Per Gaussian it accumulates:
       - g_z (5) = Σ w ∇_z α and H_z (15) = Σ w ∇²_z α, with w = ∂L/∂α = (∂L/∂I·Δ)/α, zero when
         α is clamped;
       - g_dup, h_dup, g_dec and h_dec.
     - Warp reductions are followed by atomics (gsplat bwd pattern).
   - Per visible Gaussian, z(φ) is re-implemented in torch with gsplat's projection (clamped
     perspective Jacobian, eps2d 0.3).
     - J_z (5×6) comes from `torch.func.jacfwd`; Hess(g_z·z) (6×6) from `jacfwd(jacrev)`.
     - The p–s block of the second term is zeroed (Eq. app_Sps_impl).
     - S_view = J_zᵀH_zJ_z + that Hessian; accumulate S, denom, ḡ_p, ḡ_s, ‖grad2d‖, g/h sums.
3. **Densify** (s > 500, s % 100 == 0, s < 15000).
   - S̄ = sym(S/denom); S̃ = JᵀS̄J; eigh only on candidates (stationary ∧ uncertain); λ_min, v_min.
   - I_split per Eq. 8, I_clone per Eq. 13 (∉ split), I_decay per Eq. 17 (∉ split ∪ clone).
   - Apply in this order: decay (in place, logit((1−η)o), keep Adam) → clone (append, Adam zero)
     → split (append 2 children with p ± Lv_p, log s ± v_s/√2, o/2; remove parent) → prune
     (o < 0.005, world-size after 3000) → reset accumulators.
   - No opacity reset.

## Probes (each fails if its mechanism is removed)
| probe | contribution | check |
|---|---|---|
| `stats-kernel` (CUDA) | C1/C3/C4 inputs | Kernel g_z, H_z, g/h for dup/dec vs an independent float64 torch compositor with autograd, on random small scenes. g_z is also checked against gsplat's own means2d/conics grads |
| `joint-split-matrix` | C1 | S_view assembly vs brute-force float64 autograd Hessian of a single-Gaussian α-sum in φ (minus the dropped mixed term). The normalised eigen direction maps to children p ± Lv_p, log s ± v_s/√2, opacity o/2 |
| `split-selection` | C2 | Only stationary ∧ λ < τ_split (∧ uncertain) split. Flipping a gradient over τ_pos removes it |
| `clone-surrogate` | C3 | η = −g/h window, child opacity η·o, not for split members. 1-D analytic check that −g/h minimises the quadratic, plus the actual loss change of a tiny clone vs the surrogate (CUDA) |
| `decay-surrogate` | C4 | Selective decay and no global reset at 3000. The predicted loss change matches the real change for a tiny decay (CUDA) |
| `proxy-curvature` | C5 | ω vs an independent per-pixel loop formula. Charbonnier ρ'' matches autograd |

## Fidelity gaps
1. **No official code.** Every numeric detail is from the text; ambiguities are listed in
   paper.md §7. The biggest is the uncertain filter.
2. **White background** (runtime and shared baseline) vs black in the paper.
3. **Update order.** The runtime takes the optimizer step before density control, whereas 3DGS
   densifies before the step. Statistics use pre-step params, so this is consistent.
4. **Shared runtime diffs.** View sampling is with replacement, SSIM uses valid convolution, and
   LPIPS comes from the `lpips` package.
5. **Rasterizer.** Statistics use gsplat's rasterizer (α clamp 0.999, T stop 1e-4) instead of the
   Inria/SteepGS one (0.99), which is a tiny effect.
6. **Speed.** The extra stats pass plus the functorch Hessians make training about 1.5–2.5× slower
   before step 15k. This is not a fidelity issue.
