# Plan — confidence-densify (`ss_confidence_densify`)

Methods registered by the plugin:
- `confidence-densify` — the paper (budget 1M by default, `cap_max` config).
- `cad-mcmc-control` — the paper's baseline row at the same budget: built-in `gsplat-mcmc`
  renderer + gsplat `MCMCStrategy` (opacity sampling, opacity-threshold relocation, noise) with the
  3DGS-MCMC recipe pieces the built-in config lacks (λ_reg 0.01 opacity/scale regulariser,
  initial opacity 0.5).

## Components (confidence-densify)
| kind | name | factory | requires / provides | config |
|---|---|---|---|---|
| renderer | cad-renderer | `renderer:create_renderer` | provides render.rgb, render.depth.expected_z, render.confidence, gradient.screen.signed, gradient.screen.absolute | – |
| loss | confidence-photometric (replaces photometric) | `losses:create_confidence_loss` | render.rgb, render.confidence | ssim_weight 0.2, beta_conf 0.075, conf_min 1e-3, conf_max 5 |
| loss | mcmc-reg | `losses:create_reg` | render.rgb | opacity_reg 0.01, scale_reg 0.01, stop 25000 |
| density | cad-density | `density:create_density` | gradient.screen.absolute, render.confidence | cap_max 1e6, refine_start 500, refine_stop 25000, refine_every 100, growth 0.05, relocate_every 400, share 0.1, share_full_start 5000, falloff_start 10000, falloff_sigma 5000, eta_ramp 5000, grad_fraction 0.8, conf_offset 1.2, min_opacity 0.005, color_init_radius 16, color_init_timeout 400, init_opacity 0.5 |

Renderer: SH → RGB with gsplat's own `spherical_harmonics` + `clamp_min(c+0.5,0)` (identical to
`rasterization(sh_degree=…)`), rasterised (RGB+ED) with `absgrad=True`; confidence Ĉ =
Σbγ̄/max(1−T,1e-3) from a second pass on detached geometry (CoMe numerics; gradient only to γ,
fixed after review-1). Returns `confidence`, `camera`, `sh_degree`.

Loss: per-pixel `pp = 0.8|I−I*| + 0.2(1−SSIM_map)` with same-padded SSIM map (CoMe/3DGS),
`mean(pp·clamp(Ĉ) − β log clamp(Ĉ))`; stashes detached target and per-pixel 1−SSIM (channel
mean) into `outputs` for the density component (runtime gives density no target).

## Field
`confidence` (1,), lr 2.5e-4, init 0 → γ̄ = exp(γ) = 1. Copied by all density ops.

## Density (per step t ≤ 25000)
1. Accumulate ‖absgrad‖ and visibility counts (avg since previous densification step).
2. Custom CUDA kernel (`csrc/cad_stats.cu`, gsplat alpha numerics, tile traversal from the
   Revising-Densification port) → per-primitive max blend weight (atomicMax) and Σ b·(1−SSIM).
   w_max = running max since previous relocation; SSIM term accumulated/averaged like ∇_abs.
3. Colour init for pending new primitives (radius<16 px in this view): bilinear GT at means2d,
   DC = (c_GT − 0.5 − SH_{l≥1}(dir))/C0, zero moments; drop after 400 it.
4. Densification step (500<t<25000, t%100==0): n_add = min(cap, ⌊1.05N⌋)−N; at t%400==0 the
   relocation set (schedule §3.2, exempt = last relocation's destinations). One draw of
   n_add+n_reloc sources (80 % Eq. pdf with min-rank ties, 20 % SSIM term) over non-relocated
   primitives; MCMC `compute_relocation` with ratio = copies+1; destinations copy source,
   mean ~ N(μ, R diag(s')² Rᵀ); moments zeroed for sources and destinations. No noise.

## Probes (each fails when mechanism removed)
| probe | checks |
|---|---|
| confidence-loss | loss equals independent formula; minimiser Ĉ=β/L; gradient lowers γ where error>β; ablation (plain L_rgb) differs |
| confidence-render (cuda) | rendered confidence = Σ b γ̄ (equals alpha when γ̄=1; equals 2·alpha when γ=log2) and RGB equals gsplat SH rasterization |
| blend-stats-oracle (cuda) | CUDA max-weight and Σ b·err equal a brute-force per-pixel python compositing loop |
| sampling-pdf | P_i formula vs independent implementation; η ramp; 1.2/0.2 range; opacity-ablation differs |
| relocation-schedule | share 0.1/0.061/0.0135/~0.011 at 10k/15k/20k/25k; w=0-only before 5k; exempt mask honoured; every 400 |
| densify-step (cuda) | budget growth 5 % to cap, compute_relocation conservation, positions differ from sources (sampled), no noise, optimizer alignment |
| color-init (cuda) | after a step, new primitive's SH colour in the view equals bilinear GT |

## Fidelity gaps
1. Rasteriser: gsplat 1.5.3 instead of Faster-GS (numerically 3DGS-like; minor).
2. Runtime: white background, random-with-replacement views, valid-conv SSIM in eval (shared
   with all baselines; compare deltas over our MCMC control).
3. Rank ties / SSIM-term reduction / moment resets: ambiguities resolved as in paper.md §7.
4. Init opacity 0.5 (MCMC code) assumed; paper silent.
5. Extra raster pass per step for statistics → slower than the paper (no fidelity effect).
6. Runtime means-LR schedule (0.01^(t/T)) matches 3DGS exponential decay 1.6e-4→1.6e-6.
