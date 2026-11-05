# Plan — Revising Densification (arXiv 2404.06109) as a native SplatStudio plugin

- Method name: `revising-densification`; module `ss_revising_densification`;
  package `splatstudio-method-revising-densification` in `plugin/`.
- No official code; everything from `paper.md`. Target row: "Ours, 3DGS".

## Components
| # | kind | name / factory | requires → provides | config (paper value) |
|---|---|---|---|---|
| R | renderer | stock `gsplat` (`splatstudio.backends.gsplat:create_renderer`) | unchanged | classic, no absgrad |
| D | density | `revdens-density` `ss_revising_densification.density:create_density` | `render.rgb` → () | see below |
| L0 | loss | stock `photometric` (L1 + 0.2 D-SSIM) | render.rgb | unchanged |
| L1 | loss | `revdens-aux` `ss_revising_densification.losses:create_aux` | render.rgb → () | `transmittance_weight` 0.1, `error` "ssim" |

Density config: `densify_from` 500, `interval` 100, `densify_until` 27000, `error_threshold`
0.1, `growth_fraction` 0.05, `max_primitives` "baseline" (table lookup) or int,
`opacity_correction` true, `opacity_mode` "decrement" (ablation "reset"), `opacity_decrement`
0.001, `min_opacity` 0.005, `percent_dense` 0.01, `big_prune_scale` 0.1, `big_prune_after` 3000,
`score` "error" (ablation "gradient", 3DGS τ with 2e-4 threshold), `reset_every` 3000 /
`reset_until` 15000 (ablation only).

### Data handoff loss → density
`density.after_backward` does not see the target image. `revdens-aux` computes the detached
per-pixel error map 𝓔(u) = 1 − SSIM(u) (11×11, σ 1.5, same padding, mean over RGB) and stores it
in `outputs["revdens_error"]` (the same dict the density receives). The density raises if the map
is missing while ADC is active. The contract cannot express this (density `requires` must be
renderer capabilities), so it is documented and probed (smoke log shows E statistics).

## Mechanisms → code
1. **C1 error attribution** (Sec. 3.2): CUDA kernel `csrc/err_stats.cu` (adapted from
   `reproductions/improving-adc/.../adc_stats.cu`, gsplat tile lists and numerics) computes
   E^π_k = Σ_u 𝓔(u)·α_k(u)·T_k(u) in one forward-only traversal, plus Σ α T (diagnostic).
   This equals ∂L_aux/∂e_k (the paper's implementation) without a second rasterization.
   Score E_k = running max over steps since the last ADC run; reset to 0 after each run; new
   primitives start at 0.
2. **C3 growth control** (Sec. 3.4): at each ADC run (step > 500, step % 100 == 0, step < 27000):
   prune first (opacity < 0.005; world size > 0.1·extent once step > 3000), then
   budget = min(⌊0.05·N⌋, N_max − N); candidates E_k > 0.1; keep top-budget by E_k; selected
   with max scale > 0.01·extent are split (3DGS split: 2 samples, scale/1.6, opacity kept),
   others cloned. Each op is +1 primitive.
3. **C2 opacity correction** (Sec. 3.3): cloned set (source and copy) get 1 − √(1 − α); split
   offspring keep α (paper; not gsplat's `revised_opacity`).
4. **C4 opacity regularisation** (Sec. 3.5): after each ADC run α ← max(α − 0.001, 1e-6) for all
   primitives (no reset); loss 0.1 · mean(1 − alpha) on every step.
5. **C5** ADC until 27000.
6. **Cap lookup**: table {initial SfM count → final Gaussian count of `baseline3dgs-*-30000`} for
   the 13 benchmark scenes (counts are unique per scene); unknown scene with "baseline" → error.

## Fields
None (e_k of the paper is never optimised; it is replaced by the kernel).

## Probes
| probe | contribution | measures / fails when |
|---|---|---|
| `error-attribution-oracle` (CUDA) | C1 | kernel E_k == brute-force per-pixel alpha-compositing loop (independent Python) == paper's L_aux autograd route (render zero scalar e_k, grad); fails if weights/T/error dropped |
| `error-score-max` (CUDA) | C1 | E_k is the max over two views (not sum/mean), reset after a run |
| `error-driven-selection` | C1,C3 | high-error/zero-gradient Gaussian densified; zero-error/high-gradient one not; threshold 0.1; gradient-score ablation differs |
| `growth-budget` | C3 | +⌊0.05N⌋ per run, top-E kept, cap N_max respected, pruned replaced |
| `clone-opacity` (CUDA) | C2 | clone source+copy = 1−√(1−α), split keeps α; rendered transmittance behind two coincident clones at the centre pixel equals 1−α (analytic); ablation gives (1−α)² |
| `opacity-regularisation` | C4 | −0.001 per run, no reset at 3000/6000; loss = 0.1·mean(1−alpha), gradient pushes opacity up; reset ablation does reset |
| `error-map` | C1 | stored map: mean over interior equals runtime `losses.ssim` (valid conv) to 1e-5; detached |
| `adc-schedule` | C5 | 264 ADC runs at 600..26900; none ≥ 27000 |
| `primitive-cap-table` | C3 | lookup on init counts, explicit int override, unknown → error |

## Files
`plugin/pyproject.toml`, `plugin/README.md`, `ss_revising_densification/{__init__,core,density,losses,kernels,probes}.py`,
`csrc/{err_stats.cu,ext.cpp,LICENSE-3DGS.md}`, `tests/test_revising_densification.py`.

## Unchanged from 3DGS
Renderer, photometric loss, LRs (runtime 3DGS recipe), SH schedule, 30k steps, split rule.

## Fidelity gaps
1. **Error formula** (1−SSIM vs D-SSIM, channel reduction) unspecified — affects only the 0.1
   threshold; budget mostly binds. Closable only with author code.
2. **Cap = our gsplat baseline count**, not the authors' baseline (median of 5). Same intent.
3. **Order prune→grow** and +1-per-op budget accounting are our reading of Sec. 3.4 / Fig. 3.
4. **Runtime**: white background (paper presumably black), Adam step before density (stats
   from the same forward, so no effect on E_k), sampling with replacement, gsplat 0.999 alpha
   clamp vs Inria 0.99, valid-conv SSIM in the photometric loss.
5. **Single run** vs paper mean of 5 (paper stdev ≈ 0.03–0.16 dB per scene).
6. **World-size pruning** of big Gaussians kept from 3DGS (paper: "other hyper-parameters
   default"); no screen-size prune (Inria no-op).
7. **Saturated-alpha gradient drop** in gsplat (TODO-runtime.md) affects the transmittance loss
   gradient on opaque pixels the same way as the photometric gradient; not patched (stock 3DGS
   renderer path, the paper does not scale opacities).
