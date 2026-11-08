# Pixel-GS native plan

Method `pixel-gs`, module `ss_pixel_gs`. Ablation variants (same package): `pixel-gs-pixel-only`
(C1 only), `pixel-gs-depth-only` (C2 only). Benchmark only `pixel-gs`.

## Components
| kind | name | factory | requires/provides | config |
|---|---|---|---|---|
| renderer | gsplat | splatstudio.backends.gsplat:create_renderer | unchanged (signed screen grad) | absgrad False |
| density | pixel-gs-density | ss_pixel_gs.density:create_density | requires gradient.screen.signed | DefaultStrategy 3DGS defaults (grow_grad2d 2e-4, refine 500..15000/100, reset 3000) + `pixel_aware` True, `depth_scaling` True, `gamma_depth` 0.37, `opacity_reset` True |
| loss | photometric | splatstudio.losses:create_photometric | unchanged | L1 + 0.2 D-SSIM |

No new fields.

## Mapping
`PixelGSStrategy(DefaultStrategy)` overrides `_update_state` only (official
`add_densification_stats`, gaussian_model.py:405-407):
- g = means2d.grad · (W/2, H/2) (Inria NDC units, as gsplat), n = ||g||
- C2 (Eq. 10): n ← n · min(1, (depths/(γ·scene_scale))²); scene_scale = runtime 3DGS camera extent
  = Eq. 9 radius. Official applies it to grad_means2D only, so means3D grads are untouched — we
  scale the statistic, never the graph.
- C1 (Eq. 7/11): m = per-Gaussian contributing-pixel count from a forward-only CUDA kernel over
  gsplat's tile lists (`csrc/adc_stats.cu`, reused from reproductions/improving-adc, validated
  there against a brute-force per-pixel loop; gsplat numerics α≤0.999, α≥1/255, exclusive stop
  T·(1−α)≤1e-4). grad2d += n·m, count += m for radii>0.
- gsplat `_grow_gs` divides grad2d / count.clamp_min(1) — equals Inria accum/denom with NaN→0
  (count=0 ⇒ grad2d=0). Clone/split/prune/reset are DefaultStrategy (+ runtime reset fix);
  population ops via gsplat.strategy.ops (no requires_grad pitfall).

## Probes
| probe | checks | fails when |
|---|---|---|
| pixel-count-kernel (CUDA) | kernel counts == brute-force per-pixel compositing loop | count rule wrong |
| pixel-aware-accumulation (CUDA) | grad2d/count after one view == Σ f·m·n / Σ m from independent formula; pixel-only and depth-only variants equal their reduced formulas; full ≠ 3DGS mean | C1 or C2 removed |
| depth-scaling (CPU-analytic within CUDA probe) | near-camera Gaussian (z < γR) attenuated by (z/γR)², far one not | C2 removed/wrong power |
| densify-schedule | grow happens only at 600..14900 multiples of 100 | (inherited from DefaultStrategy config check) |

## Fidelity gaps
1. Renderer numerics: gsplat (α clamp 0.999) vs Inria (0.99); pixel counts follow gsplat. Small.
2. White background (runtime) vs black (official). Known runtime difference; baseline shares it.
3. Random-with-replacement view sampling vs Inria shuffled stack (runtime). Shared with baseline.
4. SSIM valid conv, LPIPS from `lpips` package (runtime).
5. `>` vs `>=` threshold (gsplat). Negligible.
6. Per-axis 3σ bbox tile assignment (gsplat) vs Inria circular radius → a handful of edge pixels.
Expected effect: the comparison should be Δ over our pilr gsplat baseline vs paper Δ over 3DGS*.
