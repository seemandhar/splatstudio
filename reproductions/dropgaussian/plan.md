# DropGaussian — native SplatStudio plan

Method `dropgaussian` (module `ss_dropgaussian`), plus the matched ablation
`dropgaussian-nodrop` (identical components, γ = 0) so the improvement is measured against a
baseline that uses the official training settings, not gsplat's defaults.

## Components
| kind | name | factory | requires / provides | config (paper/code value) |
|---|---|---|---|---|
| renderer | dropgaussian-renderer | `ss_dropgaussian.renderer:create_renderer` | state.gaussian3d, camera.pinhole / render.rgb, gradient.screen.signed | gamma 0.2, total_steps 10000, background black, clamp_rgb true |
| density | dropgaussian-density | `ss_dropgaussian.density:create_density` | gradient.screen.signed | grad 5e-4, from 500, until 10000, every 100, percent_dense 0.01, min_opacity 0.005, reset every 3000 to 0.01, no size pruning |
| loss | dropgaussian-photometric | `ss_dropgaussian.losses:create_photometric` | render.rgb | l1_weight 1.0, ssim_weight 0.2, same-padding SSIM |

No new FieldSpecs. Base params/optimizers/LRs of the runtime = 3DGS values (checked in state.py);
means LR decays log-linearly 1.6e-4 → 1.6e-6 (×extent) over `--steps 10000` = official.

## Mapping
- **C1 (Eq. 5)**: in `Renderer.render`, if `torch.is_grad_enabled()` (training), draw
  `M = F.dropout(ones(N), p=r_t, training=True)` (identical to the official `nn.Dropout`) and
  rasterize with `sigmoid(opacity) * M`. Without grad (runtime `measure`/`ss-eval`), render all
  Gaussians with unscaled opacity. A fresh mask per render, no population change.
- **C2 (Eq. 6)**: `r_t = gamma * t / total_steps`; `t` = 1-based count of training renders
  held by the renderer (the runtime renders exactly once per step). The density checks in
  `before_backward` that the renderer's counter equals the runtime `step` and raises otherwise,
  so a desynchronised schedule cannot pass silently.
- Densification visibility: gsplat culls Gaussians whose opacity < 1/255, i.e. every dropped
  Gaussian; the Inria rasterizer does not (radius from covariance only). The renderer therefore
  also returns `radii` of an opacity-undropped projection (no grad) and the density counts
  visibility from it, so dropped-but-in-frustum Gaussians increment the denominator with a zero
  gradient exactly as in the official code.
- Density = official `densify_and_prune` with `size_threshold=None`: clone (g ≥ τ, max scale ≤
  0.01·extent), split (g ≥ τ, max scale > 0.01·extent, N=2, scale/1.6, samples from N(0,Σ)) via
  gsplat ops, prune opacity < 0.005, opacity reset (min(o,0.01)) at 3000/6000/9000.

## Probes
| probe | checks (fails if mechanism removed) |
|---|---|
| drop-compensation | mask formula vs independent Bernoulli statement: kept opacities = o/(1-r) exactly, dropped = 0, keep fraction ≈ 1-r (N=200k), E[õ]=o; CUDA: grad-enabled render differs from all-Gaussian render and dropped Gaussians get zero grad |
| drop-schedule | r_t = 0.2 t/10000 at t=1, 5000, 10000; counter increments only for grad-enabled renders; density raises on counter/step mismatch |
| test-time-all-gaussians | CUDA: no-grad render equals an independent plain gsplat render (black bg, all Gaussians, no compensation) |
| official-densification | fixture: clone/split/prune counts, no size pruning of a huge Gaussian, reset schedule {3000,6000,9000}, dropped Gaussians counted in denominator |
| loss-weights | loss equals L1 + 0.2(1-SSIM_same) from an independent formula; ≠ 0.8 L1 + 0.2 D-SSIM |

## Data / protocol
Official LLFF protocol = existing `llff-fewview3` split + init, **but at 1/8 resolution
(504×378)** instead of 1/4. New dataset `llff-fewview3-r8` (all 8 scenes, paper reports
averages only) at `~/datasets/llff/fewview3-r8`, built with
`reproductions/ad-gs/tools/prepare_llff_fewview.py --downscale 8` (horns/fortress/room reuse
the existing COLMAP clouds, which are resolution independent).

## Fidelity gaps
1. View sampling: runtime samples with replacement; official pops a shuffled stack (each of
   the 3 views once per 3 its). Cannot be changed from a component. Expected effect small
   (±0.1 dB), mostly variance.
2. Optimizer/densify order: runtime steps Adam before density; official densifies before the
   step (skipping the update of the densified iteration). Negligible.
3. gsplat vs Inria rasterizer: alpha clamp 0.999 vs 0.99, opacity-aware radii for non-dropped
   low-opacity Gaussians (<1/255), SH-degree change one step later. Small.
4. LPIPS VGG from `lpips` package vs lpipsPyTorch (same weights). Negligible.
5. Seeds/variance: the paper's own numbers for the default disagree by 0.2 dB (Tables 1 vs 5).
6. COLMAP 3.9.1 MVS cloud may differ from the authors' (version unstated) — affects init.
