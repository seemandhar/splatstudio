# PGSR native plan

Method names: `pgsr` (Mip-NeRF 360 / official `run_mip360.py` settings), `pgsr-tnt` (official
`run_tnt.py` settings: exposure compensation on, abs threshold 0.00015, opacity cull 0.05).
Module `ss_pgsr`. Benchmark: `mipnerf360`, all 9 scenes (paper reports averages only).

## Components
| component | kind | factory | requires → provides | config (paper/code values) |
|---|---|---|---|---|
| `pgsr-plane` | renderer | `ss_pgsr.renderer:create_renderer` | state, camera → render.rgb, render.depth.plane, render.normal.camera, render.distance.plane, render.alpha, render.normal_from_depth.camera, gradient.screen.signed, gradient.screen.absolute, render.observe_count, render.camera | background black; geometry rendered once step > 7000 (via link) |
| `pgsr-density` | density | `ss_pgsr.density:create_density` | gradient.screen.signed, gradient.screen.absolute, render.observe_count | official densify (percent_dense 0.001, grad 2e-4, abs 2e-4 for mip360, radii 20, 50k, 6M), opacity cull 0.005, world prune > 0.1·extent after 3000, screen prune no-op (official), trim every 1000 (< 2 observing views), reset every 3000 < 15000, xyz LR expon; loads training images for neighbour graph + grey images; steps exposure optimizer |
| `pgsr-rgb` | loss | `ss_pgsr.losses:create_rgb` | render.rgb, render.camera | λ_dssim 0.2, official same-padding SSIM, exposure (off / on for tnt) from step > 1000, ssim gate 0.5, lr 0.001 |
| `pgsr-flatten` | loss | `ss_pgsr.losses:create_flatten` | gradient.screen.signed | λ1 = 100, visible = radii > 0 |
| `pgsr-single-view` | loss | `ss_pgsr.losses:create_single_view` | render.normal.camera, render.normal_from_depth.camera | λ2 = 0.015, from > 7000, (1 − grad)^2 weight |
| `pgsr-mv-geo` | loss | `ss_pgsr.losses:create_mv_geo` | render.depth.plane, render.camera | λ4 = 0.03, from > 7000, φ < 1 px, w = exp(−φ) detached |
| `pgsr-mv-ncc` | loss | `ss_pgsr.losses:create_mv_ncc` | render.normal.camera, render.distance.plane, render.camera | λ3 = 0.15, 7×7, 102400 samples, mask ncc < 0.9 |

The two multi-view losses share one neighbour render per step (memoised on the outputs dict).
The neighbour is drawn uniformly from the reference view's ≤8 neighbours (official graph:
dist-sorted, ties by angle, angle < 30°, 0.01 < dist < 1.5). The density component reloads
the **training split only** from `--data` (MVGS precedent) and publishes cameras, grey images
and the graph through `ss_pgsr.link.LINK`; the losses identify the reference view by its
viewmat. No test image is loaded by the plugin.

## Fields
None (exposure coefficients are per-image, held inside `pgsr-rgb` with their own Adam
(betas 0.9/0.99), stepped by the density's after_backward; `knn_f` is unused upstream).

## Probes
| probe | contribution | fails if |
|---|---|---|
| plane-depth (CUDA) | C2 | kernel depth ≠ analytic ray–plane intersection of a flattened Gaussian (oblique), normal ≠ plane normal, or plane depth equals expected-z blending |
| flatten-loss | C1 | value ≠ 100·mean(min scale of visible), invisible counted, gradient on non-min axes |
| single-view | C3 | depth→normal on analytic plane wrong; loss ≠ independent formula with edge weight |
| mv-geometry | C4 | φ on consistent analytic plane depths not ≈ 0, or equals homography φ formula mismatch, or perturbed neighbour depth not rejected/down-weighted |
| mv-ncc | C5 | NCC on correct plane not ≈ 0 / not lower than wrong plane distance; homography ≠ paper Eq. |
| exposure | C6 | L1 not on exp(a)I+b when ssim < 0.5 & step > 1000, SSIM on raw render, no gradient to (a,b) |
| neighbour-graph | C4/C5 | sort order/filters differ from official lexsort rule |
| density-recipe | C7 | clone resample / percent_dense / abs-split radius+cap / prune / screen-size no-op / trim / reset deviate |
| state-transitions | base | gsplat ops misalign optimizer |

## Unchanged from 3DGS
Base LRs, SH schedule (runtime: degree up at 1001 vs official 1000), initial scales
(recipe 3dgs = distCUDA2), opacity 0.1 init, opacity reset 0.01.

## Fidelity gaps
1. Runtime applies the optimizer step before densification (official: after). Small.
2. View sampling with replacement (official shuffled stack). Small.
3. Mip-NeRF 360 images: our `mipnerf360` uses the dataset's `images_4/2` JPGs; official uses
   PIL `-r 4/2` resize of full-res images (`mipnerf360-pilr` exists only for some scenes and
   has no baseline). Expected ≤ 0.1–0.3 dB on some scenes; same for our baseline.
4. ncc_scale ≠ 1 (DTU/TnT use 0.5 = NCC on 2× resolution grey images) not supported →
   `pgsr-tnt` uses 1.0. Not relevant to the Mip-NeRF 360 benchmark.
5. Mesh extraction (TSDF), DTU Chamfer and TnT F1 are not part of this benchmark (full-view
   DTU images not available locally; FatesGS release has 3 views per scan).
6. Seeds differ (official 22); evaluation metrics from SplatStudio (SSIM valid conv., lpips
   package VGG) vs official (same-padding SSIM, lpipsPyTorch VGG) — PSNR identical definition.


## Review-1 resolution
Benchmark dataset is `mipnerf360-pilr` (official run_mip360.py: -r 4/-r 2 PIL resize of full-res images) vs `baseline3dgs-mipnerf360-pilr-30000`; earlier images_4/2 statements are superseded.
