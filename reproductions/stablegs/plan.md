# StableGS — native SplatStudio plan

Method `stablegs` (= Table 1 "StableGS w/o L_prior", the configuration of `run.sh` block 1),
module `ss_stablegs`. The DUSt3R prior variant (`run.sh` block 2) is not built (see gaps).

## Runtime mapping
The runtime renders one random view per step and gives losses one target. StableGS trains on a
**pair** of views per iteration. Following the MVGS precedent (`reproductions/mvgs`), the density
controller owns the official pair stack and the training images (reloaded from `--data` in
`set_training_cameras`) and publishes the pair for the coming step through a process-local link;
with gradients enabled the renderer ignores the runtime camera and renders the pair. Evaluation
(`no_grad`, or no density attached, e.g. `ss-eval`) renders the given camera with the appearance
path α·α'. The losses replace the base photometric loss and read the pair targets.

## Components
| component | kind | factory | requires | provides | config |
|---|---|---|---|---|---|
| `stablegs-raster` | renderer | `ss_stablegs.renderer:create_renderer` | state.gaussian3d, camera.pinhole | render.rgb, render.depth, render.depth.median, render.normal.camera, render.pair_batch, gradient.screen.gof | background [0,0,0], kernel_size 0.0, eval_dual true |
| `stablegs-density` | density | `ss_stablegs.density:create_density` | render.pair_batch, gradient.screen.gof | — | Taming defaults (budget table = run.sh budgets), gof true, dual_from 15000, pair criteria (covis 30, 16–60°, H 0.8, 4 px) |
| `pair-color` | loss | `ss_stablegs.losses:create_pair_color` | render.pair_batch | — | λ_dssim 0.2, ssim_views "official" (view 0 twice) |
| `geometric-color` | loss | `…:create_geometric_color` | render.pair_batch | — | λ_s 1.0 |
| `depth-normal` | loss | `…:create_depth_normal` | render.pair_batch, render.normal.camera, render.depth.median | — | λ 0.05, depth_ratio 0.6 |
| `depth-consistency` | loss | `…:create_depth_consistency` | render.pair_batch, render.depth | — | λ_consis 0.05, form "official" ("paper" = two-sided L1 with z-transform) |

Field: `opacity_aux` (α' logit) shape (), lr 0.025, init logit(0.99) = 4.59512; copied on
clone/split by gsplat ops, never reset, not used in pruning.

## Renderer (C1)
Vendored `source/code/submodules/diff-gaussian-rasterization_radegs` (= RaDe-GS@ec9199f kernel,
Inria licence) JIT-built; Python wrapper = RaDe-GS@ec9199f `diff_gaussian_rasterization` +
`gaussian_renderer.render` (require_coord False). Per pair view: appearance render
(opacity α·α' if dual else α, require_depth False); if geo: geometric render (α, require_depth
True) → expected depth, median depth, normal. means2D grad [N,3] (xy + GOF abs).

## Density (Taming + StableGS changes)
Subclass of `ss_taming_3dgs.density.TamingDensity` (matched reproduction): score, growth curve,
sampled clone/split, guided prune, reset, α freeze at 15000, IntervalAdam(16) for shN, lrs.
Changes: stats from both pair views (Inria NDC grads, no pixel rescale), GOF abs-grad qualifier
(quantile), all training images cached for the score (official uses all train cameras), pair
stack (selected pairs + equal number of inverse-frequency random pairs, popped at random).

## Pairs (C2 support)
`pairs.py`: COLMAP `images.bin` reader; for training image pairs: shared 3D points > 30, relative
rotation in [16°, 60°], homography RANSAC (4 px) inlier fraction < 0.8.

## Probes
| probe | measures | fails if |
|---|---|---|
| `dual-opacity-paths` (cuda) | appearance render with α'=1 equals geometric render; α'→0.5 halves alpha of a single splat; depth-path gradients reach α but not α' and appearance-path gradients reach α' | dual opacity removed / paths swapped |
| `depth-consistency-formula` (cuda-free) | official loss vs independent per-pixel loop on a toy 2-camera scene; zero on a consistent plane; positive with a floater; paper form two-sided | warp/form wrong |
| `pair-selection` | synthetic COLMAP tracks: covis, rotation, homography thresholds | criteria removed |
| `pair-stack` | stack = P + P random pairs, inverse-frequency weights, membership flag | sampler changed |
| `loss-schedule` | geometric losses zero (graph-attached) ≤ 15000, active after; colour loss = official formula incl. duplicated SSIM | schedule wrong |
| `gof-qualifier` | abs-grad quantile adds qualifiers that the signed grad misses; `gof: false` removes them | GOF removed |

## Fidelity gaps
1. Pair retrieval: paper criteria instead of the code's BoQ retrieval (+dz<0.9) — different P. Medium.
2. Missing official renderer/arguments reconstructed (kernel_size 0, Taming lrs). Low–medium.
3. DUSt3R prior (C3) not built: weights/pipeline (dust3r, g2o, open3d, pycolmap, BoQ) unavailable in env; paper Δ from prior on T&T +0.03 dB. Benchmark the w/o-prior row.
4. Optimizer step before density control (runtime order), random-with-replacement runtime frame unused (pair stack drives sampling). Low.
5. Taming score rendered with gsplat (Taming plugin), stats kernel from Taming plugin. Low.
6. LPIPS/SSIM evaluation from runtime. Shared with baseline.
