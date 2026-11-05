# Octree-GS native plan

Method `octree-gs`, module `ss_octree_gs`, plugin `reproductions/octree-gs/plugin/`.

Octree-GS (Our-Scaffold-GS variant) = Scaffold-GS anchors + octree LOD levels + view-dependent anchor selection +
LOD-aware growing + progressive training. The anchor infrastructure follows the HAC / ContextGS native ports
(`reproductions/hac/plugin`, `reproductions/contextgs/plugin`, both matched): anchors are the native population,
the global MLPs live in a process-local bank owned by the density (`model.ACTIVE`), Scaffold's `visible_filter` is
re-implemented, and a per-anchor `ogs_blob` field carries the MLP weights **and the octree scalars** (voxel size,
standard distance, origin, K, fork) into ss-eval. Code copied from those ports is attributed in the docstrings.

## Feasibility
`feasible_natively: true` with the HAC workaround (global state transported in `ogs_blob` because
`runtime.evaluate()` does not restore component state). The octree needs the training cameras at initialisation;
the runtime calls `density.initialize()` before `set_training_cameras()`, so `initialize()` stores the SfM points and
`set_training_cameras()` builds the octree and replaces the population (both before step 1).

## Components
| component | kind | factory | requires → provides | config |
|---|---|---|---|---|
| ogs-renderer | renderer | ss_octree_gs.renderer:create_renderer | state.gaussian3d, camera.pinhole → render.rgb, render.depth.expected_z, gradient.screen.signed | feat_dim 32, n_offsets 10, opacity_cap 0.999 |
| ogs-density | density | ss_octree_gs.density:create_density | gradient.screen.signed | `density.DEFAULTS` (paper.md §3) |
| photometric | loss | splatstudio.losses:create_photometric | render.rgb | 0.8 L1 + 0.2 D-SSIM (unchanged) |
| ogs-scaling-reg | loss | ss_octree_gs.losses:create_scaling_reg | render.rgb | 0.01·mean(prod s) |

## Fields (per anchor)
| field | shape | lr | init |
|---|---|---|---|
| means | [3] | 0 | octree voxel centres (all levels) |
| ogs_feat | [32] | 0.0075 | 0 (grown same-level: scatter_max of parents; next-level: 0) |
| ogs_offset | [10,3] | 0.01·s → 1e-4·s (40k) | 0 |
| ogs_scaling | [6] | 0.007 | log sqrt(3-NN mean d²) / log(voxel size) when grown |
| ogs_level | [1] | never stepped (no grad) | octree level L |
| ogs_extra_level | [1] | never stepped (no grad) | ΔL = 0, += 0.01 by density |
| ogs_blob | [8] | never stepped | header + MLPs, packed every step |
Base fields quats/opacities/scales/sh0/shN unused (no gradient).

Bank (`model.OctreeBank`): mlp_opacity/cov/color (35 → 32 → 10 / 70 / 30), own Adam eps 1e-15 with exp LR schedules;
octree scalars; training-camera centres; coarse intervals; visible_threshold.

## Mechanism → code
- `octree.py` (pure torch, CPU testable): `set_level`, `octree_sample`, `weed_out` (chunked over cameras),
  `coarse_intervals`, `coarse_index`, `int_level`, `anchor_mask` — verbatim numerics of gaussian_model.py.
- `renderer.py`: training: `anchor_mask(step)` (progressive) & `visible_anchors` → decode visible anchors with the
  live bank; no-grad / fresh process: blob path, all K levels. Neural decode input `[f, dir]` (no distance).
  Opacity straight-through cap at 0.999 (TODO-runtime: gsplat drops gradients where opac·G > 0.999; the Inria
  fork clamps alpha to 0.99 and always back-propagates).
- `density.py`: init (stores points) / `set_training_cameras` (octree build + weed + fields), Scaffold statistics
  (NDC-scaled means2d grads), `anchor_growing` per level (same-level + next-level + ΔL + weed_out), pruning +
  scaling clamp, LR schedules, bank step (skipped at total_steps), blob pack every step.

## Probes
| probe | fails if… |
|---|---|
| octree-init | K, standard_dist ≠ independent quantile formula; per-level voxels ≠ brute force; adaptive weed threshold ≠ mean view frequency; init_level ≠ ⌊K/2⌋ |
| lod-selection | anchor mask ≠ independent Eq. 3 (code `round` variant, half-voxel shift); ΔL does not raise the selected level; progressive coarse index wrong at interval boundaries; eval path ignores progressive limit |
| progressive-training | intervals not geometric (ratio 1/1.5) or not summing to 10000; renderer at step 1 uses a level > init_level; after 10000 uses all |
| neural-gaussians | decode ≠ per-anchor loop reference; input includes distance; not view dependent |
| visibility-filter | prefilter ≠ Scaffold visible_filter on a constructed case (copied from ContextGS) |
| anchor-growing | mid-gradient offset grows a same-level anchor at its voxel with parent feature; high-gradient offset grows a level+1 anchor at the finer voxel with zero feature; ΔL += 0.01 per exceeded level; never-visible candidates weeded; low-opacity anchor pruned; fields / Adam aligned |
| blob-eval | fresh renderer (no bank, checkpoint round trip) ≠ live no-grad render; blob ≠ trained weights / scalars; empty blob not refused |

## Unchanged from 3DGS / runtime
photometric loss, gsplat rasteriser, random view sampling, white background.

## Fidelity gaps
1. **Rasteriser**: gsplat instead of the Scaffold fork of the Inria rasteriser (alpha clamp 0.99 vs 0.999 with the
   0.999 opacity straight-through cap; pixel-centre conventions); `visible_filter` re-implemented.
2. **Background** white (runtime) vs black (official).
3. **Sampling / SSIM**: random-with-replacement views; valid-convolution SSIM in the loss.
4. **Update order**: runtime steps field Adam before `after_backward` (official: stats + densify before the step).
5. **3-NN init**: exact cdist 3-NN instead of simple-knn's approximate `distCUDA2`.
6. **Global-state transport** via `ogs_blob` (as HAC / ContextGS).
7. **Per-dataset presets**: registered config is mip360's (`base_layer 10`, adaptive visible threshold); T&T needs
   `visible_threshold 0.9`, DB `base_layer 12, visible_threshold 0.9` (not benchmarked).
8. **Run guards**: `--recipe 3dgs`, `--num-gaussians` > SfM size, `--steps 40000` (total_steps).
9. **Data**: `mipnerf360-1600` (images/ PIL-resized to width 1600 = `-r -1`). Our gsplat baseline on these images
   is 30k steps (paper's 3DGS row: 40k, retrained by authors).
10. Paper-vs-code choices follow the code (paper.md §7): `round` LOD without blending, adaptive τ_v, next-level
    growing during progressive training, update_until 25000.
