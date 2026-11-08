# Plan: native `scaffold-gs` plugin (module `ss_scaffold_gs`)

Extraction of the Scaffold-GS backbone from `reproductions/hard-gs/plugin/ss_hard_gs` (HGS plugin, review-approved,
written line-by-line against official Scaffold-GS @59c833b; its `scaffold-gs-hgsbase` control = this backbone).
All HGS code (PGHGS top-k, REHGS max-contribution CUDA kernel, SSIM tap) is removed. Only the vendored simple-knn
(distCUDA2, Inria licence) remains as custom CUDA (JIT). Field prefix `sgs_`, blob `sgs_blob`.

## Components
| kind | name | factory | requires / provides | config |
|---|---|---|---|---|
| renderer | scaffold-renderer | ss_scaffold_gs.renderer:create_renderer | state.gaussian3d, camera.pinhole / render.rgb, render.depth.expected_z, gradient.screen.signed | feat_dim 32, n_offsets 10, opacity_cap 0.999 |
| density | scaffold-density | ss_scaffold_gs.density:create_density | gradient.screen.signed | voxel_size, growth/prune schedule, LRs (paper.md §3) |
| loss | photometric (built-in) | 0.8 L1 + 0.2 D-SSIM | | |
| loss | scaffold-scaling-reg | ss_scaffold_gs.losses:create_scaling_reg | render.rgb | weight 0.01 |

Methods: `scaffold-gs` (voxel 0.001, Mip-NeRF 360), `scaffold-gs-tandt` (0.01), `scaffold-gs-db` (0.005).

## Fields (per anchor)
means (anchor, lr 0), sgs_feat [32] lr 0.0075, sgs_offset [10,3] lr 0.01·s→1e-4·s, sgs_scaling [6] lr 0.007,
sgs_blob [8] (MLP transport for ss-eval). quats/opacities/sh0/shN carried, unused.
Global: 3 MLPs + own Adam in a density-owned bank (`model.ACTIVE`), packed into `sgs_blob` every step.

## Probes
neural-gaussians (decode = per-anchor loop reference, view dependent), visibility-filter (fork's visible_filter on
constructed cases), anchor-density (grows at the voxel of a high-gradient offset, prunes the low-opacity anchor,
alignment + trainable leaves), opacity-selection (only neural Gaussians with tanh opacity > 0 are rasterized;
removing selection changes the count), blob-eval (fresh renderer reproduces the live render from checkpoint).

## Change vs. the hgsbase control (so its runs cannot be reused)
1. Benchmark resolution: official scripts → width-1600 (`mipnerf360-1600`); hgsbase ran on `mipnerf360-pilr`.
2. Straight-through opacity cap 0.999 (TODO-runtime gsplat saturated-alpha backward; tanh opacity can exceed
   0.999); matches Octree-GS plugin practice. Config `opacity_cap: null` restores the hgsbase behaviour.
3. Different plugin source (module/field names), so the runtime source hash differs anyway.

## Fidelity gaps
1. gsplat rasterizer (eps2d 0.3, alpha clamp 0.999) vs the Scaffold fork (0.99); expected small.
2. White background (runtime) vs black.
3. Views sampled with replacement vs shuffled stack; valid-conv SSIM loss.
4. Runtime steps Adam before growing; official grows before optimizer.step() (new anchors' first update differs).
5. LR schedules fixed to total_steps 30000 (config), independent of --steps.
6. MLP weights travel in `sgs_blob` (needs ≥ ~600 anchors).
7. Only 2 of 7 reported Mip-NeRF 360 scenes are run (batch rule); no averages claimed.
Expected effect of 1–4: ≤ 0.2 dB (hgsbase control was within 0.12 dB of HGS's Scaffold rerun at pilr).
