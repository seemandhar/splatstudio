# Plan — hierarchical-3dgs (plan id 74)

## Feasibility decision
- Paper datasets: SmallCity (released, full scene zip 3.0 GB + 2.5 GB pretrained), Campus,
  BigCity, Wayve (not released). Tab. 1 numbers are for **one unnamed chunk** of the full
  hierarchical pipeline (scaffold + skybox + depth + exposure masks, 4 chunks for SmallCity), so
  even SmallCity cannot be compared like-for-like (chunk unknown; full pipeline incl. COLMAP
  chunk BA, DPT/Depth-Anything depth and masks lives outside the runtime). Out of scope.
- **Feasible target**: §6.2 text, Mip-NeRF 360 *entire dataset average* with single-chunk
  optimization: 3DGS 29.11 / leaves 28.96; τ1/τ2/τ3 (3/6/15 px) basic 28.86/28.05/24.82,
  optimized 28.87/28.52/27.25. Data on disk; cost ≈ 9 × (30k + 15k) + 9 × 30k steps.
  Average-only → all 9 scenes (brief rule). Compare deltas (their 3DGS 29.11 ≠ 27.21 of the 3DGS
  paper; our baseline `baseline3dgs-mipnerf360-30000`).

## Method / module
`hierarchical-3dgs` (`ss_hierarchical_3dgs`), plus `hierarchical-3dgs-quick` (pipeline smoke only).

| component | kind | factory | notes |
|---|---|---|---|
| hierarchy-renderer | renderer | renderer:create_renderer | vendored hierarchy rasterizer (Mip 2D filter, α′ interpolation); leaves mode (random bg in training) / cut mode (expand_to_size + get_interpolation_weights, python attribute lerp, abs falloff) |
| hierarchy-density | density | density:create_density | train_single schedule (max-grad densification, sparse Adam, big-Gaussian shrink, official lrs) → creator at `leaves_steps` → train_post (FrozenAdam, anchors frozen) |
| photometric | loss | splatstudio.losses | 0.8 L1 + 0.2 D-SSIM (unchanged) |

Field: `h3d` [N,16] — packed tree (nodes, boxes, flags) so ss-eval can render the hierarchy
from `checkpoint.pt` (runtime has no component-state persistence).

## Probes
| probe | contribution | fails if |
|---|---|---|
| h3dgs-hierarchy-build | C1 | merge weights/falloff/covariance differ from Eq. merge of leaf descendants; broken tree |
| h3dgs-cut-selection | C2 | selection not a cut, ε rule violated, t/K differ from independent formula |
| h3dgs-alpha-interpolation | C2 | α′ correction absent (K children at t=0 ≠ parent) |
| h3dgs-hierarchy-optimization | C3 | anchors move, no interior node optimized, τ sampling range wrong |
| h3dgs-leaf-training | C4 | mean instead of max statistic, wrong selection/split/prune/reset/shrink/sparse Adam/lr |

## Evaluation
`ss-eval` → leaves (eval_tau −0.5 = threshold 0, every leaf, t = 1).
`python -m ss_hierarchical_3dgs.eval_taus --run R --taus -0.5 3 6 15` → test_tau*.json.
Runs: `--steps 30000` (basic hierarchy) and `--steps 45000` (optimized).

## Fidelity gaps
1. Depth regularization not used (no monocular depth for Mip-NeRF 360 in the runtime; paper
   ablation: depth does not raise metrics, −0.12 dB avg). Unknown whether the authors used it here.
2. Exposure optimization off (paper disables it for single-chunk comparisons).
3. No scaffold/skybox/chunking/consolidation/streaming viewer (large-scene parts, C5).
4. Runtime differences: random-with-replacement view sampling (official: shuffled epochs via
   DataLoader), densify/reset applied after the optimizer step of the same iteration (official
   before), SH degree increments one step later, SSIM valid conv, LPIPS from `lpips`.
5. Resolution: our images_4/images_2 (official default `-r -1` caps width at 1600 px); the
   paper's protocol for this experiment is unstated → compare deltas.
6. Creator's appearance filter uses training cameras only (official uses all COLMAP cameras of
   the source incl. test views; affects only which nodes are kept/anchored).
7. Evaluation background black (official) rather than the runtime's white convention.
