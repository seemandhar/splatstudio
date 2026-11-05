# SafeguardGS — native SplatStudio plan

Method `safeguardgs`, module `ss_safeguardgs`, package `splatstudio-method-safeguardgs`.
**Prunes during training** (once, at step 20000 of 30000), not post-training.

## Components
| kind | name / factory | requires → provides | config (paper value) |
|---|---|---|---|
| renderer | `sg-renderer` `ss_safeguardgs.renderer:create_renderer` (stock gsplat + target pass-through, after the mini-splatting plugin) | state.gaussian3d, camera.pinhole → render.rgb, render.depth.expected_z, gradient.screen.signed, training.target_image | absgrad false, antialiased false |
| density | `sg-density` `ss_safeguardgs.density:create_density` (wraps `splatstudio.backends.gsplat.Density` = 3DGS densify + opacity-reset fix) | gradient.screen.signed, training.target_image | `prune_steps [20000]`, `top_k 1`, `score_function 0x24 (36)`, `selection "tile"` (official kernel; `"pixel"` = paper-text per-ray ablation), `p_dist_coef 1.0`, `c_dist_coef 1.0`, `base` = gsplat 3DGS density config |
| loss | `photometric` (unchanged L1 + 0.2 D-SSIM) | render.rgb | — |

No new fields.

## Implementation notes
- `csrc/sg_topk.cu` (adapted from official `forward.cu:compute_score` / `renderCUDA_topk_color`,
  tile traversal after `reproductions/efficientgs/.../egs_raster.cu`, gsplat numerics):
  per tile, per 256-Gaussian batch b, per pixel t: key = `floor((1-score)·65535)` of the last
  contributing Gaussian of the batch (official overwrite semantics), written to a tile-private
  padded slot `pad_off[tile] + 256 b + t` (official writes `range.x + 256 b + t` and races across
  tiles in partial last batches — gap). Score = official `compute_score` (all bit-mask ids).
  `selection="pixel"`: running per-pixel top-k (k ≤ 16) of the float score over all contributors.
- `core.py`: torch reference of the score, quantisation, per-tile top-k from sorted keys
  (stable sort on `(tile << 32) | q`, ties by slot order = official stable radix sort,
  `markingTopKMasks` = first k per tile), padded offsets.
- Density: forwards to the base 3DGS density; caches every training view's target (uint8,
  matched by viewmat) from renderer outputs; at `step ∈ prune_steps` renders each training view
  (SH degree of that step, 3 at 20k) with gsplat, runs the kernel with the GT image, ORs the keep
  masks and `gsplat.strategy.ops.remove(params, optimizers, base.state, ~keep)` (Adam moments of
  survivors kept, as 3DGS `prune_points`). Logs an event with counts.
- Runtime differences: density runs after the optimizer step of step 20000 (official: before);
  white background (irrelevant to the score: α, T, colours and GT do not depend on it).

## Probes
| probe | measures | fails if |
|---|---|---|
| `sg-topk-kernel` | CUDA keep masks (tile and pixel modes, k = 1, 2, ids 0x24 / 0x04) equal an independent torch per-pixel blend + official slot/sort logic on a random scene; tile ≠ pixel semantics | kernel score, slot rule, tile top-k or union wrong |
| `sg-color-score` | analytic tile: front blue Gaussian (higher α·T) vs back red Gaussian on a red GT → SG_4 keeps front, SG_36 keeps back; every covered tile keeps ≥ 1 Gaussian (safeguard) | colour term removed / wrong |
| `sg-prune-schedule` | density on a toy scene: no prune before 20000, one prune at 20000 using cached GT, survivors = union of per-view masks, params/Adam/strategy state aligned, error if a view's GT is missing | schedule or bookkeeping wrong |

## Fidelity gaps
1. Official cross-tile slot race / Gaussian-0 artefact not reproduced (deterministic tile-private
   slots). Effect: small change in survivors; cannot be closed (nondeterministic upstream).
2. gsplat numerics (alpha cap 0.999 vs 0.99, `next_T <= 1e-4` vs `< 1e-4`) — scores match the image
   we train; tiny effect.
3. Resolution: our images_4 / images_2 vs the released sweep's 1600-px auto-downscale (paper:
   "original resolutions"). Survivor count scales with the number of 16×16 tiles, so our counts are
   expected lower than the paper's 350k; compare compression ratio and ΔPSNR vs our baseline.
4. Native runtime: white background, view sampling with replacement, SSIM valid conv, LPIPS from
   `lpips`, density after optimizer step.
5. Baseline is our gsplat 3DGS (`baseline3dgs-mipnerf360-30000`), not the authors' retrained 3DGS
   (27.434). Compare ΔPSNR: paper −0.361 dB on Mip-NeRF 360.
6. Only dataset averages are published → benchmark must use all 9 Mip-NeRF 360 scenes.
