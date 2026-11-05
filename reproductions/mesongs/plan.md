# MesonGS — native SplatStudio plan

Method family `mesongs-*`, module `ss_mesongs`, package `splatstudio-method-mesongs`.
Spec: `paper.md`. Precedents (copied with attribution, never imported): c3dgs (run shape, bank,
renderer, VQ + official `weighted_distance` kernel, run-length guard), pup-3dgs (pretrain means
LR), reduced-3dgs / rdo-gaussian (encode → decode → replace params, real bitstream size).

## Run shape (single run of total_steps = 30000 + 8000)
* 1..30000: unchanged SplatStudio 3DGS density (opacity-reset fix), means LR of a 30000-step run
  = recipe of `baseline3dgs-mipnerf360-1600-30000` (the paper protocol is width-1600 images).
* after step 30000 (`compress`): C1 importance over all training cameras (I_d via gsplat:
  d Σ render / d c_i of a 1-channel colour = Σ_p α_i T_i), volume term, prune τ; C2 octree
  voxelisation (official `octreecodes`, mean merge, lower-corner positions); RAHT tables
  (official 3DAC port); 10·n_block `VanillaQuan` block quantisers; C6 VQ of the 45 SH-rest values
  (c3dgs `vq_features`, official misaligned importance). Params replaced by the voxelised model
  (fresh Adam = official `finetuning_setup`), shN field = codebook[idx] (LR 0, unused while the
  bank is active).
* 30001..38000 (fine-tune, C8): renderer renders through the `MesonBank`: normalise quats →
  Euler (C3) → RAHT of [opacity logit, Euler, SH DC] (C4) → DC kept, AC block-fake-quantised (C5)
  → inverse RAHT; log-scales block-fake-quantised; covariance from Euler; SH rest = codebook[idx].
  Runtime LRs set to the official fine-tune LRs (× per-scene `lr_scale·4`, scaling × extent),
  means LR 0; codebook has its own Adam in the bank. LRs are 0 for the last runtime step and the
  bank skips it (official `iteration < iterations`).
* step 38000 (`encode`, C7): official 6 npz files (`np.savez_compressed`) zipped (DEFLATE) in
  memory, size = zip bytes; decode the zip (official `load_npz`) into the base fields (Euler →
  rotation matrix → quaternion), so checkpoint/validation/`ss-eval` measure the decoded model.
* `mesongs-noft-*`: total = base = 30000 (compress + encode at 30000) = paper "Our" row.

RAHT forward/inverse are linear and orthonormal (I∘F = id), so they are custom autograd
Functions whose backward is the other transform (memory O(N) instead of 60 saved copies).

## Components
| contribution | kind | factory | config |
|---|---|---|---|
| base 3DGS | density (wrapped) | `splatstudio.backends.gsplat.Density` | make_method config |
| C1–C8 | density | `ss_mesongs.density:create_density` | scene table (prune, cb, depth, n_block, beta, lr_scale), bits 8, vq batch 262144 / steps 1000 / decay 0.8, base 30000, total 38000 |
| bank render | renderer | `ss_mesongs.renderer:create_renderer` | gsplat caps (covars from Euler) |
| loss | loss | photometric (0.8 L1 + 0.2 D-SSIM = official) | — |

Variants (entry points): `mesongs-<scene>` (config3 = released script) for all 13 real scenes,
`mesongs-c1-<scene>` and `mesongs-noft-<scene>` for garden/bicycle/counter, `mesongs-smoke`
(garden table, 1500 + 500 steps).

## State
No FieldSpecs. Bank (process-local `ACTIVE`): RAHT tables on GPU, reorder, codebook [K,45]
(trainable), indices, quantiser running min/max buffers [10, n_block]. Positions, opacities,
quats, sh0, scales stay runtime fields.

## Probes
| probe | fails if |
|---|---|
| `importance` (CUDA) | I_d != brute-force α·T accumulation from an independent per-pixel compositor; volume term / prune quantile wrong |
| `octree` (CPU) | voxel keys / mean merge / lower-corner decode differ from the verbatim official `octreecodes` port |
| `raht` (CPU+CUDA) | fast tables != verbatim port; I∘F != id; F not orthonormal; custom backward != autograd of the functional version; DC != weighted mean·√W |
| `block-quant` (CPU) | scale/zero-point/running range/STE differ from VanillaQuan formulas; split_length wrong |
| `euler` (CPU) | Euler round trip R(q) != R(euler(q)) via the official matrix |
| `codec-roundtrip` (CUDA) | decoded render != bank render; size != real zip bytes; layout differs |
| `pipeline` (CUDA) | tiny end-to-end: pruning count, merge, fine-tune moves attributes not means, codebook changes, last step frozen, encode replaces params |
| `schedule` (CPU) | events / run-length guard |

## Fidelity gaps
1. Base model = our gsplat 3DGS (white bg), not the authors' 3DGS checkpoints; compare deltas.
2. Fine-tune length 8000 taken from the released mic script; M360 value unstated.
3. Official saves the best model by test PSNR every 400 FT iterations; we report the final model.
4. Runtime samples views with replacement (official shuffled stack).
5. gsplat rasteriser for I_d and rendering (official modified 3DGS rasteriser).
6. Euler `sqrt` argument clamped at 1e-12 (official nan_to_num only) to avoid NaN gradients.
7. Background white (paper) while the released block script trains/evaluates black.
8. Decoded model renders from quaternions recovered from the Euler rotation matrix (float-level
   difference from the official covariance-from-Euler render).
