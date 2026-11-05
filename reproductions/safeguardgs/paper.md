# SafeguardGS — implementation spec

Paper: *SafeguardGS: 3D Gaussian Primitive Pruning While Avoiding Catastrophic Scene Destruction*,
Yongjae Lee, Zhaoliang Zhang, Deliang Fan. arXiv [2405.17793](https://arxiv.org/abs/2405.17793)
(v2, 22 Nov 2024; LaTeX in `source/latex/`), final version WACV 2026 (`source/wacv.pdf`,
supplementary `source/wacv_supp.pdf`). Official code
https://github.com/ASU-ESIC-FAN-Lab/SafeguardGS (commit `539ca22e`, 2026-03-05; Inria 3DGS
licence, non-commercial research) in `source/code/`.

## 1. Method summary
- Pruning study for vanilla 3DGS: scene-level (threshold) vs pixel-level (rank) pruning.
- **SafeguardGS** = pixel-level importance-rank (PLIR) pruning: for every training ray, rank
  the Gaussians intersecting it by a score and keep the top-k; a Gaussian survives if it is in
  the top-k of *any* ray of *any* training view (OR of masks); all others are removed.
- Score: `Score_SG_36 = CS1(c_r,gt, c_i) · α_i · T_i` (colour similarity × blending weight).
- Setting: top-k = 1, applied **once during training at iteration 20 000** of a 30 000-iteration
  vanilla 3DGS run (so **during training, not post-training**); ≈ 89 % of primitives removed.
- Everything else is vanilla 3DGS (densification 500–15k, opacity reset every 3k, LR, SH schedule).

Contributions:
- C1 (§3.2/3.4, code `forward.cu:renderCUDA_topk_color` + `rasterizer_impl.cu:markingTopKMasks`):
  per-ray top-k rank pruning ("safeguard": every ray keeps ≥ 1 primitive), union over views.
- C2 (§3.3, Tab. 1/5): score function family; selected SG_36 = CS1 · α · T (bit mask `0x24`).
- C3 (§4.1): schedule — prune once at 20k of 30k, top-1.
- (Analysis only, not a method component: SLIT vs PLIR categorisation; Tab. 4 "-SG" variants
  that substitute SafeguardGS into LightGaussian / PUP-3DGS / Speedy-Splat pipelines.)

## 2. Equations
Rendering (Eq. 1):
```latex
\mathbf{c}_r = \sum_{i \in M} c_i w_i,\quad w_i = \alpha_i T_i,\quad \alpha_i = \sigma_i \mathcal{G}_i,\quad T_i = \prod_{j=1}^{i-1}(1-\alpha_j)
```
Squashing functions (Eq. 2, supp. §6):
```latex
PC(X_r, X_i) = \exp(-||X_r - X_i||_2)
CS_1(\mathbf{c}_{r,gt}, c_i) = 1 - \tfrac{1}{C}||\mathbf{c}_{r,gt} - c_i||_1
CS_2(\mathbf{c}_{r,gt}, c_i) = \exp(-\tfrac{1}{C}||\mathbf{c}_{r,gt} - c_i||_1)
CS_3 = (\mathbf{c}_{r,gt}\cdot c_i)/(||\mathbf{c}_{r,gt}||_2\,||c_i||_2)
```
Selected score:
```latex
Score_{SG\_36}(\mathbf{G}_i, r) = CS_1(\mathbf{c}_{r,gt}, c_i) \cdot \alpha_i \cdot T_i
```
`c_{r,gt}` = ground-truth pixel colour of ray r (training image), `c_i` = the Gaussian's
view-dependent SH colour (3DGS `computeColorFromSH`, clamped ≥ 0, not ≤ 1), `X_i` projected
centre, `T_i` transmittance *before* Gaussian i. C = 3 channels. Function ids are bit masks
(supp. §6): low nibble 0x0–0xf = factor combination (0x04 = α·T), high nibble 0x10 ×CS3,
0x20 ×CS1, 0x30 ×CS2, 0x40 +CS1, 0x50 +CS2. SG_36 = 0x24, SG_4 = 0x04 (= EfficientGS).

Pruning rule: keep G_i iff ∃ view v, ray r: rank of G_i among Gaussians on r by score ≤ k.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| score function | 0x24 (SG_36) | paper §3.4; `benchmark/benchmark.yaml`; `arguments/__init__.py:129` |
| top-k | 1 | paper §3.4/§4.3; `benchmark.yaml` (code default `safeguard_gs_purne_topk=10`) |
| prune iterations | [20000] | paper §4.1, supp. §7; `benchmark.yaml prune_iterations` (code default 15000) |
| p_dist / c_dist activation coef | 1.0 / 1.0 | `arguments/__init__.py:154-155` (unused by 0x24) |
| score quantisation | `floor((1-score)·65535)` in a 32-bit key | `forward.cu:1338-1339` |
| iterations | 30000 | paper §4.1 |
| densification | 3DGS: from 500, until 15000, every 100, grad 2e-4, opacity reset 3000, min opacity 0.005, screen size 20 after 3000, world size 0.1·extent | `arguments/__init__.py:96-102`, `train.py` |
| background | black (`white_background` False, `random_background` False) | `arguments` |
| views used for scoring | all training cameras, each once, with GT image | `gaussian_model_safeguard_gs.py:topk_gaussians` |

## 4. Schedules
- 1–15000: vanilla 3DGS densification. SH degree +1 every 1000 iterations.
- 20000 (inside the training loop, after densification bookkeeping, **before** that iteration's
  optimizer step; `prune_after_densify`): render every training view with the top-k kernel,
  OR the masks, `prune_points(~mask)` (3DGS prune keeps Adam state of survivors).
- 20001–30000: plain optimisation of the survivors.

## 5. What it builds on
Vanilla 3DGS (Inria code), unchanged except the single pruning event. The rasterizer is the Inria
one extended with score kernels.

## 6. Protocol
Datasets: Synthetic NeRF, Mip-NeRF 360 (9 scenes), Tanks&Temples (truck, train), Deep Blending
(drjohnson, playroom). llffhold-8 test split. Paper text: "at their original resolutions"; the
released sweep (`benchmark/*.yaml`) passes no `-i/-r`, so 3DGS's loader uses `images/` with the
default auto-downscale to 1600 px width; the repo's `full_eval.py` uses images_4 / images_2 (3DGS
convention). Metrics: PSNR, SSIM, LPIPS (3DGS `metrics.py`, lpipsPyTorch VGG), #primitives,
compression ratio, PSNR-per-primitive. A100. **Only dataset averages are reported** (main Tab. 3
and supp. sweep figures; no per-scene tables in the arXiv or WACV supplementary).

## 7. Ambiguities / paper vs code
1. **"Per pixel" is per tile in the official kernel.** `renderCUDA_topk_color` writes one key per
   (tile, batch b of 256 Gaussians, pixel thread t) into slot `range.x + 256·b + t`; inside a batch
   each contributing Gaussian overwrites the slot, so the slot holds the *last* (deepest)
   contributing Gaussian of that batch for that pixel. Keys `(tile_id << 32) | floor((1-s)·65535)`
   are radix-sorted (stable) and `markingTopKMasks` keeps the first k entries **per tile**. With
   k = 1 each view keeps ≤ 1 Gaussian per 16×16 tile. This matches the reported counts (≈ 350k
   survivors on Mip-NeRF 360, ≈ 89 % removed); the paper text describes a true per-ray top-k.
   Code overrides paper → port the kernel semantics; offer the paper-text per-pixel top-k as a
   config ablation.
2. Official race: in the last partial batch, threads with `t ≥ toDo` write slots beyond the tile's
   range (into the next tile's slots, or past the buffer for the last tile); which block wins is
   nondeterministic. Not reproducible; we give each tile its own padded slot range.
3. Empty slots (key −1) form a pseudo-tile after the sort; `markingTopKMasks` marks the first one,
   whose value is 0 → Gaussian 0 always survives. Inconsequential; not reproduced.
4. Opacity / scales in the prune render: `get_opacity_and_scales_for_train` = plain activations.
5. Resolution: see §6.
6. Code defaults (15000 / top-10) differ from the paper experiments (20000 / top-1, set by the
   sweep yaml); the paper values are used.
