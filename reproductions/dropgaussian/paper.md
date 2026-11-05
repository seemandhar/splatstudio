# DropGaussian — implementation spec

Park, Ryu, Kim. *DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting*,
CVPR 2025, arXiv 2504.00773. Official code: https://github.com/DCVL-3D/DropGaussian_release
(cloned to `source/code/`, commit at clone time; built on 3DGS + FSGS data tooling).
Sources: `source/latex/sec/*.tex`, `source/paper.txt`, `source/code/`.

## 1. Method summary
- Plain 3DGS trained on sparse views (3 LLFF views), with **no priors, no extra losses**.
- During training, each render randomly drops a fraction `r_t` of all Gaussians (dropout mask
  over the population) and scales the opacity of the kept ones by `1/(1-r_t)`.
- `r_t` grows linearly with the iteration: `r_t = γ t / t_total`, γ = 0.2.
- Test-time rendering uses **all** Gaussians with their **unscaled** opacity (no compensation).
- Gaussians are only temporarily dropped (no population change, no optimizer change).

Contributions
- **C1 DropGaussian with opacity compensation** (§3.2, Eq. 5).
- **C2 Progressive dropping-rate schedule** (§3.2, Eq. 6).

## 2. Equations
Eq. (5): `\tilde{o}_i = M(i)\cdot o_i`, with `M(i) = 1/(1-r)` for kept Gaussians and 0 for
dropped ones.  Code (`gaussian_renderer/__init__.py:93-103`):
```python
if is_train:
    compensation = torch.ones(opacity.shape[0])
    drop_rate = 0.2 * (iteration/10000)
    compensation = torch.nn.Dropout(p=drop_rate)(compensation)   # Bernoulli keep(1-p), x 1/(1-p)
    opacity = opacity * compensation[:, None]                      # opacity = sigmoid(logit)
```
`o_i` is the activated (sigmoid) opacity; `\tilde o_i` may exceed 1 (rasterizer clamps alpha to 0.99).
A fresh mask is drawn for every training render. Gradients flow through the mask (dropped
Gaussians receive zero gradient for every attribute in that iteration).

Eq. (6): `r_t = \gamma \cdot t / t_{total}`, γ ∈ [0,1] ("γ∈{0,1}" in the paper is a typo for an
interval), default γ = 0.2. Code hard-codes `t_total = 10000` (= iterations) and `t` = 1-based
iteration, so `r_1 = 2e-5`, `r_10000 = 0.2`.

Loss (Eq. 7): `L = L_1 + λ L_{D-SSIM}`, λ = 0.2. Code `train.py:81`:
`loss = Ll1 + 0.2 * (1 - ssim)` — **L1 weight is 1.0, not 0.8** as in 3DGS's (1-λ)L1.
SSIM: 3DGS 11×11 Gaussian window σ=1.5, same-padding (`utils/loss_utils.py`).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 10000 | paper §4.1, `arguments/__init__.py` |
| γ (drop scale) | 0.2 | paper Table 5, code renderer:98 |
| t_total in schedule | 10000 (hard-coded) | code renderer:98 |
| λ_dssim | 0.2, L1 weight 1.0 | train.py:81 |
| position lr | 1.6e-4 → 1.6e-6 × extent, max_steps 10000, delay_mult 0.01 (delay_steps 0) | arguments |
| feature/opacity/scale/rot lr | 0.0025 (shN /20) / 0.05 / 0.005 / 0.001 | arguments |
| densify interval / from / until | 100 / >500 / <10000 | arguments, train.py:109-114 |
| densify grad threshold | 5e-4 | paper §4.1 (as FSGS), arguments |
| percent_dense | 0.01 | arguments |
| prune min opacity | 0.005 | train.py:113 |
| screen/world size prune | none (`size_threshold = None` disables both) | train.py:112, gaussian_model.py:398 |
| opacity reset | every 3000 its while densifying (3000, 6000, 9000), to min(o, 0.01) | train.py:116 |
| init opacity / scale | 0.1 / sqrt(mean 3-NN d²) | gaussian_model.py |
| SH | degree 3, +1 every 1000 its | train.py |
| background | black (white_background False) | arguments |
| rendered image | clamped to [0,1] in training and test (renderer:117) | code |
| view sampling | random permutation stack (each view once per 3 its) | train.py |
| Adam | β=(0.9,0.999), eps 1e-15 | gaussian_model.py |

## 4. Schedules
`r_t` linear in t (every iteration); densification 600..9900 every 100; opacity reset 3000/6000/9000;
SH degree +1 at 1000/2000/3000; means LR log-linear decay over 10000.

## 5. Builds on
Vanilla 3DGS (Inria rasterizer, no depth), with FSGS's data pipeline (split, COLMAP MVS init)
and FSGS's grad threshold 5e-4. Everything else is unchanged 3DGS except: L1 weight 1.0,
no size-based pruning, clamped renders, densification until the last iteration.

## 6. Protocol (LLFF 3-view — the main table)
- Scenes: all 8 LLFF scenes (fern flower fortress horns leaves orchids room trex); **only
  averages reported** (no per-scene table, no supplementary in arXiv source).
- Split (`scene/dataset_readers.py:readColmapSceneInfo`): sort cameras by image name; test =
  idx % 8 == 0; train = `round(np.linspace(0, len(rest)-1, 3))` of the remaining images.
- Resolution: `-r 8` on full-res `images/` → PIL `resize` (bicubic default) to round(W/8)×round(H/8)
  = 504×378 (§4.2 "downsampling rates are set to 8 for LLFF").
- Cameras: COLMAP SIMPLE_RADIAL treated as pinhole with centred principal point.
- Init: `tools/colmap_llff.py` (FSGS script): SIFT on the 3 full-res train images, exhaustive
  guided matching, point_triangulator with known poses, image_undistorter, patch_match_stereo,
  stereo_fusion → `3_views/dense/fused.ply`.
- Metrics (`render.py`): PSNR/SSIM (3DGS) on clamped renders, LPIPS **VGG** (lpipsPyTorch),
  averaged over test views, final iteration 10000. Seeds not stated.
- Mip-NeRF 360: 12/24 views, downsample 8 (averages only). Blender: 8 views, /2.

## 7. Ambiguities
- Default config is 20.76 dB in Table 1 but 20.56 dB in ablation Table 5 (same setting);
  Mip-NeRF 360 text says 23.92 dB for 24 views vs 24.13 in Table 2. Treat ±0.2 dB as the
  paper's own run-to-run noise.
- Opacity compensation can push opacity > 1; the official rasterizer clamps alpha at 0.99.
- Dropped Gaussians still count as "visible" for densification statistics in the official
  rasterizer (radius independent of opacity) → their denominator increments with zero gradient.
- 3DGS baseline numbers (19.22) are copied from FSGS/CoR-GS, not retrained with this code.
