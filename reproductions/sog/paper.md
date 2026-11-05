# Self-Organizing Gaussians (SOG) — implementation spec

Paper: *Compact 3D Scene Representation via Self-Organizing Gaussian Grids*, Morgenstern, Barthel,
Hilsmann, Eisert, ECCV 2024. arXiv 2312.13299 (v2, 2 May 2024; LaTeX in `source/latex/`).
Official code: https://github.com/fraunhoferhhi/Self-Organizing-Gaussians @ `5f2b360`
(`source/code/`, fork of graphdeco 3DGS, Inria non-commercial license). PLAS sorter:
https://github.com/fraunhoferhhi/PLAS (Apache-2.0), submodule pin `782a3ae` (`plas/core.py`
identical to HEAD `4f1109c`; `source/plas/`).

## 1. Method summary
- All Gaussians are arranged in a square 2D grid (row-major order of the parameter tensors);
  every attribute is one multi-channel "image" with the same layout.
- The grid is (re-)sorted with PLAS (parallel linear assignment sorting) at the start and after
  every densification interval, using position, DC colour and scale as sort keys.
- A smoothness ("neighbor") loss pulls opacity and rotation towards a blurred version of their
  grid, making the grids compressible.
- Positions are trained in a log-contracted space `x = sign(y)(exp|y| - 1)`.
- After training the grids are quantized and stored as JPEG XL images; the size is the sum of the
  compressed files; quality is measured on the decoded model.

Contributions:
- **C1 Grid + PLAS sorting** (§3.1, supp. "Sorting"): prune to the largest square, sort into the
  grid at init and after every densification step.
- **C2 Smoothness regularization** (§3.2, supp. Table "params"): Huber loss between grid and
  detached blurred grid.
- **C3 Log coordinate activation** (§3.3).
- **C4 Quantization + JPEG XL compression; decoded evaluation** (§3.4, §4 "choice of compression").
- **C5 Modified densification** (supp. "Parameter selection"): interval 1000, grad thr 7e-5,
  min opacity 0.1, no opacity reset, percent_dense 0.1.

## 2. Equations
(§3.3) `x = \text{sign}(x_{\text{log}}) \times (\exp(|x_{\text{log}}|) - 1)`
code: `inverse_log_transform(y) = sign(y)*expm1(|y|)`, `log_transform(x) = sign(x)*log1p(|x|)`
(`scene/gaussian_model.py:27-32`). Means param `_xyz` stores `log_transform(points)`; renderer
uses `get_xyz = inverse_log_transform(_xyz)`; densify-split samples in world space then maps back
(`gaussian_model.py:415`); clone copies `_xyz`.

Smoothness loss (§3.2; `gaussian_model.py:576-606`, `train.py:121-140`):
```
L_nb = λ Σ_a (w_a / Σ_b w_b) · Huber_δ=1( G_5x5,σ=3( grid(a).detach() ), grid(a) )   (mean)
```
`grid(a)` = un-activated attribute (activated: false, normalize: false) reshaped
`(S, S, C)` → `(1, C, S, S)`; blur = kornia `gaussian_blur2d` separable: x pass kernel (1,5)
border **circular**, then y pass kernel (5,1) border **reflect**, sigma (3,3). `F.huber_loss(blurred, img)`.
Weights (code yaml): opacity 1, rotation 10, others 0 → normalised 0.09 / 0.91 (supp. table).
Total loss: `(1-0.2) L1 + 0.2 (1-SSIM) + L_nb`.

Sorting (§3.1; `gaussian_model.py:515-540`): keys = activated attributes (get_xyz world coords,
features_dc raw, exp(scaling)), each standardized `(t - mean(t)) / std(t)` over the whole tensor,
times weight (xyz 1, dc 1, scaling 1; opacity/rotation/rest 0) → 9 channels; random shuffle →
grid `(C,S,S)` → `sort_with_plas(grid, improvement_break=1e-4)` (min_block_size 16,
min_blur_radius 1, border x circular / y reflect) → permutation applied to all params and
optimizer state (`prune_all_but_these_indices`).

Prune to square (`gaussian_model.py:490-506`): `S = floor(sqrt(N))`; keep the `S²` highest
opacities (topk), in original order.

Quantization / compression (§3.4, `compression/compression_exp.py:compress_attr`, config
`config/compression/umbrella_sh.yaml` experiment `jxl_quant_sh`):
```
quantize(q): a_n = (a - min a)/(max a - min a); a = round(a_n 2^q)/2^q (max-min) + min   (float32)
normalize:   a = (clip(a, lo, hi) - lo)/(hi - lo)  (_xyz: lo,hi = its own min,max)
```
| attribute (stored un-activated) | quantize | normalize [lo,hi] | JPEG XL level |
|---|---|---|---|
| `_xyz` (log space) | 14 bits | own min/max | 101 (lossless) |
| `_features_dc` | — | [-2, 4] | 100 (lossy) |
| `_features_rest` (45 ch) | 5 bits | [-1, 1] | 101 |
| `_scaling` (log) | 6 bits | none | 101 |
| `_rotation` (raw quat) | 6 bits | [-1, 2] | 101 |
| `_opacity` (logit) | 6 bits | [-6, 12] | 101 |
Images are float32 `(S,S,C)` arrays written with `imagecodecs.imwrite(f, a, level=L)`
(imagecodecs **2023.9.18**; README/issue #3: 2024.9.22 gives 4× larger files). Decode:
`imread` → `img·(hi-lo)+lo` with lo/hi read back from `compression_info.csv` (float64).
Size = sum of all files in the compressed dir: 6 `.jxl` + `compression_info.csv` +
`compression_config.yml` (`decompress.py:get_size_of_files_in_dir`); MB = bytes / 1e6.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | yaml `optimization.iterations` |
| position lr | 1.6e-4 → 1.6e-6 ×extent (on log-space xyz) | yaml |
| feature/opacity/scaling/rotation lr | 0.0025 / 0.05 / 0.005 / 0.001, rest /20 | yaml, 3DGS |
| lambda_dssim | 0.2 | yaml |
| densification_interval | 1000 (3DGS 100) | supp. Table, yaml |
| densify_from / until | 500 / 15000 | yaml |
| densify_grad_threshold | 7e-5 | supp. Table, yaml |
| densify_min_opacity (prune) | 0.1 | supp. Table, yaml |
| opacity_reset_interval | 1e7 (disabled) → `size_threshold` None (no screen/world size pruning) | yaml, train.py:226 |
| percent_dense | 0.1 | supp. Table, yaml |
| λ_neighbor | 1.0 | supp. Table, yaml |
| blur kernel / sigma | 5 / 3.0 | supp. Table, yaml |
| neighbor weights | opacity 1, rotation 10 (normalized 0.09/0.91) | yaml / supp. Table |
| neighbor loss_fn / activated / normalize | huber / false / false | yaml |
| sorting weights | xyz 1, dc 1, scaling 1 | supp. Table, yaml |
| sorting normalize / activated / shuffle | true / true / true | yaml |
| PLAS improvement_break | 1e-4 | supp. fig. caption, yaml |
| SH degree | 3, +1 every 1000 | yaml `use_sh: true` |
| compression | `jxl_quant_sh` (table above) | yaml `compression: umbrella_sh` |

## 4. Schedules
- init: prune to square + sort (train.py:65-67).
- every iteration: neighbor loss on the current grid.
- `500 < it < 15000`, `it % 1000 == 0`: densify (clone/split, prune opacity < 0.1).
- `it > 500`, `it % 1000 == 0` (through 30000): prune to square + sort (after densification).
- final (30000): compress → decode; reported metrics are on the decoded model.
- No opacity reset; no size-based pruning.

## 5. Builds on
Official 3DGS (graphdeco) training loop; only densification params, the log xyz activation,
sorting, neighbor loss and compression change. Photometric loss, LRs, SH schedule unchanged.

## 6. Protocol
Mip-NeRF 360 (images_4 outdoor / images_2 indoor, `full_eval.py`), T&T, Deep Blending,
Synthetic-NeRF; llffhold 8 (`eval: true`); black background; 30k iterations; metrics with the
original 3DGS `render.py`/`metrics.py` on the decompressed model (LPIPS VGG).

## 7. Ambiguities
- Paper says DC is stored "as an 8-bit image, quality 100": code passes float32 [0,1] to
  imagecodecs (lossy JXL on float). Code followed.
- Paper Table 1 PSNRs (v2) vs repo per-scene CSVs (2024-09-16, decoded models): CSV used as the
  reference (see paper_results.json notes).
- Paper quantization says "2^q values"; code rounds `a_n·2^q` → 2^q+1 levels. Code followed.
- Neighbor-loss weights in yaml (1, 10) vs supp. Table (0.09, 0.91): identical after the code's
  normalisation by the weight sum.
- Initial prune to square at init: all opacities equal (0.1) → topk tie order arbitrary.
- Official optimizer step is skipped at every sort/densify iteration (params replaced before the
  step lose `.grad`), and no step at iteration 30000.
