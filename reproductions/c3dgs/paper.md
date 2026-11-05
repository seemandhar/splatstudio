# Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis — implementation spec

arXiv 2401.02436 (CVPR 2024), Niedermayr, Stumpfegger, Westermann (TUM).
Official code: https://github.com/KeKsBoTer/c3dgs (cloned to `source/code/`, Gaussian-Splatting
non-commercial research licence). LaTeX in `source/latex/` (`sec/4_method.tex`,
`sec/5_results.tex`, `sec/X_suppl.tex`).

## 1. Method summary
- Post-training compression of a trained 3DGS model (the 3DGS authors' 30k-iteration models).
- Per-parameter **sensitivity** = mean over all training pixels of |dE_i/dp|, E_i = sum of RGB of
  rendered training image i (one backward per training view, colours not clamped).
- **Pruning**: Gaussians whose maximum colour sensitivity is 0 are removed (supp. "Sensitivity
  Calculation and Pruning"; ablation "+ Pruning").
- **Sensitivity-aware vector clustering** (weighted mini-batch k-means with EMA updates) of
  (a) the full 48-dim SH vector into a colour codebook and (b) the 6-dim normalised covariance
  into a shape codebook (4096 entries each). Vectors whose sensitivity exceeds beta_c / beta_g are
  not clustered and are appended to the codebook. Shape codebook entries are decomposed back into
  rotation + normalised scale by eigendecomposition; each Gaussian keeps a scalar scale factor eta.
- **Quantization-aware fine-tuning** (5000 steps, 3DGS losses/LRs): 8-bit min-max fake
  quantisation of everything but positions (float16), codebook entries receive accumulated
  gradients.
- **Entropy encoding**: Morton-order sort, DEFLATE (`np.savez_compressed`).

Contributions:
- C1 sensitivity measure + zero-sensitivity pruning (§4.1 Eq. 1, supp. §Sensitivity)
- C2 sensitivity-aware vector clustering of colour (§4.1 "Color Compression") and of shape via
  normalised covariance (§4.1 "Gaussian Shape Compression", supp. §Covariance Matrix Clustering)
- C3 quantization-aware fine-tuning (§4.2)
- C4 Morton-ordered entropy encoding (§4.3)
- (renderer: WebGPU hardware rasteriser, §5 — out of scope, speed claim not reproduced)

## 2. Equations (verbatim)
Eq. (sensitivity):
`S(p) = \frac{1}{\sum_{i=1}^N P_i} \sum_{i=1}^N \big| \frac{\partial E_i}{\partial p} \big|`
N training images, P_i pixels of image i, E_i = sum of RGB components over all pixels.
Code (`compress.py:calc_importance`): hooks `grad.abs()` on `_features_dc`, `_features_rest`
and on the normalised covariance `cov3d` (6 upper-triangular entries, computed from
`scaling_qa(normalize(relu(_scaling)))` and `get_rotation`); per camera
`render(..., clamp_color=False, cov3d = cov3d * scaling_factor**2)` on a **black** background,
`loss = rendering.sum(); loss.backward()`; divide by `num_pixels = sum H*W`.

Vector sensitivity: `S(\boldsymbol{x}) = \max_{d\in [1..D]} S(x_d)` (code: `.amax(-1)` over the
48 SH values / the 6 covariance values).

Weighted distance: `\mathcal{D}(\boldsymbol{x},\boldsymbol{c}_k) = S(\boldsymbol{x}) \lVert \boldsymbol{x}-\boldsymbol{c}_k \rVert_2^2`
(the argmin over k is independent of S(x); code `weighted_distance.cu` computes the plain
squared L2 distance with a first-minimum loop).

Centroid update: `\boldsymbol{c}_k = \frac{1}{\sum_{x_i\in A(k)} S(x_i)} \sum_{x_i\in A(k)} S(x_i) x_i`,
batched (Sculley 2010): random batch, EMA with decay lambda_d. Code (`compression/vq.py`):
`importance_n = importance / importance.max()`; codebook init `rand * (amax - amin) + amin`
with scalar min/max over all elements; per step batch `torch.randint(N, (batch,))` (with
replacement); `new = scatter_sum(x * w) / (scatter_sum(w) + 1e-5)`;
`c = 0.8 c + 0.2 new` (empty clusters decay towards 0); shape codebook re-normalised after
every step by its trace (entries 0, 3, 5); final assignment of all clustered vectors.

Shape: `s = \eta \hat{s}`, `\eta = \lVert s \rVert_2`, `\hat\Sigma = (R\hat S)(R\hat S)^T = \Sigma/\eta^2`.
Decomposition (`utils/splats.py:extract_rot_scale`): `eigh(cov + 1e-8 I, UPLO="U")`,
`scale = sqrt(eigvals).nan_to_num(1e-6)`, `rot = normalize(matrix_to_quaternion(R * det R))`.

QA (§4.2): forward `\hat p = (clamp(\lfloor p/s + z \rceil, 0, 2^b-1) - z) s`, min/max as moving
averages; backward straight-through. Code: `torch.ao.quantization.FakeQuantize(dtype=torch.qint8)`
(MovingAverageMinMaxObserver, averaging constant 0.01, per-tensor affine, range [-128, 127],
zero always inside the range; gradient zeroed for clamped values) for f_dc, f_rest (codebook
entries), opacity (after sigmoid), normalised scaling (after relu+normalize), log scale factor
(before exp), rotation (before normalize); positions: `x.half().float()` with identity backward.
Observers stay enabled for the whole compression run (every attribute access updates them).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| colour codebook size | 4096 | §5.2, `arguments CompressionParams.color_codebook_size` |
| shape codebook size | 4096 | same, `gaussian_codebook_size` |
| beta_c (colour keep threshold) | 6e-7 | §5.2, `color_importance_include` |
| beta_g (shape keep threshold) | 3e-6 | §5.2, `gaussian_importance_include` |
| prune threshold | 0.0 (prune colour sensitivity <= 0) | `prune_threshold` |
| colour clustering steps / batch | 100 / 2^18 | §5.2, code |
| shape clustering steps / batch | 800 / 2^20 | §5.2, code |
| EMA decay lambda_d | 0.8 | §5.2, `vq_features(decay=0.8)` |
| centroid eps | 1e-5 | `VectorQuantize.eps` |
| colour vector | all 48 SH values incl. DC | `color_compress_non_dir = True` |
| QA fine-tune iterations | 5000 (4999 optimizer steps) | §5.2, `finetune.py` |
| fine-tune LRs | f_dc 0.0025, f_rest 0.000125, opacity 0.05, scaling (normalised codebook) 0.005, scaling_factor (log eta) 0.005, rotation 0.001, Adam eps 1e-15, fresh Adam | `gaussian_model.training_setup` |
| fine-tune position LR | **0** in code (`spatial_lr_scale` is never set when loading a .ply, so `position_lr * 0`); paper text says positions are optimised | code overrides paper |
| fine-tune loss | 0.8 L1 + 0.2 (1 - SSIM) | `finetune.py` |
| quantisation | 8-bit min-max (qint8) all but position; position float16 | §4.2 |
| Morton grid | 21 bits per axis | `_sort_morton` |
| index storage | int32 per codebook index | §4.3, `save_npz` |

## 4. Schedules
3DGS 30k unchanged (authors' released models) → sensitivity (1 pass over the training views) →
prune → colour clustering → shape clustering → 5000 fine-tune iterations, no densification, no SH
schedule (degree 3), xyz LR from `get_expon_lr_func(max_steps=30000)` evaluated at 30001..35000
(= lr_final * spatial_lr_scale = 0) → Morton sort → npz.

## 5. Builds on
3DGS (Kerbl et al. 2023). 3DGS training is untouched; the paper uses the published 3DGS
reconstructions and their training settings (black background for Mip-NeRF 360).

## 6. Protocol
Mip-NeRF 360 (9 scenes, 3DGS resolution: images_4 outdoor / images_2 indoor), T&T (train,
truck), Deep Blending (drjohnson, playroom), NeRF-Synthetic; 3DGS llffhold-8 test split.
PSNR/SSIM/LPIPS-VGG on test views. Size = `os.path.getsize(point_cloud.npz) / 1024**2` (MiB),
3DGS size = `point_cloud.ply` file size. Per-scene tables in the supplementary.
The metrics printed by `compress.py` are computed on the in-memory fake-quantised model; this
equals the stored npz (same scales/zero-points) up to the second fake-quantisation pass of
`render.py`.

## 7. Ambiguities / code vs paper
- Positions: paper "we optimize for the position", code LR 0 (frozen float16 positions).
- Pruning only appears in the supplementary and ablation; code default `prune_threshold=0.`.
- Sensitivity is computed on the fake-quantised model (observers initialise there).
- Last fine-tune iteration does no optimizer step (`if iteration < max_iter`).
- Stored rotation is `Q(normalize(r))` while QA renders `normalize(FQ(r))` (`save_npz`).
- `load_npz` + `render.py` renormalise the decoded scale (`normalize(relu)`) and rotation.
- `opacity_qa` is never disabled even when quantization is off (irrelevant: QA on by default).
- T&T average PSNR: main Table 1 prints 23.324, supp. table 23.343.
