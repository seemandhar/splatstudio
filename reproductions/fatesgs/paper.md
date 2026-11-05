# FatesGS — implementation spec

arXiv 2501.04628 (AAAI 2025 oral). Official code: https://github.com/yulunwu0108/FatesGS
(commit a721d318, cloned to `source/code/`, built on 2DGS). LaTeX + supplementary PDF in `source/latex/`.

## 1. Summary and contributions
- Sparse-view (3 views) **surface reconstruction** on a 2DGS backbone; mesh by TSDF fusion of
  rendered depth of the training views. Metric: Chamfer distance (mm) on DTU.
- **C1 Patch-based depth ranking loss** L_r (§Intra-View Depth Consistency, Eq. 8): inside
  M×M patches, random pixel pairs must keep the ordering of the monocular (Marigold) depth.
- **C2 Edge-aware depth smoothness** L_s (Eq. 9): hinge on neighbouring rendered-depth
  differences, only where monocular depth is continuous.
- **C3 Multi-view multi-level feature alignment** L_f (§Multi-View Feature Alignment,
  Eq. 10-14): back-project the rendered depth of the reference view, project to its 2 source
  views (pair.txt), compare frozen Vis-MVSNet features (levels l=1,2) with cosine distance,
  with a z-buffer visibility term v.
- Base: 2DGS (L_c = 0.8 L1 + 0.2 D-SSIM, depth distortion L_d, normal consistency L_n),
  3DGS adaptive density control, COLMAP MVS (`dense/fused.ply` from the 3 views) init.

## 2. Equations (verbatim LaTeX, paper) and code
Rendered depth: $\hat{D}(\bm{r}) = \frac{\sum_{i} \omega_i d_i}{\sum_{i} \omega_i + \epsilon}$ (2DGS
ray-splat intersection depth; code `surf_depth` with depth_ratio = 0 = expected depth).

Eq. 8 ranking: $\mathcal{L}_{r}=\sum_{i,j}\sigma\left(\mathrm{sgn}\left(\widetilde{D}^{(i)}_{k_j}-\widetilde{D}^{(i)}_{j}\right)\cdot\left(\hat{D}^{(i)}_{j}-\hat{D}^{(i)}_{k_j}\right)+m\right)$, σ = ReLU.
Code (`utils/loss_utils.py:93-129`): for each of two CenterCrops (576×768) and (544×736) of the
581×777 depth map, `F.unfold(32, stride 32)` patches; one `randperm` of the 1024 patch pixels
(shared by all patches), first half vs. second half:
`relu(sign(mono[:, a] - mono[:, b]) * (surf[:, b] - surf[:, a]) + m).mean()`; result
`0.5*crop1 + 0.5*crop2`. Mean, not sum. Mono depth = Marigold `<name>_pred.npy` at training res.

Eq. 9 smoothness: $\mathcal{L}_{s}=\sum_{i,j,k}\sum_{|\widetilde{D}_k-\widetilde{D}_{(i,j)}|<m_e}\sigma\left(|\hat{D}_k-\hat{D}_{(i,j)}|-m_t\right)$, k ∈ {(i+1,j),(i,j+1)}.
Code `TVLoss(surf_depth, mono, edge_margin=1e-2, margin=1e-4)`: mean over vertical diffs + mean
over horizontal diffs (two separate means, summed). Weight 1 (added unweighted).

Eq. 12-13: $\bm{x}_{r,i}=\bm{o}_{r}+\hat{D}_{r,i}\cdot\bm{d}_{r,i}$, $\bm{p}_{s,i} = KP_{s}^{-1}\bm{x}_{r,i}$.
Code `depths_to_points`: rays_d = K⁻¹[u,v,1] rotated to world (NOT normalised, z-depth), u,v
integer pixel indices (no +0.5).
Eq. 14 feature loss: $\mathcal{L}_{f} = \sum_{s,i,l}\frac{1}{l}\cdot v_{r,s,i}\left|1-\cos\left(\bm{F}^{(l)}_{r}(\bm{p}_{r,i}),\bm{F}^{(l)}_{s}(\bm{p}_{s,i})\right)\right|$
Eq. 15 visibility: $v_{r,s,i}=[i=\arg\min_t \|\bm{x}_{r,t}-\bm{o}_{s}\|]$ over points projecting to the same source pixel.
Code (`utils/feat_utils.py:get_feat_loss[_corr]`): pixels with surf_depth > 0; for l ∈ {1,2}:
grid = projected pixel / l in feature coords; bilinear grid_sample (align_corners False,
normalisation p/size*2-1, clamp ±1.1); valid = in range in ref and src; v from rounding the
source uv, sort by distance to source centre, first of each unique uv = 1; corr_loss = |1-cos|;
additional **diff_mask corr_loss < 0.5** (not in paper); `mean` over (2 src × all masked pixels);
levels combined `(1·L1 + 0.5·L2) / 1.5`. Weight λ3 = 1.5.
Features: Vis-MVSNet `FeatExt` (weights shipped as `utils/vismvsnet.pt`), output [2] (32 ch,
H/2). Level 1: full-res 1554×1162 image zero-padded into 1600×1200, normalised as
`(x/2 + 0.5 - mean)/std` (ImageNet mean/std, x∈[0,1] — a quirk inherited from MVSDF),
feature map 600×800 cropped to 581×777. Level 2: image resized to 777×581, padded to 800×600,
feature 300×400 cropped to 291×389.

Eq. 16 total: $\mathcal{L} = \mathcal{L}_{c} +\lambda_{1} \mathcal{L}_{r}+\lambda_{2} \mathcal{L}_{s} +\lambda_{3} \mathcal{L}_{f} +\lambda_{4} \mathcal{L}_{d}+\lambda_{5} \mathcal{L}_{n}$
$\mathcal{L}_{c} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{D-SSIM}$, λ = 0.2.
$\mathcal{L}_{d} = \sum_{i,j}\omega_i\omega_j| d_i-d_j|$ (2DGS rasterizer `rend_dist`, mean),
$\mathcal{L}_{n} = \sum_{i}\omega_i(1-\bm{n}^T_i\bm{N})$ (code: `(1 - (rend_normal*surf_normal).sum(0)).mean()`,
surf_normal = depth_to_normal(surf_depth) · alpha.detach()).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 15000 | code arguments/__init__.py |
| λ1 rank / λ2 smooth / λ3 feat / λ4 dist / λ5 normal | 10 / 1 / 1.5 / 1e4 / 0.05 | supp. A; code |
| dist start / normal start | >3000 / >7000 | code train.py (2DGS) |
| m, m_e, m_t | 1e-4, 1e-2, 1e-4 | supp. A; code |
| patch M | 32; crops 576×768 & 544×736 | code |
| feature levels | l = 1, 2; 2 source views (pair.txt[:2]) | supp. A; code |
| position lr | 1.6e-4 → 1.6e-6 × extent, max_steps 30000 (ends at 1.6e-5 at 15k) | code |
| feature/opacity/scaling/rotation lr | 0.0025 (shN /20) / 0.05 / 0.005 / 0.001 | code |
| densify | from 500, every 100, until 15000, grad 2e-4, percent_dense 0.01 | code |
| prune opacity | < 0.1 (opacity_cull) ; size prune radii > 20 px & world > 0.1 extent after it. 1000 | code |
| opacity reset | every 1000 to min(o, 0.1) | code |
| resolution | `-r 2`: 1554×1162 → 777×581 | README |
| background | black (white_background False) | code |
| SH | degree +1 every 1000 → 3 | code |
| TSDF | depth_trunc = 2·radius, voxel = depth_trunc/1024, sdf_trunc = 5·voxel; depth zeroed outside alpha mask; SH degree 0 colours; keep 50 largest clusters | render.py |

## 4. Schedules
dist loss after 3000, normal after 7000; densification 500–15000; opacity reset every 1000
(white bg extra reset not active); SH degree up every 1000.

## 5. Builds on
2DGS (official diff-surfel-rasterization CUDA shipped in the repo), unchanged L_c, L_d, L_n,
2DGS clone/split/prune. Gaussians init from `dense/fused.ply` (3-view COLMAP MVS).

## 6. Protocol
DTU 15 scans, 3 views. Large-overlap (SparseNeuS) set 23/24/33 = main Table 1; little-overlap
(PixelNeRF) set 22/25/28 = supp. Table 2 and ablations. Authors' preprocessed data (Google Drive,
downloaded to `~/datasets/dtu_fatesgs/`): RGBA 1554×1162 crops (alpha = object mask), COLMAP
`sparse/0` with known poses in the scale_mat-normalised frame, `dense/fused.ply`, Marigold depth
for set 23/24/33. Mesh: TSDF over the 3 training views → `fuse_post.ply`; evaluation
(`eval/eval_dtu/dtu_eval.py`): cull vertices outside the dilated (disk 24) object masks of the 3
views (1600×1200, `DTU_EVAL/<set>/scanX`), map to DTU world with scale_mat, DTUeval-python
(downsample 0.2, patch 60, max_dist 20, ObsMask, ground plane) → CD = (acc + comp)/2 in mm.
GT: official DTU `SampleSet` (ObsMask) + `Points` (STL). NVS (supp. Table 5): pixelNeRF split,
25 unseen views, 1/4 res, averages only.

## 7. Ambiguities / paper vs code
- Paper Eq. 8/9/14 are sums; code uses means (code wins).
- diff_mask (corr loss < 0.5) only in code.
- Paper says images 800×600; code `-r 2` of the 1554×1162 crop gives 777×581.
- Feature normalisation `(x/2+0.5-mean)/std` looks like a bug inherited from MVSDF (expects
  [-1,1] input); kept for fidelity.
- `use_mask` False: training uses raw RGB (background visible); masks only for TSDF/culling.
- Paper's NVS protocol (pixelNeRF, 25 test views) needs images not in the authors' release.
