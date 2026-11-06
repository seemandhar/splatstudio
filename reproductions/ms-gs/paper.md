# Multi-Scale 3D Gaussian Splatting for Anti-Aliased Rendering (MS-GS) — implementation spec

- arXiv 2311.17089 (CVPR 2024), Yan, Low, Chen, Lee. LaTeX in `source/latex/`, PDF `source/paper.pdf`.
- Official code: https://github.com/JokerYan/MS-GS @ `50e5040` (`source/code/`), rasterizer submodule
  https://github.com/JokerYan/MS-GS-rasterizer @ `e7a2b4f` (`source/rasterizer/`). Inria 3DGS licence
  (non-commercial research). Built on the official 3DGS code base.
- **Code overrides paper where they disagree** (marked ⚠ below).

## 1. Method summary
- 3DGS's per-Gaussian EWA dilation makes small foreground Gaussians bloat at low resolution
  (aliasing + slowdown). MS-GS keeps Gaussians at several "levels of detail" and *selects* which to
  render per view from each Gaussian's on-screen **pixel coverage**.
- After a 1000-iteration warm-up, small fine-level Gaussians are **aggregated** (voxel average
  pooling) into larger coarse-level Gaussians, inserted for levels 4×, 16×, 64× (linear
  downsampling).
- The model is then **trained multi-scale** (1× plus downsampled images); each Gaussian stores its
  level and a running max/min pixel coverage measured at its own level, which drive the
  **selective rendering** rule.

Contributions:
- C1 Pixel coverage of a splatted Gaussian (§4.1 "Pixel Coverage of Gaussian", Fig. 3).
- C2 Aggregate-to-insert large Gaussians (§4.1 "Aggregate to Insert Large Gaussians", Alg. 1, supp. "Details of Gaussian Aggregation").
- C3 Selective rendering by relative pixel coverage with per-Gaussian S_max/S_min (§4.1 "Multi-Scale Training and Selective Rendering", Eq. 6–7, Alg. 2).
- C4 Multi-scale training schedule (§4.1, §5; official `train.py`).

## 2. Equations
Pixel coverage (paper text; equation commented out in LaTeX):
the smaller of the horizontal/vertical extent of the 2D Gaussian until the level set
$\sigma_k\,\mathcal G(\cdot)=\sigma_T$, $\sigma_T = 1/255$.

⚠ Code (`rasterizer/cuda_rasterizer/forward.cu:240-259`): with the *undilated* 2D covariance
$V$ (3DGS `computeCov2D` minus its 0.3 px² dilation), conic $Q=V^{-1}$,
$$\ell = \max(0,\,-2\ln\tfrac{1}{255\,\sigma_k}),\quad d_x=\sqrt{\ell/Q_{11}},\ d_y=\sqrt{\ell/Q_{22}},\quad S_k=\min(d_x,d_y)$$
($S_k$ is a **half**-extent along the screen axes; $S_k=0$ for Gaussians outside the frustum).

Selection (Eq. 7): $(S_k/S_k^{max}\le S^{max}_{rel}) \land (S_k/S_k^{min}\ge S^{min}_{rel} \lor S_k\ge S_T)$,
paper $S^{max}_{rel}=1.5$, $S^{min}_{rel}=0.5$, $S_T=2$ px.
⚠ Code (`forward.cu:261-283`): ratio := 1 when the stored value ≤ 0 (uninitialised);
cull if `rel_min < 0.5 && S < 2.0` (filter_small) or `rel_max > 2.0` (filter_large, **2.0 not 1.5**);
the `base_mask` exemption is always false (`preserve_large` off). Culled Gaussians get radius 0
(not rendered, no gradient, not "visible"), but their $S_k$ is still output.

Range update (Eq. 6): $S'^{max}=\max(\lambda_1 S^{max}, S)$, $S'^{min}=\min(\lambda_2 S^{min},S)$, λ1=0.95, λ2=1.05.
⚠ Code (`scene/gaussian_model.py:663-688 update_pixel_sizes`), every iteration ≥250 except within
250 its after an opacity reset, for Gaussians that are *visible* (radii>0) **and** whose level ==
current training resolution index r:
- if r>0: `max = max(0.95*max, S)` (so level-0 max stays −1 → finest never culled as too large);
- if r < n_levels−1: `min = clip(1.05*min, −1)`; if min<0 (uninit): min=S if S>0; else min=min(min,S) if S>0
  (coarsest training level never gets a min → never culled as too small).

Aggregation (Alg. 1, supp.; ⚠ code `train.py:269-387`, `gaussian_model.py:insert_large_gaussians`):
at iterations 1010/1020/1030 create levels r∈{2,4,6} (4×,16×,64×):
1. render all train cams at 1× → visible mask $v_c$; render at scale $2^r$ → $S_{k,c}$.
2. $S^{min}_k = \min(1, \min_{c: S_{k,c}>0 \land v_c} S_{k,c})$; selected = $S^{min}_k<1$ ⚠(threshold 1, paper 2) **and level==0** ⚠(paper: levels 1..l_m−1).
3. normalised position $p = x/B$ ($B$ = 3DGS cameras_extent); per axis where $p>1$: $p=2-1/p$
   ⚠(negative side left linear).
4. voxel size $0.02\cdot r/4$ (0.01, 0.02, 0.03) ⚠(paper $(400/l_m)^3$ grid → 0.02/0.03/0.04);
   voxel id = floor(p / size) (Open3D-ML VoxelPooling); per voxel **average of raw parameters**
   (xyz, f_dc, f_rest, logit opacity, raw quaternion, log scale) and of $S^{min}$.
5. scale ×(2 / clip(avg $S^{min}$, 0.25, 2)); opacity unchanged; new level r; S_max=S_min=−1.
6. render all train cams at scale $2^r$ and run the range update for level r.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 40000 | §5; train_all.py `iterations` |
| densify from / until / interval | 500 / 15000 / 100 (1× iterations) | arguments, train.py:253 |
| grad threshold, percent_dense | 2e-4, 0.01 | 3DGS defaults |
| opacity reset | every 3000 it (<15000), level 0 only, min(σ,0.01) | gaussian_model.py:346 |
| position lr | 1.6e-4→1.6e-6 × extent over **30000** steps, then constant | arguments (position_lr_max_steps) |
| other lrs | 3DGS defaults | arguments |
| training scales | 2^i, i=0..max_scale; 360: max_scale=7 (1..128), TnT/DB 6 | train_all.py |
| P(1× epoch) | 0.75 | train.py:160 |
| non-1× level choice | least-trained among enabled {0}∪{2,3}(it>1010)∪{4,5}(>1020)∪{6,7}(>1030) | train.py:98-183 |
| loss multiplier | 1 at 1×, 0.1 otherwise | train.py:214 |
| insertion its / levels | 1010, 1020, 1030 / 2, 4, 6 | train.py:92-109 |
| σ_T | 1/255 | §4.1, forward.cu:251 |
| S_T | 2 px | §4.1, forward.cu:271 |
| S_rel^min / S_rel^max | 0.5 / 1.5 (code 2.0) | Eq. 7 / forward.cu:279 |
| λ1, λ2 | 0.95, 1.05 | Eq. 6, gaussian_model.py |
| insertion threshold | 1 px (paper S_T=2) | train.py:288 |
| enlargement | 2/clip(S_avg,0.25,2) | gaussian_model.py insert_large |
| stats warm-up | pixel-size update from it 250, skip 250 its after reset | train.py:244 |

## 4. Schedules
- it 1–1010: 1× only. Then epochs (a full shuffled stack of train cams) at one resolution;
  per new epoch: 1× w.p. 0.75 else least-trained enabled level (scale 2× **never** trained).
- Densify/prune only level-0 Gaussians, only on 1× iterations, when the 1×-iteration count is a
  multiple of 100, it ∈ (500,15000); gradient stats only from 1× renders. Screen-size prune is
  inert (3DGS max_radii2D reset bug, kept); world-size prune (>0.1·extent) after it 3000.
- SH degree +1 every 1000 its.

## 5. Builds on
Official 3DGS (same loss L1 + 0.2 D-SSIM (same-padded SSIM), Adam, ADC). Unchanged except the above.
`multi_occ`, `multi_dc`, `grow_large`, `prune_small`, `preserve_large` are off in the paper config.

## 6. Protocol
- Datasets: Mip-NeRF 360 (9 scenes), T&T (truck, train), DB (drjohnson, playroom); 3DGS splits
  (llffhold 8). ⚠ Code loads `images` (full res) with `-r -1` → **width 1600** base for every 360
  scene (indoor and outdoor), downsampled by s: size = (int(W/(g·s)), int(H/(g·s))), g=W/1600,
  PIL `resize` (bicubic) from the full-res JPEG; FoV camera (focal ×w/W, centred principal point).
- Eval: test views at each scale; PSNR averaged over views per scale; tables at 1×,4×,16×,64×
  (+2,8,32,128 in supp.); LPIPS (lpipsPyTorch VGG) "N.A." when it fails on tiny images.
  3DGS baseline row = single-scale 3DGS trained at 1× (30k), evaluated at all scales.
- Background black; RTX 3090 timing.

## 7. Ambiguities
- README example uses `--ms_train_max_scale 6`; `train_all.py` uses 7 for Mip-NeRF 360 (the
  per-scene table reports 128×, so 7). We use 7.
- Paper Eq. 7 S_rel^max 1.5 vs code 2.0; insertion threshold S_T=2 vs code 1; voxel sizes; level set
  "length" vs code half-extent — code values used.
- Average pooling of raw quaternions/log-scales/logit-opacity (code) — used.
