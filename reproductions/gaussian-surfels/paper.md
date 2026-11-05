# Gaussian Surfels — implementation spec

**High-quality Surface Reconstruction using Gaussian Surfels** — Pinxuan Dai, Jiamin Xu, Wenxiang
Xie, Xinguo Liu, Huamin Wang, Weiwei Xu. SIGGRAPH 2024 (conference papers, art. 22).
arXiv [2404.17774](https://arxiv.org/abs/2404.17774). Official code
<https://github.com/turandai/gaussian_surfels> (cloned at commit `01d1249`, "dual visible", into
`source/code/`; untrusted, read only). Data: HuggingFace `turandai/gaussian-surfels-dtu`
(15 IDR scans with masks **and precomputed Omnidata normals**) and `turandai/gaussian-surfels-bmvs`.

## 1. Method summary
- 3DGS with the third scale forced to 0 (`s = [s_x, s_y, 0]`): each Gaussian is a flat ellipse
  ("surfel") with normal `n_i = R(r_i)[:,2]`; it is still splatted with the 3DGS EWA projection
  (Jacobian, +0.3 low-pass), not with 2DGS ray–splat intersection.
- Per-pixel depth uses the ray/tangent-plane intersection via a local homography (Eq. 8), alpha-
  normalised depth and (un-normalised in CUDA, normalised in Python) normal maps (Eq. 7).
- Losses: photometric, depth-normal consistency, monocular-normal prior + normal-gradient (curvature)
  term, opacity (binary) term, mask term.
- Custom CUDA: the gradient into the surfel normal `R[:,2]` from the rendered normal map is scaled ×10.
- Density control à la 3DGS with surfel-specific clone/split and pruning of invisible Gaussians.
- Mesh: render depth+normal at the training views, back-project to oriented points, remove points
  in voxels with low accumulated Gaussian opacity (volumetric cutting, 512³), screened Poisson
  (depth 10), prune far vertices, close holes.

Contributions (paper sections):
- **C1 Gaussian surfels representation + per-pixel surfel depth** (§3.2, Eq. 2–8).
- **C2 Depth-normal consistency loss L_c** (§3.3, Eq. 10) + ×10 normal-gradient scaling (§3.5).
- **C3 Normal-prior loss L_n** (Omnidata prior + normal-gradient L1; §3.3, Eq. 11).
- **C4 Opacity loss L_o** (§3.3, Eq. 12) (+ mask loss L_m).
- **C5 Volumetric cutting + screened Poisson meshing** (§3.4).
- **C6 Optimisation recipe** (§3.5): random init in bbox, coarse-to-fine resolution warm-up,
  invisible-point pruning every N iterations, LRs.

## 2. Equations (verbatim from `src/tex/03.method.tex`)
Eq. (sigma): `\mathbf{\Sigma }_i=\mathbf{R}(\mathbf{r}_i)\mathrm{Diag}\left[ \left( \mathbf{s}_{i}^{x} \right) ^2,\left( \mathbf{s}_{i}^{y} \right) ^2,0 \right] \mathbf{R}(\mathbf{r}_i)^{\top}`

Eq. (render_rgb): `\tilde{C}=\sum_{i=0}^n{T_i\alpha _i\mathbf{c}_i}, T_i=\prod_{j=0}^{i-1}{(}1-\alpha _j), \alpha_i=G'(\mathbf{u};\mathbf{u}_i,\mathbf{\Sigma }'_i)o_i`,
`\mathbf{\Sigma }'_i=\left( \mathbf{J}_k\mathbf{W}_k\mathbf{\Sigma }_i\mathbf{W}_{k}^{\top}\mathbf{J}_{k}^{\top} \right) \left[:2,:2\right]`

Eq. (render_geom): `\tilde{N}=\frac{1}{1-T_{n+1}}\sum_{i=0}^n{T_i\alpha _i\mathbf{R}_i\left[ :,2 \right]}, \tilde{D}=\frac{1}{1-T_{n+1}}\sum_{i=0}^n{T_i\alpha _{i}d_i\left( \mathbf{u} \right)}`

Eq. (pix_depth): `d_i\left( \mathbf{u} \right) =d_i\left( \mathbf{u}_i \right) +\left( \mathbf{W}_k\mathbf{R}_i \right) \left[2,:\right]\mathbf{J}^{-1}_{pr}\left( \mathbf{u}-\mathbf{u}_i \right)`

Total: `\mathcal{L}=\mathcal{L}_{\mathrm{p}}+\mathcal{L}_{\mathrm{n}}+\lambda_{\mathrm{o}}\mathcal{L}_{\mathrm{o}}+\lambda_{\mathrm{c}}\mathcal{L}_{\mathrm{c}}+\lambda_{\mathrm{m}}\mathcal{L}_{\mathrm{m}}`

Eq. (loss_photo): `\mathcal{L}_{\mathrm{p}}=0.8 \cdot {L}_1(\tilde{\mathbf{I}}, \mathbf{I})+0.2 \cdot {L}_{DSSIM}(\tilde{\mathbf{I}}, \mathbf{I})`

Eq. (loss_geom): `\mathcal{L}_{\mathrm{c}}=1-\tilde{\mathbf{N}}\cdot N(V(\tilde{\mathbf{D}}))` (V: back-project, N: cross product of neighbours)

Eq. (loss_normal): `\mathcal{L}_{\mathrm{n}}=0.04 \cdot (1-\tilde{\mathbf{N}}\cdot \hat{\mathbf{N}})+0.005 \cdot L_1(\nabla{\tilde{\mathbf{N}}}, \mathbf{0})`

Eq. (loss_solid): `\mathcal{L} _{\mathrm{o}}=\exp \left( {-(o_i-0.5)^2}/{0.05} \right)`

Mask loss (text): BCE between `\sum T_i\alpha_i` and the mask, λ_m = 1. λ_o = 0.01, λ_c linear 0→0.1.

Volumetric cutting: 512³ grid over the bbox; accumulate `G(x;x_i,Σ_i)·o_i` of every surfel at the
voxel centres it intersects; drop depth points that fall in voxels with accumulated value < λ = 1.

## 3. What the official code actually does (overrides the paper; `train.py`, `scene/gaussian_model.py`, `utils/*`)
| item | paper | code (used) |
|---|---|---|
| L_p | 0.8 L1 + 0.2 DSSIM | same; GT = `img*mask + bg*(1-mask)`, **random bg per step** (`random_background=True`), render with same bg; SSIM 11×11 σ1.5 same-padding |
| L_m | BCE, λ_m=1 | `0.1 * mean(opac * (1 - maxpool9x9(mask)))` (opac = 1−T) |
| L_c | λ_c 0→0.1 linear | weight `0.01 + 0.1*min(2*it/15000, 1)`; `cos_loss(N, d2n)` = mean over pixels with cos<1 of (1−cos); N = normalize(rendered normal)·[opac>1e-5]; d2n from 4-neighbour cross products (`depth2normal`) |
| L_n prior | 0.04 | weight `0.04 − 0.02*it/15000`; `cos_loss(N, mono·mask, weight=mask)` |
| ∇N term | 0.005 L1 | `0.005 * mean(curv)`, curv = L1-norm over channels of Σ_4-neighbours (n_nb − n_c) (masked) |
| L_o | λ_o=0.01, all Gaussians | `0.01 * mean(exp(-20(o-0.5)^2) * w)`, w = 0.01 for o∈(0.01,0.5], 1 for (0.5,0.99], 0 else |
| iterations | 15k | 15000 (`opt.iterations`); optimizer step skipped at it=15000 |
| resolution | "starting at lower resolutions" | /4 for it 1–2001, /2 for 2002–5001, full afterwards (`train.py:73-80`) |
| init | random in bbox | 1,000,000 uniform points in [−1.2,1.2]³ (IDR-normalised space), random normals (`rand−0.5`, normalised) → `normal2rotation`, scales `log(sqrt(knn3_d2/4))` (third = −1e10), opacity 0.1, SH dc `rand/255`; then visual-hull prune: keep points whose projection lies inside the 9×9-max-pooled mask in all training views |
| LRs | xyz 1.6e-4→1.6e-6, r 1e-3, s 5e-3, o 5e-2, C 2.5e-3 | same (× camera extent for xyz, `get_expon_lr_func`, no delay); f_rest 2.5e-3/20; Adam eps 1e-15 |
| SH | | degree +1 every 1000 its, max 3 |
| densify | 3DGS-like | from it>200, every `max(100, N_train)` its (always 100 on DTU): **prune** (o<0.1, max(s_xy)>0.5·extent, s_min·s_max<1e-8·extent², never visible (denom==0)) **then densify**: clone if grad≥1e-4, max(s)≤0.01·extent, mean scale-grad≤1e-7 and mean opacity-logit≤2; split (N=2, in-plane samples, s/1.6) if grad≥1e-4 and max(s)>0.01·extent; stats reset after each densify; densify_until 30000 > 15000 → active the whole run |
| grad stat | | ‖∂L/∂means2D[:2]‖ accumulated where radii > 1 |
| opacity reset | 3DGS | at it ∈ {1001, 2001, …, 14001}: `o ← o·0.12` (multiplicative), Adam moments of opacity zeroed |
| screen-size prune | 3DGS | none (max_radii2D tracked but unused) |
| normal grad ×10 | ×10 | `backward.cu:727` `dL_dnormal += dchannel_dcolor * dL_dchannel * 10` |
| backfaces | | `dual_visible=True`: normals flipped to face the camera; grazing surfels culled |
| mesh | Poisson depth 10 + cutting | `render.py`: render train views at full res, black bg; points where opac>1e-5, inside GT mask, 0<depth<20; grid (512, bbox of means ±10%, cutoff max(0,1/255)) prune `grid>1`; pymeshlab screened Poisson (depth 10, preclean, samplespernode 1.5) → `_plain.ply`; remove vertices whose squared NN distance to the points > 3e-5 (pytorch3d knn → squared), close holes ≤300 → `_pruned.ply` |
| eval | DTU CD | `eval.py`: vertices `/S + T` (scale_mat) to DTU world, DTUeval-python (ObsMask, Plane; data→stl uses data_in_obs, stl→data uses data_in; max dist 20, downsample 0.2) |

## 4. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 15000 | §3.5 / `arguments/__init__.py:86` |
| position_lr init/final | 1.6e-4 / 1.6e-6 (×extent) | §3.5 / `arguments:87-88` |
| feature/opacity/scaling/rotation lr | 2.5e-3 / 5e-2 / 5e-3 / 1e-3 | §3.5 / `arguments:91-95` |
| lambda_dssim | 0.2 | `arguments:98` |
| densify_grad_threshold | 1e-4 (0.0002·0.5) | `arguments:104` |
| densification_interval | max(100, N_train) | `arguments:99`, `train.py:46` |
| opacity_reset_interval | 1000 (×0.12) | `arguments:101`, `train.py:178` |
| densify_from_iter / until | 200 / 30000 | `arguments:102-103` |
| percent_dense | 0.01 | `arguments:97` |
| min_opacity (prune) | 0.1 | `train.py:174` |
| init points | 1e6 in [−1.2,1.2]³ | `dataset_readers.py:469-481` |
| loss weights | see §3 table | `train.py:137-149` |
| volumetric cutting | 512³, thresh 1 | §3.4 / `render.py:31,61` |
| Poisson | depth 10, prune 3e-5 (squared NN dist), close holes 300 | `render.py`, `general_utils.poisson_mesh` |

## 5. Builds on
3DGS (code base, rasterizer, SH, Adam, densification skeleton). Unchanged from 3DGS: SH colour
model and its schedule, EWA projection with 0.3 dilation, alpha clamp 0.99 / 1/255 skip / T<1e-4
stop, Adam per field, exponential xyz LR.

## 6. Protocol
- DTU: 15 IDR scans (24 37 40 55 63 65 69 83 97 105 106 110 114 118 122), 49 or 64 images,
  1600×1200, masks, IDR `cameras.npz` normalisation. Test split "following NeuS2": sorted image
  indices {8,13,16,21,26,31,34} (+56 if >56 images) held out (`readIDRSceneInfo` with `--eval`).
  The ablation Table 4 reports CD 0.882 and PSNR 32.51 for "Full" = Table 1 mean CD 0.88 and
  Table 3 PSNR 32.51 → the CD numbers come from the same held-out runs (assumption).
- PSNR: per-view on masked GT over black background vs render with black background, full
  resolution, averaged (render.py / training_report). SSIM reported (Table 3); no LPIPS.
- BlendedMVS: 18 scenes, CD with the authors' fused GT point clouds (data on HF); all images train.
- Chamfer: DTUeval-python, mm.
- Seeds: not stated. GPU: RTX 4090. Time 6.67 min/scan (DTU).

## 7. Ambiguities
- CD runs trained with or without the test views: code default trains on all images (`eval=False`);
  Table 4 couples CD and PSNR in one row → we use the held-out split for both (record).
- Paper L_m = BCE λ=1, code uses 0.1·mean(opac·(1−dilated mask)) → code.
- λ_c and prior weight schedules differ (code used).
- `mono *= mask_gt` is in-place on the cached camera prior → at soft mask borders the prior decays
  over epochs (mask^k); we multiply once.
- `depth2normal` uses fy for x and fx for y (swapped focal); DTU fx≈fy, so negligible; replicated.
- The paper's "×10 normal gradient" lives in CUDA (kept by vendoring).
- PSNR "12.5%" setting = 7–8 held-out views of 49/64.
