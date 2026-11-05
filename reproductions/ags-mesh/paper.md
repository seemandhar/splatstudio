# AGS-Mesh — implementation spec

Ren, Turkulainen, Seiskari, Melekhov, Kannala, Rahtu. *AGS-Mesh: Adaptive Gaussian Splatting and
Meshing with Geometric Priors for Indoor Room Reconstruction Using Smartphones*, 3DV 2025,
arXiv 2411.19271. LaTeX in `source/latex/author-kit-3DV2025-v1/`.
Official code: the Splatfacto variant ("Splatfacto + Ours", the row we target) is the
`ags-mesh` method in github.com/maturk/dn-splatter **main @ 97588b4** (`source/code/`, same commit
as our DN-Splatter port): `dn_config.py:ags_mesh`, `regularization_strategy.py:AGSMeshRegularization`,
`dn_model.py` (surface normals), `scripts/depth_to_normal.py` (DNC masks, called by the MuSHRoom
dataparser). The 2DGS variant lives in branch `ags-mesh-2dgs` / XuqianRen/AGS_Mesh (not ported).
**Code overrides paper where they differ; both recorded.**

## 1. Summary and contributions
- Plug-in regularization for Gaussian splatting with smartphone (iPhone LiDAR) sensor depth and
  Omnidata monocular normals; applied to Splatfacto (=DN-Splatter pipeline) and 2DGS.
- **C1 Depth-Normal Consistency (DNC)** (§4.1, Eq. 1–3): per frame, normals N_d from the sensor
  depth point cloud (KNN k=200 covariance eigenvector) are compared with Omnidata N_p; depth pixels
  with angle > τ_d are dropped (D_f). Raw depth supervises until T_d, D_f afterwards.
- **C2 Adaptive Normal Regularization (ANR)** (§4.2, Eq. 4–6): normals from rendered depth N̂ are
  supervised by N_p until T_n, and afterwards only where angle(N̂, N_p) ≤ τ_N.
- C3 loss composition (§4.3, Eq. 7): L = L_color + λ_d L_D + λ_n L_N.
- C4 depth-aware-truncation TSDF + IsoOctree meshing with a point-cloud hint (§4.4, supp. §1.2)
  — used only for the 2DGS rows; the Splatfacto+Ours row uses Open3D TSDF. Mesh evaluation only,
  not needed for PSNR (gap).

## 2. Equations (verbatim, sec/4_method.tex)
- Eq. 1: $D_f = 0$ if $\theta_d > \tau_d$ else $D$; Eq. 2:
  $\theta_d=\arccos\left(\frac{\mathbf{N}_d \cdot \mathbf{N}_p}{\|\mathbf{N}_d\|\|\mathbf{N}_p\|}\right)$.
- Eq. 3: $\mathcal{L}_D = \|\hat D - D\|_1$ if step $< T_d$, else $\|\hat D - D_f\|_1$.
- Eq. 4: $\mathcal{L}_N = \|\hat N - N_p\|_1$ if step $< T_n$, else $\|\hat N - N_f\|_1$;
  Eq. 5: $N_f = 0$ if $\theta_n > \tau_N$ else $N_p$.
- Eq. 6: $\hat N = \frac{\nabla_x D \times \nabla_y D}{|\nabla_x D \times \nabla_y D|}$ (normals of rendered depth).
- Eq. 7: $\mathcal{L} = \mathcal{L}_\text{color} + \lambda_d\mathcal{L}_D + \lambda_n\mathcal{L}_N$.
- Supp. isofunction $f(x)=\sum_j w_j(d_j(x)-\tilde d_j(x))$, truncation $-\tau d_j(x)$, τ=0.05,
  $w'_j = (d_j-\tilde d_j)(-r_j\cdot n_j)/d_j^2$; IsoOctree subdivide if ≥ 50 hint points, depth 10.

### Official code (Splatfacto variant), what actually trains
`AGSMeshRegularization.get_loss` (ns = Nerfstudio 0-based step):
- **depth**: `EdgeAwareLogL1` (DN-Splatter Eq. 3 of 2403.17822, *not* the paper's plain L1) on the
  alpha-normalised expected depth, mask gt > 0.1, weight **λ_d = 0.2 (single, no DN double count)**;
  for ns ≥ 7000 gt := where(confidence > 0, gt, 0) with confidence = 1 − mask_jpg/255.
- **normal**: λ_n = 0.1 **only if ns > 7000** (supp.: "normal regularization only after 7k"), else 0.
  `normal_l1 = L1(surf[~edges(gt)], gt[~edges(gt)])` for ns < 15000, afterwards
  `L1(surf[:, conf], gt[:, conf])` with conf = arccos(clip(Σ surf·gt)) ≤ **0.1 rad (5.73°)**
  (paper: τ_N = 10°). `surf` = normals of the **detached** rendered depth (dn_model.py:589
  `depth_im.detach()`, `pcd_to_normal` central differences, zero 1-px border) → this ANR term
  carries **no gradient** in the official Splatfacto variant (it is a constant in the loss).
  Plus `L1(pred_normal, gt)·λ_n` on the Gaussian-shortest-axis normals (DN-Splatter C2 renderer),
  unmasked. All three normal maps in the GL camera frame, values in [−1, 1].
  `find_edges(gt)`: per channel Laplacian [[0,1,0],[1,−4,1],[0,1,0]] (zero pad) of 1/(g+1e−6),
  edge = lap > 0.01, one dilation by a 3×3 ones kernel (zero pad), > 0 → excluded (per element).
  L1 = `torch.abs(pred-gt).mean()` (losses.py L1 scalar).
- **scale**: min(exp(s)).mean(), weight 1 (DN-Splatter Eq. 6).
- **No normal TV** (not called by AGSMeshRegularization).
- DNC mask (MuSHRoom path: dataparser → `DepthToNormal`): angle threshold **10°** but computed
  between the **[0,1]-encoded** vectors (n+1)/2 (the standalone `depth_normal_consistency.py`
  uses 20° on the signed normals); KNN 200 Open3D; mono normal png treated as camera frame,
  rotated by R_c2w(OpenCV); mask saved as JPEG (`frame_xxxxx.jpg`), read via IMREAD_ANYDEPTH.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| T_d (switch to D_f) | 7000 | supp. §1.1; code `step < 7000` |
| T_n (switch to filtered normals) | 15000 | supp. §1.1; code `normal_mask_steps` |
| normal loss start | ns > 7000 | supp. §1.1; code `lambda_normal_l1` |
| τ_d | 10° (encoded-space angle) | supp. §1.1; `depth_to_normal.py` |
| τ_N | 10° paper / **0.1 rad code** | supp. §1.1 vs regularization_strategy.py:300 |
| λ_d | 0.2 | supp.; README `--depth-lambda 0.2` |
| λ_n | 0.1 | supp.; `normal_lambda` |
| depth loss type | EdgeAwareLogL1 (code) / L1 (paper Eq. 3) | DNSplatterModelConfig default |
| depth tolerance | 0.1 m | `depth_tolerance` |
| KNN | 200 | §4.1, code |
| scale loss | 1.0 | `get_scale_loss` |
| iterations | 30000 | supp., `ags_mesh` config |
| optimizer / density / init | identical to DN-Splatter (`ags_mesh` config = `dn_splatter` config) | dn_config.py |
| init | 1M TSDF sensor-depth points, load_pcd_normals True (MuSHRoom dataparser default) | mushroom_dataparser.py:93 |

## 4. Schedules
ns < 7000: depth on raw sensor depth, no normal loss. 7000 ≤ ns: DNC-filtered depth; ns > 7000:
normal losses (edge-masked surf term + pred term). ns ≥ 15000: surf term masked by ANR confidence.
Splatfacto density/SH/LR schedules as DN-Splatter.

## 5. Builds on
DN-Splatter / Splatfacto 1.1.3 (same renderer with Gaussian normals, density, init, optimizer).
Changes vs DN-Splatter: depth weight 0.2 (not 1.2), DNC filter after 7k, normal loss 0.1 after 7k
(not 1.0 from step 0), no TV, extra (gradient-free in code) surface-normal ANR term.

## 6. Protocol
MuSHRoom iPhone, 6 rooms (coffee_room, honka, kokko, sauna, computer, vr_room; batch_run.py),
train = long_capture minus test.txt; "within" test = test.txt; "different sequence" = short_capture.
**NVS reported as 6-scene averages only** (Table nvs). Metrics as DN-Splatter (torchmetrics
PSNR/SSIM, LPIPS-AlexNet). ScanNet++ (licence) is mesh-only. Mesh metrics need MuSHRoom Faro GT.

## 7. Ambiguities
- Paper Eq. 3 L1 vs code EdgeAwareLogL1 → code. Paper τ_N 10° vs code 0.1 rad → code.
- Paper ANR (Eq. 6) gradients through rendered depth vs official Splatfacto code detaching depth
  (ANR term constant) → code default; a `surface_grad` option gives the paper's variant.
- DNC: dataparser (10°, encoded angle, JPEG mask) vs README script (20°, signed) → dataparser
  (MuSHRoom path, matches paper's 10°).
- The NVS split of the table = DN-Splatter's within/cross protocol.
