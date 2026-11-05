# SuGaR — implementation spec

Guédon & Lepetit, "SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction
and High-Quality Mesh Rendering", CVPR 2024, arXiv 2311.12775. Official code
https://github.com/Anttwo/SuGaR (cloned to `source/code/`, read only). The paper's Mip-NeRF 360
results use the `"sdf"` regularization (README: "results presented in the paper were obtained with
the density regularization for object-centered scenes and sdf for scenes with a challenging
background, such as the Mip-NeRF 360 dataset"), so this spec follows `sugar_trainers/coarse_sdf.py`.

## 1. Method summary
- Stage A (vanilla 3DGS, 7k it): official `gaussian_splatting/train.py --iterations 7000`.
- Stage B (coarse SuGaR, it 7000→15000, `coarse_sdf.py`): no densification; opacity-entropy loss
  7000–9000; prune opacity < 0.5 at 9000; SDF-estimation regularizer R (Eq. R) + normal
  regularizer R_Norm (Eq. R_norm) from 9000 to 15000, using 16-NN density queries (neighbours
  recomputed every 500 it).
- Stage C (mesh extraction, `coarse_mesh.py`): λ=0.3 level-set points found along view rays from
  the Gaussian depth maps of all training views, normals = −∇d/‖∇d‖; fg/bg Poisson (depth 10),
  10 % low-density vertex removal, quadric decimation to 1,000,000 triangles each, cleanup,
  vertices snapped to the nearest level-set point.
- Stage D (refined SuGaR, `refine.py`, 15k it for R-SuGaR-15K): one thin Gaussian per triangle
  (high-poly config), mean = barycentre (1/3,1/3,1/3) of its triangle (vertices learnable), first
  axis = face normal with fixed thickness, 2 learnable in-plane scales, learnable 2D rotation as a
  complex number, opacity + SH (deg 3, warm-up from 0) learned; mesh normal-consistency loss 0.1.

Contributions
- C1 Surface-alignment SDF regularization R (§3.1, Eq. R, plus opacity entropy and pruning, §4.1).
- C2 Normal regularization R_Norm (§3.1, Eq. R_norm).
- C3 Efficient level-set mesh extraction + Poisson (§3.2, supp. "Mesh extraction").
- C4 Mesh-bound Gaussian refinement (§3.3, supp. §"Parameterisation", "Joint refinement").

## 2. Equations
Density (Eq. 1): $d(p)=\sum_g \alpha_g\exp(-\tfrac12(p-\mu_g)^T\Sigma_g^{-1}(p-\mu_g))$.
Ideal flat density (Eq. 5): $\bar d(p)=\exp(-\frac{1}{2s_{g^*}^2}\langle p-\mu_{g^*},n_{g^*}\rangle^2)$.
SDF (Eq. 7): $f(p)=\pm s_{g^*}\sqrt{-2\log d(p)}$.
R (Eq. 8): $\mathcal R=\frac1{|\mathcal P|}\sum_{p}|\hat f(p)-f(p)|$, $\hat f$ = depth-map
difference along the line of sight; $p\sim$ Gaussians.
R_Norm (Eq. 10): $\frac1{|\mathcal P|}\sum_p\|\frac{\nabla f(p)}{\|\nabla f(p)\|}-n_{g^*}\|^2$.

Code forms (override the paper where they differ):
- `get_field_values` (sugar_model.py:1278): for sample x of Gaussian g, neighbours
  $\mathcal N(g)$ = 16-NN of g's centre (incl. g). $d(x)=\sum_{j\in\mathcal N}\rho\,\alpha_j
  \exp(-\frac12\|S_j^{-1}R_j^T(x-\mu_j)\|^2)$ with **density_factor ρ = 1/16**; where d ≥ 1,
  d ← d / stopgrad(d). β = mean over neighbours of min scale ("average" beta mode).
  $f(x)=\beta(\sqrt{-2\log\max(d,10^{-16})}-\sqrt{-2\log\min(\text{thr},1)})$, thr = 1 ⇒ offset 0.
- SDF estimation loss (coarse_sdf.py:600–611): depth map D = rendered Σ w_i z_i + (1−A)·max z
  (colors = camera z of centres, bg = max z, gradients through depth). Sampling mask = visible
  (radii>0) ∧ |D(π(μ_g)) − z_g| < 2·σ_g(v) (σ_g(v) = ‖S_g R_g^T v‖, v = unit vector to camera).
  1,000,000 samples, Gaussians picked uniformly among the mask (not volume-weighted),
  $x=\mu_g+R_g(1.5\,s_g\odot\epsilon)$. $\hat f = D(\pi(x)) - z_x$ (bilinear, border padding,
  samples with z > znear). Loss $=0.2\cdot\text{mean}(\min(|f(x)-|\hat f||/(E/10),\,10E))$, E =
  camera extent (1.1 × max camera distance to their mean).
- Normal loss (coarse_sdf.py:614–640): n_j = smallest-scale axis of neighbour j, sign-aligned to
  n_g; $w_j = \text{sg}(\rho\alpha_j e^{-\frac12 q_j}) \cdot \text{sg}(|\langle x-\mu_j, n_j\rangle|)
  / \text{sg}(s_{j,\min})^2$ clamp 1e-6; $w_j \leftarrow w_j/\text{sg}(\sum w)$ clamp 1e-6;
  loss $= 0.2\cdot\text{mean}\,\|n_g-\sum_j w_j n_j\|^2$.
- Opacity entropy (coarse_sdf.py:540): $0.1\cdot\text{mean}_{vis}(-o\log(o+10^{-10})-(1-o)\log(1-o+10^{-10}))$
  for 7000 < it < 9000.
- Level-set points (sugar_model.py:1885 `compute_level_surface_points_from_camera_fast`): per
  view, 2·10M/n_views random valid depth pixels backprojected to p; σ = σ_{g*}(v) of p's closest
  Gaussian; 21 samples $t\in[-3\sigma,3\sigma]$ along the camera ray; density with ρ = 1, d ≥ 1
  normalized to 1; first crossing (needs sample 0 below λ) linearly interpolated; normal
  $-\nabla d/\|\nabla d\|$, $\nabla d=\sum_j \rho\alpha_j e^{-q_j/2} R_jS_j^{-1}(S_j^{-1}R_j^T(x-\mu_j))$
  (sign as coded). Per view keep 10M/n_views points; colour = rendered sh0 RGB at the pixel.
- Bound Gaussian (supp. §1, sugar_model.py:440–505): $R_0=\hat n_f$,
  $\bar R_1=\text{normalize}(v_0-v_1)$, $\bar R_2=\text{normalize}(R_0\times\bar R_1)$;
  $(x',y')=(x,y)/\|(x,y)\|$, $R_1=x'\bar R_1+y'\bar R_2$, $R_2=-y'\bar R_1+x'\bar R_2$;
  scales $(\epsilon, s_1, s_2)$, ε = E/10^6 (fixed); μ = (v0+v1+v2)/3.
- Normal consistency (pytorch3d `mesh_normal_consistency`): mean over face pairs sharing an
  edge of $1-\cos(n_0,n_1)$.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| vanilla 3DGS iterations | 7000 | §4.1, train_full_pipeline.py:121 |
| entropy window / weight | 7000<it<9000, 0.1 | §4.1, coarse_sdf.py:107-110 |
| prune opacity threshold | 0.5 at it 9000 (applied at it 9001) | §4.1, coarse_sdf.py:186,481 |
| SDF reg start | it > 9000 | coarse_sdf.py:115,135 |
| sdf estimation / normal factor | 0.2 / 0.2 | train.py coarse_args |
| knn / reset | 16 / every 500 (and at 7001, after prune) | §4.1, coarse_sdf.py:174-177 |
| density_factor (reg.) | 1/16 | coarse_sdf.py:144 |
| samples / sampling scale | 1,000,000 / 1.5 | coarse_sdf.py:148-149 |
| close-gaussian threshold | 2σ | coarse_sdf.py:131 |
| sdf normalisation | E/10, clamp 10E | coarse_sdf.py:600-611 |
| coarse densification | none (densify_until 0), no opacity reset | coarse_sdf.py:73-75 |
| coarse LR | 3DGS (pos 1.6e-4·E → 1.6e-6·E over 30k from it 7000) | coarse_sdf.py:55-63 |
| surface level λ | 0.3 | §4.1, train.py |
| level samples / range | 21 / ±3σ | coarse_mesh.py:30-31 |
| total level points | 10,000,000 | coarse_mesh.py:26 |
| Poisson depth / density quantile | 10 / 0.1 | §4.1, coarse_mesh.py:41-42 |
| outlier removal | nb 20, std 20 | coarse_mesh.py |
| fg / bg bbox | ±1·E / ±4·E around mean camera centre | coarse_mesh.py:39-40, center_bbox True |
| decimation target | 1,000,000 **triangles** each for fg and bg (paper: "1M vertices") | train.py `-v` |
| Gaussians / triangle | 1 (high-poly) | train.py --high_poly |
| refinement iterations | 15,000 (headline R-SuGaR-15K) | Table 1 |
| refine vertex LR | 1.6e-4·(10E/√1e6) → exp decay to 1.6e-6·(…) over 30k (it from 1) | refine.py:426 |
| refine other LR | sh0 2.5e-3, shN 1.25e-4, opacity 0.05, scales 5e-3, rot2d 1e-3 | refine.py:44-52 |
| refine init | opacity 0.1, sh0 = vertex colour interp., shN 0, scales = min edge·1/(2√3), rot (1,0) | sugar_model.py |
| refine SH warm-up | degree 0, +1 every 1000 it | refine.py:166-170 |
| normal consistency weight | 0.1 | train.py refined_args |
| refine densification | none | refine.py:115 |
| background | black (white_background False) | train.py |
| loss | 0.8 L1 + 0.2 D-SSIM | both trainers |

## 4. Schedules
it 1–7000 3DGS (densify 500–7000 interval 100, opacity reset 3000, 6000). 7001: knn. 7001–8999:
entropy. 9001: prune <0.5 + knn. 9001–15000: R + R_Norm, knn every 500. Mesh extraction at 15000.
Refinement 15001–30000 (refine it 1–15000) with SH warm-up from degree 0.

## 5. Builds on
3DGS (Inria rasterizer, official optimizer). Photometric loss unchanged in all stages.

## 6. Protocol
Mip-NeRF 360 7 scenes (garden, kitchen, room, bicycle, counter, bonsai, stump; flowers/treehill
excluded), llffhold 8 (`eval_split_interval=8`). Resolution: 3DGS loader `resolution=-1` on
`images/` ⇒ SuGaR's own `load_gs_cameras` rescales so the longer side is 1920 (cameras.py:16, 113-117; review-1 major — the earlier 1600 claim was wrong; only the vanilla 3DGS 7k stage uses -r -1 = 1600) — our
`mipnerf360-1920` protocol. Metrics: PSNR (3DGS `psnr`), SSIM, LPIPS-VGG (lpipsPyTorch).
Black background. Paper tables report "1M vertices" and "200K vertices"; Table 1 = 1M.

## 7. Ambiguities
- Paper's Table 1 3DGS row (28.69 avg) is the 3DGS paper's per-scene numbers (images_4/2), not a
  3DGS run at SuGaR's 1600-px resolution.
- "1M vertices" is a triangle target per fg/bg mesh (≈2M Gaussians total).
- Paper says sample density from neighbours "of g"; code also normalizes d ≥ 1 and uses ρ=1/16.
- Mesh-extraction depth: code rasterizes splatted diamond proxies with PyTorch3D (2σ triangles,
  z-buffer, pix_to_face gives g*); the alternative coded branch `use_gaussian_depth=True` uses
  the splatted Gaussian depth and the K-NN of the back-projected point.
