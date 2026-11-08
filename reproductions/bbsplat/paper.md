# BillBoard Splatting (BBSplat) — implementation spec

Svitov, Morerio, Agapito, Del Bue. arXiv 2411.08508 (v-latest LaTeX, ICCV format, in
`source/latex/arxiv_ICCV.tex`). Official code: github.com/david-svitov/BBSplat @ cff8d6a
(2025-03-20), rasterizer submodule github.com/david-svitov/diff-bbsplat-rasterization @ 2dbc336.
Licences: BSD-3 (BBSplat additions, `LICENSE_BBSplat.txt`) on top of the Inria
Gaussian-Splatting licence (non-commercial research use) — vendoring for research is permitted.

## 1. Method summary
- Scene = set of planar 2DGS-style primitives {mu, s (2D), r (quat), SH} **without scalar opacity**;
  each carries a learnable 16x16 alpha map T^alpha and a 16x16 RGB texture T^RGB (§3.2).
- Rendering: 2DGS ray-splat intersection gives (u,v) in [-1,1] on the plane (support is the
  square |u|,|v|<=1, no Gaussian falloff); alpha = bilinear(T^alpha)(u,v), colour =
  bilinear(T^RGB)(u,v) + RGB(SH, d_centre); front-to-back alpha blending (§3.2, Eq. 5-6).
- Training from SfM points (no pre-trained 3DGS/2DGS model): FPS subsample + optional 10K
  Fibonacci "sky sphere" points (Supp. §Initialization); textures frozen for 500 its.
- MCMC-style densification (relocate dead / add 5%), adapted: only T^alpha is rescaled on
  cloning, scales unchanged; "dead" = mean(T^alpha) <= 0.005 (§3.4).
- Impact-weighted texture sparsity regularisation (§3.3, Eq. 8-9) + MCMC opacity regulariser.
- After 30K its geometry/textures/DC are frozen and only SH rest is fine-tuned for 2K its (§4.1).

Contributions:
- C1 textured billboard primitive + CUDA rasterizer (bilinear texture sampling with gradients
  to texels and to (u,v) -> plane transform) (§3.2, Supp. Alg. 1-2).
- C2 SH + texture colour composition (Eq. 6).
- C3 texture regularisation weighted by billboard impact (§3.3).
- C4 BBSplat-MCMC density control (alpha-only rescaling, mean-alpha dead test) (§3.4).
- C5 sky-sphere + FPS initialisation (Supp.).
- C6 SH-only fine-tuning stage (30K -> 32K) (§4.1).
- (Texture quantisation/zip compression §3.5 — storage only; affects reported metrics only via the
  8-bit quantisation of textures at render time, see Ambiguities.)

## 2. Equations
Plane (2DGS, Eq. 1): `P_i(u,v) = H_i (u,v,1,1)^T`, `H_i = [[R_i S_i, mu_i],[0,1]]`.
Ray-plane intersection (Eq. 3-4):
`u(x) = (h_u^2 h_v^4 - h_u^4 h_v^2)/(h_u^1 h_v^2 - h_u^2 h_v^1)`,
`v(x) = (h_u^4 h_v^1 - h_u^1 h_v^4)/(h_u^1 h_v^2 - h_u^2 h_v^1)`; code: `p = k x l`, `s = p.xy/p.z`
with `k = -Tu + x Tw`, `l = -Tv + y Tw` (pixel centre x+0.5).
Blending (Eq. 5): `c(x) = sum_i c_i[u(x)] T_i^alpha[u(x)] prod_{j<i} (1 - T_j^alpha[u(x)])`;
code: `alpha = min(0.99, bilinear(T^alpha))`, skip if alpha < 1/255, stop when T < 1e-4.
Colour (Eq. 6): `c_i[u] = T_i^RGB[u] + RGB(SH_i, d_i)`, d_i from camera to mu_i; code: SH result
+0.5 clamped >= 0 (3DGS), texture is sigmoid-activated, sum not clamped.
Bilinear sampler (code `grid_sample.h`): `x = (u+1)/2 (S_T-1)`, align_corners=True style, zero
outside the texel grid.
Impact (Eq. 7): `I_i = sum_{x in R_i} T_i^alpha[u(x)] prod_j (1 - T_j^alpha[u(x)])` (atomicAdd of
alpha*T in the forward kernel).
Weights: `w_i = sigma - min(I_i, sigma)` if `I_i > 0` else 0.
Texture reg (Eq. 9): `L_RGB = 1/N sum w_i ||T_i^RGB||`, `L_alpha = 1/N sum w_i ||T_i^alpha - G||`,
`L_texture = lambda_RGB L_RGB + lambda_alpha L_alpha`.
**Code** (train.py): over visible (I>0) primitives only,
`reg = mean_vis(mean_{c,h,w}(sigmoid(T^RGB)) * w) * 1e-4 + mean_vis(|mean_{h,w}(sigmoid(T^alpha) - A0)| * w) * 1e-4`
with `A0 = 0.5 * G` (the initial alpha map), sigma = `max_impact_threshold` = 100.
MCMC opacity reg (code): `0.01 * mean(sigmoid(T^alpha))` over all primitives and texels.
Photometric (Eq. 10): `(1-0.2) L1 + 0.2 (1 - SSIM)`.
MCMC cloning (Eq. 13): `T^alpha_{1..N} = 1 - (1 - T^alpha_old)^{1/N}` per texel; scales unchanged.
MCMC noise (code): `noise = randn * sigmoid_k100(1 - o - 0.995) * noise_lr * xyz_lr`, with o fixed to 1
("results in the paper obtained this way") -> coefficient `1/(1+exp(99.5))` = 0 in float32: **no noise**.
2DGS normal/distortion losses: available (`--lambda_normal`, `--lambda_dist`, after 7000/3000 its),
used only for DTU in `scripts/train_all.sh`.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 32000 | code arguments; paper §4.1 (30K + 2K SH fine-tune) |
| position lr | 1.6e-4 -> 1.6e-6 expon over 30000, x camera extent | code / Supp. |
| SH dc lr / rest lr | 0.005 / 0.00025 | code feature_lr (paper §4.1 "lr_SH = 5e-3") |
| scaling lr | 0.005 | code / Supp. |
| rotation lr | 0.001 | code / Supp. |
| texture alpha lr | 0.001 | code / paper §4.1 |
| texture color lr | 0.0025 | code / paper §4.1 |
| Adam eps | 1e-15 | code training_setup |
| texture size S_T | 16 | code asset `alpha_init_gaussian_small.png` / paper §4.1, Supp. Tab. 8 |
| texture training | iterations [500, 30000) | code texture_from_iter / texture_to_iter |
| freeze (all but SH rest) | iteration > 30000 | code `position_lr_max_steps` |
| densify | 500 < it < 25000, every 100 | code |
| dead threshold | mean alpha <= 0.005 | code dead_opacity / paper gamma |
| growth | +5% per densify step up to cap_max | code add_new_gs |
| cap_max | Tab.1 setting: T&T 30K, M360 60K ("Points" column); scripts (Tab.2 setting): 300K bicycle/truck | paper Tab. 1 / scripts/train_all.sh |
| max_read_points | cap_max - 10K (scripts: 290K for 300K) | scripts convention |
| sky sphere | 10K Fibonacci points, radius max|xyz| of SfM, white colour | code create_from_pcd |
| lambda_dssim | 0.2 | code / paper |
| lambda_RGB, lambda_alpha | 1e-4 | code / paper §4.1 |
| sigma (max impact) | **100** code default (paper says 500, Supp. Tab. 7) | code arguments |
| opacity_reg | 0.01 | code |
| lambda_normal / lambda_dist | 0 / 0 for M360, T&T | scripts/train_all.sh (paper §4.3 claims they were used) |
| background | black | code (no --white_background) |
| init alpha | logit(0.5 * G), G = 16x16 Gaussian PNG (22 corner texels exactly 0 -> logit -inf) | code |
| init RGB texture | logit(0.1) (paper: "zeros") | code |
| init SH dc | RGB2SH(colour) - 0.1 | code |
| init rotation | torch.rand(N,4) (normalised) | code |
| init scales | log sqrt(mean sq. dist to 3 NN), two columns | code (distCUDA2) |

## 4. Schedules
SH degree +1 every 1000 its; position lr expon (delay mult 0.01 inert, delay steps 0); textures lr 0
before 500 and from 30000; all groups except f_rest lr 0 for it > 30000; densify window (500,25000);
normal/dist losses after 7000/3000 if enabled. Densification happens after backward and before
the optimizer step; relocation creates new Parameters (no grad) so that iteration's Adam step is
skipped for all fields (side effect of the official code).

## 5. Builds on
2DGS rasterizer (ray-splat intersection, transform matrices, normal/depth/distortion maps) with
the Gaussian replaced by the texture lookup; 3DGS-MCMC densification (relocation/addition,
multinomial sampling by mean alpha, Adam moment reset at sampled indices). No 3DGS adaptive
density, no opacity reset, no scalar opacity.

## 6. Protocol
Datasets: T&T (Train, Truck, Francis, Horse, Lighthouse; ~900 px wide, i.e. 3DGS's 979 px),
Mip-NeRF 360 (7 public scenes; outdoor /4, indoor /2), DTU (15 scenes). llffhold 8 (every 8th
test). PSNR/SSIM/LPIPS (official `metrics.py`, LPIPS VGG as 3DGS). Per-scene numbers are given
only for the fixed-storage experiment (Tab. 1; Supp. Tab. "Quantitative metrics for all scenes");
the large-budget per-scene table is commented out in the LaTeX source. RTX 4090.

## 7. Ambiguities
- The scout note "initialised from a 3DGS/2DGS model" is wrong: training starts from SfM points.
- Exact cap_max/max_read_points for Tab. 1 are not published; we use cap = Tab.1 "Points"
  (T&T 30K, M360 60K) and max_read = cap - 10K (same split as the published scripts).
- sigma: paper 500, code 100 — code used.
- Paper says Tab. 1/2 BBSplat numbers used 2DGS depth-normal regularisation; published scripts
  use it only for DTU — scripts used (lambda 0 for M360 / T&T).
- RGB texture init: paper "zeros", code sigmoid^-1(0.1) with SH dc lowered by 0.1 — code used.
- Evaluation: official render.py loads the 8-bit quantised textures (`save_texture`/`load_texture`)
  before rendering test views, i.e. reported metrics include texture quantisation.
- FPS subsampling uses pytorch3d `sample_farthest_points` (start at point 0); point order follows
  the COLMAP ply.
