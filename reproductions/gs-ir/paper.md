# GS-IR: 3D Gaussian Splatting for Inverse Rendering (CVPR 2024) — implementation spec

arXiv 2311.16473 · official code https://github.com/lzhnb/GS-IR @ e5a030b (cloned to
`source/code/`, MIT license file in repo; its `pbr/renderutils` is a copy of NVIDIA nvdiffrec
renderutils, NVIDIA Source Code License). Sources: `source/latex/main.tex`,
`source/latex/sec/X_suppl.tex`.

## 1. Method summary
- Three stages on one Gaussian set: (1) **initial stage** — standard 3DGS training plus a
  per-Gaussian normal supervised by normals derived from a linearly interpolated depth; (2)
  **baking** — the stage-1 Gaussians are frozen and SH occlusion (visibility) volumes are baked
  on a regular grid from 6-face depth cubemaps; (3) **decomposition stage** — per-Gaussian
  albedo / roughness / metallic + a learnable cube-map environment light + learnable SH
  irradiance volumes are optimised through a split-sum PBR shader of the rasterised G-buffer.
- Novel views are rendered by the PBR shader (that is what the paper's PSNR measures).

Contributions
- **C1 Depth by linear interpolation** (§4.1 "Depth Generation", Eq. 7): D = Σ w_i d_i / Σ w_i.
- **C2 Normal reconstruction with Gaussians as normal proxy** (§4.1 "Normal Derivation",
  Eq. 8–9, supp. Eq. 1–2): per-Gaussian normal n_i, rendered n̂ = Σ T_i α_i n_i, tied to the
  depth-gradient normal by L_{n-p} plus an edge-aware TV term.
- **C3 Baked occlusion volumes** (§4.2 Eq. 10–11, supp. §2 Eq. 6–7): 6-pass depth cubemap per
  grid probe → binary occlusion → SH coefficients; masked trilinear recovery at shading.
- **C4 Indirect illumination volumes** (§4.2 last paragraph, Eq. 13): learnable SH irradiance
  grid used inside occluded directions: I_d ≈ (1-O) I_d^dir + O I_d^indir.
- **C5 Split-sum PBR decomposition** (§4.3 Eq. 12–16, supp. Eq. 3–5): Cook–Torrance with IBL,
  pre-filtered cube-map mips + BRDF LUT; loss L_d = L_shade + λ_M TV_mat + λ_E TV_light.

## 2. Equations (verbatim) and how the code implements them
- Eq. 7: `\hat{D} = \sum^N_{i=1} \hat{w}_i d_i, \quad \hat{w}_i = \frac{T_i \alpha_i}{\sum^N_{i=1} T_i \alpha_i}`
  code: rasterizer `out_depth = D / O` (O = Σ w, zero if O ≤ 1e-6), d_i = view-space z.
- Pseudo normal `\hat{\bm{n}}_{\hat{D}} = \nabla_\text{uv}\hat{D}`; code
  (`depthmapToNormalCUDA`): depth is 3×3 median filtered (kornia, zero pad), back-projected
  (x−W/2)/f·D, one-sided differences choosing the side with the smaller |ΔD|, n = ddx × ddy,
  normalised, rotated to world; invalid (0) if depth < 0.01 anywhere in the 5×5 window or
  within 2 px of the border. Depth receives **no gradient** (the autograd backward ignores
  grad_depth) — normal_from_depth is a constant target.
- Eq. 8: `\mathcal{L}_{n\text{-}p}=\left\|\hat{\bm{n}}-\hat{\bm{n}}_{\hat{D}}\right\|`, code:
  `F.l1_loss(normal_map[:, mask], normal_from_depth[:, mask])`, mask = all 3 comps of
  normal_from_depth ≠ 0; `normal_map` = normalize(Σ w_i n_i) (not alpha-divided before norm).
- Eq. 9 / supp Eq. 1: `\mathcal{L}_n=\mathcal{L}_{n\text{-}p} + \lambda_{n\text{-}TV}\mathit{TV}_\text{normal}`,
  `\mathcal{L}_\text{init} = \mathcal{L}_c + \mathcal{L}_n` (L_c = 0.8 L1 + 0.2 D-SSIM).
- supp Eq. 2: `\triangle^{\hat{N}}_{ij} = \exp(-|I_{i,j}-I_{i-1,j}|)(\hat N_{i,j}-\hat N_{i-1,j})^2 + \exp(-|I_{i,j}-I_{i,j-1}|)(\hat N_{i,j}-\hat N_{i,j-1})^2`;
  code `get_tv_loss`: weights use the channel-mean |ΔI|; loss = mean_h + mean_w.
- **Normal/material gradient isolation** (code, `backward.cu`): rendered normal / albedo /
  roughness / metallic maps send gradients only to those per-Gaussian attributes, never to
  α, means or covariances ("do not contribute to the alpha/opacity"). So stage-1 geometry is
  driven only by L_c, exactly as vanilla 3DGS.
- Eq. 10 (occlusion SH): `O(\theta, \phi) = \sum^\mathit{deg}_{l=0} \sum^{l}_{m=-l} \bm{f}^\text{o}_{i(lm)} Y_{lm}(\theta, \phi)`.
- Eq. 11: `\bm{f}^\text{o}_{i(lm)} = \int_{S^2} \hat{\bm{O}}(\bm{\omega}) Y_{lm}(\bm{\omega}) d \bm{\omega}`,
  code `baking.py`: probe grid res 256 over [−16,16]^3 (Mip-NeRF 360), probes only at the 8
  grid corners around every Gaussian centre; per probe, Gaussians with |μ−p|∞ < 1.5 are
  rendered into 6 90° faces 256², depth = (D/O)·(opacity>0.5)·‖ray‖ (ray distance);
  cube → lat-long 256×512 (nearest); value = 1 − [0<d<thr] (thr 0.4, d==0 counts as visible:
  the stored SH is **visibility**); f = Σ value·Y_lm·ΔΩ with ΔΩ = (cos θ − cos(θ+dθ))dφ; 16
  coefficients (bands 0..3, "degree 4"). Empty grid cells receive the id of a 6-neighbour
  (iterative dilation) until all cells have an id.
- supp Eq. 6 masked trilinear: `\tilde{w}_k = 0 if (p_k - x)\cdot n \le 0 else w_k`, normalised,
  `f_x = \sum \hat w_k f_k`; code shifts x by n·bound/res first, clamps x to the AABB.
- supp Eq. 7: `\hat{O}_x(\theta,\phi) = \sum f_{x(lm)} Y_{lm}`; code `SH_reconstruction`: the
  value used in shading is the mean over 256 cosine-weighted (GGX importance sampling with
  roughness 1) Hammersley directions, curand jitter seeded (1234, pixel), of clamp(Ô,0,1).
- Eq. 12/14/15 split sum; Eq. 13: `I_\text{d}(\bm{x}) \approx \left(1 - \text{O}(\bm{x})\right) I^\text{dir}_\text{d}(\bm{x}) + \text{O}(\bm{x}) I^\text{indir}_\text{d}(\bm{x})`;
  code: `diffuse_light = env_diffuse(n)·V + (1−V)·irradiance` with V the reconstructed
  visibility and irradiance = clamp(Σ_lm c_lm(x) Y_lm(n), 0) from a 64³ grid of 9 SH
  coefficients, **single channel**, trilinear with clamped coordinates, AABB ±1.5 (train.py
  `--bound` default; the README does not pass it for Mip-NeRF 360).
- Eq. 15 (final): `L_o \approx (1 - m)\frac{a}{\pi}[(1-O)I^{dir}_d + O I^{indir}_d] + R I_s`;
  code: diffuse = diffuse_light·albedo (**no (1−m), no 1/π**: the diffuse cube is already
  cosine-convolved /π); F0 = (1−m)·0.04 + m·albedo; specular = prefiltered_env(r, refl)·(F0·A+B)
  with (A,B) = BRDF LUT(NoV, roughness); rgb = clamp(diffuse+specular, 0, 1) (no tone map, no
  gamma for Mip-NeRF 360); pixels whose rendered normal is exactly 0 → black background.
- Environment light: 6×256×256×3 cube, init U(0.25, 0.75); mips by 2×2 average to 16²
  (custom backward: bilinear cube lookup of the coarse gradient ×0.25); diffuse cube = cosine
  convolution of the 16² mip (Σ L·clamp(N·L,0,0.999)·ΔΩ/π); specular mip k of 4 (256…32) =
  GGX(roughness_k)-weighted normalised convolution, roughness_k = 0.08 + k/3·0.42, the 16² mip
  with roughness 1, energy cutoff 0.99; mip level from roughness: r<0.5 → (clamp(r,.08,.5)−.08)
  /.42·3, else (clamp(r,.5,1)−.5)/.5+3; sampled linear-mipmap-linear in "cube" boundary mode.
- Eq. 16 / supp Eq. 3–5: `\mathcal{L}_d = \|I-\hat I^{shade}\| + \lambda_M TV_{mat} + \lambda_E TV_{light}`;
  code: L1(pbr, gt) + 1.0·TV_mat([albedo, roughness, metallic]) (masked by normal_mask if any
  pixel is unmasked) + 0.001·[(1−roughness)[mask].mean() + metallic[mask].mean()] (not in the
  paper) + 0.01·TV_light (plain squared differences of the base cube sampled to a 512×1024
  lat-long map). No D-SSIM in stage 2.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| stage-1 iterations | 30000 | supp. Training Details; README |
| stage-2 iterations | 10000 (Mip-NeRF 360), 5000 (TensoIR) | README `--iterations 40000/35000`; supp says 10K |
| pbr_iteration | 30000 | code:train.py:751 |
| λ_dssim | 0.2 | code:arguments |
| normal_loss weight | 1.0 | code:train.py:264 |
| λ_n-TV | 5.0 | code:train.py:752 (default), supp. |
| λ_M (brdf_tv) | 1.0 | code:train.py:753, supp. |
| λ_E (env_tv) | 0.01 | code:train.py:754, supp. |
| lamb (roughness/metallic prior) | 0.001 | code:train.py:348 (not in paper) |
| roughness range | [0.04, 1] during training (r·0.96+0.04) | code:train.py:243 |
| normal / albedo / roughness / metallic lr | 0.05 (opacity_lr) | code:gaussian_model.py:training_setup |
| normal init | (0,0,1) raw; albedo/roughness/metallic raw 1 (sigmoid → 0.731) | code:create_from_pcd |
| light optimizer | Adam lr 0.05 (cube base + irradiance coeffs), eps 1e-8, clamp cube ≥ 0 after step | code:train.py:160–168, 437 |
| cube resolution | 256 | code:train.py:155 |
| irradiance volume | 64³ grid, 9 coeffs, 1 channel, AABB ±1.5, init 0 | code:volumes.py, train.py:157 |
| occlusion probes | res 256, bound 16, threshold 0.4, crop 1.5, 16 coeffs (Mip-NeRF 360) | README baking command; baking.py defaults |
| occlusion probes (TensoIR) | res 128, bound 1.5, threshold 0.25 | README |
| SH recon samples | 256 | gs_ir/__init__.py |
| densification | 3DGS defaults (500–15000, every 100, grad 2e-4, reset 3000) | code:arguments |
| position lr | 1.6e-4·extent → 1.6e-6·extent over 30000 steps | code:arguments |
| background | black (both stages, Mip-NeRF 360) | train.py (white_background False) |
| `--metallic` | on (Mip-NeRF 360), off (TensoIR) | README |
| `--indirect` | on | README |
| `--gamma`, `--tone` | off for NVS numbers | README |

## 4. Schedules
- SH degree +1 every 1000 steps (3DGS). Densification as 3DGS until 15000.
- Step 30000: stage switch. Baking happens offline between the two `train.py` runs; light
  optimiser and material gradients only from step 30001. Stage-2 Gaussian optimiser still
  steps but geometry/SH get no gradient (geometry is frozen in practice).

## 5. Builds on
3DGS (official rasterizer extended with normal/material channels). Stage 1 = unchanged 3DGS
for RGB/geometry (+ normal field). PBR pieces from nvdiffrec (cube map, split sum, BRDF LUT).

## 6. Protocol
- Mip-NeRF 360: 9 scenes, `-i images_4` outdoor / `-i images_2` indoor, `-r 1` (no further
  rescale), `--eval` (llffhold 8), black background. NVS metrics from `render.py --pbr
  --metallic --indirect`: PBR image clamped [0,1]; PSNR = mean of per-channel PSNRs; SSIM =
  3DGS ssim; LPIPS-VGG fed [0,1] images (no [-1,1] mapping). Eval render differs from the
  training render: `inference=True` (roughness map += final transmittance T, i.e. background
  roughness → 1) and `pad_normal=True` (normal_map·O + (1−O)·(0,0,1) after thresholding O at
  0.004 / 0.996, before normalisation; normal_mask computed before padding), and the
  roughness map is **not** remapped to [0.04,1] at eval.
- TensoIR Synthetic: 4 objects, normal MAE, NVS, albedo (per-channel scaled), relighting.

## 7. Ambiguities / code-vs-paper
1. Paper says 10K decomposition iterations; README uses 10K for Mip-NeRF 360 (40000 total).
2. Paper Eq. 15 has (1−m)a/π diffuse; code uses albedo·diffuse_cube (metallic only in F0).
3. Paper says depth-gradient normal; code median-filters depth first and stops depth gradients.
4. lamb prior (0.001) and roughness remap [0.04,1] are code-only; eval skips the remap and adds T.
5. Points for volume queries are cam_center + normalised_ray·z_depth (z depth used as a ray
   distance) — code quirk, replicated.
6. Irradiance AABB ±1.5 for Mip-NeRF 360 (default arg) while the occlusion AABB is ±16.
7. Official `train.py` skips the optimizer step on the final iteration (`iteration < iterations`).
8. Text says "7 publicly available scenes"; tables report 9 and the average is the 9-scene mean.
