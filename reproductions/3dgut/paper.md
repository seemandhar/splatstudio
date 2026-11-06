# 3DGUT — implementation spec

Paper: *3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting*, Wu, Martinez
Esturo, Mirzaei, Moenne-Loccoz, Gojcic (NVIDIA), CVPR 2025, arXiv 2412.12507v2.
Sources: `source/latex/` (arXiv e-print), `source/paper.pdf`, official code `source/code/`
(github.com/nv-tlabs/3dgrut @ bb203b1, Apache-2.0).

## 1. Method summary
- Replace 3DGS's EWA linearised projection (Eq. 3, Jacobian J) with an **Unscented Transform**:
  7 sigma points of each 3D Gaussian are projected exactly through an arbitrary (distorted /
  fisheye / rolling-shutter) camera model; the weighted sample mean/covariance give the 2D
  mean + conic used only for tiling/culling (bounding rectangles).
- The particle response is **evaluated in 3D** along each pixel's camera ray at the point of
  maximum response (3DGRT formulation), so no gradient flows through the projection function.
- Optional "sorted" variant: per-ray k-buffer (MLAB, k=16) ordering by t_max (not used for the
  unsorted "Ours" row).
- Densification uses the 3D positional gradient scaled by half the camera distance (no 2D
  screen gradients exist), densify every 300 its.
- Everything else from 3DGS.

Contributions:
- **C1 UT projection** (§4.1, Eq. 5–9, Alg. Estimate2DGaussian): sigma points + weights → 2D
  mean/covariance; supports any camera model g(·) incl. rolling shutter (camera pose interpolated
  per sigma point).
- **C2 3D particle evaluation** (§4.2, Eq. 10, Supp. B): α = σ·exp(-½ ω_g²) at τ_max along the ray;
  backward in canonical Gaussian space.
- **C3 per-ray sorting / MLAB k-buffer** (§4.3) — only in "Ours (sorted)"; not reproduced (we target
  the unsorted "Ours" row).
- **C4 training adaptation** (§4.4): densification on 3D position gradients ×(dist/2), every 300
  iterations; UT α=1, β=2, κ=0.
- (Secondary-ray hybrid rendering §6, generalized kernels Supp. A: applications, not benchmarked.)

## 2. Equations (verbatim)
Eq. sigma points:
```
x_i = μ                          i = 0
x_i = μ + sqrt((3+λ)Σ)_[i]       i = 1,2,3
x_i = μ - sqrt((3+λ)Σ)_[i-3]     i = 4,5,6
```
`sqrt(Σ)` read from the factorisation Σ = R S Sᵀ Rᵀ (columns of R S).
Weights:
```
w_0^μ = λ/(3+λ),  w_i^μ = 1/(2(3+λ))
w_0^Σ = λ/(3+λ) + (1 - α² + β),  w_i^Σ = 1/(2(3+λ))
λ = α²(3+κ) - 3
```
2D mean/covariance: `v_μ = Σ_i w_i^μ g(x_i)`, `Σ' = Σ_i w_i^Σ (g(x_i)-v_μ)(g(x_i)-v_μ)ᵀ`.
With α=1, κ=0: λ=0, w_0^μ=0 (!), w_i^μ=1/6, w_0^Σ=2, w_i^Σ=1/6.

Eq. 10 (max response): `τ_max = (μ-o)ᵀΣ⁻¹d / (dᵀΣ⁻¹d) = -o_gᵀd_g / d_gᵀd_g`,
`o_g = S⁻¹Rᵀ(o-μ)`, `d_g = S⁻¹Rᵀd`; α = σ·ρ(o+τ_max d) = σ exp(-½‖o_g × d̂_g‖²).
Volume rendering Eq. 4: `c = Σ_i c_i(d) α_i Π_{j<i}(1-α_j)`.
Supp. B: ∂α/∂ω² = -½σe^{-½ω²}, ∂ω²/∂o_g = 2o_g + 2τ_max_g d̂_g, ∂o_g/∂μ = -S⁻¹Rᵀ.
Loss (paper §4.4 text): `L = L2 + 0.2 L_SSIM`; **code** (`configs/base_gs.yaml` loss):
`0.8 L1 + 0.2 (1-SSIM)` (use_l2: false). Code overrides → standard 3DGS photometric loss.
Densification statistic (code `threedgrut/strategy/gs.py:130`):
`accum += ‖∂L/∂μ · ‖μ - o_cam‖‖ / 2` for Gaussians with nonzero grad; densify if accum/count ≥ 2e-4.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| UT α, β, κ | 1.0, 2.0, 0.0 | paper §4.4; `configs/render/3dgut.yaml` |
| ut_in_image_margin_factor | 0.1 | code 3dgut.yaml |
| ut_require_all_sigma_points_valid | false | code 3dgut.yaml (gsplat python default: true) |
| kernel degree | 2 (Gaussian) | code 3dgut.yaml |
| particle_kernel_min_response | 0.0113 | code 3dgrt.yaml |
| min α / max α | 1/255 / 0.99 | code 3dgrt.yaml |
| min transmittance | 1e-4 | code 3dgut.yaml |
| tile culling | rect + tight-opacity bounding, tile-based culling (StopThePop) | code |
| k_buffer_size | 0 (unsorted, "Ours"); 16 for sorted | code base_unsorted / base_sorted |
| densify freq / start / end | 300 / >500 / <15000 | code strategy/gs.yaml, paper §4.4 |
| clone/split grad threshold | 2e-4 (on 3D stat above) | code |
| percent dense | 0.01 · scene extent | code |
| split N, scale shrink | 2, /(0.8·2) | code gs.py |
| prune: opacity < 0.005 | every 100, 500<s<15000 | code |
| screen-size / world-size prune | **none** (prune_scale, prune_weight disabled) | code |
| opacity reset | every 3000, 0<s<15000, to min(σ, 0.01), Adam state zeroed | code |
| lrs | pos 1.6e-4·extent→1.6e-6·extent exp, opacity 0.05, sh0 0.0025, shN 0.000125, rot 0.001, scale 0.005 | code base_gs.yaml |
| Adam eps | 1e-15 | code |
| SH degree | 3, +1 every 1000 | code |
| init (COLMAP) | SfM points, opacity 0.1, **scale = 0.01·dist to nearest camera center** (isotropic), **random quaternion (U[0,1)^4, normalised)** | code model.py:708 (`use_observation_points: true`) |
| background | black | code base_gs.yaml |
| iterations | 30000 | paper/code |

## 4. Schedules
SH degree +1 every 1000; densify 500<s<15000 every 300 (3D-grad stat accumulated every step
s<15000); prune opacity every 100 (500<s<15000); opacity reset at multiples of 3000 before 15000;
position LR exponential decay over 30k.

## 5. Builds on
3DGS (all optimisation parameters) + 3DGRT particle evaluation + StopThePop culling. Kept from
3DGS: SH colour, Adam lrs, opacity reset, clone/split rules, loss (code), 30k its.

## 6. Protocol
- Mip-NeRF 360: 9 scenes, outdoor images_4, indoor images_2, rectified 3DGS images, every 8th
  test (code `test_split_interval: 8`). Tanks&Temples truck/train. Metrics PSNR/SSIM/LPIPS.
- ScanNet++ (6 scenes, fisheye 1752×1168 converted to equidistant fisheye per FisheyeGS) —
  licensed dataset; Waymo 9 static scenes with lidar depth/opacity losses — licensed.
- Paper Table 1 3DGS row (27.26 MipNeRF360 / 23.64 T&T) differs from the 3DGS paper (27.21 /
  23.14); source unspecified.
- Per-scene tables in supp. (Tab. 6 T&T, Tab. 7 MipNeRF360, Tab. 8 ScanNet++).
- Official README re-run on RTX 5090 (unsorted, GS strategy): garden 27.18, counter 29.14,
  average 27.43 — slightly higher than the paper table (26.90 / 29.03 / 27.26).

## 7. Ambiguities
- Loss: paper says L2 + 0.2 SSIM; code uses 0.8 L1 + 0.2 D-SSIM. → code.
- Init: paper "adopt all parameters from 3DGS" (kNN scales, identity rotations); code uses
  camera-distance scales and random rotations. → code (configurable).
- Densify cadence: 3DGS uses 100; 3DGUT 300 (paper and code agree).
- Scene extent: code computes from all cameras (train+test); 3DGS uses train cameras.
- Camera position for the densification distance: code uses sensor pose at t=0
  (`batch.T_to_world[0,:3,3]`), identical for global shutter.
- The "3DGS" baseline row in Table 1 is not the 3DGS paper's number.
