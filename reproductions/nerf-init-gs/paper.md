# Evaluating Alternatives to SFM Point Cloud Initialization for Gaussian Splatting

Foroutan, Rebain, Yi, Tagliasacchi — arXiv 2404.12547 (v-latest LaTeX in `source/latex/`).

**Official code**: none. The abstract's "source code" link (theialab.github.io/nerf-3dgs →
github.com/theialab/nerf-3dgs) is the project page only (`href="URL_TO_CODE"` placeholder).
The first author's fork `Youlenda/gaussian-splatting` branch `3dgs_rf` (Jan 2024, cloned to
`source/code-3dgs_rf/`) is an early prototype that trains **TensoRF on NeRF-synthetic lego**
jointly with 3DGS (`train_rfgs_depth.py`: depth computed but no depth loss, empty rasterizer
submodule). It is not the paper's NerfAcc pipeline; used only to confirm the 3DGS base code
(official Inria 3DGS, black background, 3DGS defaults). Everything else is from the paper text.

## 1. Method summary
- Replace SfM point-cloud initialization of 3DGS by points sampled from a briefly trained NeRF.
- NeRF = NerfAcc's proposal-network iNGP (Mip-NeRF 360 contraction + proposal sampling), 5k its.
- Sample one point per ray from the volume-rendering-weight distribution for 500k uniformly
  drawn training rays; colour = NeRF radiance at that point/direction.
- Optionally distil NeRF depth into GS rendering with an L1 loss with exponentially decayed weight.
- Study of random init: a 50×50×50 box (50k Gaussians) beats the 3×camera-bbox random init.

Contributions:
- C1 Random-box initialization study (Sec. 3.1, 4.2, Table 1) — `50^3` box, 50k points.
- C2 Volumetric initialization from NeRF (Sec. 3.2, 4.3, Table 2).
- C3 Volumetric structure (depth) distillation (Sec. 3.3, 3.4, 4.4, Table 4, Supp. Eq. 13).
- (C4 COLMAP-free SLAM pipeline on TUM/Replica — out of scope, data/ORB-SLAM3 not available.)

## 2. Equations
Eq. 2 (surfaceness): $\surfaceness(t; \ray) = \sigma(\ray(t)) \exp(-\int_{t_{min}}^{t}\sigma(\ray(s))ds)$

Eq. 3 (CDF): $\cdf(t; \ray) = \int_{t_{min}}^{t} \surfaceness(s; \ray) ds$

Eq. 4 (init): $\mean = \ray(\cdf^{-1}(u; \ray)),\ c = c_{NeRF}(\mean; \ray),\ u\sim\mathcal U[0,1]$ —
"we uniformly sample a subset of all training rays and draw a single sample from each ray".

Eq. 5 (GS depth): $D_{GS}(\ray) = \sum_i d_i \opacity_i G_i \prod_{j<i}(1-\opacity_j G_j)$, $d_i$ = mean
of the 1D conditional Gaussian along the ray: $d = -\frac{d^T\Sigma^{-1}o}{d^T\Sigma^{-1}d}$ (Supp.).

Eq. 6 (NeRF depth): $D_{NeRF}(\ray) = \int t\,\sigma(\ray(t)) \exp(-\int\sigma) dt$, precomputed for all training rays.

Eq. 7: $\mathcal L_{GS+Depth} = \mathcal L_{GS} + \lambda \mathcal L_{Depth}$, $\mathcal L_{Depth} = \mathbb E_{\ray}|D_{NeRF}-D_{GS}|$.

Supp. Eq. 13: $\lambda(i) = \lambda_{init}\, d^{i/S}$, $\lambda_{init}=0.9$, $d=0.9$, $S=100$.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| NeRF | NerfAcc proposal iNGP (`train_ngp_nerf_prop.py`) | Sec. 3.2 |
| NeRF iterations | 5000 (main), also 10k/30k in Table 2 | Sec. 4.3 |
| NeRF config | nerfacc 360 defaults: 4096 rays, prop [256,96]+48 lindisp, near 0.2, far 1e3, Adam 1e-2 eps 1e-15 wd 0, LinearLR(0.01,100)+MultiStep(½,¾,9/10;0.33), opaque bkgd, factor 4 | nerfacc example (paper: "off-the-shelf") |
| init points | 500,000 | Sec. 4.3 |
| random-box init | 50k Gaussians, 50³ box at origin | Sec. 4.2, Table 1 caption |
| λ_init, d, S | 0.9, 0.9, 100 | Sec. 3.4, Supp. Eq. 13 |
| GS training | official 3DGS, 30k iterations, standard settings | Sec. 4.3, Table 1 caption |

## 4. Schedules
λ(i) decays continuously: 0.81 @100, 0.31 @1000, 0.038 @3000, 0.0046 @5000 — effectively a
first-~3k-iteration structure prior. All 3DGS schedules unchanged.

## 5. Builds on
Official 3DGS (Inria code, "current official public code release"), all densification /
optimisation settings unchanged; the point cloud "is loaded in exactly the same way as COLMAP
point clouds".

## 6. Protocol
Mip-NeRF 360, 7 scenes (garden bicycle stump counter kitchen bonsai room; no flowers/treehill),
COLMAP cameras, 3DGS protocol (llffhold 8, images_4 outdoor / images_2 indoor implied by
re-running 3DGS: their COLMAP re-run matches 3DGS numbers). Deep Blending (2 scenes, Supp.).
OMMO, TUM, Replica not available. Metrics PSNR/SSIM/LPIPS (3DGS code).

## 7. Ambiguities (choice made)
1. How the inverse CDF is evaluated: we use nerfacc's own piecewise-linear inverse CDF in the
   proposal sampler's lindisp s-space on the 48 final intervals (the "importance samples").
2. Opaque background: the last interval absorbs the remaining transmittance (nerfacc
   `opaque_bkgd=True`), so sky rays give samples/depth up to the far plane (1e3 nerf units).
   Paper does not mask or clip; we keep them (L1 gradients are bounded).
3. "5k iterations": we run nerfacc's schedule with max_steps = 5000 (milestones scale).
4. NeRF input resolution: nerfacc loader uses factor 4 for every scene; we train on the GS
   training images (images_4 outdoor, images_2 indoor) — depth must be at GS resolution anyway.
5. D_GS: whether the depth is ray distance or z, and whether accumulated or normalised: Eq. 5
   is un-normalised; we use un-normalised Σ w z with gsplat centre z (and convert NeRF ray
   distance to camera z) — see plan fidelity gaps.
6. λ decay: continuous exponent (Eq. 13 has no floor).
7. "Hyper-parameters ... can lead to an excessive number of Gaussians if the wrong parameters
   are used" — no changed densification parameters are listed, so 3DGS defaults are kept.
