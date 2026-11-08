# AuGS — Augmented Radiance Field (ICLR 2026, arXiv 2602.19916)

Sources: `source/latex/` (arXiv e-print), `source/paper.pdf`, official code `source/code/`
(github.com/XiaoXinyyx/AuGS_Code, commit 57cca00, 2026-04-21 — **real code, not page-only**;
includes a modified gsplat 1.4.0 in `submodules/gsplat`, a 2D Gaussian rasterizer
`diff-gaussian-2d-rasterization`, and the WPCA fork `source/wpca_fork/`). Code overrides paper.

## 1. Summary and contributions
Post-hoc enhancement of a converged 3DGS (or 3DGS-MCMC) scene:
- C1 (§3.1) **View-dependent opacity kernel** (Phong-like opacity lobe) with 5 extra params
  per augmented Gaussian: lobe direction d (3), angular span T, sharpness β.
- C2 (§3.2) **Error-driven 2D Gaussian training**: per training view, 2D Gaussians are fitted
  over the rendered image (as background) where the error is large; the kernel budget (10 % of
  the base Gaussian count) is split across views ∝ view loss; new kernels sampled ∝ loss².
  Depth maps for sorting / back-projection: "median" depth (T threshold) + max-response point.
- C3 (§3.3) **Inverse Gaussian splatting**: back-project each 2D kernel: SLINK clustering of the
  covered pixels' 3D points (fg/bg separation), weighted PCA → rotation/variances, scale
  calibration k (Frobenius fit of projected covariance), lobe initialisation (d → camera,
  T = 7θ̄/π clipped [0.01, 1], β = 0).
- C4 (§3.4) **Joint optimisation**: 30 × |train views| iterations; augmented kernels fully
  optimised; original kernels: only opacity (sparse Adam, visible/used kernels only).

## 2. Equations
Opacity lobe (Eq. 1):
$\hat\alpha(\theta,\beta,T,\alpha)=\alpha\left(\frac{\cos(\max(0,\min(\theta/T,\pi)))+1}{2}\right)^{\exp(\beta)}$,
θ = angle between normalised (camera − μ) and normalised d. Code (`gaussian_renderer/__init__.py`
compute_opacity): `acos(clip(dot, -0.9999, 0.9999))`, `base >= 1e-5` else attenuation 0,
T = `theta.clamp(max=pi)`, view dir uses the *learnable* aug means.

Sampling probability (Eq. 3): $p(u,v)\propto[(1-\lambda)\mathcal L_1(u,v)+\lambda\mathcal L_{SSIM}(u,v)]^2$,
λ = 0.2; L1 averaged over channels; L_SSIM = 1 − SSIM map (3DGS same-padding, channel mean); clamp ≥ 0.

Projected covariance (Eq. 4) Σ2D = J W R S (J W R S)^T, S = k·diag(σ); scale calibration (Eq. 6)
$k=\sqrt{\sum_{ij}Q_{ij}\hat\Sigma_{2D,ij}/\|Q\|_F^2}$, Q = Σ2D/k². Code (projector.compute_scale):
q = (JWR) diag(V) (JWR)^T with V = WPCA variances; k' = ⟨Ĉ,q⟩/⟨q,q⟩, scale = sqrt(k'·V);
Ĉ = raw 2D covariance (no +0.3); J uses tx/tz, ty/tz clamped to ±1.3 tan(fov/2).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| ratio aug / base count | 0.1 | §4.1, benchmark.py:7 |
| 2D lr xy / color / opacity / scale / rot | 0.001·(H+W)/2 / 0.01 / 0.02 / 0.001 / 0.02 | supp. §A, train_2d.py argparse |
| 2D init opacity, scale, color | 0.02, 0.005·min(W,H) px, render at pixel clamped [1/255, 244/255] | train_2d.py:37,165-172 |
| add fraction / interval | 20 % of per-view target every 200 steps (COARSE_OPT_STEP) | train_2d.py:40 |
| finetune steps | 1000 | train_2d.py:41 |
| prune | opacity ≤ 0.005 or pix_covered ≤ 25 (after each phase) | train_2d.py:83 |
| 2D scale clamp | ≤ min(50, 0.01(H+W)) px | gaussian_2d_renderer end_optimizer_step |
| 2D AA | +0.3 px² with opacity compensation sqrt(det/det_h), alpha ≤ 0.99 | diff-gaussian-2d forward.cu |
| depth-map T threshold | **0.7 in code** (paper says 0.5) | submodules/gsplat rasterize_to_pixels_fwd.cu:173 |
| cluster threshold | 5·δd, δd = tan(fovx/2)·max(0.2, depth)/(W/2); ×1.5 on failure | supp. §A, projector.py |
| WPCA | weighted mean, covar = (wX)^T(wX)/(w^Tw), eigh descending; var ≥ 1e-7 | wpca_fork |
| skipped kernels (<3 pts) / depth < 0.2 | depth := zfar = 100 | projector.project |
| T init | clip(7θ̄/π, 0.01, 1.0) | supp. §A, refine.py:51 |
| β init / higher SH | 0 / 0 | §3.3 |
| joint iters | 30 × |train| (bicycle 169 → 5070, room 272 → 8160) | benchmark.py |
| joint lr | xyz 1.6e-5·extent (const), f_dc 1e-3, f_rest 5e-5, opacity 0.02, scale 2e-3, rot 5e-4, d 1e-3, T 2e-4, β 2e-3 | supp. §A, benchmark.py |
| joint optimiser | SparseGaussianAdam (**no bias correction**) for 3DGS params (aug rows always; original rows only opacity, only where `is_used`); torch Adam (bias-corrected) for d/T/β | refine.py |
| loss | (1−0.2) L1 + 0.2 (1 − fused SSIM) | refine.py |

## 4. Schedules
Base: 30000 iterations of 3DGS (train_3dgs.py, vanilla 3DGS densification) at sh=3.
At 30000: render all train views (RGB + depth) → 2D stage → projection → joint stage of
30·|train| iterations (no densification, constant LRs).

## 5. Builds on
3DGS (`train_3dgs.py`, Inria pipeline with a gsplat renderer, opacity_lr 0.025) or 3DGS-MCMC
(`train_mcmc.py`, cap = scene_points/1.1, opacity_lr 0.05, densify until 25000, opacity/scale reg 0.01).

## 6. Protocol
Mip-NeRF 360 9 scenes, `--resolution 4` outdoor / `2` indoor (PIL resize of full-res images —
our `mipnerf360-pilr`), llffhold8 (`--eval`), black background, PSNR/SSIM/LPIPS(VGG) via render.py;
T&T, DB, NeRF-Synthetic. Seed 0. Per-scene tables in supp. (per_scene_m360, diff_ps_mipnerf).

## 7. Ambiguities / code-vs-paper
- Depth threshold 0.7 (code) vs 0.5 (paper) → use 0.7.
- 2D depth for sorting: bilinear resample during optimisation (code) vs nearest (paper text); nearest in projection.
- Pivot handling: argmax weight before and after the validity filter (code).
- Per-view kernel budget `ceil(loss_i/Σ · 0.1·N)`; add count `min(round(0.2·target), target − size)` loops forever for targets ≤ 2 in code (we use ≥ 1).
- compute_scale asserts ⟨q,q⟩ > 0 and ⟨Ĉ,q⟩ > 0 — a failure aborts the whole view in code.
- is_used = Gaussians that contributed (alpha ≥ 1/255 before termination) — modified gsplat.
