# Plan — Analytic-Splatting (`analytic-splatting`, module `ss_analytic_splatting`)

## Components
| kind | name / factory | requires → provides | config (paper/code value) |
|---|---|---|---|
| renderer | `analytic-renderer` / `ss_analytic_splatting.renderer:create_renderer` | `state.gaussian3d`, `camera.pinhole` → `render.rgb`, `gradient.screen.signed` | `background` [0,0,0] (official `white_background` False) |
| density | `analytic-density` / `ss_analytic_splatting.density:create_density` | `gradient.screen.signed` | densify_from 500, until 15000, every 100, grad_threshold 2e-4, percent_dense 0.01, min_opacity 0.005, big_world 0.1 after 3000, reset_every 3000, reset_value 0.01, filter_coef 0.2, near 0.2, frustum_margin 0.15 |
| loss | base `photometric` (L1 + 0.2 D-SSIM) | unchanged | — |

No new FieldSpecs (the 3D filter used by the opacity reset is recomputed on the fly from the
training cameras at each reset step; the official code recomputes it after every densify step,
and every reset step is also a densify step, so the value used is identical).

## C1/C2 → code
- Renderer = official rasterizer `submodules/diff-gaussian-rasterization` (commit a905939)
  vendored **unmodified** in `ss_analytic_splatting/csrc/` (Inria licence + repo MIT licence
  copied), glm copied from the same submodule (`third_party/glm`). JIT-built with
  `torch.utils.cpp_extension.load` through `tools/ssenv`. `raster.py` reproduces
  `diff_gaussian_rasterization/__init__.py` + `gaussian_renderer.render` (no 3D filter, SH in
  kernel, means2D [N,3] screen-gradient tensor). Camera conversion: OpenCV K/viewmat →
  Inria FoV camera (centred principal point required; znear 0.01, zfar 100), as in SteepGS port.
- Eq. 13 (forward), supp. Eq. 8–12 (backward) live in `csrc/cuda_rasterizer/{forward,backward}.cu`.
  `reference.py` contains an independent float64 PyTorch implementation of the same shading
  (EWA projection without 0.3 dilation, eigen-decomposition, logistic window integral,
  α-blending with 0.99 clamp / 1/255 skip / T<1e-4 stop) used only by probes.

## Density (official train.py + gaussian_model.py, `dense=False`)
After each step s < 15000: accumulate ‖∂L/∂means2D[:, :2]‖ for radii>0; at s>500, s%100==0:
clone (grad ≥ 2e-4 & max scale ≤ 0.01·extent), then split (grad ≥ 2e-4 & max scale >
0.01·extent; 2 children, N(0, s) samples rotated, scale/1.6), prune opacity<0.005 and
(s>3000) max scale > 0.1·extent. No screen-size prune (official no-op). Opacity reset at
s % 3000 == 0 (< 15000): logit(min(o, 0.01/coef)), coef = sqrt(|S²|/|S²+f²|) with the
Mip-Splatting 3D filter f from the training cameras (log-space evaluation, as in the
Mip-Splatting port). Runtime calls `after_backward` after the optimizer step (official
densifies before it) — same as the other native ports.

## Probes
| probe | contribution | measures | fails if |
|---|---|---|---|
| `analytic-forward-integral` (CUDA) | C1, C2 | kernel α/o for one Gaussian per pixel vs (a) independent Eq. 13 (float64), (b) brute-force pixel-area quadrature, (c) the 3DGS centre sample | kernel ≠ Eq.13 (rel>2e-3), or not closer to the true integral than the centre sample for σ≈0.4 px |
| `analytic-backward` (CUDA) | C1, C2 | kernel gradients (means, log-scales, quats, opacity logits, sh0, means2D) for 3 composited Gaussians vs float64 autograd of the reference, and vs central finite differences of the kernel itself | rel err > 2e-2 (autograd) / 5e-2 (FD) |
| `analytic-logistic-cdf` (CPU) | C2 | max |S−Φ| over [−6,6] (paper: ~1e-4), analytic dS/dx vs autograd, Python Eq.13 vs quadrature | S replaced by another CDF or Eq. 13 wrong |
| `analytic-density-schedule` (CPU) | 3DGS training kept | densify cadence, clone/split split by percent_dense, filter-aware reset ceiling 0.01/coef, no screen-size prune | schedule/reset deviates |

## Fidelity gaps
1. **MTMT via dataset conversion** (`mipnerf360-ms`, tools/make_multiscale.py). The official
   train_ms.py is train.py on the union of the 4 scales (no radii rescale: commented out at
   train_ms.py:209-211); its only extra mechanism, `--sample_more_highres` (P(1×)=0.475), is
   approximated by 3 copies of each 1× train frame (P(1×)=0.50; exact 19:7 weighting needs ~40×
   frames in RAM). With-replacement sampling instead of the without-replacement stack.
2. Densify/reset after the optimizer step instead of before (all native ports). Small.
3. Random view sampling with replacement, white-bg/SSIM/LPIPS runtime differences (CLAUDE.md).
4. Reset uses log-space coef (avoids the official fp32 0/0); identical where official finite.
5. "Ours + 3D filter" variant not ported (not the main row).

## Files
`plugin/pyproject.toml`, `plugin/README.md`, `plugin/ss_analytic_splatting/{__init__,raster,
renderer,density,filters,reference,probes}.py`, `plugin/ss_analytic_splatting/csrc/**`,
`plugin/tests/test_analytic_splatting.py`.

## Benchmark
Primary (claim test): MTMT `mipnerf360-ms` garden + counter, 30k, `analytic-splatting` and
`gsplat` (same data, same duplication) in one exp; per-scale report with
tools/multiscale_summary.py. Target: gain over our gsplat vs the paper's gain over its MTMT
3DGS row (garden +0.97/+0.99/+2.07/+3.53, avg +1.89; counter +0.78/+0.74/+1.18/+3.24, avg +1.48).
Secondary (no-regression): STST `mipnerf360` garden + counter vs `baseline3dgs-mipnerf360-30000`
(paper: 27.39 / 29.11; 3DGS row 27.41 / 28.70). Optional `mipnerf360-pilr`
garden + counter vs `baseline3dgs-mipnerf360-pilr-30000` (paper Ours* 27.73 / 29.24, 3DGS*
27.70 / 29.08 — same-author retrained baseline).
