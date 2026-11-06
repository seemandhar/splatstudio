# 3D-HGS — native SplatStudio plan

Method `3dhgs`, module `ss_3dhgs`, package `splatstudio-method-3dhgs`.

## Architecture
The half-Gaussian alpha (Eq. 10) and its backward through erf, the plane normal and the
conditional z-statistics of the ray-space covariance cannot be expressed with gsplat. We vendor
the official rasterizer (`submodules/diff-gaussian-rasterization` of lihaolin88/3DHGS @ dde1bd5)
**unmodified** in `ss_3dhgs/csrc/` (Gaussian-Splatting licence, `csrc/LICENSE.md`; glm from the
same submodule), JIT-built with `torch.utils.cpp_extension.load` via `tools/ssenv`.
Python side re-implements `diff_gaussian_rasterization/__init__.py` + `gaussian_renderer.render`.

## Components
| kind | name / factory | requires | provides | config (paper/code value) |
|---|---|---|---|---|
| renderer | `hgs-renderer` `ss_3dhgs.renderer:create_renderer` | state.gaussian3d, camera.pinhole | render.rgb, gradient.screen.signed | background [0,0,0] (official), test_time_cull true (official test path loads a .ply → non-empty `cov3D_precomp_small` → half-aware tile rectangles) |
| density | `hgs-density` `ss_3dhgs.density:create_density` | gradient.screen.signed | – | densify_from 500, densify_until 20000, every 100, grad_threshold 2e-4, percent_dense 0.01, min_opacity 0.01 (on max of halves), big_world_after 3000, big_world 0.1, reset_every 3000, reset_value 0.02, clone_opacity_rescale true, opacity_lr_decay 1.4, opacity_lr_decay_every 5000 |
| loss | `photometric` (base) | render.rgb | – | 0.8 L1 + 0.2 D-SSIM |

## Fields
| field | shape | lr | init | density treatment |
|---|---|---|---|---|
| `opacities` (base) = α1 logit | () | 0.05 (÷1.4 every 5k) | logit 0.1 | clone: 1−√(1−α); split: copy; reset min(α,0.02) |
| `opacity2` = α2 logit | () | 0.05 (÷1.4 every 5k) | logit 0.1 = −2.1972246 | same as α1 |
| `hgs_normal` (raw, unnormalised) | (3,) | 0.003 | 0.6 | copied on clone/split |

## Implementation notes (equations → code)
- renderer: Inria raster settings from K/viewmat (znear 0.01, zfar 100, centred principal point
  check, as SteepGS port). Inputs: means, means2D zeros (retain grad), normal = raw `hgs_normal`,
  sh = cat(sh0, shN), opacities = stack(σ(opacities), σ(opacity2)) [N,2], exp(scales),
  normalize(quats). `cov3D_precomp_small`: empty in training; when autograd is disabled and
  `test_time_cull`, a non-empty [N,6] tensor (kernel only tests the pointer and recomputes the
  splitting ellipse itself via `calculate_small`; official values from `generate_small_cov` are
  never read). Output rgb HWC, info {means2d, radii}.
- density: official `densify_and_prune` with two opacities: stats vgn += ‖∇means2D[:, :2]‖,
  denom += 1 for radii>0; at s>500, s%100==0, s<20000: clone (grad ≥ τ, max s ≤ 0.01·extent)
  appending copies with α'=1−√(1−α) per half (original unchanged — official `copy_` into an
  advanced-index temporary is a no-op), split (padded grads, max s > 0.01·extent) 2 samples
  R·N(0,s)+μ, scale s/1.6, opacities/normal/SH copied, originals removed; prune max(α1,α2)<0.01
  ∨ (s>3000 ∧ max s>0.1·extent); screen prune inert (max_radii2D zeroed). s%3000==0: both halves
  → min(α,0.02), Adam moments zeroed. s%5000==0: lr of `opacities` and `opacity2` ÷1.4.
- Adam moments of appended rows are zero (`cat_tensors_to_optimizer`).

## Probes
| probe | contribution | check (fails if mechanism removed) |
|---|---|---|
| `hgs-equal-opacity` (CUDA) | C1 | α1=α2: vendored kernel = independent float64 3DGS compositor (no erf) to 1e-4, invariant to random normals; vs gsplat mean |Δ| reported |
| `hgs-half-plane` (CUDA) | C1 | single Gaussian α1=0.9, α2=0: per-pixel alpha = ½α1 G·P(half) where P(half) is a numerical z-quadrature of the 3D ray-space Gaussian over the half-space (not the erf formula); α2-side pixels ≈ 0; swapping α1/α2 complements |
| `hgs-gradients` (CUDA) | C2 | vendored backward vs float64 autograd of an independent compositor (same kernel conventions incl. normal-translation quirk) for opacities, sh, means, scales, quats, normal; random scene with opacity ≤ 0.45 (avoids the forward-only −4.8 cutoff) |
| `hgs-test-cull` (CUDA) | C2 | test-time half-aware culling changes the tile assignment only (render close to training-path render) and is active only without grad |
| `hgs-density` | C3 | clone rescale per half on copies only, split /1.6 + copied normal, prune on max half, reset 0.02 both halves w/ zero moments, opacity LR ÷1.4 at 5000, densify stops at 20000 |

## Fidelity gaps
1. Runtime order: optimizer step precedes density; LR decay at s%5000 applies from step s+1. Small.
2. SH degree switch one step later; random-with-replacement sampling; SSIM valid conv; LPIPS
   via `lpips` (official also VGG). Small.
3. Means LR schedule: runtime 0.01^(s/steps) ≈ Inria exp schedule (delay_steps 0). Small.
4. Paper-vs-code (follow code): normal LR 0.003 everywhere (paper DB 0.3 is ignored by the code),
   no normal LR decay, normal transform with translation, forward-only −4.8 cutoff. Recorded.
5. Initialization: `--recipe 3dgs` (3-NN scales, opacity 0.1, all SfM points) = official.
6. Evaluation split: llffhold 8 (`--eval`), as `full_eval.py`.

## Benchmark choice
Per-scene numbers exist (supp. tab:all). Protocol = `full_eval.py` images_4/images_2 → native
`mipnerf360`, baseline `baseline3dgs-mipnerf360-30000`. Scenes: garden (outdoor; paper 27.50 vs
3DGS 27.30, +0.20) and counter (indoor; 29.84 vs 28.70, +1.14). Our baseline: garden 26.89,
counter 28.85.
