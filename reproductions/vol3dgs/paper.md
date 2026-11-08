# Volumetrically Consistent 3D Gaussian Rasterization (Vol3DGS) — implementation spec

- arXiv 2412.03378 (CVPR 2025 highlight). Talegaonkar, Belhe, Ramamoorthi, Antipa.
- Official code: https://github.com/chinmay0301ucsd/Vol3DGS (MIT), rasterizer submodule
  https://github.com/chinmay0301ucsd/slang-gaussian-rasterization @5eca118 (Apache-2.0, slangtorch).
  Cloned read-only in `source/code/` (submodule checked out). LaTeX in `source/latex/main.tex`.

## 1. Method summary
- Replace 3DGS's splatted 2D alpha `o * G2D(p)` by the analytic opacity of the 3D Gaussian
  density integrated along the pixel ray: `alpha = 1 - exp(-kappa * G(gamma d) * sqrt(2 pi) * beta)`.
- Keep 3DGS alpha blending (sorted, non-overlapping assumption), tiles/sorting from the 2D splat.
- Reparameterise the density: `kappa = -log(1 - 0.99 theta) * mean(1/s) [/ sqrt(2 pi) in code]`, theta = sigmoid.
- Modified adaptive density control: world-space position gradients, separate clone/split
  thresholds, halve density of new Gaussians, split high-density (theta >= 0.99) Gaussians,
  prune by theta, no opacity reset, scene-extent multiplier, world-size prune 0.01 * extent.
- Softplus (beta = 10) colour activation; different LRs; SH degree +1 every 2000 its.

Contributions
- C1 Volumetric alpha by analytic ray integral (Sec. 4.1-4.2, Eq. 7-19; code `utils.slang:evaluate_splat_volr`).
- C2 Density reparameterisation (Sec. 5 "Reparameterizing density", Eq. 20; code `utils/general_utils.py:alpha2density`).
- C3 Densification heuristics (Sec. 5 "Heuristics", Supp. D; code `scene/gaussian_model_volr.py`, `train_volr.py`).
- C4 Training recipe: softplus RGB, LRs, SH schedule (Supp. D; `scripts/all_datasets.sh`, `arguments/__init__.py`).

## 2. Equations
Density field (Eq. 3-4): $\sigma(x)=\sum_i \kappa_i G_i(x)$, $G_i(x)=\exp\{-\tfrac12(x-\mu_i)^T\Sigma_i^{-1}(x-\mu_i)\}$.

Alpha blending (Eq. 13): $C(p)=\sum_i c_i\alpha_i\prod_{j<i}(1-\alpha_j)$, $\alpha_i = 1-\overline T_i$.

1D Gaussian along ray (Eq. 15-16), ray $o + t d$ with $o$ the camera centre (camera space: $o=0$), unit $d$:
$$\gamma_j=\frac{(\mu_j-o)^T\Sigma_j^{-1}d}{d^T\Sigma_j^{-1}d},\qquad \beta_j=\frac{1}{\sqrt{d^T\Sigma_j^{-1}d}}$$
Transmittance (Eq. 19, boxed): $\overline T_j=\exp(-\kappa_j G_j(\gamma_j d)\sqrt{2\pi}\beta_j)$.

Code (`utils.slang:evaluate_splat_volr`, camera space, eps = 1e-7):
```
den = max(eps, d.A.d);  num = mu.A.d;  if num <= 0: alpha = 0   (A = inverse 3D cov, camera space)
tmax = num/den; diff = tmax d - mu; pdfmax = exp(-0.5 diff.A.diff)
sigma_d = sqrt(1/(den+eps)); alpha = min(0.99, 1 - exp(-2.506 kappa sigma_d pdfmax))
skip if alpha < 1/255; stop (exclusive) when T*(1-alpha) < 1e-4
d = normalize(((x - W/2 + 0.5)/fx, (y - H/2 + 0.5)/fy, 1))   (pixel centre, principal point at centre)
```
Backward: slang autodiff of the same; per-Gaussian gradients skipped when d(inv_cov3d) has a NaN.

Density reparameterisation (Eq. 20 paper): $\kappa=-\log(1-0.99\theta)\frac13(\frac1{s_x}+\frac1{s_y}+\frac1{s_z})$, $\theta\in[0,1]$.
**Code** (`alpha2density`, reparam "ours"): `kappa = -log(1 - 0.99 theta) * mean(1/s) / 2.506`, theta = sigmoid(raw).
The /sqrt(2 pi) makes a ray through the centre of an isotropic Gaussian give alpha = 0.99 theta. Code wins.
Code `density2alpha` (general_utils.py:56) as written: `clamp(1 - exp(-kappa * 2.506 / mean(1/s)) / 0.99, 0, 0.99999)`
(operator precedence: only exp(.) is divided by 0.99). It is NOT the inverse of alpha2density: the
round trip is theta - 1/99, so the "split theta >= 0.99" test never fires, and halving gives
1 - sqrt(1 - 0.99 theta)/0.99 (~0.0102 below the intended (1 - sqrt(1 - 0.99 theta))/0.99; 0 for
theta < ~0.02). We port this as the default (it produced the paper's numbers); the intended inverse is
config `density2alpha: inverse`.

## 3. Hyper-parameters (Mip-NeRF 360; code = scripts/all_datasets.sh unless noted)
| name | value | source |
|---|---|---|
| iterations | 30000 | arguments default |
| position lr | 1.6e-4 -> 1e-5 (x spatial_lr_scale = camera extent), exp. decay, max_steps 30000 | script, Supp. D |
| feature lr (sh0) | 0.003; shN 0.003/20 | script, Supp. D, gaussian_model_volr.training_setup |
| density (theta) lr | 0.03 (constant) | script `--opacity_volr_lr 0.03`, Supp. D |
| scaling lr / rotation lr | 0.005 / 0.001 | arguments default |
| lambda_dssim | 0.2 (fused_ssim) | train_volr.py |
| theta init | 0.1 | create_from_pcd |
| scale init | log sqrt(mean 3-NN dist^2) | create_from_pcd (distCUDA2) |
| SH degree | +1 every 2000 its (max 3) | train_volr.py |
| softplus rgb | softplus(SH + 0.5, beta = 10) | `--softplus_rgb`, spherical_harmonics.slang |
| densify from / until / interval | >500, <15000, every 200 | script; paper Supp. D |
| clone grad threshold | 5e-5 | script, Supp. D |
| split grad threshold | 1e-7 (code) / 1e-8 (paper) | script vs Supp. D -> code |
| gradient for densification | world-space ‖dL/dμ‖ averaged over visible steps | train_volr.py (`viewspace_point_tensor = get_xyz`) |
| percent_dense | 0.01 (x scene_extent_mult 2 = 0.02 x extent) | script; old supp says 0.02 |
| scene_extent_mult | 2 (Mip-NeRF 360), 5 (T&T), 1.5 (DB) | script, Supp. D |
| min opacity (theta) prune | 0.05 (Mip, T&T in paper; T&T script 0.01), 0.03 DB | script |
| world-size prune | max scale > 0.01 x (2 x extent), after step 3000 | gaussian_model_volr.densify_and_prune, Supp. D |
| screen-size prune | max_radii2D > 20 after step 3000 — **no-op** (postfix zeros max_radii2D before the check) | code |
| opacity reset | disabled | `--disable_opacity_reset` |
| max points | 7e6 (stop densifying above) | train_volr.py |
| abs_gs | off for Mip-NeRF/DB, on for T&T | script |
| near plane | view z > 0.2 | vertex_shader.slang |
| tile radius | ceil(3 sqrt(lambda_max(Sigma2D + 0.3 I))), lambda via max(0.1, mid^2 - det) | utils.slang:splat_radius |
| background | black (renderer asserts black) | inria_3dgs_volr.py |

## 4. Schedules
- Position LR exponential decay 1.6e-4 -> 1e-5 over 30k (Inria get_expon_lr_func, no delay).
- SH degree: `if iteration % 2000 == 0: oneupSHdegree()` -> degree = min(it // 2000, 3).
- Densification statistics accumulate every step while `it < 15000` and N < 7e6.
- Densify-and-prune at it in (500, 15000) with it % 200 == 0; the size prunes only for it > 3000.

## 5. Densification details (code `densify_and_prune`)
1. grads = accum / denom (NaN -> 0).
2. Clone: grad >= 5e-5 and max scale <= 0.01 * ext (ext = 2 x camera extent). New copy identical
   except theta_new = halve(theta) where halve(theta) = density2alpha(alpha2density(theta, s)/2, s)
   = 1 - sqrt(1 - 0.99 theta) / 0.99 (official expression as written). **The original keeps its theta** (code; paper says "halve").
3. Split (on the post-clone population; clones get grad 0): (grad >= 1e-7 and max scale > 0.01 ext)
   OR density2alpha(kappa) >= 0.99 (no scale condition; unreachable with the official expression). 2 children: positions sampled from N(mu, R diag(s)),
   scale / 1.6, theta_child = halve(theta) (computed with the parent scale), SH copied; parent removed.
4. Prune: theta < 0.05; for it > 3000 also max scale > 0.01 ext (and the no-op screen prune).
5. Accumulators reset (postfix). Adam moments of new Gaussians = 0.

## 6. Builds on
Official 3DGS code base (Inria) + slang.D rasterizer. Unchanged: L1 + 0.2 D-SSIM loss, Adam, random
view order (shuffled stack), 3DGS COLMAP init, Inria clone/split geometry.

## 7. Protocol
- Mip-NeRF 360 (9 scenes), T&T (truck, train), DB (drjohnson, playroom); 3DGS split (every 8th test).
- Resolution: script passes no `-r`/`-i` -> Inria loader default `-r -1`: full-res `images/` PIL-resized
  to width 1600 when wider (all Mip-NeRF 360 scenes). Our matching dataset key: `mipnerf360-1600`.
- Black background, 30k its, metrics PSNR/SSIM(fused)/LPIPS-VGG (`metrics.py`), 3090 Ti.
- Per-scene numbers: Supp. E tables (Mip-NeRF 360, T&T, DB). 3DGS-Slang baseline: averages only.

## 8. Ambiguities
- kappa normalisation: paper Eq. 20 has no 1/sqrt(2 pi); code divides by 2.506 -> use code.
- split threshold 1e-8 (paper) vs 1e-7 (script) -> code. Both are effectively "split every visible
  large Gaussian".
- "halve density after clone": code halves only the new copy -> code.
- percent_dense: old supplement 0.02 / CVPR supplement 0.01 x extent multiplier 2 -> identical.
- Garden PSNR differs between the two supplement copies in the LaTeX (27.29 vs 26.87); bicycle/room
  are identical in both.
- Principal point: kernel assumes cx = W/2, cy = H/2; our port uses K's cx, cy (identical for centred
  intrinsics).
- Densify happens before optimizer.step in Inria (new params skip that step); native runtime steps first.
