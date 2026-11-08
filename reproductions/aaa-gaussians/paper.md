# AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering — implementation spec

Steiner, Köhler, Radl, Windisch, Schmalstieg, Steinberger. ICCV 2025, arXiv 2504.12811.
Sources: `source/latex/` (arXiv e-print), `source/paper.pdf`, official code
`source/code/` = github.com/DerThomy/AAA-Gaussians @ dc5b3f2 with the rasterizer submodule
`source/code/submodules/diff-gaussian-rasterization/` = github.com/DerThomy/AAA-Gaussians-Rasterization
@ d03ce2d. Licences: Gaussian-Splatting licence (non-commercial research), StopThePop headers
`cuda_rasterizer/stopthepop/` MIT.

## 1. Method summary
- A 3DGS renderer that evaluates every Gaussian **in 3D** (max contribution along each pixel ray via
  screen-space planes transformed into Gaussian space, Hahlbohm et al. "HTGS"), no 2D EWA splat.
- Inside the **StopThePop hierarchical per-ray sorting rasteriser** (sort_mode HIER, PTD_MAX,
  queues 64/8/4), per-pixel blending order = depth of the max-contribution point.
- **Adaptive 3D smoothing filter** (anti-aliasing): per view, Σ̂ = Σ + k/v̂'² I with
  v̂' = min(v̂_train, f/z) and an amplitude normalisation that only accounts for the scale change
  *perpendicular to the view ray*.
- **View-space angle bounding** (screen AABB from planes through the camera in view space)
  instead of HTGS's projective bounding, so Gaussians reaching behind the image plane are not
  discarded (no frustum-edge pop-in).
- **3D frustum culling**: exact max-contribution test of a Gaussian against the whole view
  frustum (preprocess) and against each tile frustum (tile-based culling).
- Densification = **3DGS-MCMC** with the per-scene caps of the official `full_eval.py`.

Contributions:
- C1 Adaptive 3D anti-aliasing filter with perpendicular amplitude scaling (§3.2, Eq. 7–11).
- C2 Perspective-correct view-space bounding (§3.3, Eq. 12–14; supp. "Derivation of Bounds").
- C3 3D frustum-based (and per-tile) culling (§3.4, Eq. 15).
- C4 (built-on) 3D evaluation through screen-space planes (§3.1 "Evaluation in 3D", Eq. 4–5)
  within the StopThePop hierarchical rasteriser (§3.1 "Blending sort order", §4 impl. details).
- C5 (built-on) 3DGS-MCMC densification with identical settings (§4 impl. details).

## 2. Equations (verbatim from 3_method.tex)
Gaussian: $\mathcal{G}(\mathbf{x}) = \exp(-\frac12 \rho(\mathbf{x})^2)$,
$\rho(\mathbf{x}) = \sqrt{(\mathbf{x}-\bm\upmu)^\top\mathbf\Sigma^{-1}(\mathbf{x}-\bm\upmu)}$,
$\mathbf\Sigma = \mathbf{R}\mathbf{S}\mathbf{S}^\top\mathbf{R}^\top$.

3D evaluation (Eq. 4–5): $\mathbf{T}' = \mathbf{M}_{vp}\mathbf{P}\mathbf{V}\mathbf{T}$,
$\mathbf{T} = \begin{pmatrix}\mathbf{RS} & \bm\upmu\\ \mathbf 0^\top & 1\end{pmatrix}$,
$\bm\uppi'_{x/y} = \mathbf{T}'^\top \bm\uppi_{x/y}$ with $\bm\uppi_x=(1,0,0,-x)^\top$,
$\bm\uppi_y=(0,1,0,-y)^\top$; contribution = distance² of the plane-intersection line to the
origin in Gaussian space (code `max_contrib_ray`): ρ²_min along the ray, alpha =
min(0.99, o·G), skipped if < 1/255; depth for sorting = NDC z of the max point, rejected outside
[-1, 1] (`hierarchical_render.cuh:536`).

Mip-Splatting 3D filter (Eq. 6–7): $\hat v_{train}=\max(\{\hat v_n\})$, $\hat v = f/d$,
$\hat{\mathbf\Sigma}=\mathbf\Sigma + k/\hat v^2\,\mathbf I$,
$\hat{\mathcal G}=\sqrt{|\Sigma|/|\hat\Sigma|}\exp(\dots\hat\Sigma^{-1}\dots)$.

Perpendicular normalisation (Eq. 9–11):
$\hat{\mathcal G}_\perp = \sqrt{|\Sigma_\perp|/|\hat\Sigma_\perp|}\exp(-\frac12(\mathbf x-\bm\upmu)^\top\hat\Sigma^{-1}(\mathbf x-\bm\upmu))$,
$\sqrt{\frac{|\Sigma_\perp|}{|\hat\Sigma_\perp|}} = \sqrt{\frac{|\Sigma|\,\mathbf d^\top\Sigma^{-1}\mathbf d}{|\hat\Sigma|\,\mathbf d^\top\hat\Sigma^{-1}\mathbf d}}
= \sqrt{\frac{d_1'^2 s_2^2 s_3^2 + d_2'^2 s_1^2 s_3^2 + d_3'^2 s_1^2 s_2^2}{d_1'^2\hat s_2\hat s_3 + d_2'^2\hat s_1\hat s_3 + d_3'^2\hat s_1\hat s_2}}$,
$\mathbf d' = \mathbf R^\top\mathbf d$, $\hat s_i = s_i^2 + k/\hat v^2$, $\mathbf d$ = unit vector camera→μ.

Effective rate (Eq. 12): $\hat v' = \min(\hat v_{train}, \hat v)$, i.e. filter variance
$\max(k/\hat v_{train}^2,\ k\,z^2/f^2)$ (code `consistent_common.cuh:39-41`:
`scale_mip_filter = max(filter_3d², (z/f)²·k)`, f = max(fx, fy)).

View-space bounds (Eq. 13–14): $\theta_{1,2}=\tan^{-1}\frac{s_{1,3}\pm\sqrt{s_{1,3}^2-s_{1,1}s_{3,3}}}{s_{3,3}}$,
same for φ with index 2; $s_{i,j}=\langle\mathbf t,\mathbf T_{(view,i)}\odot\mathbf T_{(view,j)}\rangle$,
$\mathbf t=(\tau_\rho,\tau_\rho,\tau_\rho,-1)$; rotate so $(\theta_\mu-\pi)<\theta_1<\theta_\mu<\theta_2<(\theta_\mu+\pi)$,
clamp to $[-(\pi/2-\epsilon),\pi/2-\epsilon]$ (code ε = 1e-5); negative discriminant → full screen
on that axis; camera inside ellipsoid (|o_gauss|² < τ) → discard.

Frustum culling (Eq. 15): $\min_{\mathbf x\in\mathcal F}\rho(\mathbf x)^2 < \tau_\rho$; only the
two closest planes + 3 edges are tested (`max_contrib_gaussian_frustum_3D`).
τ_ρ (code `cutoff`) = min(11.11, 2 ln(o_dil / (1/255))) and the comparison in code is
0.5·ρ²_min ≤ ln(o_dil·255).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| filter kernel k | 0.3 | §4 impl.; `forward.cu:149`, `mip_filter.py` `0.3**0.5` |
| sort | HIER, order PTD_MAX, queues tile_4x4 64, tile_2x2 8, per_pixel 4 | `configs/aaa.json` |
| culling | rect, tight opacity, tile_based_culling, hierarchical_4x4 = true; load_balancing true | `configs/aaa.json` |
| proper_ewa_scaling / eval_3D / new_aabb | true / true / true (new_aabb default in C++) | `configs/aaa.json`, `rasterizer.h:139` |
| z near / far | 0.01 / 100 | `scene/cameras.py:48-49` |
| iterations | 30000 | `arguments` |
| lr xyz | 1.6e-4 → 1.6e-6 ×extent, exp. over 30k | arguments |
| lr f_dc / f_rest / opacity / scaling / rotation | 0.0025 / 0.0025/20 / 0.05 / 0.005 / 0.001 | arguments |
| λ_dssim | 0.2 (fused_ssim, same padding) | train.py |
| opacity_reg, scale_reg | 0.01 · mean(σ(o)), 0.01 · mean(exp(s)) | train.py:107-108 |
| MCMC | relocate+add every 100, 500 < it < 25000, dead σ(o) ≤ 0.005, growth 5%/step to cap | train.py:130-134 |
| noise_lr | 5e5 × xyz_lr, op_sigmoid(k=100, x0=0.995) | train.py:141-147 |
| cap_max | bicycle 6131954, flowers 3636448, garden 5834784, stump 4961797, treehill 3783761, room 1593376, counter 1222956, kitchen 1852335, bonsai 1244819, truck 2541226, train 1026508, drjohnson 3405153, playroom 2546116 | `full_eval.py:22-25` |
| init | SfM points, opacity 0.5, scale = log(0.1·sqrt(mean 3-NN dist²)) | `gaussian_model.py:136-141` |
| SH | degree 3, +1 every 1000 it | train.py |
| background | black | arguments (white_background False) |

## 4. Schedules
- SH degree +1 at iterations 1000, 2000, 3000.
- MCMC relocation/addition at it % 100 == 0 for 500 < it < 25000; noise every iteration.
- **3D filter v̂_train**: `filter_3D` is `None` during the whole training (only per-view
  adaptive filter, `filter_3D ? filter_3D[idx] : 0`); `compute_3D_filter(trainCameras)` runs
  once at the final save (it 30000) and is stored in the PLY; `render.py` then renders test
  views with max(filter_3D², k z²/f²). `mip_filter.py`: min view z over train cameras with
  z > 0.2 and projection inside a 15% margin, divided by max focal_x, × sqrt(0.3); Gaussians
  seen by no camera get filter 0.

## 5. What it builds on / unchanged
3DGS (Inria) training loop + 3DGS-MCMC densification/noise/regularisers; StopThePop
hierarchical rasteriser; HTGS 3D evaluation. Loss L1 + 0.2 D-SSIM unchanged.

## 6. Protocol
Mip-NeRF 360 (images_4 outdoor, images_2 indoor, "pre-downscaled", no further resize since
widths < 1600), T&T, Deep Blending; llffhold 8 (`--eval`); 30k iterations; black background;
PSNR/SSIM/LPIPS (LPIPS-VGG via lpipsPyTorch, "unnormalized images"). Single seed. The main
table (Tab. 1 `image_metrics.tex`) and per-scene supp. Table (`tab:suppl_perscene_full`) are
the single-scale standard-resolution protocol; the multi-resolution table
(`tab:ablation_multires`, ½×/1×/2× for bonsai/bicycle) is separate (its 1× rows equal Table 6).

## 7. Ambiguities / code-vs-paper
- Paper says filter rate min(v_train, v); during training the code uses only the per-view
  term (filter_3D = None) — v_train only enters at test time. Port follows code.
- Backward (`backward.cu:650`) recomputes the per-view filter with focal_x (forward uses
  max(fx, fy)), treats the filter as constant w.r.t. the mean, and in `preprocessCUDA_3D` the
  `computeGauss2Screen_ProperEWA` *assigns* `dL_dmean`, overwriting the SH view-direction mean
  gradient accumulated just before. Vendored unchanged (faithful), documented.
- Quaternion normalisation is done in PyTorch (`get_rotation`); the kernel's gradient is w.r.t.
  the normalised quaternion.
- "All compared methods optimize to the same number of primitives" → caps in full_eval.py.
- Popping/FLIP, large-FOV and multi-resolution evaluations are out of scope for the PSNR gate.
