# 2D Gaussian Splatting for Geometrically Accurate Radiance Fields (2DGS)

Huang, Yu, Chen, Geiger, Gao — SIGGRAPH 2024 — arXiv 2403.17888.
Official code: github.com/hbb1/2d-gaussian-splatting @f3e3b9f (cloned to `source/code/`), rasterizer
github.com/hbb1/diff-surfel-rasterization @e0ed020 (`source/code/submodules/`). Licence: Inria/MPII
Gaussian-Splatting licence (non-commercial research and evaluation use) — vendoring for research is allowed
with the licence kept and cited.

## 1. Method summary
- Scene = set of 2D oriented planar Gaussian disks (surfels): centre p_k, two tangents t_u, t_v (from a
  quaternion; normal t_w = t_u × t_v), two scales (s_u, s_v), opacity, SH colour (degree 3).
- Rendering by explicit ray–splat intersection in the splat's uv frame (homogeneous plane intersection),
  object-space low-pass filter max{G(u(x)), G((x−c)/σ)}, σ = √2/2, then 3DGS tile sort + alpha blending.
- Renderer also outputs alpha-blended normal, expected/median depth of the *per-pixel intersection*, and a
  depth-distortion map (NDC-mapped depths).
- Losses: L = L_c + α L_d + β L_n (L_c = 0.8 L1 + 0.2 D-SSIM); distortion from step 3000, normal from 7000.
- 3DGS adaptive density control with the projected-centre gradient, opacity cull 0.05.

Contributions:
- C1 2D surfel representation + perspective-correct ray–splat intersection rasteriser with low-pass filter (§4.1–4.2, App. A/B).
- C2 Depth distortion loss L_d (§5, App. eq. distortion, efficient CUDA single-pass form).
- C3 Normal consistency loss L_n between splat normal and depth-gradient normal (§5).
- C4 Densification with projected 3D-centre gradient approximation (§6 Implementation).

## 2. Equations (verbatim from `source/latex/main/04_2DGS.tex`, `07_appendix.tex`)
- Splat: $P(u,v) = \bp_k + s_u\bt_u u + s_v\bt_v v = \bH(u,v,1,1)^T$, $\bH = [s_u\bt_u, s_v\bt_v, 0, \bp_k; 0,0,0,1]$; $\cG(\bu)=\exp(-\frac{u^2+v^2}{2})$.
- Screen: $\bx = (xz, yz, z, z)^T = \bW P(u,v) = \bW\bH(u,v,1,1)^T$ (eq:splat_to_screen).
- Planes: $\bh_u = (\bW\bH)^T\bh_x$, $\bh_v = (\bW\bH)^T\bh_y$, $\bh_x=(-1,0,0,x)$, $\bh_y=(0,-1,0,y)$.
- Intersection (eq:uv_intersection): $u=\frac{h_u^2h_v^4-h_u^4h_v^2}{h_u^1h_v^2-h_u^2h_v^1}$, $v=\frac{h_u^4h_v^1-h_u^1h_v^4}{h_u^1h_v^2-h_u^2h_v^1}$.
- Low-pass: $\hat\cG(\bx)=\max\{\cG(\bu(\bx)), \cG(\frac{\bx-\bc}{\sigma})\}$, σ=√2/2 (code: FilterInvSquare=2, rho=min(rho3d, 2|d|²)).
- Blend: $\bc(\bx)=\sum_i \bc_i\alpha_i\hat\cG_i(\bu(\bx))\prod_{j<i}(1-\alpha_j\hat\cG_j(\bu(\bx)))$.
- Distortion: $\cL_d = \sum_{i,j}\omega_i\omega_j|z_i-z_j|$ (main text); implemented (App.) as
  $\sum_i\sum_{j<i}\omega_i\omega_j(m_i-m_j)^2 = \sum_i \omega_i(m_i^2A_{i-1}+D^2_{i-1}-2m_iD_{i-1})$ with
  NDC depth $m = \frac{f}{f-n}(1-\frac{n}{z})$, n=0.2, f=100 (forward.cu:401).
- Normal: $\cL_n=\sum_i\omega_i(1-\bn_i^T\bN)$, $\bN=\frac{\nabla_x\bp_s\times\nabla_y\bp_s}{|\cdot|}$.
  Code: `normal_error = 1 - sum_c(rend_normal * surf_normal)` with rend_normal = Σ ω_i n_i (unnormalised,
  world frame) and surf_normal = depth_to_normal(surf_depth) · alpha.detach(); mean over pixels.
- Mean depth $z_{mean}=\sum\omega_iz_i/(\sum\omega_i+\epsilon)$ (code: D/alpha, nan→0); median $z_{median}=\max\{z_i|T_i>0.5\}$.
- surf_depth = (1−depth_ratio)·expected + depth_ratio·median; depth_ratio=0 (unbounded/default), 1 (DTU).
- Total: $\cL = \cL_c + \alpha\cL_d + \beta\cL_n$.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | code:arguments/__init__.py:79; paper Tab. M360 |
| λ_dist (α) | paper: 100 unbounded / 1000 bounded; **code default 0.0**, m360_eval.py no override; dtu_eval.py 1000 | paper §5; code:arguments:93; scripts |
| λ_normal (β) | 0.05 | paper §5; code:arguments:94 |
| dist start | step > 3000 | code:train.py:79 |
| normal start | step > 7000 | code:train.py:78 |
| depth_ratio | 0.0 (expected depth) | code:arguments:65, README |
| λ_dssim | 0.2 | code |
| lr xyz | 1.6e-4·extent → 1.6e-6·extent, 30k, delay_mult 0.01 (no delay steps) | code |
| lr feature / rest / opacity / scaling / rotation | 0.0025 / 0.0025/20 / 0.05 / 0.005 / 0.001 | code |
| percent_dense | 0.01 | code |
| densify from / until / interval | 500 / 15000 / 100 | code |
| grad threshold | 0.0002 (on ‖dL/dmean2D‖, mean2D grad = dL/dT[:,2]·depth·W/2 hack) | paper §6, backward.cu:634 |
| opacity cull | 0.05 | paper §6, code:arguments:95 |
| opacity reset | every 3000 (to min(o, 0.01)) | code |
| size threshold | 20 px after 3000 (no-op: postfix zeroes max_radii2D), world 0.1·extent | code |
| init | SfM points, scales log√(mean 3-NN dist²) ×2, rots torch.rand(N,4) (uniform [0,1)), opacity 0.1 | gaussian_model.py:124 |
| near / far (kernel) | 0.2 / 100 per-pixel depth cut; camera znear 0.01 zfar 100 | auxiliary.h:37, cameras.py |
| alpha clamp / min / T stop | 0.99 / 1/255 / T<1e-4 | forward.cu |
| background | black (white_background False) | code |

## 4. Schedules
SH degree +1 every 1000 steps; xyz LR exponential; densify 500–15000 every 100; opacity reset every 3000;
L_d on after 3000, L_n on after 7000.

## 5. Builds on
3DGS (Inria code): optimiser, densification (clone/split N=2 with /0.8N scaling, samples on the disk plane
with zero std on the normal axis), SH, losses. Changed: representation (2 scales), rasteriser, losses.

## 6. Protocol (NVS)
Mip-NeRF 360, 9 scenes, `-i images_4` outdoor / `-i images_2` indoor (pre-downsampled JPGs, no further resize
since widths < 1600), `--eval` llffhold 8, 30k iterations, black background, metrics.py (3DGS: PSNR, SSIM,
LPIPS-VGG) on saved PNG renders. Per-scene table in the supplementary. DTU/TnT geometry (Chamfer/F1) are not
NVS numbers.

## 7. Ambiguities
- λ_dist for Mip-NeRF 360: paper 100, released code 0. Code wins (rule); λ_dist kept configurable.
- Released code ≠ paper code: README says the low-pass-filter gradient was removed from densification
  (fewer Gaussians, "PSNR slightly dropped"). Only the released code is available → expect slightly lower PSNR.
- Paper says normal from median-depth points; code uses expected depth (depth_ratio 0) for unbounded scenes.
- Paper says "remove splats with opacity lower than 0.05 every 3000 step"; code prunes at every densification (100).
