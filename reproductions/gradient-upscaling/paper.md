# Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images — implementation spec

Niedermayr, Neuhauser, Westermann (TU Munich), ICCV 2025, arXiv 2503.14171.
Sources: `source/latex/main.tex`, `source/latex/sec/X_suppl.tex`, `source/paper.pdf`.

**Official code**: https://github.com/KeKsBoTer/upscale3dgs contains only README + project page
("Training code is coming soon", checked 2026-11-05; `source/code/`). The authors' web viewer
fork https://github.com/KeKsBoTer/brush (Apache-2.0, `source/brush/`) contains the
*inference* path in WGSL: `crates/brush-render/src/shaders/rasterize.wgsl` (forward image
gradients) and `upscale.wgsl` (spline upscaler). No CUDA, no backward, no training code exists
publicly; the paper's CUDA rasterizer and SLANG.D upscaler are unreleased.

## 1. Method summary
- Render a 3DGS model at a reduced resolution (factor s = 2, 3, 4, 8) and, in the same blending
  loop, the **analytic spatial derivatives** I_x, I_y, I_xy of the splatted image.
- Upscale to target resolution with a **bicubic (Hermite) spline** per low-res pixel cell whose
  corner values *and* derivatives are the rendered ones (classical bicubic would estimate
  derivatives with finite differences).
- Both stages are differentiable; **training with upscaling** renders at low res, upscales,
  and evaluates the 3DGS loss at the full ground-truth resolution, back-propagating through the
  spline and the derivative rendering ("Spline (Ours)").
- Inference-only use on a normally trained low-res model is also evaluated ("Spline (3DGS)").
- Densification: Taming-3DGS budgeted growth (Mallick et al.) to a fixed count (5 M Mip-NeRF360,
  1 M T&T / DB) for every method in the tables.

Contributions
- C1 Gradient-aware spline upscaling (Sec. 5 "Spline Interpolation", Eq. F=CAC^T; supp. "Spline
  Image Interpolation").
- C2 Analytic image gradients in the 3DGS forward pass (Sec. 5 "3DGS Image Gradients"; supp.
  "Forward pass").
- C3 Backpropagation of the analytic gradients / training with upscaling (Sec. 5 "Differentiable
  Image Gradients", "Scene Reconstruction"; supp. "Backward Pass", "Inversion Trick").
- C4 (evaluation) speed: render low-res + upscale vs full-res render (Tab. 2, supp. Tab. render
  times).

## 2. Equations (verbatim)
Blending (Eq. 3–5):
```
I(x,y) = \sum_{i=1}^{N} T_i(x,y) \alpha_i(x,y) c_i,  \alpha_i(x,y) = \sigma_i \exp(g_i(x,y)),
T_i(x,y) = \prod_{j=1}^{i-1}(1-\alpha_j(x,y));   g_i = -d_i\Sigma_i d_i^T,  d_i=[x-\mu_x, y-\mu_y]
```
Image gradients (main Eq. 9–11):
```
\partial I/\partial x = \sum_i c_i (\partial T_i/\partial x\,\alpha_i + T_i\,\partial\alpha_i/\partial x)
\partial^2 I/\partial x\partial y = \sum_i c_i(\partial^2T_i/\partial x\partial y\,\alpha_i + \partial T_i/\partial x\,\partial\alpha_i/\partial y
                                   + \partial T_i/\partial y\,\partial\alpha_i/\partial x + T_i\,\partial^2\alpha_i/\partial x\partial y)
```
Front-to-back recurrences (supp. "Forward pass"):
```
A_i = A_{i-1} + \alpha_i(1-A_{i-1}),  T_i = 1-A_{i-1}
\partial A_i/\partial x = \partial A_{i-1}/\partial x (1-\alpha_i) + (1-A_{i-1}) \partial\alpha_i/\partial x
\partial^2 A_i/\partial x\partial y = \partial^2 A_{i-1}/\partial x\partial y (1-\alpha_i) + (1-A_{i-1})\partial^2\alpha_i/\partial x\partial y
                                    - \partial A_{i-1}/\partial x\,\partial\alpha_i/\partial y - \partial A_{i-1}/\partial y\,\partial\alpha_i/\partial x
```
Per-Gaussian derivatives (brush `rasterize.wgsl`, consistent with the paper's g):
`dg/dx = -(a dx + b dy)`, `dg/dy = -(b dx + c dy)`, `d2g/dxdy = -b` (conic (a,b,c));
`α_x = α g_x`, `α_y = α g_y`, `α_xy = α (g_x g_y + g_xy)`.

Backward (supp. "Backward Pass"): back-to-front `B̂_i = (1-α_i) B̂_{i+1} + α_i c_i`,
`∂I/∂α_k = (1-A_k)(c_k - B̂_{k+1})`, its x/y/xy derivatives (supp. eqs.), colour
`∂B̂/∂c_k = (1-A_k) α_k` and its derivatives, chain rule to σ, Σ, μ ("computed analogously");
inversion trick (Weiss & Westermann 2022) recovers A_{i-1} and its derivatives from the stored
final A_N jet. (Typo in the paper: `A_{i-1} = A_i α_i/(1-α_i)` should be `(A_i-α_i)/(1-α_i)`.)

Spline (supp. "Spline Image Interpolation"):
```
p(x,y) = \sum_{i=0}^{3}\sum_{j=0}^{3} a_{ij}x^i y^j = [1 x x^2 x^3] A [1 y y^2 y^3]^T,
F = C A C^T,  A = C^{-1} F (C^T)^{-1},
F = [[f(0,0), f(0,1), f_y(0,0), f_y(0,1)], [f(1,0), f(1,1), f_y(1,0), f_y(1,1)],
     [f_x(0,0), f_x(0,1), f_xy(0,0), f_xy(0,1)], [f_x(1,0), f_x(1,1), f_xy(1,0), f_xy(1,1)]]
C = [[1,0,0,0],[1,1,1,1],[0,1,0,0],[0,1,2,3]]
```
This is exactly the tensor-product cubic Hermite patch on one low-res pixel cell, with
derivatives in units of low-res pixels.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| upscale factors | 2, 4 (main Tab. 1), 8 (supp.), 3 (timing Tab. 2) | Tab. 1, supp. |
| Gaussians | 5 M (Mip-NeRF360), 1 M (T&T, DB) | Tab. 1 caption |
| densification | Taming-3DGS predefined growth to that count | §6.2 |
| iterations / lr / loss | 3DGS defaults (code based on Kerbl et al.); L1 + 0.2 D-SSIM at full GT res | §6.2, §6.3 "SSIM on the full-resolution image" |
| bicubic baseline | PyTorch differentiable bicubic (`F.interpolate(mode='bicubic')`) | §6.2 |
| low-res size | brush: `floor(W/s)`; Inria `-r`: round | brush render.rs:102 |
| alpha clamp / thresholds | 0.999, 1/255, T<=1e-4 (brush = gsplat numerics) | brush rasterize.wgsl:89-94 |

## 4. Schedules
None beyond 3DGS/Taming (densify 500–15000, opacity reset every 3000, SH degree +1 per 1000).

## 5. Builds on
Inria 3DGS training code + Taming densification. Everything except the renderer (low-res
rasterization with derivative jets + upscaler) is unchanged.

## 6. Protocol
- Datasets: Mip-NeRF360 all 9 scenes at **full resolution 5187x3361**, T&T truck/train
  (979x546), DB drjohnson/playroom (Tab. 1 header "1264x832") — "we used the full-resolution
  images provided by the datasets". Test split = 3DGS llffhold-8 (assumed).
- Rows: *3DGS* / *Mip*: train at W/s, render at full res. *Bicubic*: per supp. Tables (durations
  > 3DGS) trained with bicubic upscaling in the loop (Tab. 1 caption says "upscales
  low-resolution renderings from 3DGS" – contradictory; the per-scene table averages match
  Tab. 1, so the training interpretation is used). *Spline (Ours)*: trained with spline
  upscaling. *Spline (3DGS)*: 3DGS trained at W/s, rendered at W/s, spline-upscaled (averages
  only).
- Metrics: PSNR/SSIM/LPIPS at full resolution; LPIPS net unspecified (3DGS code: VGG).
- GPUs: NVIDIA L4 (timings).

## 7. Ambiguities and choices
1. No released training code: the backward pass is derived from the paper's equations
   (implemented as reverse-mode over the derivative jets; equivalent to the supp. equations).
2. Clamped alpha (0.999): brush differentiates α unclamped; we use the exact derivative of the
   clamped α (zero) — consistent with gsplat's backward, which also drops gradients there.
3. Background: brush ignores the background in the image gradient ("TODO"); the paper's Inria
   base uses a black background. Our runtime uses white; the derivative includes `bg·∂T`.
4. Border handling of the upscaler: brush clamps samples to the edge with a mirrored fraction;
   we use the edge cell and extrapolate the cubic (first-order consistent with the analytic
   derivative). Affects a ≤ s/2-pixel strip.
5. Taming options besides the growth curve (batched SH, opacity freeze, lr changes) — not
   mentioned; only the densification is used.
6. Densification interval for an absolute budget: Taming default 100 (assumed).
