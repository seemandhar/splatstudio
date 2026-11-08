# MSAA-GS — Multi-Sample Anti-Aliasing and Constrained Optimization for 3D Gaussian Splatting

arXiv 2508.10507 (Zhou, Zhang, Xiong, Xia; Shanghai Univ. of Engineering Science, Aug 2025).
Sources: `source/latex/sn-article.tex` (LaTeX), `source/paper.pdf`.
**Official code: none found** (no URL in the paper/LaTeX — the "Code availability" item is
commented out; arXiv, web search and GitHub repository search for "msaa gaussian" return nothing;
`papers_100.json` `code_url` empty). Implementation follows the paper text.

## 1. Method summary
- 3DGS with its rasterizer changed to 4x multi-sample anti-aliasing: each pixel is alpha-blended
  independently at 4 sub-pixel positions and the 4 colours are averaged (training *and* test).
- Photometric L1 replaced by an error-adaptive weighted L1 (AWS); D-SSIM kept.
- An added gradient-difference loss (GDC) on forward-difference image gradients.
- Densification and all other hyper-parameters "consistent with 3DGS".
- "Constrained optimization" in the title = the two loss constraints (AWS, GDC); there is **no**
  constraint on scales/opacities.

Contributions
- C1 MSAA rasterization (Sec. 3.2, Eq. 4-5, Sec. 3.5 Eq. 14).
- C2 Adaptive Weighting Strategy (Sec. 3.3, Eq. 6-9).
- C3 Gradient Difference Constraints (Sec. 3.4, Eq. 10-12) with composite objective Eq. 13.
(The third bullet in the intro, "comprehensive benchmarking", is not a mechanism.)

## 2. Equations (verbatim; numbering = order in the paper)
Splat weight and compositing (3DGS, Eq. 2-3):
`w_n(u) = exp(-1/2 (u-p_n)^T (Σ_n^{2D})^{-1} (u-p_n))`; standard front-to-back alpha compositing.

C1 (Eq. 4-5):
```
\mathbf{u}_k = \mathbf{u} + \boldsymbol{\delta}_k, \quad k = 1..n,   δ_k ∈ [0,1]×[0,1]
\mathbf{C}^\text{MSAA}(\mathbf{u}) = \frac{1}{n} \sum_{k=1}^{n} \mathbf{C}_k(\mathbf{u})
```
"At each subpixel u_k, we perform alpha compositing ... to yield C_k(u)." n = 4 ("4×MSAA").
Eq. 14: `I_pred(i,j) = C^MSAA(u)`, u = (j,i). Fully differentiable through all n paths.

C2 (Eq. 6-8):
```
e_{i,j} = \lVert I_\text{pred}(i,j) - I_\text{gt}(i,j) \rVert_1
w_{i,j} = \alpha + (1-\alpha)\, \frac{e_{i,j}}{\max_{(p,q)\in\Omega} e_{p,q} + \varepsilon}
\mathcal{L}_w = \frac{1}{HW} \sum_{i,j} w_{i,j}\, \lVert I_\text{pred}(i,j) - I_\text{gt}(i,j)\rVert_1
\mathcal{L}_\text{recon} = \lambda_1 \mathcal{L}_w + \lambda_2 \mathcal{L}_\text{D-SSIM}
```
C3 (Eq. 10-12):
```
\nabla_x I(i,j) = I(i,j+1) - I(i,j),   \nabla_y I(i,j) = I(i+1,j) - I(i,j)
\mathcal{L}_\text{grad} = \frac{1}{(H-1)(W-1)} \sum_{i=1}^{H-1}\sum_{j=1}^{W-1} \lVert g^\text{pred}_{i,j} - g^\text{gt}_{i,j} \rVert_1
```
Composite (Eq. 13): `L = λ1 L_w + λ2 L_D-SSIM + λ3 L_grad`.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| sub-samples n | 4 | Sec. 3.2, Sec. 4.1.2 |
| sub-sample offsets δ_k | not given | — (we use the D3D/OpenGL standard 4x rotated-grid pattern, see §7) |
| λ1, λ2, λ3 | 0.8, 0.2, 0.1 | Sec. 4.1.2 ("Implementation") |
| α (AWS weight floor) | not given | — (0.5, the value of the older local Nerfstudio port) |
| ε | not given ( >0 ) | — (1e-6) |
| iterations | 30000 | Sec. 4.1.1 |
| everything else (LRs, densification, SH, opacity reset) | 3DGS defaults | Sec. 4.1.1 |

## 4. Schedules
None besides 3DGS's. The paper says "λ3 can be gradually increased during training if edges remain
overly smoothed" but gives no schedule (a commented-out LaTeX line mentions 0.1→0.5); the stated
implementation value is a constant 0.1, which we use.

## 5. Builds on
Vanilla 3DGS (Inria code; "Ours(3DGS)"). Also applied on Pixel-GS and AbsGS ("Ours(Pixel-GS)",
"Ours(AbsGS)", Table 1 — averages only). 3DGS training kept unchanged except the rasterizer
(MSAA) and the loss (L1 → AWS-weighted L1, + GDC).

## 6. Protocol
Mip-NeRF 360 (9 scenes), Tanks&Temples (truck, train), Deep Blending (drjohnson, playroom);
resolution "same as 3DGS" (images_4 outdoor / images_2 indoor), 3DGS split (llffhold 8), 30k iters,
single RTX 4090. Metrics SSIM/PSNR/LPIPS (net not stated; 3DGS uses VGG). Per-scene results:
Table 2 (Mip-NeRF 360, Ours(3DGS) vs an author-retrained 3DGS). Averages: Table 1; ablation
(3DGS+MSAA, 3DGS+AWS+GDC): Table 3.

## 7. Ambiguities and choices
1. **Sub-sample positions.** "δ_k within [0,1]×[0,1], a fractional displacement from the pixel
   centre", u = pixel centre. Read literally (u + δ, δ ∈ [0,1]²) every sample lies in the
   lower-right quadrant and the mean shifts the image by +0.5 px, misaligning it with the GT; we
   read δ as a position inside the pixel square [j, j+1]×[i, i+1] (sample = pixel corner + δ)
   and use the standard 4x MSAA rotated-grid pattern (0.375,0.125), (0.875,0.375),
   (0.125,0.625), (0.625,0.875), centroid = pixel centre.
2. **"Fused at the input layer or within intermediate feature fusion modules"** (Sec. 4.1.2) —
   meaningless for 3DGS; we average the colours (Eq. 5).
3. **Per-sample alpha / depth**: alpha and the depth channel are averaged the same way.
4. **ℓ1 norm scale.** e_{ij} is the ℓ1 norm over RGB; w only depends on e/max e so the
   sum-vs-mean choice does not change w. For L_w we use the per-channel mean (|·|_1 / 3) so that
   λ1 = 0.8 multiplies an L1 of the same scale as 3DGS's `l1_loss` (λ1/λ2 = 0.8/0.2 copy 3DGS).
5. **w is treated as a constant (detached)** — the max-normalised weight map is a weighting, the
   paper does not differentiate it.
6. **GDC reduction.** Eq. 12 divides by (H-1)(W-1) and sums ||g_pred - g_gt||_1 over both
   directions and channels. We implement mean over channels of |Δ∇x| + mean of |Δ∇y| on the
   (H-1)×(W-1) domain (common code form); the literal channel-sum would be 3x larger.
7. **Table inconsistency.** Table 1's 3DGS row = 3DGS paper averages (27.21), Table 2's per-scene
   3DGS row averages 27.49. Compare our improvement over our baseline to Table 2's per-scene
   delta (e.g. garden +0.13, kitchen +0.13).
8. Densification gradient: with MSAA inside the rasterizer, the screen-space mean gradient is the
   sum over the four sample paths (one means2d per Gaussian), as a fused kernel produces.
