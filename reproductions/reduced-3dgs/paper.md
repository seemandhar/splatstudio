# Reducing the Memory Footprint of 3D Gaussian Splatting (Reduced-3DGS)

Papantonakis, Kopanas, Kerbl, Lanvin, Drettakis. Proc. ACM CGIT 7(1) (I3D 2024),
doi:10.1145/3651282. **Not on arXiv** (the plan's slot had no arXiv id; `2404.04211` is an
unrelated paper). Sources: `source/paper.pdf` (project page PDF, `pdftotext` → `source/paper.txt`),
official code `source/code/` = github.com/graphdeco-inria/reduced-3dgs @ `dc905c1`
(Inria 3DGS licence, non-commercial research use).

## 1. Summary and contributions
- 3DGS + three memory reductions applied at different times of one 30k-step training run, and
  the reported quality is that of the **decoded** compressed model (`point_cloud_quantised_half.ply`).
- **C1 (§4.1) Resolution-aware redundant-primitive pruning** ("mercy points" in code): every
  1000 its, per-Gaussian redundancy score from a pixel-footprint sphere vs. 30-NN ellipsoid test;
  prune half (lowest opacity) of the Gaussians above an adaptive threshold, plus the 3% lowest
  opacity (≤0.05); an L1 opacity regulariser; dead-point pruning (opacity < 1/255) after 15k.
- **C2 (§4.2) Adaptive SH bands**: once at 15k, per-Gaussian band reduction from (a)
  transmittance-weighted colour std over training views (→ 0 bands) and (b) transmittance-weighted
  colour distance to the truncated SH (→ 1 or 2 bands); an L1 sparsity term on SH-rest throughout.
- **C3 (§4.3) Codebook quantisation + half floats** after training: 1-D K-means (256 entries,
  1-byte index per scalar component) for opacity, scale, rotation-real, rotation-imag, DC colour,
  and each of the 15 SH-rest coefficients; positions and codebook entries stored as float16.

## 2. Equations
Colour statistics (Eq. 1–2), per Gaussian, over the N training views where it is visible:
$$\mu_c=\frac{\sum_i c_i\bar T_i}{\sum_i\bar T_i},\quad
\sigma_c=\frac{\sum_i (c_i-\mu_c)^2\bar T_i}{\sum_i\bar T_i},\qquad
\bar T_i=\frac{\sum_k^P T_{ik}}{P}$$
$P$ = pixels the Gaussian is splatted to in view $i$, $T_{ik}$ = transmittance *in front of* the
Gaussian at pixel $k$ (code: accumulated only for pixels where it is blended, i.e. alpha≥1/255
and before early termination). Code computes the weighted variance with West's incremental
algorithm on the full (degree-3) clamped colour $c=\max(\mathrm{SH}(d)+0.5,0)$, then
$\mathrm{std}=\sqrt{\sigma^2}$ averaged over RGB; std < ε_σ → DC := (μ−0.5)/C0, degree 0, SH-rest := 0.

Band distance: $d_q=\frac{\sum_i \bar T_i\,\lVert c_3-c_q\rVert_2}{\sum_i \bar T_i}$, $c_q$ = clamped
colour from bands ≤ q. Choose the lowest q with $d_q<\epsilon_{cdist}$; **code** only tests q∈{2,1}
(loop `range(active-1, 0, -1)`), never q=0, with $\epsilon_{cdist}=6\sqrt3/255=0.04075$.

Redundancy (§4.1): $a_{min}$ = smallest world-space pixel footprint over training views whose
frustum contains the centre (code: NDC x,y∈[−1,1], z∈[0,1] ⇒ depth∈[0.01,100]; footprint
= z/f_x for landscape images). Sphere radius $r=a_{min}\sqrt3/2$. Neighbour j of i
(exact 30-NN excluding i) intersects if $\sum_k \frac{(R_j^T(\mu_i-\mu_j))_k^2}{(s_{jk}+r_i)^2}<1$
(code adds r to the axes; paper says "scale by"). score_i = 1 + #intersecting neighbours;
propagated score = min over all spheres (incl. own) the Gaussian's ellipsoid intersects.
Threshold $\tau_p=\max(\mu+\lambda_r\sigma,3)$ (σ = unbiased std); candidates score>τ_p;
prune candidates with opacity < median(candidate opacity) **or** any Gaussian with
opacity < min(q_{0.03}(opacity), 0.05) (`mercy_type=redundancy_opacity_opacity`).

Losses: $L=(1-\lambda)L_1+\lambda L_{D-SSIM}+\lambda_\alpha\,\mathrm{mean}_{vis}\,\sigma(o)$ and a
backward-only SH sparsity gradient $\frac{\lambda_{sh}}{45 V}\,\mathrm{sign}(f_{rest})$ for
the active coefficients of the V visible Gaussians (`rasterizer_impl.cu:566`) — i.e. the gradient
of $\frac{\lambda_{sh}}{45V}\sum_{vis}\sum_{active}|f_{rest}|$.

Quantisation: per codebook, Lloyd 1-D K-means, init = 256 random data values, ≤500 iterations,
stop when Σ|Δcentre| < tol (1e-3 DC, 1e-4 others), empty cluster centre := 0. Clustering spaces:
opacity in sigmoid space, scale in exp space (centres stored as logit / log), rotation on the
normalised quaternion, DC and SH-rest raw (3 channels share one codebook). Encoded size
(`generate_results.py:memory_results`, MiB):
$3P\cdot 2 + 3(P_0+4P_1+9P_2+16P_3) + 256\cdot16\cdot4 + 8P + 256\cdot 8\cdot 4$ bytes.

## 3. Hyper-parameters (`full_eval.py` `full_final`)
| name | value | source |
|---|---|---|
| lambda_sh_sparsity | 0.1 | `full_eval.py` high_sh_sparsity |
| cull_SH | iteration 15000 | full_eval / paper §4.2 |
| std_threshold ε_σ | 0.04 | paper §4.2, code |
| cdist_threshold | 6 → ×√3/255 = 0.0408 | code (paper: 0.04) |
| mercy_points, prune_dead_points | on | code |
| lambda_alpha_regul | 0.001 | code (paper: "L1 sparsity on opacity", no value) |
| mercy_type | redundancy_opacity_opacity | code |
| lambda_mercy λ_r | 1 | paper, code |
| mercy_minimum | 3 | paper, code |
| mercy_interval | 10 × densification_interval(100) = 1000 | code |
| box_size (pixel_scale) | 1 | code |
| knn | 30 | paper, code |
| opacity cull quantile / cap | 0.03 / 0.05 | paper, code |
| codebook size | 256 | paper |
| 3DGS defaults | lr, densify 500..15000/100, grad 2e-4, reset 3000, λ_dssim 0.2 | code |

## 4. Schedules
- Mercy at it%1000==0, it ≤ 27000 (`fine_tune_start = iterations − 3000`), skipped at opacity
  resets (it<15000 and it%3000==0) → 1k,2k,4k,5k,7k,8k,10k,11k,13k,14k,15k,16k,…,27k.
- Dead-point prune (opacity<1/255) every 100 its once it ≥ 15000, and at save its 7000, 30000.
- SH culling once at 15000 (after that iteration's optimizer step).
- After the last step: K-means codebooks → quantised (+half) PLY; evaluation renders the decoded
  half-float quantised model.
Order inside an official iteration: densify/prune → opacity reset → mercy → optimizer step →
save (dead prune) → SH cull.

## 5. Builds on
Original 3DGS (Inria code), all 3DGS training unchanged otherwise. Renderer modified to use a
per-Gaussian SH degree (`_degrees`), returns average transmittance for SH culling.

## 6. Protocol
Mip-NeRF360 (9 scenes, images_4 outdoor / images_2 indoor), T&T truck/train, DB
drjohnson/playroom; `--eval` llffhold 8, 30k its, black background, LPIPS-VGG
(`lpipsPyTorch`), A6000. Metrics of the quantised-half model.

## 7. Ambiguities / code-vs-paper
- Official SH-cull statistics feed the logit opacity to the rasterizer and alias mean_old (variance = Σ T (c − mean_new)²); replicated by default (`official_shcull_quirks`).
- cdist threshold 0.04 (paper) vs 6√3/255 = 0.0408 (code); distance culling never goes to 0 bands
  in code.
- Sphere/ellipsoid test: code adds the radius to the axes.
- Opacity regulariser weight only in code (0.001).
- SH sparsity normalisation (λ/(45V)) only in code.
- Paper reports 89% / 0.1% / 2.7% / 8.2% Gaussians in 0/1/2/3 bands for its main model.
- During training after culling the official zeroes the dropped coefficients but leaves Adam
  moments; the renderer ignores them via `_degrees`.
- Memory formula counts codebook entries as 4-byte floats even for the half-float model.
