# 3DLS — Beyond Gaussians: Fast and High-Fidelity 3D Splatting with Linear Kernels

arXiv 2411.12440 (v1, CVPR-format LaTeX in `source/latex/main.tex` + `supplementary.tex`).
Project page https://hche8927.github.io/3DLS/. **No official code**: github.com/hche8927/3DLS
(re-checked 2026-11-07) holds only the project page and an image-cropping `toolbox.ipynb`;
the author's other repos are unrelated. Everything below comes from the paper text/figures.

## 1. Method summary
- Replace the 3DGS Gaussian falloff exp(-D²/2) by a bounded linear (tent) falloff
  max(0, 1 − D/λ) of the 2D Mahalanobis distance D; blending unchanged.
- λ = 2.5 ("Distribution Alignment") widens the tent so its footprint matches the Gaussian.
- "Adaptive Gradient Scaling": the backward pass multiplies the kernel's (constant-magnitude)
  derivative by ω = exp(−D²), i.e. a custom (non-exact) gradient.
- Built on gsplat v1.3 with custom CUDA forward/backward rasterization kernels.
- Loss 0.6·L1 + 0.2·L2 + 0.2·(1−SSIM); 3DGS densification with three changed thresholds.

Contributions
- C1 Linear kernel (§3.2 "Transition to Linear Kernels", Eq. 9–11).
- C2 Distribution Alignment, λ (§3.3.1, Eq. 12, Fig. 3a).
- C3 Adaptive Gradient Scaling (§3.3.2, Eq. 13–14, Fig. 3b).
- (training recipe, §4.1 + supp. A) loss 6:2:2 and densification thresholds (Table 6/supp.).

## 2. Equations (verbatim)
Gaussian in Mahalanobis form (Eq. 7–8):
`G(x) = \exp(-\frac12 D_M^2),  D_M = \sqrt{(x-\mu)^\top \Sigma^{-1}(x-\mu)}`
Linear kernel (Eq. 9): `L(x) = \max(0, 1 - D_M)`
2D projection (Eq. 10–11): `L'(x') = \max(0, 1 - D'_M)`,
`D'_M = \sqrt{(x'-\mu')^\top \Sigma_{2D}^{-1}(x'-\mu')}`
DA (Eq. 12): `L(x) = \max(0, 1 - D_M/\lambda)`, λ = 2.5.
AGS (Eq. 13–14): `\omega(D'_M) = \exp(-{D'_M}^2)`,
`\Delta\theta \propto -\nabla_\theta L'(x') \cdot \omega(D'_M)`.
Blending (Eq. 6, unchanged): `C = \sum_i c_i \alpha_i \prod_{j<i}(1-\alpha_j)`, `α_i = o_i L'_i(x')`.
Loss (supp. Eq. 1–4): `L = α L1 + β L2 + γ (1 − SSIM)`, α=0.6, β=0.2, γ=0.2;
L1, L2 are per-pixel means.

Variables: Σ_2D is the EWA-projected 2D covariance (gsplat adds eps2d=0.3 blur; gsplat
conic = Σ_2D⁻¹). In gsplat notation `sigma = ½ dᵀ conic d` so `D'_M = sqrt(2·sigma)`.

Derived gradient used here (kernel k = 1 − D/λ, α = o·k):
∂α/∂o = k; ∂α/∂D = −o/λ; ∂D/∂sigma = 1/D. AGS multiplies the D-path by ω = exp(−D²)
(D unscaled: Fig. 3b shows native magnitude 1/λ = 0.4 on |x|<2.5 and the scaled curve
0.4·exp(−x²), e.g. ≈0.15 at x=1). Opacity/colour gradients do not pass through ∇L' and are not
scaled (paper scales "∇_θ L'(x')").

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| λ (alignment) | 2.5 | §3.3.1, supp. B |
| ω | exp(−D²), D unscaled | Eq. 13, Fig. 3b |
| loss weights L1/L2/SSIM | 0.6/0.2/0.2 | §4.1, supp. A Eq. 1 |
| grow grad2d τ_∇g | 0.0002 | supp. Table 6 |
| grow scale2d τ_s2g | 0.05 | supp. Table 6 (gsplat default; only active with refine_scale2d_stop_iter>0, gsplat default 0) |
| grow scale3d τ_s3g | **0.006** (3DGS 0.01) | §4.1, Table 6 |
| prune scale2d τ_s2p | 0.15 | Table 6 (gsplat default, inactive by default) |
| prune scale3d τ_s3p | **0.4** (3DGS 0.1) | §4.1, Table 6 |
| prune opacity τ_op | **0.025** (3DGS 0.005) | §4.1, Table 6 |
| everything else | 3DGS/gsplat v1.3 defaults (30k steps, densify 500–15000 every 100, reset every 3000, SH +1 per 1000, LRs) | §4.1 "All other hyperparameters ... consistent with the baseline 3DGS" |

## 4. Schedules
None new. gsplat DefaultStrategy schedule (refine 500→15000 every 100, opacity reset every
3000 to 2·prune_opa = 0.05 in gsplat; prune_scale3d only after the first reset).

## 5. Builds on
gsplat v1.3 `simple_trainer.py` default (3DGS) strategy; projection, SH, sorting, blending
unchanged. Only the rasterizer falloff + its backward, the loss and three thresholds change.
3DLS+AA (Mip-Splatting 2D AA) is a separate variant, not our target.

## 6. Protocol
Mip-NeRF 360 (9 scenes), Tanks&Temples (truck, train), Deep Blending (drjohnson, playroom).
30k (and 7k) iterations; SSIM/PSNR/LPIPS-VGG. Resolution not stated; gsplat v1.3 benchmark
scripts use data_factor 4 outdoor / 2 indoor (images_4 / images_2), and the paper's own
3DGS (gsplat) row (garden 27.41, kitchen 31.40) is consistent with that, so we assume our
standard images_4/images_2 protocol. Test split: gsplat test_every=8 (= llffhold8).
Background: gsplat trainer default (black, no random bg). Hardware RTX 3090, PyTorch 2.1, CUDA 11.8.
Seeds not stated.

## 7. Ambiguities
- No code → kernel bounding radius not specified. Exact choice: support is D<λ, and α ≥ 1/255
  needs D ≤ λ(1 − 1/(255·o)); we use that as the extend factor (image identical to any larger
  radius; only speed differs).
- Whether gsplat's `opac·vis ≤ 0.999` backward gate and the 0.999 alpha clamp are kept:
  presumably (minimal modification of gsplat) — kept.
- AGS scope: applied to the kernel-derivative path (means2d, conics). Scaling also changes
  the means2d gradient that drives densification (consistent with ablation: AGS reduces N
  3.27M→1.85M). Opacity/colour gradients unscaled.
- D in ω: unscaled D'_M (Eq. 13 + Fig. 3b), not D/λ.
- Singularity of ∂D/∂sigma at D=0: gradient magnitude w.r.t. means is finite (unit
  direction); we clamp D ≥ 1e-6 inside the kernel.
- Truck "27.57" in the scout notes is the Mip-NeRF 360 *mean*; per-scene truck = 25.49.
