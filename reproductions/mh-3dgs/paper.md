# MH-3DGS — Metropolis-Hastings Sampling for 3D Gaussian Reconstruction

Kim, Jung, Park — NeurIPS 2025 — arXiv 2506.12945.
Official code: https://github.com/SNU-VGILab/MH-3DGS (cloned at `source/code`, commit
71ef8f9). License: Inria Gaussian-Splatting license (non-commercial research) — ported for
research reproduction with citation. Built on the Inria 3DGS code + 3DGS-MCMC rasterizer
(the MCMC fork only adds the `compute_relocation` kernel; rendering = classic 3DGS).

## 1. Method summary
- Replaces 3DGS ADC (clone/split/prune/opacity reset) by an MH "birth" sampler run every 100
  iterations in (500, 15000): candidates are copies of existing Gaussians picked ∝ importance and
  displaced by Gaussian noise (coarse + fine batch), accepted with prob. `σ(I-0.5)·D(v')`.
- Importance per Gaussian = sigmoid of a weighted sum of z-normalised opacity, aggregated SSIM
  map value and aggregated L1 map value (maps averaged over a round-robin view subset whose size
  anneals from |C| to 1).
- Voxel crowding prior: `D(v') = 1/(1+λ_v c(v'))`, λ_v = 5 (code `alpha=5.0`), voxel size anneals
  0.02 → 0.005 (absolute world units).
- Dead Gaussians (opacity ≤ 0.005) are relocated onto alive ones (3DGS-MCMC relocation, Eq. 9 of
  MCMC) at every densification step.
- Loss: 3DGS L1 + 0.2 D-SSIM + 0.01·mean(opacity) + 0.01·mean(scale) (MCMC regularisers). No SGLD
  noise, no Gaussian cap — the final count (~0.62–0.81 M) emerges from the sampler.

Contributions:
- C1 Multi-view importance field + per-Gaussian importance (§4.2.1–4.2.2, Eq. importance / surr-importance).
- C2 Coarse–fine MH birth proposals with voxel-crowding acceptance (§4.2.3, §4.3, Alg. 1).
- C3 Relocation of low-opacity Gaussians (§4.2.4, from 3DGS-MCMC).
- C4 Annealed schedules (view subset size, proposal scales, voxel size, batch sizes) (§5.1).
- (C0, not novel) MCMC opacity/scale regularisers in the loss (Eq. likelihood).

## 2. Equations (paper, LaTeX verbatim) and what the code does
Loss (Eq. likelihood):
`\mathcal{L}(\Theta) = (1-\lambda) \mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}} + \lambda_{\text{opacity}} \overline{\alpha} + \lambda_{\text{scale}} \overline{\Sigma}`,
λ=0.2, λ_opacity=λ_scale=0.01. Code (train.py): `opacity_pen = sigmoid(opa).mean()`,
`scale_pen = exp(scales).mean()` (all 3 axes).

View subset: `k_t=\max(1,\lfloor(1-\eta_t)|\mathcal C|\rfloor),\ \eta_t=\frac{t-t_{\min}}{t_{\max}-t_{\min}}`.
Code: `progress=(it-500)/(15000-500)`, `k=max(int(n*(1-progress)),1)`, round robin
`start=(it*k) % n`, cameras `start..start+k-1 (mod n)` in the (shuffled, 3DGS `Scene`) train list.

Aggregated maps: `SSIM_agg(p)=1/k Σ[1-SSIM]`, `L1_agg(p)=1/k Σ|I-Î|`.
**Code**: aggregates the SSIM *similarity* map (`ssim(..., size_average=False)`, 11×11 Gaussian
window σ=1.5, zero "same" padding, per channel) — not 1−SSIM — and the per-channel |I−Î| map.
Renders use the training background (black).

Importance (Eq. importance): `s(p)=\sigma(\alpha O(p)+\beta SSIM_{agg}(p)+\gamma L1_{agg}(p))`,
α=0.8, β=0.5, γ=0.5, each input "robust normalised". **Code**: per-Gaussian, not per-pixel;
`robust_normalize(x)=(x-mean)/(std+1e-6)` (unbiased std) of (a) Gaussian opacity sigmoid,
(b) SSIM map value, (c) L1 map value; values (b),(c) are read at pixel
`(x=int(μ_x·(W−1)), y=int(μ_y·(H−1)))` clamped to the image, i.e. the **world** x/y coordinates
of the mean are used as normalised image coordinates (no camera projection; the `projection()`
result in train.py is unused); channel mean of the 3 map channels.
Paper Eq. surr-importance `I(i)=s(\lfloor\Pi(\mathbf x_i)\rfloor)` (projection) differs.

Proposal: `g'_i=(\mathbf x_i+\delta_i,\Sigma_i,\mathbf c_i,\alpha_i),\ \delta\sim\mathcal N(0,\sigma^2 I)`.
**Code**: `idx = unique(multinomial(I, B, replacement=True))`;
`std = clamp(scale_phase · exp(scales)[idx] · 1/(1+ln(1+denom[idx])), min=1e-6)` per axis
(world axes); `denom` = number of densification-step visibilities since the last population
append (3DGS `add_densification_stats`, called only on densification iterations → 0 or 1 in
practice). New Gaussian copies the parent's raw sh0/shN/opacity/scale/quat; parent unchanged;
new Adam moments zero.

Acceptance (§4.3): `\rho(i)=\sigma(I(i))\,D(v')`, `D(v')=\frac{1}{1+\lambda_v c_\Theta(v')}`.
**Code**: `ρ = sigmoid(I − 0.5) · 1/(1 + 5·c(v'))`, `c` = count of *pre-step* Gaussians in the
voxel `floor(x'/voxel)` (same counts for coarse and fine), accept if `U(0,1) < ρ`.

Relocation (§4.2.4): dead `α ≤ 0.005`; sample alive j ∝ α_j with replacement; dead ← params of j
with MCMC `compute_relocation(α_j, s_j, N=count_j+1)` (N clamped to [1,50]), opacity clamped to
[0.005, 1−eps]; j gets the same new opacity/scale; Adam moments of the sampled j reset to 0
(not of the dead slots). Identical to `gsplat.strategy.ops.relocate`.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | code arguments |
| densify_from / until / interval | 500 / 15000 / 100 (strict `>500`, `<15000`) | code arguments, train.py |
| coarse scale | 10.0 → 5.0 linear in progress | paper §5.1, code |
| fine scale | 2.0 → 1.0 | paper §5.1, code |
| voxel size | 0.02 → 0.005 | paper §5.1, code |
| coarse / fine batch | int((1−p)·4500) / int((1−p)·16000), ≥1 | paper §5.1 (no decay stated), code |
| λ_v (`alpha`) | 5.0 | code train.py (method default 6.0 overridden) |
| α/β/γ importance weights | 0.8 / 0.5 / 0.5 | paper, code |
| opacity threshold τ | 0.005 | paper, code |
| λ_opacity, λ_scale | 0.01, 0.01 | paper, code |
| λ_dssim | 0.2 | |
| lrs | means 1.6e-4→1.6e-6 ×extent, sh0 0.0025, shN 0.0025/20, opacity **0.025**, scale 0.005, quat 0.001 | code arguments |
| init | SfM points, 3-NN scales, opacity 0.1 | 3DGS |
| SH | +1 degree every 1000 | 3DGS |
| background | black | full_eval (no `--white_background`) |
| reloc_until_iter | 25000 (declared, **unused**; relocation only inside the densify window) | code |

## 4. Schedules
All MH/relocation work happens at the 144 iterations it ∈ {600, 700, …, 14900}; progress p=(it−500)/14500 drives
the linear anneals; p is never 1, so the last batch sizes are int(0.0069·B) = 31 / 110.

## 5. Builds on
Inria 3DGS training loop (losses, lrs, SH schedule, shuffled-stack view sampling) with the
3DGS-MCMC rasterizer and relocation. Removed: clone/split, opacity reset, prune, SGLD noise.

## 6. Protocol
Mip-NeRF 360 (9 scenes; outdoor `images_4`, indoor `images_2`), T&T truck/train, DB drjohnson/playroom;
`--eval` llffhold8; 30k iterations; RTX 3090; LPIPS from `lpipsPyTorch` (VGG); single run.
Main Table compares at the **same number of Gaussians** as MH-3DGS (3DGS-S and 3DGS-MCMC with
the MH-3DGS count) → Mip-NeRF 360 numbers are well below full 3DGS (e.g. garden 26.49 vs 27.31,
bicycle 24.53 vs 25.10) — this is the ~0.7 M budget, not a resolution/protocol difference.

## 7. Ambiguities / paper–code disagreements (code wins)
1. Pixel lookup uses world x/y as normalised image coordinates (code) vs. projection (paper).
2. SSIM similarity (code) vs 1−SSIM (paper).
3. Acceptance `σ(I−0.5)` (code) vs `σ(I)` (paper); λ_v=5.
4. Proposal std is relative to the parent's scale and the `denom` factor (code) vs. absolute σ (paper).
5. `unique()` after multinomial sampling ⇒ effective batches < B (code).
6. Batch sizes decay linearly (code), paper states fixed 4500/16000.
7. Relocation runs only during 600..14900 despite `reloc_until_iter=25000`.
8. Paper Table shows 3DGS-MCMC trained "with the same number of Gaussians as ours".
