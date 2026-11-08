# DecompGS — Decomposing Densification in Gaussian Splatting for Faster 3D Scene Reconstruction

arXiv 2507.20239 (Huang, Liu, Wong; AAAI 2026 template). Sources: `source/latex/aaai2026.tex`
(LaTeX e-print), `source/paper.pdf`/`paper.txt`.

**Official code: none found** (no link in the paper/arXiv page; GitHub API search for
"decomposing densification" and the arXiv id returned nothing; `papers_100.json` `code_url` is
empty). Everything below comes from the paper text; implementation defaults that the paper
omits are taken from the 3DGS / 3DGS-accel code base the paper says it builds on
("open-source accelerated version of 3DGS" = 3DGS + Taming-3DGS per-splat backprop + sparse
Adam, footnote 1).

## 1. Method summary
- Analysis (Sec. 3.1): split moves Gaussians far from their SfM parent (global diffusion),
  clone stays local (<2 % of the scene extent) and produces ~80 % of new Gaussians.
- **C1 Global-to-local densification (G2L, Sec. 3.2 "Global-to-local framework")**: phase 1
  (global spread) applies *only the split operation*; phase 2 (local refinement) applies split
  and clone as in 3DGS. Phase boundary = T_2 of the C2F schedule (Ablation §, "employ iteration
  T_2 as the critical boundary"). Fig. 2 (pipeline2.png) shows small (clone-candidate) Gaussians
  left untouched during the global phase and large ones split.
- **C2 Energy-guided coarse-to-fine multi-resolution training (C2F, Sec. 3.3)**: train with
  images downsampled by r ∈ {K, …, 2, 1}; the iteration at which each scale ends is set by the
  ratio of the FFT L1 energy of the downsampled images to that of the full-resolution images.
  Positional LR held constant while training at reduced resolution, reduced after restoring
  full resolution (Implementation details).
- **C3 Adaptive opacity pruning (Sec. 3.4)**: the fixed opacity prune threshold is replaced by
  τ = min(p-quantile of opacities, τ_u).
- Densification extended to T_densify = 25K (from 15K).

## 2. Equations (verbatim)
Energy spectrum: $\mathcal{E}(\mathbf{I}) = \sqrt{\Re(\mathcal{F}(\mathbf{I}))^2 + \Im(\mathcal{F}(\mathbf{I}))^2}$ (2D FFT magnitude).

(1) $\mathbf{I}_r = \mathcal{D}_{r}(\mathbf{I}) = \textit{Bilinear}(\mathbf{I}, \text{scale}=1/r)$ — bilinear with anti-aliasing.

(2) $\mathcal{E}_r = \|\mathcal{E}(\mathbf{I}_r)\|_1 \cdot r^2$

(3) $\mathbf{T}_r = \text{Round}( (\mathcal{E}_r / \mathcal{E}_1) \cdot \mathbf{T}_{\text{densify}}), r \in \mathcal{A}$,
$\mathcal{A} = \{1,2,\dots,K\}$. "The training proceeds from coarsest (r=K) to finest (r=1). For scale
factor r, the training starts at $\mathbf{T}_{r+1}$ and ends at $\mathbf{T}_{r}$." (T_{K+1} := 0; T_1 = T_densify.)

Pruning (unnumbered): $\tau_k = \boldsymbol{\alpha}_{\text{sorted}}[k],\ k = \lfloor N p \rfloor$;
$\tau = \min(\tau_k, \tau_u)$; Gaussians with opacity below τ are pruned (replacing 3DGS's
fixed threshold).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | Impl. details |
| T_densify (densify_until) | 25000 | Impl. details; Table 5 ablation 15K/20K/25K |
| K (max scale factor) | 8, A = {1..8} | Impl. details, Eq. 3 |
| p (prune ratio) | 0.03 | Impl. details, Table 6 |
| τ_u (prune upper limit) | 0.05 | Impl. details, Table 6 |
| densify grad threshold | 0.0002 | 3DGS default (not stated) |
| percent_dense | 0.01 | 3DGS default (not stated) |
| densify from / interval | 500 / 100 | 3DGS default (not stated) |
| opacity reset | every 3000 to 0.01, while densifying | 3DGS default; Sec. 3.4 says reset is part of 3DGS, not removed |
| world-size prune | max scale > 0.1·extent after step 3000 | 3DGS default |
| position LR | 1.6e-4·extent → 1.6e-6·extent; held constant during low-res | 3DGS + Impl. details |
| SSIM weight λ | 0.2 | 3DGS |

## 4. Schedules
- Resolution r(t): r for T_{r+1} ≤ t < T_r; r = 1 from T_2 on (T_1 = 25K, after which 3DGS
  full-res continues to 30K).
- G2L: t < T_2 → split only; t ≥ T_2 → split + clone.
- Position LR: constant until full resolution (t = T_2), then decayed (exact form unstated).
- Densification 500–25000 every 100; reset every 3000 while densifying; SH degree +1 / 1000.

## 5. Builds on
3DGS-accel (3DGS + Taming per-splat backprop + sparse Adam). Loss L1 + 0.2 D-SSIM unchanged;
3DGS clone/split rules (clone if max scale ≤ 0.01·extent, split into 2 with scale/1.6 otherwise)
unchanged except as above.

## 6. Protocol
Mip-NeRF 360 (9 scenes), Deep Blending (2), Tanks & Temples (2), 3DGS preprocessing,
llffhold 8, 30K iterations, SfM init. RTX 3090. Only dataset averages are reported (Table 2);
no per-scene table, no supplementary in the source.

## 7. Ambiguities (choices made)
1. **Global phase, small Gaussians**: "only apply the split operation". Choice: clone candidates
   are *skipped* (Fig. 2; the paper's point is avoiding redundant early clones — splitting them
   instead would add the same number of Gaussians). Alternative `split_all` kept as a config.
2. **Energy aggregation over images**: unspecified; choice: E_r summed over all training views
   (per scene, "we calculate the number of iterations ... of each scene"), RGB channels summed.
3. **Downsampled size**: floor(H/r)×floor(W/r) antialiased bilinear (as DashGaussian, which the
   paper cites as inspiration); training render uses the same size with scaled intrinsics.
4. **Position-LR reduction**: form unstated. Choice: hold lr_init until T_2, then log-linear
   decay to lr_final at 30K (DashGaussian's helper, which this paper says it follows); the
   alternative "jump to the 3DGS schedule value" is a config.
5. **Where τ is computed**: over all Gaussians at each densify/prune step (after growth, as 3DGS
   computes its prune mask), strict `<`.
6. Round: Python `round`. Monotonicity of T_r in r is guaranteed in practice (energy loss under
   low-pass); the scale lookup is defined robustly as r(t)=max{r: T_r > t} ∪ {1}.
7. Base code 3DGS-accel's sparse Adam / per-splat backprop are speed features; dense Adam kept
   (fidelity gap, small quality effect: 3DGS-accel vs 3DGS ≈ −0.04…+0.10 dB in the paper's table).
