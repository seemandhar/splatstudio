# Efficient Confidence-Aware Densification for 3D Gaussian Splatting (arXiv 2610.13541)

Sources: arXiv LaTeX (`source/latex/`, Eurographics 2027 template), PDF `source/paper.pdf`.
**Official code: none found** (paper, arXiv listing, GitHub/web search on 2026-11-06). The
learned-confidence component comes from CoMe (Radl et al., arXiv 2603.24725), whose code
(`github.com/r4dl/CoMe`, cloned to `source/come/code/`, read only) gives the confidence loss
numerics and LR.

## 1. Summary and contributions
- Keeps the fixed-budget 3DGS-MCMC framework (densify every 100 it from 500 to 25k, +5 % per
  step up to the cap, MCMC opacity/scale-preserving copy rule `compute_relocation`, opacity/scale
  L1 regulariser λ_reg = 0.01).
- Changes *where* primitives are placed: sources are drawn from an error distribution instead of
  opacity, low-contribution primitives are relocated, new primitives get sampled positions and a
  colour from a training image. No position noise.
- C1 (§3 Prelim, CoMe): learned per-primitive confidence γ̄_i = exp(γ_i), alpha-blended to a
  confidence map Ĉ, trained with the confidence-weighted photometric loss (Eq. confidence).
- C2 (§3.1): source sampling — 80 % from P_i (Eq. pdf: rank of absolute screen-space gradient ×
  confidence-rank factor, η ramp over 5000 it), 20 % from the per-primitive SSIM term
  (Σ_p b_i(p)(1−SSIM(p))). One draw serves growth and relocation.
- C3 (§3.2): relocation every 400 it (every 4th densification step) of the primitives with the
  lowest maximum blending weight w_i = max_{k∈V,p} b_i^k(p); share schedule (w=0 only and ≤10 %
  before 5k; lowest 10 % for 5k–10k; 10 %·exp(−½((t−10k)/5k)²) for 10k–25k); primitives placed by
  a relocation are exempt from the next one.
- C4 (§3.3): new (added or relocated) primitives: MCMC opacity/scale adjustment; rotation, SH,
  confidence copied; mean drawn once from N(μ_i, Σ'_i) (Σ' from R_i and the adjusted scale);
  3DGS-MCMC continuous position noise disabled.
- C5 (§3.3): colour init — first training view after placement where the new primitive is
  visible with projected radius < 16 px: bilinear GT lookup at the projected mean, set SH DC so
  the SH colour in that view equals c_GT (higher SH kept), reset its optimiser moments; give up
  after 400 it.

## 2. Equations (verbatim)
Photometric loss (Eq. photoloss), λ_SSIM = 0.2:
`\mathcal{L}_\text{rgb} = (1 - \lambda_\text{SSIM}) \mathcal{L}_1 (\bm{C}, \bm{C}_\text{GT}) + \lambda_\text{SSIM} (1 - \mathrm{SSIM}(\bm{C}, \bm{C}_\text{GT}))`

Confidence loss (Eq. confidence), per pixel:
`\mathcal{L}_\mathrm{conf}(\bm{p}) = \mathcal{L}_\mathrm{rgb}(\bm{p}) \cdot \hat{C}(\bm{p}) - \beta_\text{conf} \log \hat{C}(\bm{p})`
with Ĉ(p) = Σ_i b_i(p) γ̄_i (CoMe Eq. conf_alphablending), γ̄_i = exp(γ_i), γ_i init 0.
CoMe code (`train.py:203-217`, `appearance_network.py:104`): `pp = 0.8*|I-I*| + 0.2*(1-ssim_map)`
per pixel and channel ([3,H,W], same-padded 11×11 Gaussian SSIM map), `C = clamp(conf_render,
1e-3, 5)`, `loss = mean(pp*C - β log C)` — this *replaces* the plain photometric loss.

MCMC regulariser (Eq. reg): `\mathcal{L}_\text{reg} = \lambda_\text{reg} \sum_i ( |o_i| + \|\bm{s}_i\|_1)`
(3DGS-MCMC code uses means over primitives; we follow the code).

Sampling pdf (Eq. pdf), 80 % of sources:
`\mathcal{P}_i = r_i(\nabla_\text{abs}) \left( (1 - \eta) + \eta \left(1.2 - r_i(\bar{\gamma})\right) \right)`, `η = min(t/5000, 1)`;
r_i(x) ∈ [0,1] normalised rank (smallest 0, largest 1). ∇_abs = AbsGS/GOF absolute screen-space
gradient, averaged over observations since the previous densification step.

Max blending weight: `w_i := \max_{k \in \mathcal{V},\, \bm{p}} b_{i}^{k}(\bm{p})`, V = training
views rendered since the previous relocation; b = α_i T_{i−1}.

Relocation share: t<5k: only w_i = 0, ≤10 % of N; 5k–10k: lowest 10 %; 10k–25k:
10 %·exp(−½((t−10k)/5k)²) (61 % at 15k, 14 % at 20k, ~1 % at 25k).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | §4 Impl. |
| budget (cap) | 100k/500k/1M/2M; **we use 1M** | §4.1 |
| densify every / start / stop | 100 / 500 / 25000 | §3 (3DGS-MCMC) |
| growth per step | 5 % until cap | §3 |
| relocation every | 400 it | §3.2 |
| relocation share | 10 % | §3.2, Supp. explore table |
| relocation fall-off | exp(−½((t−10k)/5k)²) from 10k | §3.2 |
| w=0-only phase | t < 5000 | §3.2 |
| η ramp | min(t/5000,1) | §3.1 |
| pdf split | 80 % P_i / 20 % SSIM term | §3.1 |
| confidence factor offset | 1.2 | Eq. pdf |
| β_conf | 0.075 | §4 Impl. (= CoMe) |
| confidence LR | 2.5e-4 | CoMe §4 / `arguments/__init__.py:196` |
| Ĉ clamp | [1e-3, 5] | §4 Impl.; CoMe train.py:208 |
| λ_reg (opacity & scale) | 0.01, active until 25k | §4 Impl. |
| MCMC min opacity clamp | 0.005 | 3DGS-MCMC |
| position noise | disabled | §3.3 |
| colour-init radius / timeout | < 16 px / 400 it | §3.3 |
| SfM init > budget | farthest point subsampling | §4 Impl. (not triggered at 1M) |

## 4. Schedules
η ramp 0→5000; relocation phases 0–5k / 5k–10k / 10k–25k; densification 500–25k; λ_reg until
25k; confidence loss whole training; SH degree +1 per 1000 it (3DGS).

## 5. Builds on
3DGS-MCMC in the Inria 3DGS codebase with the Faster-GS rasterizer and fused Adam. Kept from
MCMC: budget/growth schedule, compute_relocation, regulariser, learning rates (3DGS defaults),
MCMC initialisation (opacity 0.5 in the MCMC code). Removed: opacity-proportional sampling,
opacity-threshold relocation, position noise.

## 6. Protocol
Mip-NeRF 360 (outdoor images_4, indoor images_2), T&T, DB (provided resolution), DL3DV(11);
every 8th image test; 30k it; SfM init; LPIPS-VGG. Per-scene tables Supp. S.1–S.4
(`tables/eval_{100K,500K,1M,2M}.tex`), also 3DGS-MCMC re-run by the authors at each budget.

## 7. Ambiguities (our choice)
- Rank ties (e.g. ∇_abs = 0 for unobserved primitives): min-rank (ties share the lowest rank), so
  never-observed primitives get P_i = 0.
- SSIM term: accumulated Σ_p b_i(p)(1−SSIM(p)) averaged over observations since the previous
  densification step (like ∇_abs); used raw (not ranked); per-pixel SSIM averaged over channels.
- "Same draw serves growth and relocation": one draw of n_reloc + n_add sources (excluding
  primitives being relocated), compute_relocation ratio = total copies per source + 1.
- Optimiser moments: zeroed for sources and all destinations (gsplat's MCMC relocate zeroes only
  sources).
- Growth-added primitives are not exempt from the next relocation (paper exempts only relocated
  ones); w accumulators of destinations reset to 0.
- Absolute gradient normalisation (gsplat absgrad in pixel units) irrelevant because ranks are used.
- Initial opacity 0.5 / 3-NN scales as in 3DGS-MCMC code (paper silent).
- Confidence map numerics (CoMe rasteriser `hierarchical_render.cuh:1231, 1513-1515, 1643`, not in
  either paper's text): Ĉ = Σ b γ̄ / max(1−T, 1e-3) (opacity-normalised, ≡ 1 at init, matching
  Fig. confidence "maps nearly white"), and dL/dĈ reaches only γ — not opacity, means, conics, or
  ∇_abs. We render Ĉ in a second gsplat pass on detached geometry (review-1 CRITICAL-1; the first
  version blended γ̄ unnormalised through the geometry and lost ~1.5 dB at 2k).
- Growth-added primitives start with w_i = 0 and their 𝒱 window is shorter than other
  primitives'; before 5k they are relocated at the next 400-it boundary if no view has hit them
  (literal reading: the paper exempts only relocated primitives).
- First densification at 600 and first relocation at 800 (`step > 500`, Inria/MCMC convention).
- Colour init uses means2d of the render before this step's optimizer update (negligible drift)
  and the current SH degree; higher bands are kept.
- Confidence loss active from step 1 ("throughout training"; CoMe's default starts at 500).
- The MCMC control keeps λ_reg for the whole 30k and its position noise (3DGS-MCMC code); our
  method stops λ_reg at 25k (paper §4).
