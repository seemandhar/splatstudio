# Improving Densification in 3D Gaussian Splatting for High-Fidelity Rendering (ImprovedGS)

Deng, Diao, Li, Yu, Xu (Zhejiang University). arXiv 2508.12313 (AAAI-26 version in the LaTeX;
README says CVPR 2026 Findings). Official code: https://github.com/XiaoBin2001/Improved-GS
(cloned `source/code/`, current version "built from the latest 3DGS"; the paper version was built
on TamingGS, Aug 2025). Licence: ImprovedGS Non-Commercial Research License + Inria 3DGS licence
for the rasterizer submodule.

The plan entry mentions "Efficient Density Control" (EDC) as an abandoned earlier version by the
same authors; LAS + opacity reduction originate there. The reported numbers (Table 1, appendix
per-scene tables) are produced by the full ImprovedGS recipe below (EAS + LAS + RAP + MU + GC);
EDC is not a separate component of this paper and is counted once (inside LAS).

## 1. Method summary
- Densification candidates = Gaussians whose view-averaged **absolute** screen gradient ≥ τ
  (AbsGS-style); among candidates, *which* to split is sampled ∝ an **edge-aware score** (EAS).
- Only one densification op: **Long-Axis Split** (LAS): 2 children placed symmetrically on the
  longest axis at ±d·L0, shrunk analytically, opacity × 0.6; no clone, no 3DGS split.
- **Growth Control** (GC): Gaussian count capped by a √-curve budget reaching N_max shortly
  before the end of densification (TamingGS-style budget, per-scene N_max).
- **Recovery-Aware Pruning** (RAP): after the opacity resets at 3k/6k (reset to ≤ 0.05), prune
  the lowest-20 % opacity Gaussians 300 iterations later (3300, 6300).
- **Multi-view Update** (MU): after densification, optimizer steps only every N iterations
  (gradients of N single-view backward passes summed): N=5 then N=20.

Contributions:
- C1 Edge-Aware Score (Sec. 3.2, Eq. 5–7)
- C2 Long-Axis Split (Sec. 3.3, Eq. 8, App. A.4/A.7/A.8)
- C3 Recovery-Aware Pruning (Sec. 3.4)
- C4 Multi-view Update (Sec. 3.5, App. A.6)
- C5 Growth Control (Sec. 3.6, Eq. 9, App. A.9)
- (impl.) hyper-parameter changes (App. A.2): position lr 4e-5→2e-6, τ=0.0003, reset opacity
  0.05, no large-Gaussian pruning.

## 2. Equations (verbatim from `sec/3_methods.tex`)
3DGS average gradient (Eq. 3/4):
`\frac{\sum_{k=1}^{M^i}{\sqrt{( \frac{\partial L_k}{\partial \mu _{k,x}^{i}} ) ^2+( \frac{\partial L_k}{\partial \mu _{k,y}^{i}} ) ^2}}}{M^i}>\tau _\mathrm{pos}`

Edge-aware score (Eq. 5): `S_{i,j} = \sum_{p \in \mathcal{P}} \omega_{p,j}^{i} \cdot \alpha_{p,j}^{i}`
with ω = per-pixel Laplacian edge weight of view j, α = rendering (blend) weight of Gaussian i at p.
Eq. 6: `\bar{S}_i = \frac{1}{N_s} \sum_{j=1}^{N} S_{i,j}` over N_s randomly sampled views.

Absolute gradient (Eq. 7): `\frac{\partial L_k}{\partial \mu_{k,x}^{i}} = \sum_{j=1}^m |\frac{\partial L_j}{\partial \mu_{i,x}}|` (same for y).
Candidates: avg abs gradient > threshold; "probability of splitting a candidate is proportional
to its average edge-aware score".

LAS (Eq. 8): `R_s = R_0 \cdot \sqrt{1 - \frac{d^2}{L_0^2}}`, child long semi-axis `L_s = L_0 - d`,
d = 0.45 L0, child opacity = 0.6 × parent.

GC (Eq. 9): `N = N_{max} \cdot \sqrt{ \frac{I - I_{start}}{I_{end} - I_{start}} }`.

### Code (authoritative numerics; `scene/gaussian_model_densification.py`, `scene/methods/*.py`)
- Edge map (`densification_methods.compute_edge_map`): rgb255 = round(img·255);
  gray = round((299R + 587G + 114B)/1000); conv2d with [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], zero
  padding 1; clamp [0,255]/255; then min-max normalize to [0,1] (PIL FIND_EDGES equivalent).
- Score (`densification_stage._compute_improvedgs_scores`): for each sampled view render with
  `pixel_weights = edge map`; kernel accumulates `accum_w[i] += ω_p · T · α` (forward.cu:416,
  α after 0.99 clamp, skipped below 1/255, stop at T<1e-4); per view
  `score[vis] += minmax01(accum_w over all N)[vis] / n_views` (vis = radii > 0).
  Views: 10 from a cycling pool (pop from the end of the shuffled train list, refill when empty);
  **all** training views when `iteration % 3000 == 400 and iteration < 9000` (i.e. 3400, 6400).
- Stats: `accum_abs[vis] += ‖(Σ_p|∂L/∂μx|, Σ_p|∂L/∂μy|)‖₂`, `denom[vis] += 1` (Inria grad units,
  means2D channels 2:4) every iteration with 500 < it < 15000. Reset only when a split happens
  (densification_postfix), sliced on every prune.
- Densify (`densify_and_prune_improved`), every 100 its, 500 < it < 15000 (600..14900):
  g = accum_abs/denom (nan→0); qualifiers = g ≥ τ; `current_budget = min(budget(it), #qual + N)`;
  `split_budget = current_budget − N`; if > 0: LAS on `min(split_budget, #positive)` Gaussians
  drawn by `torch.multinomial(score·qual, k, replacement=False)`; if it < 14900: prune
  opacity < 0.005. (`it > densify_until−100` branch — score := g, τ/1.5 — never fires because
  14900 is not > 14900; ported for fidelity.) If EAS has no scores, score := g.
- LAS (`long_axis_split`): offsets = ±(s_max · 3 · d) along the rotated longest axis;
  `new_scales = s · sqrt(1−d²)` on short axes, `s_max · (1−d)` on the long axis;
  `opacity = logit(sigmoid(o) · 0.6)`; rotation, SH copied; parent removed; children's Adam
  moments zero.
- Budget (`_compute_current_budget`): start = 500, end = 15000 − 500 = 14500,
  progress = clamp((it−500)/14000, 0, 1); `budget = max(int(progress**0.5 · N_max), 1)`,
  N_max at progress ≥ 1.
- RAP (`pruning_methods.run_3dgs_pruning_method`, after densification in the same iteration):
  it == 300: prune opacity < 0.02. Opacity reset at it % 3000 == 0, 500 < it < 15000
  (3000, 6000, 9000, 12000): opacity = min(opacity, 0.05), opacity Adam moments zeroed. After the
  first 2 resets schedule a prune at reset+300 (3300, 6300): remove opacity < quantile(opacity, 0.2).
- MU (`training_loop.should_run_parameter_update`): optimizer step + zero_grad only if
  `it < 20000` or (`20000 ≤ it < 24000` and it % 5 == 0) or (`it ≥ 24000` and it % 20 == 0);
  skipped iterations accumulate (sum) gradients. LR schedule still advances per iteration.
  **Paper text says 15000–22500 with N=5, then N=20; code defaults 20000/24000** (code wins;
  both recorded).
- Order per iteration: lr update → render/backward → optimizer step (MU-gated) → densify →
  RAP/prune/reset.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | arguments/__init__.py:154 |
| position_lr_init / final | 4e-5 / 2e-6 (× scene extent), delay_mult 0.01 (no delay steps), max_steps 30000 | App. A.2; arguments:155-158 |
| feature_lr / f_rest | 0.0025 / 0.0025/20 | arguments:160; gaussian_model_optimizer.py:76-77 |
| opacity_lr | 0.025 | arguments:161 |
| scaling_lr / rotation_lr | 0.005 / 0.001 | arguments:163-164 |
| initial_opacity | 0.1 | arguments:162 |
| lambda_dssim | 0.2 (fused_ssim) | arguments:170 |
| densify_from / until / interval | 500 / 15000 / 100 | arguments:171-174 |
| densify_grad_threshold τ (abs) | 0.0003 | App. A.2; arguments:175 |
| min_opacity (prune) | 0.005 | arguments:176 |
| opacity_reset_interval | 3000 | arguments:172 |
| improvedgs_reset_min_opacity | 0.05 | App. A.2; arguments:219 |
| edge_sample_cams N_s | 10 | arguments:218 |
| split_distance d | 0.45 | Sec. 3.3; arguments:220 |
| opacity_reduction | 0.6 | Sec. 3.3; arguments:221 |
| budget N_max | bicycle/garden/stump 3M; flowers/treehill/drjohnson/truck 1.5M; others 1M | Sec. 4.2; configs/improvedgs.json |
| budget_warmup_until_offset | 500 | arguments:224 |
| rap_prune_ratio / offset / rounds | 0.2 / 300 / 2 | Sec. 3.4; arguments:225-227 |
| early prune | it 300, opacity < 0.02 | pruning_methods.py:191 |
| mu_start / interval / second_start / interval | 20000 / 5 / 24000 / 20 (paper: 15000 / 5 / 22500 / 20) | arguments:228-231; Sec. 3.5 |
| optimizer | "ours_adam" (CUDA Adam, all-visible → dense Adam, eps 1e-15) | arguments:180; diff_gaussian_rasterization/__init__.py:521 |
| SH degree | 3, +1 every 1000 its | training_loop.py |
| background | black | white_background false |
| antialiasing | off | pipeline |

## 4. Schedules
300 early prune; 501–14999 stats; 600..14900 densify every 100; budget √-curve 500→14500;
3000/6000/9000/12000 opacity reset to 0.05; 3300/6300 RAP 20 % prune; 3400/6400 EAS over all views;
20000 MU N=5; 24000 MU N=20; position lr exponential 4e-5→2e-6 over 30000.

## 5. Builds on
3DGS (current Inria code) with the TamingGS-derived rasterizer (per-bucket backward, CUDA Adam,
SpeedySplat tight boxes). Kept from 3DGS: L1 + 0.2 D-SSIM, SH schedule, SfM init with 3-NN
scales, opacity reset cadence, lrs except position/opacity. Removed: clone, 3DGS split,
large-Gaussian (world/screen size) pruning.

## 6. Protocol
Mip-NeRF 360 (9 scenes), Tanks&Temples (train, truck), Deep Blending (drjohnson, playroom);
every 8th image test (llffhold 8); PSNR/SSIM/LPIPS(VGG) "using the methods provided by 3DGS";
"best of three runs" (paper) / `select_best_repeat_by_psnr: true, repeat: 3` (config).
Resolution: code loads `images` with `resolution=-1` (auto-downscale to width 1600 when wider);
the authors' data package (Baidu link) is not inspectable, the paper says "as with 3DGS" and the
paper version was built on TamingGS whose full_eval uses images_4 / images_2. Their 3DGS row
(garden 27.36, bicycle 25.21, room 31.45) is consistent with the standard 3DGS protocol.
→ we use `mipnerf360` (images_4 outdoor / images_2 indoor).

## 7. Ambiguities
- Paper vs code MU start (15000/22500 vs 20000/24000): code used.
- Paper Eq. 6 says "randomly sample N_s views"; code cycles a fixed shuffled pool, full set at
  3400/6400. Code used.
- Per-view min-max normalisation of the score is code-only (not in Eq. 5/6).
- "Best of 3 runs" inflates reported numbers slightly vs a single seed.
- Released results CSV (new code) differ slightly from the paper tables (e.g. garden 28.061 vs
  28.098, room 32.657 vs 32.652); paper tables are the target.
- Data resolution (see §6).
