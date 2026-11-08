# LiteGS — implementation spec

arXiv 2503.01199 (v-latest LaTeX = ICLR 2026 submission, `source/latex/iclr2026_conference.tex`).
Official code: https://github.com/MooreThreads/LiteGS (cloned `source/code/`, Gaussian-Splatting
licence = Inria non-commercial research licence; port for research with citation).
**Code defaults override the paper where they disagree; both are recorded.**

## 1. Method summary
- A system + algorithm co-design for fast 3DGS training. System part: warp-based fp16 rasteriser
  with scanline evaluation (§3.1), Morton-sorted Cluster-Cull-Compact (§3.2, cluster = 128
  primitives, AABB frustum culling, compaction, cluster-granular sparse Adam). These change speed,
  not the image model (Table 2: w/o culling 28.28 vs 28.25 dB).
- Algorithm part (§3.3, quality-affecting): densification driven by the **variance of the
  per-pixel opacity gradient** of each primitive, inside a TamingGS-style budgeted growth
  (multinomial sampling of a fixed number of candidates per densification).
- **Opacity decay** (halve opacities) every 10 epochs instead of the 3DGS hard reset, with
  densification every 5 epochs from epoch 3 until 80 % of training.
- Learning rates = TamingGS; epoch-based schedules (SH degree +1 every 5 epochs).

Contributions
- C1 Opacity-gradient-variance densification metric (§3.3 "Opacity Gradient Variance Metric").
- C2 Opacity decay + epoch-based densification schedule (§3.3 last two paragraphs, §4 Impl. details).
- C3 (system, out of scope) warp-based rasteriser (§3.1) — speed only.
- C4 (system, out of scope) Cluster-Cull-Compact (§3.2) — speed only (Table 2).
- Training recipe choices found in code that change results: sparse (visible-only) Adam
  without bias correction, Adam-state clear at each decay, budgeted growth + weight pruning,
  "split" that keeps the parent, SH schedule per 5 epochs, rgb clamp before loss, lrs.

## 2. Equations
Rendering (Eq. 1–2, unchanged 3DGS):
$I_{\pi_j}(x,y)=\sum_i T_{s(i)}\,\alpha_{s(i)}\,c_{s(i)}$, $\alpha_i=o_i G(\cdot)$,
$G(\Delta x,\Delta y,\Sigma)=e^{-0.5(\Delta x,\Delta y)\Sigma(\Delta x,\Delta y)^T}$.

Densification metric (§3.3, verbatim):
$S_i=\sum_{\pi\in\Pi}\sum_{(x,y)}\big(\frac{\partial \mathrm{Loss}(I_\pi(x,y),gt_\pi(x,y))}{\partial o^i}\big)^2$,
$M_i=\sum_{\pi}\sum_{(x,y)}\frac{\partial \mathrm{Loss}}{\partial o^i}$, fragment count $C_i$,
metric $C_i\cdot\mathrm{Var}=S_i-\frac{M_i^2}{C_i}$.

Code (`litegs/utils/statistic_helper.py:get_var`, `training/densify.py:DensityControllerTamingGS.get_score`):
`mean=M/(C+1)`, `sq_mean=S/(C+1)`, `var=clamp_min(sq_mean-mean^2,0)`,
**score = var · C · σ(o)²** (opacity² factor is code-only), nan→0, clamp ≥ 0.
`o` is the activated opacity; gradients are w.r.t. the activated opacity, per pixel
(`raster.cu` backward: `grad_a += d_alpha * G`). Statistics accumulate only during
densification epochs (one epoch per 5), over every view of that epoch.

Opacity decay (`densify.py:reset_opacity`, mode 'decay'):
$o\leftarrow \sigma^{-1}(\max(0.5\,\sigma(o),\,1/128))$ followed by `optimizer.state.clear()`
(**all** Adam moments of all parameters dropped).

Budget (`split_and_clone`): `cur_target=(target-init)/(until-from)*(epoch-from)+init`,
`budget=min(max(int(cur_target-N),1)+prune_num, N)`, `idx=multinomial(score,budget,replace=False)`.
Clone if `max(scale) <= percent_dense*extent` (copy), else split: new primitive at
`xyz + R·n, n~N(0,diag(s))`, scale `s/1.6`, other attributes copied; **the parent is kept unchanged**
(only the child is appended; 3DGS removes the parent). New primitives get zero Adam moments.
Prune (`prune_mode='weight'`): primitives (existing before the densify) whose summed blend weight
Σ T·α over the statistics epoch is 0. Order: split/clone → prune → (decay).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 (total_epoch=int(30000/n_train), runs total_epoch·n_train steps) | code arguments.py, trainer.py |
| T&T iterations (quality) | 40000 | full_eval.py custom_config, Table 1 `*` |
| position lr | 1.6e-4·extent → 1.6e-6·extent, log-linear over 30000 steps | optimizer.py |
| feature (sh0) lr | 0.0025 | arguments.py |
| sh_rest lr | 0.00025 (feature_lr/10; 3DGS uses /20) | optimizer.py |
| opacity lr | 0.025 (runtime default 0.05) | arguments.py |
| scaling lr | 0.005 | arguments.py |
| rotation lr | 0.001 | arguments.py |
| Adam | betas (0.9,0.999), eps 1e-15, **no bias correction**, update only visible primitives | optimizer.py, compact.cu adamUpdate |
| lambda_dssim | 0.2 (fused_l1_ssim_loss; submodule not vendored) | arguments.py |
| densify_from | epoch 3 | arguments.py |
| densification_interval | 5 epochs | arguments.py, paper §3.3 |
| opacity_reset_interval | 10 epochs (decay) | arguments.py, paper §3.3 |
| densify_until | int(total_epoch·0.8/10)·10+1 | trainer.py |
| decay rate / floor | 0.5 / 1/128 | densify.py |
| percent_dense | 0.01 · extent (extent = 1.1·max cam distance) | arguments.py, data.py get_norm |
| target_primitives (quality) | bicycle 5987095, flowers 3618411, garden 5728191, stump 4867429, treehill 3770257, room 1548960, counter 1190919, kitchen 1803735, bonsai 1252367, truck 2584171, train 1085480, playroom 2326100, drjohnson 3273600 | full_eval.py (last entry of each list) |
| target_primitives default | 1,000,000 | arguments.py |
| init | SfM points, opacity 0.1, scale=log sqrt(mean 3-NN dist²), identity rot, SH0 from rgb | scene/point.py |
| SH | degree = min(epoch//5, 3) | trainer.py |
| background | black (white_background False) | arguments.py |
| image clamp | rendered image clamped to [0,1] before the loss | render/__init__.py |
| reg_weight (scale reg) | 0 | arguments.py |
| cluster_size / sparse_grad | 128 / True | arguments.py |

## 4. Schedules
- Epoch = one shuffled pass over training views (DataLoader shuffle=True, batch 1).
- SH degree 0 for epochs 0–4, 1 for 5–9, 2 for 10–14, 3 from 15.
- Statistics (S, M, C, blend weight) collected during epoch e iff 3 ≤ e < until and e % 5 == 0;
  at the end of that epoch: split/clone + prune. Decay at the end of epoch e iff 3 ≤ e < until and
  e % 10 == 0. Mip-NeRF 360 garden (161 train views): 186 epochs, until=141, densify epochs 5..140,
  decays 10..140. Room (272 train views): 110 epochs, until=81.
- Morton re-sort every 5 epochs (speed only).

## 5. Builds on
3DGS + TamingGS (budgeted, score-sampled densification, sparse Adam, TamingGS lrs). Kept from 3DGS:
the rendering model, L1+0.2 D-SSIM, xyz lr schedule, SfM init.

## 6. Protocol
Mip-NeRF 360 (images_4 outdoor / images_2 indoor, the dataset-provided downsampled images),
T&T, Deep Blending; llffhold 8 (idx%8==0 test); PSNR/SSIM/LPIPS (example_metrics.py); RTX 3090.
Variants: turbo (18K iters, small budgets — scripts not in repo), balance (30K, mid budgets),
quality (30K, 3DGS-sized budgets; full_eval.py). Repeat 3 in full_eval.py.

## 7. Ambiguities
- Paper: "C·Var = S − M²/C"; code uses var·C·σ(o)² with (C+1) denominators. Port follows code.
- Official kernel's `err_square` squares the *running* per-thread sum of `grad_a` across the
  thread's 2 pixel rows (`cur_err = grad_a`), i.e. not exactly per-pixel squares. Port uses exact
  per-pixel squares (the paper's formula); recorded as a fidelity gap.
- Sparse Adam visibility is cluster-granular (128-primitive Morton clusters intersecting the
  frustum) in the code; port uses per-primitive gsplat visibility (radii > 0).
- Clustered appends are truncated to multiples of 128 (speed artefact) — not ported.
- The paper mentions scale regularisation for "extreme" indoor (commented-out text); code default
  reg_weight=0 and full_eval.py passes none — not used.
- Turbo per-scene budgets (Table 5 primitives: bicycle 680k, garden 730k, room 400k, …) and its
  18K schedule are not scripted in the repo.
