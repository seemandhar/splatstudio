# Faster and Better 3D Splatting via Group Training (ICCV 2025) — implementation spec

- arXiv 2412.07608 (LaTeX source in `source/latex/`), official code
  https://github.com/Chengbo-Wang/3DGS-with-Group-Training (`source/code/`, `__init__.py` +
  `grouping_method.py`, a drop-in for the graphdeco 3DGS `train.py`). No license file in the repo.
- Benchmark host: **vanilla 3DGS** (graphdeco-inria/gaussian-splatting). Mip-Splatting is a
  second host with its own tables (Tab. 4/5, supp.); we reproduce the 3DGS-host rows.

## 1. Method summary
- The Gaussian set G is split every 500 iterations into an **Under-training group** (sampled,
  UTR = 60 % of |G|) and a **Cached group** (the rest). Only the under-training group exists in
  the model between regroupings: it alone is rendered, receives gradients, is densified/pruned
  and opacity-reset. The cached group is physically removed and re-appended at the next
  grouping (merge → resample).
- Sampling is **Opacity-based Prioritized Sampling (OPS)**: multinomial without replacement
  with p_i ∝ σ(opacity_i).
- **Schedule**: groupings at 1000, 1500, …, 29000 (from 500 iters after densify_from_iter=500).
  At I_D = 14 500 and I_O = 29 000 the groups are merged **without resampling** (global
  densification 14.5k–15k, global optimisation 29k–30k); SH rest coefficients are frozen at
  those merges.
- Densification itself is the unchanged 3DGS ADC, applied to the under-training subset only
  (the "densification changes" are: subset-only densify/prune/opacity-reset, density
  statistics reset at every merge, and the global-densification window).

Contributions
- C1 Group training: subset rendering + optimisation, cached group frozen (Sec. 3.1, Eq. 1–2, Fig. 3).
- C2 Opacity-based prioritized sampling (Sec. 3.2/3.3, Eq. 11 `p_i = α_i / Σ α_i`).
- C3 Cyclic resampling schedule with global densification (I_D) / global optimisation (I_O)
  merges and SH freezing (Sec. 4.1, Fig. 5 "schedule", ablation Tab. 6).

## 2. Equations
Eq. (1) `G_{UT} = { g_i | g_i ∈ G, i ∈ I, I ⊆ {1..|G|} }`, Eq. (2) `G_{Cached} = G \ G_{UT}`.
Eq. (11) `p_i = \frac{\alpha_i}{\sum_{i=1}^N \alpha_i}`, α_i = activated opacity; |I| = int(N·UTR),
sampling **without replacement** (`torch.multinomial(p, int(N*UTR), replacement=False)`).
Densification criterion unchanged: `\sqrt{(∂L/∂x_m)^2 + (∂L/∂y_m)^2} > τ_grad` (averaged over views).
Props. 1–2 (∂L/∂x ∝ o_m; E[T_N] = (1−E[o]E[G])^N) are motivation only.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| grouping_method | Opacity-weighted (OPS) | code `__init__.py` GroupingParams |
| UTR | 0.6 | code; paper §4.1 |
| grouping_from_iter | 1000 | code; paper "500th iteration after densification begins" |
| grouping_until_iter | 29000 (= I_O) | code |
| grouping_interval | 500 | code; paper §4.1 |
| I_D | 14500 (hard-coded) | code `_UTR = 1.0 if _iteration in [14500, until]` |
| SH freeze | `f_rest.requires_grad=False` at merge-only groupings | code; paper §4.1 |
| 3DGS ADC | densify 500–15000 every 100, τ=2e-4, opacity reset /3000, prune opa 0.005 | 3DGS defaults |

## 4. Schedules (exact official behaviour, iteration = 1-based 3DGS iteration)
- Grouping happens **before render** of iterations g ∈ {1000, 1500, …, 29000}.
  1. If a cache exists: `densification_postfix(**cache)` appends the cached Gaussians with
     **zero Adam moments** (cat_tensors_to_optimizer) and **resets xyz_gradient_accum, denom,
     max_radii2D of all Gaussians to 0**.
  2. g ∈ {14500, 29000}: no resampling, cache = None, f_rest frozen (`requires_grad=False`).
  3. otherwise: sample mask with OPS, cache = all fields of the unsampled Gaussians,
     `prune_points(cache_mask)` (removes them **and their Adam state** from the optimizer).
- Consequences: cached Gaussians are neither stepped nor decayed; their moments are dropped
  and restart from zero on merge (Adam `step` count is shared per tensor, so bias correction
  stays at the global step). A Gaussian that was under training before the regroup keeps its
  moments regardless of the new sample (it was never removed); only Gaussians coming back
  from the cache restart from zero moments.
- f_rest freeze lasts until the next op that rebuilds the parameter (3DGS clone/split/prune
  creates a new `nn.Parameter(...requires_grad_(True))`): at 14500 the densify at the end of
  iteration 14500 unfreezes it (≈1 iteration frozen); at 29000 it stays frozen to 30000.
- Opacity reset (3000, 6000, 9000, 12000) and low-opacity/oversize pruning act only on the
  under-training group present at that iteration.
- Densify at g (1000, 1500, …, 14500) uses one iteration of statistics (accumulators were
  zeroed by the merge before render).

## 5. Builds on
Vanilla 3DGS train.py; everything else unchanged (L1+0.2 D-SSIM, LR schedule, SH schedule).

## 6. Protocol
Mip-NeRF 360 (images_4 outdoor / images_2 indoor via full_eval.py), T&T, DB, llffhold 8,
30k iterations, RTX 3090, LPIPS-VGG (3DGS metrics.py). Times in minutes, size in MB.
Per-scene numbers in supp. (tables/X_suppl/3dgs_*.tex).

## 7. Ambiguities
- 0~15K variant: only the default 0~30K config is released; the 0~15K variant would need
  `grouping_until_iter=15000` (the code would then also freeze f_rest from 15k to 30k).
- README uses `use_trained_exp` (new 3DGS) while the cache dict lacks `new_tmp_radii` required
  by the newest `densification_postfix`; the exact 3DGS commit is unknown.
- Paper text "Group Training was activated at the 500th iteration after densification
  begins" = 1000 in code.
