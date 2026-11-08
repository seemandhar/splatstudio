# BOGausS — native SplatStudio plan

Method `bogauss` ("Ours (Big)": final count = per-scene 3DGS count), module `ss_bogauss`.
No official code exists; everything follows paper.md (choices for ambiguities in paper.md §7).

## Components
| component | kind | factory | requires | provides | config |
|---|---|---|---|---|---|
| `bogauss-renderer` | renderer | `ss_bogauss.renderer:create_renderer` | state.gaussian3d, camera.pinhole | render.rgb, render.depth.expected_z, gradient.screen.signed, training.target_image | gsplat classic; forwards the frame GT image |
| `bogauss-density` | density | `ss_bogauss.density:create_density` | training.target_image | — | `DEFAULTS` in density.py |
| `photometric` | loss | built-in | render.rgb | — | 3DGS L1 + 0.2 D-SSIM (unchanged) |

No new FieldSpecs. The density replaces every per-field Adam with `BOGausSAdam` (optim.py) in
`initialize`; per-Gaussian state lives in the optimizer state (`exp_avg`, `exp_avg_sq`, `t`) and
in `density.state` (`delta`, `pmax`, `psnr`), all kept aligned by the plugin's own ops
(`ops.py`: remove / split_children) built from `p.detach()` with `requires_grad` copied.

## Contribution → code
- C1 precision Δ_i (`core.precision_delta`): for each training camera, Gaussians inside the frustum
  add (fx²+fy²)/(z² σ_uv²); Δ = α·(sum)^-1/2. Recomputed at start and after every densification
  (and every 500 steps after the window). Minimal size σ ≥ Δ in the scale update.
- C2 `optim.BOGausSAdam`: per-Gaussian t; α_s = max(1/t, 1-β_s); sparse (only `visible` rows
  updated, visibility = radii>0 set by the density in `before_backward`); `mode="position"`
  (step ×Δ), `mode="size"` (linear σ: grad_σ = grad_logσ/σ, σ ← max(σ − ηΔ m/(√v+ε), Δ), store
  log σ), `mode="plain"`. η for means/sizes: `lr_means`/`lr_scales` "auto" (median-matched to
  3DGS, paper.md §7.1) or numbers; the runtime's ×0.01 exponential decay acts on the means lr.
- C3 inheritance in `ops.split_children`: children moments α_g m, α_g² v, t·α_t.
- C4/C6 statistics: custom CUDA kernel `csrc/bogauss_stats.cu` (adapted from our Taming port's
  kernel, gsplat numerics) per training step: Σ_p Tα, Σ_p Tα·SE(p), #pixels. P^c = blend/count;
  `pmax = max(pmax, P^c)`; `psnr += se_i / Σ_j se_j`.
- C4 pruning at every densification: candidates pmax < 0.02, lowest first, at most 1 % of N.
- C5 `core.split_adaptation` table (offline fit at import, 1-D alpha-profile matching), children at
  μ ± 0.3 σ_long along the longest axis, longest-axis size ×r(o), opacity o_c(o).
- C6 split top-k P_SNR to reach the Taming parabolic count target (`core.count_target`), steps
  1000..15000 every 500, last target = budget. Budget from Taming big-budget table keyed by SfM count.

Unchanged from 3DGS: loss, SH schedule, init (recipe 3dgs), means-lr decay shape, other lrs.

## Probes
| probe | measures | fails if |
|---|---|---|
| `bogauss-precision` | Δ vs independent per-camera loop; Δ shrinks ∝1/√#cams and ∝ z | Δ removed/wrong |
| `bogauss-unbiased-sparse-adam` | first update = ηΔ·sign; invisible rows unchanged, t only counts visible; differs from torch Adam | optimizer replaced by Adam |
| `bogauss-linear-size` | size step is linear σ − ηΔ·…, clamp at Δ | log update / no clamp |
| `bogauss-state-inheritance` | children moments 0.2 m / 0.04 v, t copied | states reset |
| `bogauss-stats-kernel` (cuda) | kernel vs brute-force pixel loop | kernel wrong |
| `bogauss-prune` | prunes only pmax<τ, lowest first, ≤1 % | criterion removed |
| `bogauss-split` | longest axis ±0.3σ, adaptation preserves alpha profile better than naive copy; only split, no clone | split changed |
| `bogauss-split-priority` | top-k by Σ_c SE_ci/SE_c (not SE sum) | priority changed |
| `bogauss-schedule` | densify steps 1000..15000/500, no opacity reset, targets reach budget | schedule changed |

## Fidelity gaps
1. No official code: all ambiguities in paper.md §7 are our choices (η calibration, σ_uv, frustum, S_i, SplitAdaptation fit, growth curve, window). Largest PSNR risk: η of Δ-scaled updates.
2. Backend shared with baseline: gsplat (alpha clamp 0.999, white background, valid SSIM, sampling with replacement); Taming "fast" kernels not ported (speed only).
3. Statistics only from views rendered during training (one per step), not from all cameras.
4. Runtime steps optimizers before density control (same as baseline).
5. Exposure-correction variant not reproduced.
