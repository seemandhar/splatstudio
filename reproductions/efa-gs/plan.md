# EFA-GS native plan

Method names: **`efa-gs`** (EFA-GS(Mip), paper default) and control **`efa-gs-base`** (same code
path with LFCF off = the official EFA repo's Mip-Splatting with GOF abs-grad densification,
"Mip-splatting (i=∞)" in Tab. speed). Module `ss_efa_gs`. Package `plugin/`.

Prior art: `reproductions/mip-splatting/plugin` (3D filter, filter-aware reset, 2D mip filter =
gsplat antialiased eps2d 0.1). Its `filters.py` is vendored as `ss_efa_gs/mip_filters.py` (cited;
vendored so this plugin's source hash does not depend on another workspace). Its density wraps
gsplat DefaultStrategy with signed gradients only (paper-era commit 1ecf7d5); EFA's base needs the
later GOF densification (signed ∨ absolute gradient with quantile threshold), so the density is
written fresh.

## Components
| kind | name | factory | requires / provides | config (paper/code values) |
|---|---|---|---|---|
| renderer | efa-mip-gsplat | `ss_efa_gs.renderer:create_renderer` | provides render.rgb, render.depth.expected_z, gradient.screen.signed, gradient.screen.absolute, render.antialiased_2d, render.filter_3d | kernel_size_2d 0.1 |
| density | efa-density | `ss_efa_gs.density:create_density` | requires gradient.screen.signed, gradient.screen.absolute | grad_threshold 2e-4, densify_from 500, densify_until 15000, densify_every 100, percent_dense 0.01, prune_opacity 0.005, prune_world 0.1, opacity_reset_every 3000, reset_value 0.01, filter_coef 0.2, near 0.2, frustum_margin 0.15, filter_interval 100, total_steps 30000, lfcf true, interval_times 2, c_max 1.5, c_min 1.0, c_last 1.0, pow 1.0, splitting_ub 1.0, split_multiplier 2.0, tolerance 1e-5, diffscale true, depth_strategy true, lfcf_children 1 |
| loss | photometric | built-in | — | L1 + 0.2 D-SSIM (unchanged) |

Field: `filter_3d` (scalar, constant, lr 1e-8; recomputed from cameras) — as the Mip plugin.

## Density step (after_backward, step = official iteration)
Every step (step < 15000): accumulate, for visible Gaussians (radii > 0),
g = ‖grad·(W/2, H/2)‖₂ and g_abs = |absgrad_x·W/2| + |absgrad_y·H/2| (= official scalar
Σ_pix |∂x|+|∂y| in NDC, because gsplat absgrad sums |·| per axis) into both the 100-step and the
LFCF (lff) accumulators, with counts.
If 500 < step < 15000 and step % 100 == 0:
- LFCF if step % (interval_times·100) == 0 and lfcf, else GOF densify (clone ∨ split N=2).
- prune (σ(o) < 0.005 ∨ (step > 3000 ∧ max s > 0.1·extent) ∨ NaN), recompute filter_3d.
Opacity reset (filter-aware) at step % 3000 == 0, step < 15000 (after densify, as official).
Filter recompute every 100 steps in (15000, 29900).
All population changes go through `_update_param_with_optimizer` (new Parameter built with
`requires_grad=p.requires_grad`, never from a no_grad clone), and per-Gaussian state tensors are
updated explicitly with the official rules (clones: zeros for lff stats, P = True). Scale edits of
LFCF are in-place on `params["scales"].data` (Adam moments untouched, like `set_attributes`).
After every population change a trainable-leaf check runs (requires_grad pitfall guard).

## Probes (each fails if its mechanism is removed)
| probe | contribution | check |
|---|---|---|
| lfcf-decision | C1 | enlarge/shrink/split masks vs an independent per-Gaussian loop of Alg. 1 + code rules on random stats; removing the gradient-ascent test changes the masks |
| depth-factor | C2 | log c per Gaussian vs independent formula; deepest gets c_min, nearest c_max; ablation (depth_strategy false) differs |
| scale-strategy | C3 | enlarge: Σ Δlog s = 0, smallest axis + log c, others −1/3, −2/3; shrink: −½(log c+ln2)·(1, ½, 0) |
| anneal-schedule | C4 | log c multiplier 1 at 500, ½ at 7750, 0 at 15000; split lb; LFCF steps = {600, 800, …, 14800} |
| lfcf-gpu-step | C1–C4 | CUDA: drive density through step 600 on a rendered fixture: scales edited in place, Adam moments unchanged for edited rows, children count = splits, fields remain trainable leaves; `efa-gs-base` makes no scale edits |
| gof-densify | base | Q quantile selection vs independent torch.quantile; clone/split masks |
| filter3d | base | analytic 3D filter value (Mip Eq. 8) |
| renderer-absgrad (CUDA) | base | renderer provides absgrad ≥ |grad|; the density's g_abs / g equal |ax|·W/2+|ay|·H/2 and ‖g·(W/2,H/2)‖ recomputed from gsplat's own grad/absgrad (checks scaling, L1 combination, visibility — not gsplat's per-pixel |·| accumulation itself) |

## Fidelity gaps
1. Runtime: random-with-replacement view sampling (official shuffled stack); densify after the
   optimizer step (official before, dropping that step's update); valid-conv SSIM. (Background is
   black via the renderer `background` config, as official; the gsplat baselines use white. The
   means LR 0.01^(t/T) equals official get_expon_lr_func since lr_delay_steps = 0.)
1b. 3D-filter / LFCF-interval projection uses K's cx, cy (official W/2, H/2): only the in-frustum
   test changes; negligible for the centred pilr PINHOLE cameras.
2. gsplat vs Inria rasterizer: gsplat drops gradients where α ≥ 0.999 (TODO-runtime), tile/culling
   differences → absgrad statistics differ slightly; quantile threshold limits the effect.
3. Initialization through `--recipe 3dgs` (camera extent, 3-NN scales) ≈ official.
4. Random number streams differ (split sampling, split probability) — single-seed noise
   (README: ±0.05 dB across GPUs).
5. Control: `efa-gs-base` is our port of the EFA repo's base, not the authors' runs; the existing
   `runs/mip-splatting-i1-pilr` (paper-era Mip) is a second, older-base reference.
