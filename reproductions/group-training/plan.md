# Plan — Group Training (plan id 94)

Method `group-training`, module `ss_group_training`, host = vanilla 3DGS (gsplat baseline
components). Smoke-only variant `group-training-smoke2k` (identical, `total_steps=2000` so the
cache is merged back before the runtime's checkpoint/eval at step 2000).

## Components
| component | kind | factory | requires/provides | config |
|---|---|---|---|---|
| gsplat | renderer | `splatstudio.backends.gsplat:create_renderer` (unchanged) | provides rgb, depth, gradient.screen.signed | absgrad False |
| group-training-density | density | `ss_group_training.density:create_density` | requires gradient.screen.signed | DefaultStrategy keys (baseline values) + grouping keys below |
| photometric | loss | unchanged | render.rgb | — |

Grouping config (defaults = official `GroupingParams`): `grouping=True`,
`grouping_method="opacity"` (OPS; `"random"` = RS), `utr=0.6`, `grouping_from_iter=1000`,
`grouping_until_iter=29000`, `grouping_interval=500`, `global_densify_iter=14500`,
`freeze_sh_at_merge=True`, `cached_moments="reset"` (official; `"keep"` = alternative),
`reset_stats_on_merge=True`, `total_steps=30000`.

No new FieldSpecs (any declared fields would be cached/merged generically: all params keys).

## Mechanism → code
The density wraps `splatstudio.backends.gsplat.Density` (DefaultStrategy + the runtime's
3DGS opacity-reset fix) so ADC is identical to the baseline, then adds grouping:
- Runtime order is render → backward → optimizer step → `after_backward(step)`. Official
  grouping happens before the render of iteration g, so we regroup in `after_backward(g-1)`
  after the regular density step.
- **Regroup at g** (`g ∈ range(from, until+1, interval)`):
  1. merge: if a cache exists, append cached rows to every param (new `nn.Parameter`,
     `requires_grad=True`), Adam `exp_avg`/`exp_avg_sq` rows = 0 (official
     `cat_tensors_to_optimizer`; `step` kept), DefaultStrategy state tensors extended with
     zeros, then all accumulators zeroed (official `densification_postfix` resets
     xyz_gradient_accum/denom/max_radii2D).
  2. g ∈ {I_D, until}: no resampling; `shN.requires_grad=False` (official f_rest freeze).
  3. else: OPS mask (`torch.multinomial(σ(o)/Σσ(o), int(N·UTR), replacement=False)`),
     cache ← detached clones of all fields of the unsampled rows, `gsplat.strategy.ops.remove`
     drops them from params/optimizers/state (official `prune_points`), shN unfrozen.
- Inactive Gaussians are not in the parameter tensors → not rendered, no gradient, no Adam
  step/decay (the runtime's per-field Adam would otherwise decay their moments and keep moving
  them with momentum when their gradient is zero — probe shows this).
- shN unfreeze after a refine event (official clone/split/prune rebuild the parameter with
  `requires_grad_(True)`): after the base density step at a refine step.
- Final merge at `step == total_steps` if a cache exists (smoke only; at 30k the cache is
  empty since 29000).
- Density statistics: all stats live in active-subset indexing between regroups (removal
  slices them); at merge they are scattered into the full population (cached rows appended)
  and then zeroed, as the official code does.

## Probes
| probe | checks | fails if |
|---|---|---|
| gt-ops-sampling | sample size = int(N·UTR), without replacement, inclusion frequency vs independent successive-sampling reference on a skewed-opacity population (high-opacity rows ≫ low), RS differs | OPS replaced by RS / with replacement / wrong ratio |
| gt-schedule | regroup steps 1000..29000/500, merge-only at 14500/29000, none at 999/29500, SH frozen exactly in official windows | schedule wrong |
| gt-subset-state | regroup→N_active=int(0.6N); cached rows bit-identical after several Adam steps; runtime Adam with zero grad would have moved them (contrast); merge restores all rows, moments of returned rows zero, retained rows' moments intact, stats zeroed, check_state alignment, gsplat ops still work | caching removed, moments not reset, misalignment |
| gt-cuda-training | CUDA: a few hundred steps on a synthetic scene with interval scaled down crosses regroups+refine; rendered N == active count, cached untouched, final merge restores population | subset render not used |

## Fidelity gaps
1. Host differences of the runtime vs 3DGS (white bg, gsplat DefaultStrategy: no screen-size
   prune, step indexing) — shared with the baseline; compare deltas.
2. Timing on shared A6000s vs paper RTX 3090; speedup ratio is what is compared, and
   bench GPU contention makes wall time noisy.
3. Order of optimizer step vs densify differs from 3DGS (runtime-wide, shared with baseline).
4. Only the 0~30K OPS config is benchmarked.
