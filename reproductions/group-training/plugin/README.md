# splatstudio-method-group-training

Native SplatStudio port of **Faster and Better 3D Splatting via Group Training** (Wang, Ma,
Xue, Lao — ICCV 2025, arXiv 2412.07608), following the official plug-in
https://github.com/Chengbo-Wang/3DGS-with-Group-Training on the vanilla-3DGS host.
Paper fidelity unverified.

## Mechanism
Every 500 iterations from 1000 to 29000 (official `GroupingParams`) the density component
regroups, in `after_backward` of the step before (the official code groups before the render):
1. **Merge**: the cached Gaussians are appended back to every field with zero Adam moments
   (official `densification_postfix` / `cat_tensors_to_optimizer`); density accumulators are
   zeroed (official postfix resets `xyz_gradient_accum`, `denom`, `max_radii2D`).
2. **Merge-only** at I_D=14500 and I_O=29000: no resampling; `shN.requires_grad=False`
   (official f_rest freeze). The next refine step (3DGS clone/split/prune rebuilds the tensor)
   or regroup unfreezes it, so in practice SH-rest is frozen for iteration 14500 and 29000–30000.
3. **Resample** otherwise: OPS mask `torch.multinomial(σ(o)/Σσ(o), int(0.6 N), replacement=False)`
   (Eq. 11); the unsampled rows are cached as detached tensors and removed from params,
   optimizers and DefaultStrategy state with `gsplat.strategy.ops.remove` (official
   `prune_points`).

Between regroups the parameters hold only the under-training group, so it alone is rendered,
receives gradients, is densified/pruned and opacity-reset. Cached Gaussians are never stepped:
the runtime's per-field Adam would otherwise decay their moments and move them by momentum
even with a zero gradient (probe `gt-subset-state` shows the drift it avoids). Their moments are
dropped and restart at zero on merge, as in the official code (`cached_moments: "keep"` is an
optional non-official alternative).

## Config (density component, defaults = official)
`grouping_method` opacity (RS: random), `utr` 0.6, `grouping_from_iter` 1000,
`grouping_until_iter` 29000, `grouping_interval` 500, `global_densify_iter` 14500,
`freeze_sh_at_merge` true, `cached_moments` reset, `reset_stats_on_merge` true,
`total_steps` 30000 (final merge if a cache is still held). 3DGS ADC keys are the gsplat
baseline's (`grow_grad2d` 2e-4 etc.).

## Methods
- `group-training` — paper rows "Group Training w/ OPS, 0~30K".
- `group-training-smoke2k` — same with `total_steps=2000` for 2000-step smoke tests only.

## Fidelity gaps
- Host = SplatStudio gsplat runtime (white background, gsplat DefaultStrategy without
  screen-size pruning, optimizer step before the density step). Shared with the baseline.
- Wall-time is measured on shared A6000s; compare the speed-up ratio to the gsplat baseline
  run in the same experiment, not the paper's RTX 3090 minutes.

## Commands
```bash
pip install -e .
ss-inspect group-training
ss-verify group-training --device cuda
python -m pytest -q tests
```
