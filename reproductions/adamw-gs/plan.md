# AdamW-GS native plan

Method names / module: `adamw-gs-gs8` (paper Table-2 headline outdoor row GS8 = `adamw-gs` + opacity reset every 3000 + 3DGS big prune), `adamw-gs` (Mip-NeRF 360 outdoor released config `mipnerf_adamwgsp`:
noise on, StSS GS2) and `adamw-gs-indoor` (released `mipnerf_adamwgs`: noise off, StSS GS1);
module `ss_adamw_gs`. Both: vanilla 3DGS host, opacity reset off, opacity correction on clone.

## How the optimizer gets in (runtime has no optimizer factory)
The runtime builds one `torch.optim.Adam` per field and calls `opt.step()` with no arguments,
before `density.after_backward`. The density component's `initialize(params, optimizers, ...)`
receives the runtime's own `optimizers` dict; it **replaces every entry in place** with a
`SparseAdamWGS` (subclass of `torch.optim.Adam`, same `exp_avg`/`exp_avg_sq` state keys, so
`check_state`, `gsplat.strategy.ops` and checkpointing keep working). The runtime iterates
`optimizers.values()` every step and reads `optimizers["means"]` for its LR schedule, so the
replacement is picked up; the LR (already multiplied by scene scale) is copied over.
Per-iteration inputs the optimizer needs but `step()` does not receive (visibility, iteration,
per-primitive step counter) live in a shared `AdamWGSContext` filled by `density.before_backward`
(visibility = `radii > 0` of this iteration's render; `asystep += visibility`).
Precedent: `reproductions/taming-3dgs` (IntervalAdam swapped in at initialize).

## Components
| kind | name | factory | requires/provides | config (paper/code value) |
|---|---|---|---|---|
| renderer | gsplat | splatstudio.backends.gsplat:create_renderer | unchanged | absgrad False |
| density | adamw-gs-density | ss_adamw_gs.density:create_density | requires gradient.screen.signed | β (0.9,0.999), eps 1e-15, λo 1e-3, λs 1e-5, cap 10/10, opacity window [3000,15000], scale window [1000,15000], RSR interval 100 window [1000,15000] ratios [0.1,0.3,0.5]@[1000,7000,10000] (indoor [0.1,0.3]@[1000,7000]), α1 0.2, α2 0.01 (code-effective; paper 0.04), noise on [1000,14000] noise_lr 5e5 (indoor off), opacity_correction True, opacity_reset_every 0 (off), 3DGS ADC: grad 2e-4, start 500, stop 15000, every 100, percent_dense 0.01, prune_opa 0.005 |
| loss | photometric | splatstudio.losses:create_photometric | render.rgb | L1 + 0.2 D-SSIM (unchanged) |

No new FieldSpecs. Per-primitive int32 `asystep` lives in the density `state` dict, so
gsplat `duplicate`/`split`/`remove` copy it from the parent / drop it exactly like the official
`asystep[selected].repeat(N)` / `asystep[valid]` (new primitives inherit the parent's count,
moments start at zero).

## Step mapping (per iteration s, runtime order)
1. render → `before_backward`: retain means2d grad; `vis = (radii>0).all(-1)`; `asystep += vis`; ctx.step = s.
2. backward; 3. `SparseAdamWGS.step()` for each field = adam.cu numerics on visible rows only (Eq. 4, 7, 8;
   App. Eq. 13–16): per-primitive bias correction, DAR term `min(reg_scale·∇R/denom, 10)` for opacities/scales
   inside their windows, `reg_scale = λ / img_scale`, `img_scale` = official `_img_scale(W·H of first train camera)`.
4. `after_backward`: 3DGS ADC (stats, clone with opacity correction of source+clone, split, prune o<0.005;
   optional reset + 3DGS big-prune when `opacity_reset_every>0`), then position noise (Σ·n·σ(−100((1−o)−0.995))·5e5·lr_means(s)),
   then RSR (uniform k=int(ratio·N) without replacement; all six optimizers' moments ×α1/×α2).
Official order is densify → step → noise → RSR; ours is step → densify → noise → RSR (runtime fixes
the optimizer step before density). Effect: on the 144 densification iterations (600..14900) the official code
performs **no** parameter update (densify leaves `.grad=None`) whereas we update then densify — the same
difference the gsplat baseline already has; asystep counts are identical (children inherit post-increment counts).

## Probes (each fails if the mechanism is removed)
| probe | checks |
|---|---|
| adamwgs-sparse-step | SparseAdamWGS vs an independent per-element loop of adam.cu: invisible rows bit-identical (dense Adam fails), per-primitive bias correction with distinct asystep (global-step Adam fails) |
| adamwgs-dar | opacity/scale DAR term equals reference, moments contain only ∇ℓ (coupled L1 fails), cap at 10, window boundaries 2999/3000/15000/15001 and 999/1000 |
| adamwgs-img-scale | `_img_scale` digits rule (1089480→100000, 1617204→160000, 1297·840) |
| adamwgs-rsr | at s=1000 exactly int(0.1N) rows of every optimizer scaled by α1/α2, others untouched; ratio 0.3 at 7000, 0.5 at 10000 (outdoor) / 0.3 (indoor); nothing at 900, 1050, 15100 |
| adamwgs-topology | clone/split/prune through the density: asystep inherited, zero moments for children, opacity correction 1−sqrt(1−o) on source and clone, check_state passes |
| adamwgs-noise | means displacement equals independent Σ·n formula (same RNG seed); absent in indoor variant and outside [1000,14000] |

## Files
`plugin/pyproject.toml`, `plugin/ss_adamw_gs/{__init__,optim,density,probes}.py`, `plugin/README.md`,
`plugin/tests/test_adamw_gs.py`.

## Fidelity gaps
1. White background / random-with-replacement views / SSIM valid conv (runtime-wide; calibrated by baseline). Effect ~ baseline offset.
2. Optimizer step runs before densification (see above); 144 extra updates (densify iterations 600..14900); with GS8 also the reset iterations 3000/6000/9000/12000, where the official code skips the opacity update (parameter replaced before the step) and we update then reset. Small; not closable without core change.
3. Means LR: runtime applies lr(s−1) at step s (official lr(s)); negligible.
4. Official fused CUDA kernel replaced by torch ops with the same float32 formula (no speed claim; time metric not reproduced).
5. α2 = 0.01 (code) vs 0.04 (paper text); configurable.
6. Paper headline "GS8" (outdoor with opacity reset) and "GS7" (StSS GS5) have no released configs; we target the released configs = per-scene rows GS9 (outdoor) and GS7/GS8 (indoor). `opacity_reset_every=3000` + `rsr_ratios` config allow GS8/GS7 runs.
7. T&T/DB configs (per-scene noise and StSS differ) not packaged as methods; only Mip-NeRF 360 is benchmarked.
8. Visibility differs from diff-gauss: gsplat 1.5.3 culls opacity < 1/255 (radii 0), uses opacity-aware per-axis radii (extend = min(3.33, sqrt(2 ln(255 o)))) and near plane 0.01 (3DGS 0.2). Near-dead primitives are invisible here, so they get no Sparse-Adam/DAR update or asystep increment. Expected small because o < 0.005 is pruned every 100 steps until 15k. 
9. Method `adamw-gs-gs8` (paper headline outdoor row): opacity reset every 3000 + world-space big prune (max scale > 0.1 extent after 3000), as official train.py:219 enables with resets. The official 20-px screen test (`big_points_vs`) is dead code: densify_and_clone/densify_and_split both call densification_postfix, which re-zeros max_radii2D before prune() reads it. Ported as inactive. A first GS8 smoke that applied a live 20-px test (3DGS radius from conics) collapsed to 7.3 dB at 3200 steps; with it disabled the run recovers (20.8 dB). gsplat's tighter radii therefore do not matter here.
