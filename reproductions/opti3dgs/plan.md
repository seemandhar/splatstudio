# Opti3DGS native plan

Methods (one package `ss_opti3dgs`, two entry points):
- `opti3dgs` — Table 1 "3DGS + Ours*": frequency-modulated loss + 3DGS density with
  scaling_lr 0.01, refine_every 500, grow_grad2d 1e-4. **Primary benchmark target.**
- `opti3dgs-filter` — Table 2 "Ours-Filtering": same loss, default 3DGS density (secondary).

## Components
| kind | name | factory | requires/provides | config (paper value) |
|---|---|---|---|---|
| renderer | gsplat | built-in `splatstudio.backends.gsplat:create_renderer` | unchanged | — |
| density | opti-density | `ss_opti3dgs.density:create_density` (subclass of runtime `Density`, i.e. gsplat DefaultStrategy + 3DGS opacity-reset fix) | requires `gradient.screen.signed` | `scales_lr` 0.01, `refine_every` 500, `grow_grad2d` 1e-4 (filter variant: 0.005/100/2e-4) |
| loss | photometric (replaces built-in) | `ss_opti3dgs.losses:create_photometric` | requires `render.rgb` | `ssim_weight` 0.2, `kernel_sizes` [15,9,5,3], `boundaries` [1000,4000,8000,12000], `padding_mode` "reflect" |

The loss blurs the target with the scheduled mean filter (Eq. 2) and evaluates the runtime's
own L1 + 0.2 D-SSIM (same SSIM as the baseline) against it. Evaluation uses the original
images (runtime evaluator untouched). `scales_lr` is applied in `Density.initialize` by writing
the scales optimizer's lr (gsplat ops keep param_groups on densification). No new fields; no
parameters are rebuilt by the plugin (requires_grad pitfall avoided; a probe checks leaves).

## Schedule mapping (C2)
`kernel(step) = kernel_sizes[i]` for the first i with `step <= boundaries[i]`, else 1 (step is
1-based, = Inria iteration). K=1 → identity (no filtering after 12000).

## Probes
| probe | checks | fails if |
|---|---|---|
| opti-mean-filter | `mean_blur` equals scipy `uniform_filter(mode="mirror")` (independent), box kernel energy reduction on a checkerboard, K=1 identity, size preserved | filter wrong/removed |
| opti-schedule | kernel at steps 1,1000,1001,4000,4001,8000,8001,12000,12001,30000 = 15,15,9,9,5,5,3,3,1,1 | schedule wrong |
| opti-loss-target | loss(step=1) == photometric(rgb, blur15(gt)) ≠ photometric(rgb, gt); loss(12001) == photometric(rgb, gt); gradient finite | modulation removed |
| opti-density-config | after initialize: scales lr 0.01, refine_every 500, grow_grad2d 1e-4; a CUDA densify run keeps every field a trainable leaf with optimizer aligned | settings dropped / frozen fields |
Plus the built-in state/photometric/renderer probes.

## Fidelity gaps
1. No official code: step boundaries read from Fig. 3; padding mode guessed (reflect). Small effect.
2. Runtime protocol differences vs Inria 3DGS (white background, random view sampling with
   replacement, valid SSIM, gsplat rasteriser) — compare Δ vs our baseline to the paper's Δ.
3. Paper baseline is the authors' 3DGS re-run on H200; the paper's "Ours*" Δ on T&T is −0.54 dB,
   DB +0.02 dB.
4. densification_interval=500 interpreted as gsplat `refine_every` only (reset_every 3000,
   refine_start 500, refine_stop 15000 unchanged).

## Benchmark
Averages only → cheapest complete averages: `tandt` [truck, train] and `db` [drjohnson,
playroom], 30k steps, vs `baseline3dgs-tandt-30000` / `baseline3dgs-db-30000`. Targets
(opti3dgs): T&T 23.15, DB 29.56; secondary opti3dgs-filter: T&T 23.71, DB 29.68.

## Files
plugin/pyproject.toml, plugin/ss_opti3dgs/{__init__,filtering,losses,density,probes}.py,
plugin/tests/test_opti3dgs.py, plugin/README.md
