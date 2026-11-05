# AD-GS native plan

Method `ad-gs`, module `ss_ad_gs`, package `splatstudio-method-ad-gs` in `plugin/`.
Protocol: LLFF 3 views (FSGS/CoR-GS split), 1008×756, 10 000 steps, `--recipe 3dgs`,
init from the training-view COLMAP MVS cloud (`tools/prepare_llff_fewview.py`).

## Components
| component | kind | factory | requires → provides | config (paper/code values) |
|---|---|---|---|---|
| `adgs-renderer` | renderer | `ss_ad_gs.renderer:create_renderer` | state, camera → `render.rgb`, `render.depth.accumulated_z`, `render.camera`, `gradient.screen.signed` | `train_background: random`, `eval_background: black` (official `--random_background`, black test bg) |
| `adgs-density` | density | `ss_ad_gs.density:create_density` | `gradient.screen.signed`, `render.camera` | schedule (N_W 1500, cycle 200, N_H 100), thresholds warm 5e-4/0.005, low 5e-4/0.1, high 2e-4/0.005, interval 100, from 500, until 10000, prune_from 500, percent_dense 0.01, FSGS `dist_thres` 10, opacity reset 0.05 at (s-501)%3000==0, `n_pseudo` 10000, `seed` 0 |
| `photometric` | loss | splatstudio built-in | `render.rgb` | λ_ssim 0.2 (G1 L_ph) |
| `adgs-depth` | loss | `ss_ad_gs.losses:create_depth_smoothness` | `render.depth.accumulated_z` | ω1 = 0.01, λ_r = 0.001, schedule (G1 Eq. 7, low phase only) |
| `adgs-cotrain` | loss | `ss_ad_gs.losses:create_cotraining` | `render.camera` | λ3 = 1, start_sample_pseudo 500, ω1/λ_r for G2, schedule; trains G2 (photo + depth in low phase + pseudo) and returns G1's L_pseudo |

The density owns the second field G2 (own ParameterDict, Adam per field, grad stats), the
pseudo-pose pool and the phase schedule; in `before_backward` it places itself in
`outputs["adgs"]` so the co-training loss can reach G2 (no global state). G2 is cloned from the
initial G1 state after the recipe's k-NN scales/LR scaling (official: identical init). G2's
loss is back-propagated inside the co-training loss (independent graph); its optimizer step and
densification run in `density.after_backward`, with the means LR the runtime used for G1 in
that step.

## Probes
| probe | contribution | fails if |
|---|---|---|
| `alternating-densification` (CPU) | C1 | phase function differs from `it % 200 >= 100 and it > 1500`; a Gaussian with grad 3e-4 is not densified at a high step / is densified at a low step; opacity-0.05 Gaussian not pruned at low step or pruned at high step |
| `phase-gated-losses` (CPU) | C2 | depth loss nonzero outside low phase or zero inside |
| `edge-aware-depth` (CPU) | C4 | value/gradient differ from an independent loop implementation of Eq. 7 (code form) |
| `pseudo-view-consistency` (CUDA) | C3 | in a low step G1 gets no gradient from L_pseudo, G2 params get no gradient, or target not detached; pseudo poses not valid rotations / outside train-camera box |
| base `state-transitions`, `photometric-gradient`, `renderer-gradient` | state | as built-in |

## Fidelity gaps (known before implementation)
1. **SH schedule**: runtime raises SH degree every 1000 steps (official every 500). Degree 3 at
   3000 vs 1500. Small effect; would need trainer change.
2. **Optimizer/densify ordering**: runtime steps before densifying (gsplat); official densifies
   before the step (densify iterations don't update parameters). Negligible.
3. **SSIM**: runtime SSIM uses valid convolution (official: same padding) for L_ph and L_pseudo,
   and for evaluation. Same as all our baselines.
4. **View sampling**: runtime samples with replacement; official pops a shuffled stack. With 3
   views this matters little.
5. **Pseudo-pose bounds**: official uses LLFF `poses_bounds.npy` near/far of the training views;
   components cannot read dataset metadata, so near/far are recomputed as the 0.1/99.9
   percentile camera depth of the initial (training-view MVS) cloud — LLFF's own definition but
   on a different cloud. horns: near 13.58 vs 12.47, far 104.9 vs 168.8 → focus distance 45.7 vs
   43.2 (+6 %); camera positions unaffected.
6. **ω2 = 0.05 pseudo-view depth smoothness** (paper) is disabled in official code → not used.
7. **Random numbers**: different RNG streams (split sampling, pseudo pose sampling, backgrounds).
8. **Eval metrics**: LPIPS-VGG from `lpips` (official lpipsPyTorch, same weights); SSIM valid.
9. **gsplat vs diff-gaussian-rasterization**: gsplat classic mode, eps2d 0.3 equivalent.
10. **Image resampling**: our bicubic PIL downscale equals the official loader's `PIL.resize`
    default (bicubic) — exact up to Pillow version.

## Iteration 1 (after benchmark: horns −0.90, fortress −0.58, room −0.59 dB vs paper; gain over our gsplat exceeds the paper's)
Exactly two code-grounded fixes, nothing else changed:
1. **Pseudo-pose bounds (gap 5 closed)**: `adgs-density` `pseudo_bounds: "official"` uses the
   LLFF `poses_bounds.npy` near/far of the 3 training views (min near / max far, the only values
   `generate_random_poses_llff` uses), recorded by `prepare_llff_fewview.py` in `prep.json` and
   stored in `scene_bounds`, matched by training-camera centres (tol 1e-3); unmatched scenes
   raise. horns focus distance now 43.18 (was 45.7). Probe `pseudo-view-consistency` fails if
   the bounds/focus do not come from the table.
2. **SH schedule (gap 1 closed)**: `adgs-renderer` counts autograd renders (one per training
   step) and uses degree min(t // 500, 3) (official train.py:191); no-grad renders use degree 3.
   New probe `sh-schedule` (contribution `official-training-protocol`) fails on the runtime's
   1000-step schedule or if `render()` ignores it.
Evidence: `evidence/verify-i1.json` (declared_checks_passed, 8 probes), 7 unit tests.
