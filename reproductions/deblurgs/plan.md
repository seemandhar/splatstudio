# DeblurGS — native SplatStudio plan

Method `deblurgs` (module `ss_deblurgs`). Variant `deblurgs-nu0` uses the literal code default
(ν lr = 0). Dataset `deblurnerf-motion`: Deblur-NeRF real_camera_motion_blur, 10 scenes,
600×400. Built by `tools/prepare_motionblur.py`. Evaluation is `python -m ss_deblurgs.evaluate`
(test-pose post-optimisation), run as `post_eval` after ss-eval.

## Components
| kind | name / factory | requires → provides | config (defaults = official code) |
|---|---|---|---|
| renderer | `deblurgs-renderer` `ss_deblurgs.renderer:create_renderer` | state.gaussian3d, camera.pinhole → render.rgb, render.depth.expected_z, gradient.screen.signed, render.motion_blur | near_plane 0.2, max_sh_degree 2, gamma 2.2, alpha_cap 0.999 |
| density | `deblurgs-density` `ss_deblurgs.density:create_density` | gradient.screen.signed | 3DGS + annealed θ (4e-4→2e-4/25k), interval 200, from 500, until 75k, reset 3000 → min(o,0.1), prune 0.005, percent_dense 0.01, xyz lr 1.6e-4→1.6e-6/150k × official extent; owns the trajectory bank (N 21, order 9, lrs 1e-2/1e-3/ν 3e-3, half 15k, start 1000, end 100k, ν start 30k) |
| loss | `deblurgs-objective` `ss_deblurgs.losses:create_objective` | render.motion_blur | L1(blur_lin, gt^2.2) + λ_t(1e-3→1e-5/150k)·mean\|Δsub\| + 0.1·hinge(opacity). Replaces photometric (code has no D-SSIM) |

## State
- No new per-Gaussian fields.
- `opacities` holds the **raw identity opacity** (official Clamp activation). The renderer uses
  clamp(o, 0, 1). `density.initialize` overwrites the runtime logit(0.1) init with 0.1.
- SH: degree ≤ 2 (`shN[:, :8]`). shN[8:] is unused and stays at 0.
- Per-image trajectory bank (`ss_deblurgs.bank`, process-local `ACTIVE`, created by the density
  in `set_training_cameras`):
  - rot / trans Bézier control points [n, 10, 3] (se(3) log in the pytorch3d convention);
  - ν raw [n, 19];
  - train viewmats (to identify the view);
  - own Adam (eps 1e-15) and schedules.
  - Exported to `density_state["bank"]` every 1000 steps and at the end.

## Per-step flow
1. The runtime picks a train frame. The renderer (grad enabled, bank active, viewmat matches
   view i) computes sub-frame times:
   - current step s = bank.step + 1;
   - s < 1000: [0], with the curve detached;
   - otherwise all 21 sorted times.
2. Poses: se3_exp(Bézier(t)) → c2w → w2c viewmat.
3. Render each sub-frame (gsplat, one random background per step, raw opacity clamp with a
   straight-through 0.999 cap). The renderer returns:
   - `blur_linear` = mean;
   - `subframes`;
   - `subframe_infos`;
   - `rgb` = γ(blur), with `info` = the first sub-frame's info.
4. density.before_backward: retain_grad on every sub-frame's means2d.
5. Loss (above). The runtime takes the Adam step for the Gaussians.
6. density.after_backward:
   - bank Adam step plus its lr schedules and freezing;
   - accumulate stats for every sub-frame (grad norm × (W/2, H/2), denom += 1/N_sub);
   - densify / prune / reset;
   - xyz lr override.

## Evaluation
Non-train cameras, or no bank (ss-eval): sharp render over a white background, then γ. ss-eval
gives "unaligned" numbers. `ss_deblurgs.evaluate`:
- loads the checkpoint;
- optimises each test pose exactly like official `optimize_test_pose`: quaternion of R_c2w
  (lr 5e-5) + w2c T (lr 5e-4), 2000 epochs, StepLR(100, 0.9), L1(clamp(γ(render)), gt);
- reports PSNR / SSIM (runtime valid-conv and 3DGS same-padded) / LPIPS (Alex = paper, VGG);
- writes `<run>/test_tpo.json`.

## Probes
| probe | checks | fails if |
|---|---|---|
| bezier-se3 | Bézier vs de Casteljau; se3 exp vs matrix_exp of the twist (row convention), log∘exp = id; init trajectory ≈ COLMAP pose | curve/SE3 wrong |
| blur-synthesis (cuda) | renderer train output == mean of independent gsplat renders at the 21 bank poses; ≠ sharp render once control points spread; step<1000 single frame at t=0 | averaging removed |
| alignment | times sorted, end points fixed, init = linspace; ν lr 0 before 30k, then decays to 1e-7 at 150k; ν gets gradient; nu0 variant keeps ν fixed | ν not learned |
| gamma | loss target = gt^2.2, eval rgb = render^(1/2.2) (independent numpy) | gamma dropped |
| smoothness | loss equals independent formula; λ_t(1)=1e-3, λ_t(150k)=1e-5, log-linear | term removed |
| densify-annealing | θ(1)=4e-4, θ(12500)=√(8e-8), θ≥25k = 2e-4; Gaussian with grad 3e-4 clones at step 25000, not at 600; reset min(o, 0.1); prune only opacity | annealing removed |
| test-pose-opt (cuda) | evaluate.optimize_test_poses recovers a perturbed pose on a synthetic scene (PSNR rises) | post-opt missing |
| state-transitions | base runtime probe | — |

## Fidelity gaps
1. View sampling with replacement (official: shuffled stack). Small effect.
2. The runtime's Adam step comes before densification (official: Gaussians densified at a step
   skip that step). Negligible.
3. SH degree raised at step 1001 instead of 1000. Negligible.
4. gsplat vs Inria rasterizer:
   - the alpha-saturation gradient is handled by a straight-through cap;
   - 3DGS antialias/0.3 dilation is the same as classic gsplat.
5. ν lr 3e-3 (paper) instead of the code's 0. A `deblurgs-nu0` variant is offered. Unknown which
   one produced Table 1.
6. Paper λ_smooth / L2 form vs the code; we follow the code.
7. The released code is "under construction"; its defaults may not match the paper runs.
8. SSIM: we report both forms; LPIPS-Alex from the `lpips` package (official: lpipsPyTorch,
   same weights).
9. Only averages are reported, so we must run all 10 scenes (150k × 21 renders each).
