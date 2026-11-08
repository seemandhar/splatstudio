# CoR-GS native plan

Method `cor-gs` (+ ablation/protocol baseline `cor-gs-single`), module `ss_corgs`, package
`splatstudio-method-cor-gs` in `plugin/`. Benchmark: LLFF 3-view at 1/8 res, all 8 scenes
(averages only; Table 1: 20.45 / 0.712 / 0.196; authors' 3DGS 19.22 / 0.649 / 0.229), dataset
`llff-fewview3-r8` (official `-r 8` from full-res `images/` with PIL bicubic — CoR-GS does NOT use
`images_8` unlike FSGS; same split and 3-view MVS init), 10 000 steps, `--recipe 3dgs`, all points.

## Two fields in one run
Field Θ¹ = runtime `params` (what `ss-eval` evaluates; paper keeps gs0 for inference).
Field Θ² is **component-owned state** of the density component `corgs-density`: a dict of
`nn.Parameter`s cloned from Θ¹ at `initialize()` (identical init, as official), with its own
Adams (same LR/eps as the runtime's), its own densification statistics. The density exposes
itself as `outputs["corgs"]` in `before_backward` (FSGS plugin pattern) so the loss components
can render Θ². Θ² is not in the checkpoint params → evaluation can only see Θ¹.

## Components
| name | kind | requires → provides | config |
|---|---|---|---|
| `corgs-renderer` | renderer | state.gaussian3d, camera.pinhole → render.rgb, render.depth.accumulated_z, render.camera, gradient.screen.signed | train bg `random` (rand(3) per training render, stored in outputs), eval bg black, SH min(t//500,3) (eval 3), ST alpha cap 0.999 |
| `corgs-density` | density | gradient.screen.signed | FSGS-style densify for both fields: grad 5e-4, interval 100, 500<t<10000, prune 0.005 (t>500), percent_dense 0.01, far split dist_thres 10, no unpooling; opacity reset 0.05 at t>500, (t-501)%3000==0; co-prune every 500 for t>500, τ=5; n_fields 2; owns 10 000 pseudo poses (`set_training_cameras`, official LLFF bounds table) |
| `photometric` | loss | render.rgb | built-in 0.8 L1 + 0.2 D-SSIM on Θ¹ (Eq. 6) |
| `corgs-field2` | loss | render.camera | renders Θ² at the same camera/bg/SH degree, 0.8 L1 + 0.2 D-SSIM vs GT (Θ²'s own Eq. 6) |
| `corgs-coreg` | loss | render.camera | 500 ≤ t ≤ 10000, every step: pseudo view, R(I'¹, sg I'²) + R(I'², sg I'¹), λ_p = 1 (Eq. 5, 7) |

Per step order (native): render Θ¹ → losses (Θ² train render, pseudo renders) → backward →
runtime steps Θ¹ Adam → `after_backward`: step Θ² Adam (means LR = the LR Θ¹ used this step),
accumulate stats + densify both, opacity reset both, co-prune both (masks from pre-prune
positions), check every field of both is a trainable leaf.

Co-prune: exact NN via chunked `torch.cdist` (open3d `evaluate_registration` = KD-tree hybrid
search, radius τ, max_nn 1); keep i iff min_j ||x¹_i − x²_j|| ≤ τ. No custom CUDA needed.

## Probes
| probe | fails if |
|---|---|
| `co-pruning` | fixture fields with known far outliers: pruned sets ≠ brute-force numpy reference, either field not pruned, masks computed after one field was pruned, optimizer state misaligned, fires outside t>500 & t%500==0 |
| `pseudo-coregularization` (CUDA) | loss ≠ independent L1/SSIM formula with detached targets, gradient leaks across fields, inactive inside [500,10000] or active outside, zero for different fields |
| `two-fields` (CUDA) | Θ² not initialised as a copy, not stepped/densified independently, no grad from `corgs-field2`, or Θ² appears in runtime params |
| `corgs-schedule` | SH degree ≠ min(t//500,3), resets not at 501/3501/6501/9501, eval background not black, train background not random |

## Variants
`cor-gs-single`: same renderer/density with `n_fields 1`, no co-prune/co-reg, photometric only
= the authors' "3DGS" row (19.22) in this pipeline; protocol-matched baseline for gains.

## Fidelity gaps
1. Runtime optimizer step happens before density control (official: densify, then step; the
   densified tensors skip that step). Θ² follows the runtime order for symmetry.
2. View sampling with replacement (official pops a shuffled stack of the 3 views).
3. gsplat vs Inria rasterizer (opacity-aware radii, 0.999 alpha cap via straight-through).
4. Runtime SSIM in losses uses valid convolution (official same padding); applied identically to
   both fields and the co-reg term. Eval: runtime SSIM/`lpips` VGG on float renders vs official
   8-bit PNGs + lpipsPyTorch.
5. Init cloud: COLMAP 3.9.1 MVS of the 3 views (authors' COLMAP version unknown).
6. Official skips the optimizer step at the last iteration; native does not (negligible).
7. One seed per scene; different RNG streams than seed 42.
8. τ = 5 is in world units: our llff-fewview3-r8 keeps the original LLFF COLMAP frame
   (FSGS bounds table matches the poses), so the units agree.
