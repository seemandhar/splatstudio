# CRiM-GS: native SplatStudio plan

- **Method:** `crim-gs`, in module `ss_crim_gs`.
- **No official code.** Details the paper leaves out come from CoMoGaussian (`source/comogaussian`, MIT, cd6d536), the authors' successor paper.
- **Datasets:**
  - `deblur3dgs-synmotion` (new): Deblur-NeRF synthetic motion blur, 600×400, hold 8. Built by `tools/prepare_synthetic.py`.
  - `deblur3dgs-motion` (existing): real motion blur, 600×400, hold files.
- **Benchmark scenes:** cozy2room (synthetic, 33.01) and ball (real, 28.25). Both have per-scene numbers in the appendix tables.

## Components
| kind | name / factory | requires → provides | config (defaults) |
|---|---|---|---|
| renderer | `crimgs-renderer` `ss_crim_gs.renderer:create_renderer` | state.gaussian3d, camera.pinhole → render.rgb, render.depth.expected_z, gradient.screen.signed, render.antialiased_2d, render.filter_3d, render.motion_blur | kernel_size_2d 0.1, background 0 (black, official) |
| density | `crimgs-density` `ss_crim_gs.density:create_density` | gradient.screen.signed | DefaultStrategy (grow 2e-4, from 500, every 100, until 20000, prune_opa 0.005, no reset, no size prune), 3D filter coef 0.2 / near 0.2 / margin 0.15 / interval 100, total_steps 40000. Owns the CRiM kernel (`kernel_*` keys: N 9, hidden 64, rtol 1e-4, atol 1e-5, lr 1e-4 → linear, start_warp 1000, start_pixel_weight 3000, CNN 64 ch / 3 layers, decoder init 1e-5) |
| loss | `photometric` (`splatstudio.losses:create_photometric`) | render.rgb | ssim_weight 0.3 (λc) |
| loss | `crimgs-regularizers` `ss_crim_gs.losses:create_regularizers` | render.motion_blur | lambda_o 1e-4, lambda_m 1e-3 |

## State
- **Field:** `filter_3d` [N], a constant Mip-Splatting filter (lr 1e-8, never receives a gradient), recomputed by the density.
- **Kernel** (`ss_crim_gs.kernel.CRiMKernel`, process-local `ACTIVE`). The density builds it in `set_training_cameras` and gives it its own Adam. It holds:
  - nn.Embedding(n_train, 64);
  - E_r, E_a: Linear(64, 64);
  - f, g: halves 32 + 32, ReLU → Linear(32, 32) each;
  - D_r: z_rot → ω (3, normalised), θ (1); z_trans → v (3);
  - D_a: z_rot → A (9); z_trans → t̃ (3);
  - CNN F with weight and mask heads.
- **Not checkpointed:** the kernel is not needed at test (sharp render). It is exported to `density.state["kernel"]` only as a small diagnostic, on the final step.

## Per-step flow
1. **Renderer, sharp render:** applies the 3D filter to scales and opacities, then renders the input pose with gsplat antialiased (eps2d 0.1) over black. Its `info` goes to the density (statistics from the sharp view only, as in the code).
2. **Renderer, trajectory:** when gradients are on, the kernel is active, the viewmat is a training view, and step > 1000:
   - kernel.poses(i): latent ODE (own dopri5) at τ = 0..8 → T̃_r, T̃_a;
   - c2w_s = c2w · T̃_r · T̃_a, viewmat_s = inv(c2w_s);
   - one batched gsplat call over the N cameras.
3. **Blend:**
   - step ≤ 3000: rgb = mean(N renders + sharp);
   - otherwise: P, M = CNN(renders.detach()), and rgb = (1 − M)·sharp + M·Σ P·I.
4. **Losses:** photometric (0.3) on rgb; regularisers λo·mean_s ‖R̃ᵀR̃ − I‖_F + λM·mean(M).
5. **density.after_backward:**
   - kernel Adam step (step > 1000), lr = 1e-4 − (step − 1000)·(1e-4 − 1e-6)/40000;
   - gsplat densification;
   - 3D filter recompute (after densify steps; every 100 after 20000 until 39900).

## Probes
| probe | checks | fails if |
|---|---|---|
| dopri5 | solver vs matrix_exp on a linear ODE (rel err < 1e-4); 5th-order convergence of the fixed-step tableau; adaptive step count > 1 | wrong tableau / Euler fallback |
| screw-se3 | Eq. 9/10 equal torch.matrix_exp of [S]θ; ‖ω̂‖ = 1; θ is independent of ‖ω‖ | Rodrigues/G wrong, ω not normalised |
| kernel-poses | poses = c2w·T̃_r·T̃_a (independent recomposition); init poses within 1e-3 of the input; R̃_a ≈ I; latent at τ0 equals E(embed) | composition wrong |
| blur-synthesis (cuda) | renderer output equals an independent combination of independent gsplat renders (sharp at step ≤ 1000; mean of N + 1 at 2000; P/M blend at 4000); kernel gets gradients | blending removed |
| objective | regulariser equals the independent formula; photometric ssim_weight 0.3 | λ / L_o form wrong |
| kernel-schedule | linear lr values; no kernel step at ≤ 1000 | schedule wrong |
| density-schedule (cuda) | no opacity reset at 3000/6000; filter steps match the code; no growth after 20000 | Mip reset leaks in |
| filter3d | filter equals an independent loop | — |
| state-transitions | base | — |

## Fidelity gaps
1. **No official code.** CoMo code fills the gaps: kernel lr, rtol/atol, CNN norm layers, mask head, the warm-up mean of N + 1, no opacity reset, densify until 20k. CRiM-GS may differ.
2. **ODE time grid:** τ = 0..8 forward from τ0, autonomous f (assumed). CoMo integrates outward from the middle with a time embedding.
3. **Solver:** our own dopri5 hits the output times exactly; torchdiffeq interpolates. Same order and tolerance. Step-size control is detached.
4. **Pose convention:** clean c2w right-multiplication, versus CoMo's mixed [R_c2w | t_w2c] matrix.
5. **Runtime differences:**
   - views sampled with replacement (official: shuffled stack);
   - SSIM loss with valid convolution;
   - the Gaussian Adam step comes before densification;
   - means lr decay 0.01^(s/S) (matches 3DGS expon from 1.6e-4 to 1.6e-6; no delay_mult in either).
6. **Metrics:** ss-eval gives runtime PSNR (same), SSIM (valid conv, not skimage) and LPIPS-VGG (paper: Alex). PSNR is the target.
7. **gsplat vs the Inria/iComMa rasterizer:**
   - pose gradients come from gsplat's viewmat backward;
   - the saturated-alpha gradient drop is not capped (opacities are sigmoid ≤ 1, same as the Mip/baseline ports).
