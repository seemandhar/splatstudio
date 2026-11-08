# Plan — Deformable Beta Splatting (`beta-splatting`, module `ss_beta_splatting`)

Methods: `beta-splatting` (cap 1M, official default), `beta-splatting-garden` (cap 5M),
`beta-splatting-room` (cap 1.5M), `beta-splatting-bicycle` (cap 6M) — caps from official
`benchmark.py`. Benchmark: `mipnerf360-pilr` garden + room (official `-r 4`/`-r 2` PIL resize),
baseline `baseline3dgs-mipnerf360-pilr-30000`.

## Components
| kind | name | factory | requires / provides | config (paper/code value) |
|---|---|---|---|---|
| renderer | beta-renderer | `ss_beta_splatting.renderer:create_renderer` | state.gaussian3d, camera.pinhole / render.rgb | sb_number 2, background 0.0 (black), eps2d 0.3, tile 16 |
| density | kernel-agnostic-mcmc | `ss_beta_splatting.density:create_density` | — | cap_max, densify_from 500, until 25000, interval 100, dead_opacity 0.005, growth 1.05, noise_lr 5e4, noise_power 100, init_opacity 0.5 |
| loss | photometric (base) | splatstudio.losses | render.rgb | ssim_weight 0.2 |
| loss | dbs-regularizer | `ss_beta_splatting.losses:create_regularizer` | render.rgb | opacity_reg 0.01, scale_reg 0.01, start 500, stop 25000 (exclusive window) |

## Fields
| name | shape | lr | init |
|---|---|---|---|
| betas (raw b; beta = 4 e^b) | () | 0.001 | 0 |
| sb_params (r,g,b raw; theta; phi; b_m) | (2, 6) | 0.0025 | 0; theta~U[0,pi], phi~U[0,2pi] set in density.initialize |
Base: sh0 lr 0.0025 (= paper base colour lr), opacity 0.05, scales 0.005, quats 0.001, means
1.6e-4 x extent with 0.01 decay (runtime schedule = official get_expon_lr_func with delay 0).
shN unused (sh_degree 0): renderer never reads it (no gradient, Adam skips).

## CUDA (`csrc/beta_raster.cu`, JIT via torch.utils.cpp_extension.load)
Adapted from the official fork's `rasterize_to_pixels_{fwd,bwd}.cu` + `spherical_beta.cuh`
(Apache-2.0) on gsplat 1.5.3 tile lists (via linear-splatting's kernel scaffold).
- raster fwd/bwd: sigma = d^T conic d, skip sigma<0||>=1, alpha=min(.999, o (1-sigma)^beta),
  no 1/255 skip, gate o*vis<=.999 in bwd, v_beta.
- SB fwd/bwd: c = c0 + sum dot^{4e^{b_m}} softplus-colour; no view-direction gradient (as official).
Projection: gsplat 1.5.3 `fully_fused_projection` (EWA, eps2d 0.3, near 0.01); radii replaced by
the 1-sigma box `ceil(sqrt(Sigma_xx)), ceil(sqrt(Sigma_yy))` (exact Beta support; official uses the
enclosing circle `ceil(sqrt(lambda_max))` — same pixels, fewer tiles).

## Probes
| probe | checks | fails if |
|---|---|---|
| beta-forward-vs-dense | CUDA render == float64 dense torch reference of (1-r^2)^beta compositing; differs from Gaussian | kernel wrong/Gaussian |
| beta-backward-vs-dense | grads (means2d, conics, colors, opacities, betas) == autograd of dense ref | bwd wrong / v_beta missing |
| beta-support | single splat alpha zero outside r>=1 and tile boxes cover support | radius/support wrong |
| spherical-beta-vs-reference | CUDA SB colour + grads == torch formula; dirs no grad; lobe colours change with view | SB removed/wrong |
| mcmc-opacity-only | relocation/add: opacity 1-(1-o)^{1/(k+1)}, scales unchanged, moments zeroed, cap & 5% growth, noise ∝ (1-o)^100 Σ | gsplat scale-adjusting MCMC / logit noise |
| regularizer-schedule | loss = 0.01 mean o + 0.01 mean exp(s) inside (500,25000), 0 outside | missing/sum vs mean |
| state-transitions, photometric-gradient | base runtime probes | |

## Fidelity gaps
1. Optimizer step order: runtime steps Adam before density; official densifies before the step and
   the re-wrapped Parameters drop `.grad`, so official skips the Adam step on ~245 densification
   iterations. Effect: negligible. Not closable without runtime change.
2. SSIM: runtime valid-conv SSIM vs fused-ssim (same padding) in the loss; LPIPS from `lpips` pkg.
3. View sampling with replacement vs official shuffled stack (runtime limitation).
4. Means LR schedule off by one step (runtime updates after the step); negligible.
5. Projection is gsplat 1.5.3 (not the 1.3.0 fork); same EWA math.
6. "full" mode (test-set early stopping) intentionally not reproduced; compare to 30K column.
7. Training clamps nothing; evaluator clamps [0,1] as official.
