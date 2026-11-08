# BBSplat — native SplatStudio plan

Method names: `bbsplat` (Mip-NeRF 360, Tab. 1 budget cap 60K), `bbsplat-tnt` (T&T, cap 30K),
`bbsplat-300k` (published `scripts/train_all.sh` budget for bicycle/truck; no per-scene numbers,
not benchmarked by default). Module `ss_bbsplat`, package `splatstudio-method-bbsplat`.
Representation `surfel2d`. Prior art reused: `reproductions/hdgs` (vendored 2DGS-family CUDA as
JIT extension, official camera matrices, depth_to_normal), `gsplat.strategy.ops` for
optimizer-aligned population edits.

## Components
| component | kind | factory | requires | provides | config (paper/code value) |
|---|---|---|---|---|---|
| bbsplat-raster | renderer | `ss_bbsplat.renderer:create_renderer` | state.surfel2d, camera.pinhole | render.rgb, render.depth.expected_z, render.alpha, render.normal.world, render.normal_from_depth.world, render.distortion.bbsplat, render.impact.bbsplat | background black, depth_ratio 0, eval_quantize true |
| bbsplat-mcmc | density | `ss_bbsplat.density:create_density` | — | — | cap_max, max_read_points, sky_points 10000, densify 500/25000/100, dead 0.005, growth 1.05, LRs, texture window [500,30000), freeze >30000, total 32000 |
| photometric | loss | built-in | render.rgb | — | 0.8 L1 + 0.2 D-SSIM |
| bbsplat-texture-reg | loss | `ss_bbsplat.losses:create_texture_reg` | render.impact.bbsplat | — | sigma 100 (code), lambda_rgb 1e-4, lambda_alpha 1e-4, opacity_reg 0.01 |
| bbsplat-geometry | loss | `ss_bbsplat.losses:create_geometry` | render.normal.world, render.normal_from_depth.world, render.distortion.bbsplat | — | lambda_normal 0, lambda_dist 0 (scripts for M360/T&T), starts 7000/3000 |

## Fields
| field | shape | lr (declared) | init | density treatment |
|---|---|---|---|---|
| bb_alpha | (16,16) | 0.001 (lr 0 outside [500,30000)) | logit(0.5 G), zero texels -> -1e4 (official -inf) | relocate/add: 1-(1-a)^(1/(n+1)) per texel |
| bb_color | (3,16,16) | 0.0025 (same window) | logit(0.1) | copied |
`opacities` (base field) is unused by the renderer; lr 0. `scales[:,2]` unused (kernel takes 2
scales). Fields are per-primitive, so the runtime checkpoint holds them; no side store needed.

## Init (density.initialize, replaces the population through gsplat ops)
FPS (pytorch3d semantics: start index 0, iterative argmax of min distance) to max_read_points;
prepend 10K Fibonacci sphere points (radius max|xyz|, colour 1.0); scales = log sqrt(mean d^2
3-NN) on the combined set (2 cols, 3rd pinned); quats U[0,1)^4; sh0 = RGB2SH(c) - 0.1; shN 0.

## Density schedule (after the runtime's optimizer step, so lrs set at step t apply to t+1)
xyz lr expon(t+1) * extent (30000 max); textures lr only for 500 <= t+1 < 30000; for t+1 > 30000
all lrs 0 except shN; final step lrs 0 (official skips the last step). Densify at
500 < t < 25000, t % 100 == 0: relocate dead (mean alpha <= 0.005) onto multinomial(mean alpha)
samples, then add min(cap, 1.05N) - N; Adam moments zeroed only at sampled indices (official).
MCMC position noise is exactly zero in the official code (fixed opacity 1 -> sigmoid coefficient
underflows) -> not applied; probe documents it.

## Probes
| probe | measures | fails if |
|---|---|---|
| raster-vs-reference (CUDA) | official kernel forward rgb/alpha/depth/impact + grads (means, scales, quats, SH, both textures) vs dense torch emulation (ray-plane intersection, bilinear align-corners sampler, 0.99 clamp, 1/255 skip, T<1e-4 stop) | texture lookup, colour composition, impact or any gradient wrong |
| texture-composition (CUDA) | constant-SH scene: rendered colour = SH colour + sigmoid texel; removing texture changes it | Eq. 6 broken |
| texture-reg | loss vs independent formula; zero for invisible; zero weight at impact >= sigma | C3 removed/wrong |
| mcmc-relocation | dead detection, alpha rescale formula, scales unchanged, moment reset at sampled ids, population cap growth 5% | C4 changed to Gaussian MCMC or 3DGS ADC |
| init-sky-fps | FPS equals brute-force farthest sampling; sphere count/radius/colour; SH dc offset | C5 removed |
| schedule | lr table at boundary steps (499/500, 29999/30000, 30001, 32000) | C6/texture window removed |

## Fidelity gaps
1. Adam step is not skipped on densify iterations with relocation (official side effect) — tiny.
2. Densification happens after the optimizer step instead of before — order effect only.
3. Random-with-replacement view sampling (runtime), SSIM valid conv, SH degree increase one step
   later — known runtime-wide differences.
4. Tab. 1 cap_max/max_read_points not published: cap = Tab. 1 "Points", max_read = cap - 10K.
5. Depth-normal regularisation off for M360/T&T (scripts) although the paper text says on.
6. FPS start point / point order follows our points.npz order (COLMAP order in conversion).
7. ±inf alpha logits of the official code are stored as ±1e4 (identical sigmoid output and zero
   gradient; runtime requires finite state).
8. Eval: 8-bit texture quantisation emulated in the renderer when gradients are disabled
   (official render.py loads quantised npz textures); rendered images are not 8-bit-quantised
   before metrics (official metrics read PNGs) — runtime-wide.
9. Paper's 3DGS baseline is budget-constrained; compare against the paper's absolute numbers and
   against its fixed-storage 3DGS-MCMC row, not our unconstrained gsplat baseline.
