# Plan — SSS native plugin (`student-splatting`, module `ss_student_splatting`)

Methods (per-scene official configs): `student-splatting` (common config: cap 3M, C_b 5e5,
C 120, burn-in 7000 — garden/most scenes), `student-splatting-bicycle` (3M, 5e8, 120, 15000),
`student-splatting-room` (1.5M, 5e5, 120, 7000), `student-splatting-truck` (2.6M, 5e5, 120, 7000).

## Components
| component | kind | factory | requires/provides | config (paper/code value) |
|---|---|---|---|---|
| `t-renderer` | renderer | `ss_student_splatting.renderer:create_renderer` | state.gaussian3d, camera.pinhole / render.rgb | background 0 (black), tile 16, near 0.2, fov_clamp 1.3, alpha_clip 0.99, exact_nu_grad False |
| `sghmc-recycling` | density | `ss_student_splatting.density:create_density` | — / — | cap_max, C_burnin, C, burnin, densify 500–25000/100, dead 0.005, recycle_frac 0.05, growth 1.05, sig k=100 t0=0.995, lr_base (0.001226442, 1.22663e-5, ref scale 7.45176), lr_max_steps 30000, opacity_lr 0.005, nu lr 10 ×0.5/5000 min 0.01, init opacity 0.5, init scale ×0.1 |
| `photometric` | loss | built-in | render.rgb | 0.8 L1 + 0.2 D-SSIM (unchanged) |
| `mcmc-regularizer` | loss | `ss_student_splatting.losses:create_regularizer` | render.rgb | opacity_reg 0.01·mean\|tanh o\|, scale_reg 0.01·mean exp s, every step |

## Fields
| field | shape | lr | init | density ops |
|---|---|---|---|---|
| `degrees` (raw nu; nu = hardtanh(raw,1,1e4)) | () | 10 (schedule in density) | 100 | copied like other fields |
`opacities` is re-interpreted as tanh pre-activation (signed opacity); init atanh(0.5).
Official `negative` attribute is a constant 1 (not learned) → not represented.

## Renderer (C1, C2)
1. Torch EWA projection = Inria `computeCov2D` (J with ±1.3·tan_fov clamp, no +0.3 low-pass),
   near cull z ≤ 0.2, custom autograd for the conic inverse with Inria's `1/(det²+1e-7)` backward.
2. Radius ceil(e(nu)·sqrt(λmax)), λ via `max(0.1, mid²-det)`, e(nu) critical-value table (forward.cu).
3. gsplat `isect_tiles`/`isect_offset_encode` with that radius (tile rect formula: gsplat's).
4. gsplat `spherical_harmonics` (dirs = μ - campos), +0.5, clamp_min 0.
5. JIT CUDA `t_raster.cu` (adapted from official `forward.cu`/`backward.cu` renderCUDA on the
   gsplat-tile scaffold of `reproductions/beta-splatting` `beta_raster.cu`): signed alpha,
   ±0.99 clamp, |α|<1/255 skip, T<1e-4 stop, straight-through clamp gradient, official partial dν.

## Density (C3, C4)
- `initialize`: opacity atanh(0.5), scales += ln 0.1, means Adam lr := 0 (Adam then only keeps the
  moments m, v that AdamSGHMC uses; runtime's means lr schedule stays 0), opacity lr 0.005.
- `before_backward`: sig = σ(100((1-|tanh o|)-0.995)), cov = R S² Rᵀ from pre-step params.
- `after_backward`: SGHMC step for means from Adam moments (sghmc.py), momentum stored in the
  means Adam state (`momentum` key) so population ops keep it aligned; nu lr schedule for next
  step; densify (recycle ≤5 % dead with t relocation, then +5 % up to cap).

## Probes
| probe | contribution | fails if |
|---|---|---|
| t-forward-vs-dense | C1, C2 | kernel ≠ float64 dense t reference, or t==Gaussian, or sign ignored |
| t-backward-vs-dense | C1 | any of means2d/conic/colour/opacity/nu grads ≠ autograd of reference (official partial dν via detached exponent, straight-through clamp) |
| t-projection | C1 | torch projection ≠ gsplat EWA (eps2d=0) means2d/conics; radius table wrong; conic backward ≠ Inria formula |
| scooping-sign | C2 | negative opacity does not subtract colour / raise T analytically on a single splat |
| sghmc-step | C3 | means update ≠ independent formula (burn-in with cov noise, friction-off; post: friction, reset at B+1; sig gate); Adam means lr ≠ 0 |
| t-relocation | C4 | relocation ≠ float64 lgamma/beta formula; N=1 not identity; recycle >5 %, growth ≠ 5 %, moment zeroing wrong, fields not trainable leaves |
| regularizer | C3 support | loss ≠ 0.01 mean|tanh o| + 0.01 mean exp s |

## Fidelity gaps
1. SSIM valid-conv and random-with-replacement view sampling (runtime) — small, global.
2. Inria vs gsplat tile rectangle (±1 tile at edges, both conservative) — negligible.
3. Last iteration: official skips the optimiser step at the final iteration; runtime steps Adam fields. negligible.
4. SH degree increases at step 1001 vs 1000 (runtime schedule). negligible.
5. Means Adam moments are computed by torch Adam (eps inside sqrt(v)/sqrt(bc2)+eps vs official
   sqrt(v)+eps) — eps=1e-15, negligible.
6. Evaluation on float renders (runtime) vs official 8-bit PNG round-trip — ~0.01 dB.
7. Reported numbers are a 3-run mean; we run 1 seed.
8. Official per-scene 40k iterations for room (bench with --steps 40000); baseline is 30k gsplat.
