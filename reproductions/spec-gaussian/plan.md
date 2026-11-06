# Spec-Gaussian native plan

Method names (variants of one package `ss_spec_gaussian`, `splatstudio-method-spec-gaussian`):

| name | decoder Ψ | resolution c2f | use |
|---|---|---|---|
| `spec-gaussian` | outdoor (Linear-ReLU-Linear) | off (official train.py) | Mip-NeRF 360 outdoor scenes |
| `spec-gaussian-indoor` | indoor (Linear-ReLU-Linear-ReLU-Linear) | off | Mip-NeRF 360 indoor scenes (`--is_indoor`) |
| `spec-gaussian-c2f` | outdoor | paper §3.3 (1/4 → 1 over 5000 its) | paper-text ablation |
| `spec-gaussian-indoor-c2f` | indoor | paper §3.3 | paper-text ablation |

Variants mirror the official per-scene flags of run_wo_anchor.sh (`--is_real --asg_degree 12`
for all, `--is_indoor` for room/counter/kitchen/bonsai). Benchmarked variant = official code path.

## Components
| component | kind | factory | requires → provides | config (paper/official value) |
|---|---|---|---|---|
| `spec-asg-raster` | renderer | `ss_spec_gaussian.renderer:create_renderer` | state.gaussian3d, camera.pinhole → render.rgb, render.depth.expected_z, gradient.screen.signed, gradient.screen.absolute | feat_dim 12, num_theta 2, num_phi 4, hidden 32, view_pe 2, indoor, spec_start 3000, c2f, c2f_start_scale 0.25, c2f_until 5000 |
| `spec-density` | density | `ss_spec_gaussian.density:create_density` | gradient.screen.absolute | wraps SplatStudio gsplat Density(absgrad, grow_grad2d 0.0005, opacity reset 3000), owns the SpecBank (Θ+Ψ, Adam eps 1e-15, lr linear 0.0025 → 0.000125 over 30000), spatial_lr_scale 5, bakes decoder into `spec_blob` every step |
| `photometric` / `spec-photometric` | loss | base / `ss_spec_gaussian.losses:create_photometric` | render.rgb | λ 0.2; c2f variants resize the target bilinearly to the rendered size |

## Fields
| field | shape | lr | init | note |
|---|---|---|---|---|
| `spec_feat` | (12,) | 0.0025 | 0 | per-Gaussian ASG feature f (asg_degree 12); copied by clone/split |
| `spec_blob` | () | 1e-12 (no grad) | 0 | global decoder weights packed into the first rows (HAC pattern) so ss-eval can render |

## Equations → code
- ASG basis: `asg.init_predefined_omega` (port of quaternion_utils.py), fixed buffers.
- ASG: `asg.asg_encoding`: a·relu(d·z)·exp(−softplus(λ−1)(d·x)² − softplus(μ−1)(d·y)²) (RenderingEquationEncoding).
- Θ: Linear(12 → 2·4·4); Ψ input = [κ (16), d (3), sin/cos(d·{1,2}) (12)].
- colour: `clamp_min(SH_deg(d)+0.5, 0) + c_s` for Gaussians in front of the camera, c_s = 0 at step ≤ 3000.
- absgrad densification: gsplat `absgrad=True` (per-pixel |∂L/∂μ'| per component = official fabsf kernel).

## Probes
| probe | measures | fails if |
|---|---|---|
| asg-basis | axes vs independent numpy port; orthonormal; ASG encoding vs float64 numpy; λ≠μ anisotropy changes output along x vs y | basis or anisotropy replaced (e.g. SG / SH) |
| decoder-architecture | layer shapes per variant, zero last bias, forward vs independent numpy | wrong depth/width/inputs |
| colour-composition (CUDA) | renderer RGB = rasterization of independently computed clamp(SH+0.5)+c_s; c_s off ≤3000, on after; grads reach spec_feat/MLP only after 3000 | specular branch dropped / schedule wrong |
| decoder-blob (CUDA) | pack/unpack exact; eval render from blob equals live-bank render | ss-eval would lose the decoder |
| absgrad-densify (CUDA) | density requires absolute grads with threshold 0.0005; absgrad ≥ |signed grad| and differs on opposing gradients | signed grads used |
| official-schedules | MLP lr linear decay values; means lr rescaled to 5·1.6e-4; spec_start | schedule differs |
| c2f-resolution | render sizes at steps 1/2500/5000 (c2f) or constant (off); loss resizes target | schedule missing in c2f variants |

## Fidelity gaps
1. White background (runtime convention) vs official black — small.
2. No 3DGS screen-size prune (max radius 20 px after 3000); gsplat DefaultStrategy default, same as our baseline.
3. Densify-then-step order of the official loop vs step-then-densify in the runtime (same as baseline).
4. SH degree increments at step 1001 not 1000 (runtime); negligible.
5. Random view sampling with replacement (runtime) vs shuffled stack.
6. Reflection direction / normal and the paper-text N=32/f=24/3×64 decoder are not used: the
   official real-scene code (which produced the table) does not use them. Synthetic-scene variant not ported.
7. Resolution coarse-to-fine is off in the benchmarked variants (absent from official train.py); `-c2f` variants implement it.
8. Paper's 3D-GS row (27.79) is a stronger 3DGS than ours/3DGS paper → compare deltas.
