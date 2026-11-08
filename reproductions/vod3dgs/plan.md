# VoD-3DGS native plan

Methods: `vod3dgs-s` (reset keeps smallest eigenvalue, benchmark target) and `vod3dgs-l`
(largest). Module `ss_vod3dgs`. No official code → implemented from the paper (paper.md).

## Components
| kind | name | factory | requires / provides | config (paper value) |
|---|---|---|---|---|
| renderer | vod-renderer | `ss_vod3dgs.renderer:create_renderer` | state.gaussian3d, camera.pinhole / rgb, depth.expected_z, gradient.screen.signed | – (C1, Eq. 9) |
| density | vod-density | `ss_vod3dgs.density:create_density` | gradient.screen.signed | DefaultStrategy 3DGS defaults (prune_opa 0.005, grow_grad2d 2e-4, refine 500–15000/100, reset 3000) + `variant` S/L (C3), `opacity_lr` 0.0125 (C5), max-view prune (C2) |
| loss | photometric | splatstudio.losses | render.rgb | ssim_weight 0.2 (unchanged) |
| loss | vod-view-consistency | `ss_vod3dgs.loss:create_vc_loss` | render.rgb | weight 1.0, view_sampling `covisibility`, covis_dir package data (C4, Eq. 12) |

## Fields
| field | shape | lr | init | density ops |
|---|---|---|---|---|
| `vod_S` | (6,) [Sxx,Sxy,Sxz,Syy,Syz,Szz] | 0.0125 (lr_γ/4) | 0 | copied by gsplat duplicate/split, removed by prune; eigen-reset in place (`.data`) at each opacity reset, Adam moments zeroed |

## Implementation notes
- Eq. 9: renderer computes cam center c = -Rᵀt from viewmat, ω = normalize(c − μ),
  q = ωᵀŜω (6-param expansion, off-diagonals ×2), opacity = σ(γ + q). Outputs `camera_center`.
- Eq. 10: `VoDStrategy(DefaultStrategy)._prune_gs` replaces σ(γ) by σ(γ + max_j q_j) over all
  training cameras (registered by `set_training_cameras`); too-big prune unchanged.
- Eq. 11: at every 3DGS opacity reset step (3000, 6000, …, <15000; runtime reset fix) after
  γ reset, Ŝ ← λ v vᵀ with λ = min (S) or max (L) eigenvalue (`torch.linalg.eigh`), chunked.
- Eq. 12: loss gets c_i from outputs, finds training index i (nearest training center), samples
  j ≠ i with p ∝ covisibility(i, j) (COLMAP shared 3D points, `covis.py`; LightGlue in paper),
  computes mean over all Gaussians of max(ω_i·ω_j, 0)(α̂_i − α̂_j)²; means detached.
- C5: density.initialize sets the opacity optimizer lr to 0.0125; vod_S FieldSpec lr 0.0125.
- Trainability guard: density.after_backward asserts every param is a leaf with requires_grad.

## Probes
| probe | checks | fails if |
|---|---|---|
| vod-opacity (CUDA) | renderer opacity == σ(γ + ωᵀSω) with an independently built 3×3 matrix; render changes when S≠0; grad reaches vod_S; sign-invariance | S ignored / wrong quadratic form |
| max-view-prune | Gaussian with low γ but high opacity from one train view survives, S=0 one is pruned | prune uses σ(γ) |
| eigen-reset | S/L variants match independent rank-1 projection, moments zeroed, field trainable, only at reset steps | reset missing / wrong eigenvalue |
| vc-loss | value == independent per-Gaussian Eq. 12 incl. cos clamp; grad reaches vod_S/opacities not means; sampling freq ∝ covisibility | loss/sampling wrong |
| opacity-lr | opacities/vod_S lr == 0.05/4 after initialize | lr not changed |

## Unchanged from 3DGS
Photometric loss, SH schedule, ADC thresholds, γ reset, means LR schedule, init.

## Fidelity gaps
1. Covisibility from SfM instead of LightGlue matches (paper-sanctioned alternative). Effect: small
   (ablation: keypoints vs uniform +0.01 dB on NeRF-Synthetic). Closable with LightGlue weights.
2. γ also reset at opacity reset (paper ambiguous). Effect unknown, likely small.
3. L_vc means detached; Ŝ moments zeroed at reset (unspecified).
4. Runtime differences vs Inria 3DGS (white bg, sampling with replacement, gsplat backward
   alpha>0.999 gradient drop); compare gains over our gsplat baseline vs paper gain over 3DGS†.
5. gsplat rasterizer instead of Inria's; no custom CUDA needed (opacity is precomputed per view).
6. Benchmark on averages only (no per-scene numbers): tandt (2) + db (2) complete averages.
