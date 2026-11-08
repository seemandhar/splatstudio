# Plan — Sort-free GS (LC-WSR) as a native SplatStudio plugin

Method `sort-free-gs` (LC-WSR + view-dependent opacity), ablation `sort-free-gs-dir` (DIR-WSR,
w = v_i). Module `ss_sort_free_gs`, package `splatstudio-method-sort-free-gs` in `plugin/`.

## Components
| component | kind | factory | requires → provides | config |
|---|---|---|---|---|
| wsr-renderer | renderer | `ss_sort_free_gs.renderer:create_renderer` | state.gaussian3d, camera.pinhole → render.rgb, render.depth.expected_z, gradient.screen.signed | weight "lc"/"dir", u_offset 0.1, cull_nonpositive true, den_eps 1e-4, clamp_output true, tile_size 16, eps2d 0.3 |
| wsr-density | density | `ss_sort_free_gs.density:create_density` (wraps `splatstudio.backends.gsplat.Density`) | gradient.screen.signed | prune_opa 0.0 (no opacity prune), opacity_reset false, grow_grad2d 2e-4 (3DGS); vi_lr 0.01, sigma_lr 0.1→0.001 (×scene_scale, exp over 30000), wbg_lr 0.01 |
| photometric | loss | runtime (0.8 L1 + 0.2 D-SSIM) | render.rgb | unchanged |

## Fields
| field | shape | lr | init | meaning |
|---|---|---|---|---|
| opacities (base) | [] | 0.01 (set by density) | logit 0.1 | v_i = sigmoid (Eq. 10 v_i) |
| opa_sh0 | [1] | 0.05 | 0 | DC of H_i (u = 0.1 + Y(dir,H), Eq. 11) |
| opa_shN | [15] | 0.0025 | 0 | higher-order H_i |
| wsr_sigma | [] | 0.1·s → 0.001·s | 10 | **global** σ stored per Gaussian (all equal; renderer uses mean) |
| wsr_wbg | [] | 0.01 | 0.02 | **global** w_B, same trick |

Global-in-field: the runtime has no global learned fields and the checkpoint stores only
`params`. Each global is a per-Gaussian field whose rows are kept identical. The renderer reads
`p[0].detach() + (p - p.detach()).sum()` (exact value; every row gets the full scalar gradient g;
the `mean*N - mean*(N-1)` form was rejected after it quantised σ in float32, smoke-2). (The
original plain `mean()` gave g/N(t), which is NOT scalar-Adam-equivalent while N changes;
see review-1 M1.) After every step the density copies the value and Adam moments of an existing
row to all rows. The probe `recipe` checks equality with a scalar Adam across clone/prune and
fails for the plain mean.

## Renderer (custom CUDA, JIT)
1. gsplat `fully_fused_projection` (opacities=None → 3.33σ per-axis radii, eps2d 0.3,
   near 0.01) → means2d, conics, depths (view z, differentiable).
2. u_i = 0.1 + Y_deg(dir_i; H_i) (torch, 3DGS SH constants), v_i = sigmoid(opacities),
   σ = mean(wsr_sigma), w_i = max(0, 1 − d_i/σ)·v_i (LC) or v_i (DIR).
3. Cull (radii=0): u ≤ 1/255 or w ≤ 0 (no contribution possible) — speed only, exact.
4. Colours: gsplat SH (+0.5, clamp ≥0); feature = [rgb, depth].
5. `isect_tiles` with **zero depths** (tile binning only, no depth ordering — sort-free) .
6. `csrc/wsr_raster.cu` (adapted from gsplat 1.5.3 RasterizeToPixels3DGS{Fwd,Bwd}.cu via
   reproductions/linear-splatting): per pixel S_f = Σ_i f_i α_i w_i, S_w = Σ α_i w_i, with
   α_i = u_i G_i, G = exp(−½ΔᵀconicΔ), skip α < 1/255 or ½ΔᵀconicΔ > ½·3.33²; no T, no early
   stop, no α clamp. Backward: s = f·v_Sf + v_Sw; v_f = α w v_Sf; v_u = G w s; v_w = α s;
   v_sigma = −u G w s → conic/means2d grads as gsplat.
7. Torch: C = (S_rgb + w_B c_B)/clamp(S_w + w_B, 1e-4), c_B = white; clamp [0,1] (rebuttal);
   depth = S_d/S_w; alpha = S_w/(S_w+w_B).

## Probes
| probe | contribution | fails if |
|---|---|---|
| wsr-forward-vs-dense | C1 | CUDA (S_rgb, S_w, final C) ≠ float64 dense torch reference; or order dependence: permuting Gaussians changes the image (> 1e-5); or equals alpha compositing |
| wsr-backward-vs-dense | C1 | CUDA grads (means2d, conics, colours, u, w) ≠ autograd of dense reference |
| lc-weight | C2 | renderer's per-Gaussian weight ≠ max(0,1−d/σ)v (independent formula; d ≥ σ ⇒ 0); σ and w_B receive nonzero grads; DIR ablation weight is depth-independent |
| view-dependent-opacity | C3 | u from renderer ≠ independent SH (gsplat spherical_harmonics on a padded tensor); u differs between two view directions for nonzero H; u unclamped (>1 reachable) |
| recipe | C4 | density has prune_opa 0, no reset, vi lr 0.01; σ lr schedule at step 1/15000/30000; global fields stay synchronised after a duplicate/split with moments |
| base probes | | state-transitions, photometric-gradient, renderer-gradient |

## Fidelity gaps
1. No official code; u/v roles, lrs, inits from the unofficial reimpl (only the product enters).
2. Non-positive u culled (paper: unconstrained) — unofficial does the same.
3. α threshold on u·G (unofficial: on u·G·w) and 3.33σ footprint (unofficial 3σ).
4. Keep 3DGS world-size prune (paper wording), unofficial removes all pruning.
5. c_B = black (renderer `background` 0.0; paper/3DGS/unofficial), not the runtime's white convention.
9. Culled Gaussians (u ≤ 1/255 or w ≤ 0) never get gradients again and are never pruned (as unofficial). `cull_nonpositive=False` is unsafe (denominator clamp maps negative values to 1e-4) and is not used.
6. EXP-WSR not implemented (averages only, not the reported per-scene variant).
7. Runtime protocol (random view sampling, SSIM valid conv, PIL resize etc.) — compare Δ vs
   our gsplat baseline: garden 26.89 (paper 3DGS 27.41, paper WSR 27.78: Δ +0.37),
   truck 25.21 (paper 25.18 → 25.28: Δ +0.10).
8. Means2d gradient magnitudes differ from alpha compositing (normalised sums) — same grad
   threshold as paper, so densification count may differ (paper: 2.88M vs 3.98M avg).

## Benchmark
Per-scene reported. Changed after review-1 M2 to `mipnerf360: bicycle` (24.20, Δ −1.05 vs paper 3DGS)
+ `tandt: train` (21.93, Δ +0.84), 30k steps, main method only, vs `baseline3dgs-*-30000`.
