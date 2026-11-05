# Mini-Splatting — native SplatStudio plan

Methods (one plugin `ss_mini_splatting`, package `splatstudio-method-mini-splatting`):
- `mini-splatting` — Mini-Splatting, `imp_metric=outdoor` (M360 outdoor scenes, T&T)
- `mini-splatting-indoor` — Mini-Splatting, `imp_metric=indoor` (M360 indoor scenes, DB)
- `mini-splatting-d` — Mini-Splatting-D (densification only)

The paper picks the importance metric per scene class (README); the runtime has no per-scene
config, so the two metrics are two registered names with otherwise identical specs.

## gsplat 1.5.3 capability check
gsplat's `rasterization` returns per-Gaussian `radii`, `means2d`, `conics`, `opacities` and the
tile intersection lists (`isect_offsets`, `flatten_ids`) but no per-Gaussian blending-weight
sums, pixel counts, argmax-weight counts or per-pixel argmax ids. `rasterize_to_indices_in_range`
can enumerate (gaussian, pixel) pairs, but only in Python-driven transmittance batches (far too
slow per training step and memory heavy for full views). → custom CUDA extension
(`csrc/ms_raster.cu`) that re-traverses gsplat's own tile lists with gsplat's exact alpha
numerics (0.999 clamp, 1/255 threshold, T ≤ 1e-4 stop, pixel centers +0.5) and accumulates the
official `render_imp` statistics (`accum_weights`, `area_proj`, `area_max`) and the official
`render_depthCUDA` outputs (mid-point of the argmax Gaussian, transmittance). Adapted from
`submodules/diff-gaussian-rasterization_ms/cuda_rasterizer/forward.cu` (Inria 3DGS license,
research use). simple-knn's `distCUDA2` (same license) is vendored for reinit scales — the
runtime's exact cdist k-NN is O(N²) and infeasible for 3.5M points.

## Components
| component | kind | factory | requires | provides | notes |
|---|---|---|---|---|---|
| ms-renderer | renderer | `ss_mini_splatting.renderer:create_renderer` | state.gaussian3d, camera.pinhole | render.rgb, render.depth.expected_z, gradient.screen.signed, training.target_image | stock gsplat Renderer + passes the frame's target image through `outputs["target_image"]`/`outputs["viewmat"]` (needed: depth reinit colors points with GT pixels) |
| ms-density | density | `ss_mini_splatting.density:create_density` | gradient.screen.signed, training.target_image | – | C1–C5; uses `set_training_cameras` |
| photometric | loss | splatstudio.losses:create_photometric | render.rgb | – | unchanged L1 + 0.2 D-SSIM |

No new FieldSpecs.

## Density (C1–C5) mapping
- per step < 15000: accumulate 3DGS grad stats (gsplat means2d.grad × (W/2,H/2), visible =
  radii>0) and blur mask from `area_max` of the stats kernel run on the step's own `info`.
- densify (step>500, %100==0, %5000≠0, N<num_max): gsplat `ops.duplicate` (clone), `ops.split`
  (grad∧big ∪ blur), `ops.remove` (opacity<0.005; after 3000 max-scale>0.1·extent).
- depth reinit at 5000/10000 over all training cameras (depth kernel + GT colors cached from
  `outputs["target_image"]` keyed by viewmat); replace all fields, fresh Adam.
- 15000: importance over all views, intersection, A-ES weighted sampling without replacement
  (≡ numpy `choice(replace=False, p)`), reinit from centers (DC colors kept). MS-D: reinit only.
- 20000: importance, intersection, CDF-1% prune, fresh Adam.
- SH: shN coefficients above the official active degree are zeroed together with their Adam
  moments after every step (exactly equivalent to the official 0-degree model + oneupSHdegree).
- LR: means LR overwritten after every step with the official `lr(iter)` / `lr(iter-10000)`.

## Probes
| probe | measures | fails if |
|---|---|---|
| ms-stats-kernel (CUDA) | kernel stats vs. brute-force per-pixel torch loop on a synthetic scene; alpha vs gsplat | kernel or numerics wrong |
| ms-depth-midpoint (CUDA) | kernel mid-points vs. independent t* = argmax of Gaussian density along ray (Σ⁻¹ form) | depth formula wrong / mid-point removed |
| ms-blur-split (CUDA) | a zero-gradient Gaussian covering the view is split by the density; not split with θ_blur→∞ | blur split removed |
| ms-depth-reinit (CUDA) | reinit replaces the population with ≈num_depth points on the depth surface, GT colors, opacity 0.1 | reinit removed / colors not GT |
| ms-simplification (CUDA) | occluded Gaussian (never argmax) gets zero importance and is dropped; kept count = int(0.5·nnz); 20k CDF prune | intersection or sampling removed |
| ms-schedule | LR at 15001 equals lr(6001); SH masking by step | schedule removed |
+ base state/photometric/renderer probes.

## Fidelity gaps
1. White background (runtime) vs black in official M360/T&T/DB — affects depth-collapse
   analysis and backgrounds; expected small (<0.1 dB).
2. Random-with-replacement view sampling vs shuffled stack; stack reset after reinit not
   reproducible. Small.
3. Optimizer step happens before density control (runtime order); 3DGS skips the Adam step for
   tensors replaced at densify iterations. Negligible.
4. gsplat rasterization numerics (0.999 alpha clamp, opacity-aware tile bounds, eps2d 0.3
   identical) vs 3DGS rasterizer (0.99 clamp). Stats kernels follow gsplat to be consistent with
   the trained model.
5. Initial SfM 3-NN scales use the runtime's exact k-NN; reinit scales use the official
   approximate simple-knn. Negligible.
6. Per-scene importance metric chosen by method name (two names).
7. LPIPS/SSIM implementations per CLAUDE.md calibration.
