# BAGS native plan

Method family `bags` / `bags-<scene>`, module `ss_bags`, package `splatstudio-method-bags`
in `plugin/`. No custom CUDA: gsplat (antialiased mode, eps2d 0.1) + Mip 3D filter in torch;
the spatially varying convolution is the official `nn.Unfold` formulation.

## Data / protocol decision
BAGS's README links the original Deblur-NeRF Drive folder (`1_TkpcJnw…`, original `sparse/0`
COLMAP) — the same source as our `deblurnerf-motion` / `deblurnerf-real` sets, not the
Deblurring-3DGS re-release (`deblur3dgs-*`, re-run COLMAP with denser points). So we use
`deblurnerf-*`: 600×400 PIL /4, sorted idx % hold == 0 → test, holds equal the official
`llffhold` per scene (ball 7, caps 5). Benchmark scenes: **ball** (motion, 27.68 dB) and
**caps** (defocus, 24.16 dB), both downloaded. Registry untouched.

## Architecture
State: `ss_bags.bank.BlurBank` (nn.Module: multi-scale BPN, optional single-scale BPN, own
Adam lr 5e-4 eps 1e-15, training viewmats for view lookup, `step`), created by the density
component in `set_training_cameras`, published as `ss_bags.bank.ACTIVE`, stepped in
`density.after_backward`, exported into `density.state["bank"]` (checkpoint). Evaluation
never needs it (test renders are sharp).

| component | kind | factory | requires → provides | config |
|---|---|---|---|---|
| `bags-raster` | renderer | `ss_bags.renderer:create_renderer` | state.gaussian3d, camera.pinhole → render.rgb, render.depth.ray_distance, render.multiscale, render.blur_view, render.antialiased_2d, render.filter_3d, gradient.screen.signed | kernel_size 0.1, background black, upsample_steps [3000, 6000] |
| `bags-density` | density | `ss_bags.density:create_density` | gradient.screen.signed | per-scene init_dgt, ms_steps, total_steps, min_opacity 0.1, dgt 2e-4, densify 500/35000/100, reset 3000, percent_dense 0.01, xyz lr 1.6e-4→1.6e-6 (60000, restarts at ms_steps), BPN lr, kernel sizes, use_another_mlp, upsample_steps |
| `bags-objective` | loss | `ss_bags.losses:create_objective` | render.rgb, render.multiscale, render.blur_view | λ_dssim 0.2, bpn_start 250, ms_steps (shared) |
| `bags-mask` | loss | `ss_bags.losses:create_mask_loss` | same | λ_mask 0.001 |
| `bags-depth-tv` | loss | `…:create_depth_tv` | + render.depth.ray_distance | use, α (0.001 / 0.01) |
| `bags-rgb-tv` | loss | `…:create_rgb_tv` | same | use, α 0.001 |

The base `photometric` loss is **replaced** (official loss is on the BPN output at the current
scale; for it ≤ 250 on the sharp render — same 0.8 L1 + 0.2 D-SSIM with same-padded SSIM).

Renderer: if grad is enabled and the camera matches a bank training view → render at scale
1/4 (step < 3000), 1/2 (< 6000), 1 otherwise (K and W/H scaled; 600×400 → 150×100, 300×200),
returning `view`, `scale`; else full-res sharp render (test/validation, ss-eval). Depth =
alpha-blended Euclidean distance (official rasterizer) via a 4th colour channel with gsplat
SH evaluated in torch. Loss resizes the target with antialiased bicubic to the render size.

## Fields
| field | shape | lr | notes |
|---|---|---|---|
| filter_3d | () | 1e-8 (never gets grad) | Mip 3D filter, from training cameras at the current scale |

## Probes
| probe | contribution | fails if |
|---|---|---|
| spatial-conv-reference | C1 | unfold conv ≠ independent float64 shift-and-add loop (values and autograd grads w.r.t. image and kernels), impulse offset orientation, delta kernel ≠ identity, gradcheck fails |
| bpn-architecture | C1 | layer shapes/no-bias/softmax/sigmoid/embedding/PE differ from the official net; mask blend formula; input detached |
| objective-terms | C1 | photometric uses sharp image when BPN active; mask/TV values ≠ independent numpy; BPN not used for it ≤ 250 |
| multiscale-schedule (cuda) | C2 | render resolution / kernel size / head per step wrong (≤2999 →150×100 K5, 3000 → 300×200 K9, 6000 → 600×400 K17); test views not sharp full-res |
| phase-schedule | C2/base | xyz LR restart, dgt switch, densify/reset/filter steps, final-step skip, ss-net switch differ from the official loop |
| official-config | base | per-scene variant values ≠ official scripts |
| density-recipe (cuda) | base | clone/split/prune(0.1)/filter-aware reset vs independent formula |
| state-transitions | base | gsplat ops alignment |

## Fidelity gaps
1. Coarse targets are resized from 600×400 (official: PIL from 2400×1600) — small.
2. Runtime samples views with replacement — official also (randint, no pop): no gap.
3. Runtime Adam step precedes densification (official skips the Gaussian Adam step on
   densify/reset steps) — negligible.
4. SH degree raised one step later — negligible.
5. Metrics: runtime ss-eval = VGG LPIPS + valid-conv Gaussian SSIM; `ss_bags.evaluate` gives the
   official Deblur-NeRF metrics (8-bit PNG, PSNR, skimage-equivalent SSIM 7×7, LPIPS-Alex).
6. Per-scene official flags are used (code overrides paper's "40k overall").
7. No shared baseline on deblurnerf-*: gsplat baseline must be run on ball/caps; paper's
   comparison row is Mip-Splatting (ball 23.22, caps 21.00).
8. Mixed-resolution Mip-NeRF 360 and drone experiments not reproduced (data protocol).
