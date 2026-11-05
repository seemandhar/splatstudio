# SparseGS native plan

Method `sparsegs`, module `ss_sparsegs`, package `splatstudio-method-sparsegs`.
Dataset: new `mipnerf360-sgs12` (`~/datasets/mipnerf360/sparsegs12/<scene>`, built by
`tools/prepare_m360_sgs12.py`; 6 released scenes, authors' 12 training views, 1/4 res, MVS init,
BMD depth sidecar `sgs_depth.npz`). Steps 10000.

## Components
| component | kind | factory | requires → provides | config (paper value) |
|---|---|---|---|---|
| sparsegs-renderer | renderer | `ss_sparsegs.renderer:create_renderer` | state.gaussian3d, camera.pinhole → render.rgb, gradient.screen.signed, render.depth.softmax, render.depth.alpha, render.depth.mode | beta 5, near_plane 0.2 (Inria in_frustum), background 0 (black) |
| sparsegs-density | density | `ss_sparsegs.density:create_density` | gradient.screen.signed | Inria density (grad 2e-4, 500..18000/100, percent_dense 0.01, min_opacity **0.05** (SparseGS), max_screen 20 after 3000, extent 0.1, reset to 0.01 every 3000), means lr expon 1.6e-4→1.6e-6 over 30000; warp GT at 4999; floater prune at `prune_step` (=total 10000) a=0.97, b=7.5, min 80, power_thresh -4, mode_range official |
| photometric | loss | `splatstudio.losses:create_photometric` | render.rgb | ssim_weight 0.2 |
| sparsegs-pearson | loss | `ss_sparsegs.losses:create_pearson` | render.depth.softmax | global 0.15, local 0.15, patch 64, p_corr 0.5 |
| sparsegs-warp | loss | `ss_sparsegs.losses:create_warp` | render.rgb | weight 0.05, start 4999, prob 0.4, angles [-2.5, 2.5] |
| sparsegs-sds | loss | `ss_sparsegs.losses:create_sds` | render.rgb | weight 5e-4, start_frac 2/3, freq 0.1, step_ratio 0.99, guidance 100, SD2.1-base fp16 |

Shared per-run context `ss_sparsegs.context.CTX` (set by the density in `initialize` /
`set_training_cameras`): training cameras, depth table, warp cameras/targets, the per-step
random draws (`draws(step)`: SDS draw `randint(1,100) <= 100*freq` then warp draw
`randint(1,10) <= 4`, official order), warp render info for densification. `total_steps`
config key on density/losses (default 10000).

## Fields
None (base 3DGS fields only).

## C1 renderer (custom CUDA, `csrc/sgs_depth.cu`, JIT `torch.utils.cpp_extension.load`)
gsplat `rasterization` (classic, packed=False) renders RGB; its `info` (means2d, conics,
opacities, depths, isect_offsets, flatten_ids — all in the autograd graph) feeds a second
rasterizer pass written after the official forward.cu/backward.cu: per pixel front-to-back,
α = min(0.99, o·exp(power)), skip α<1/255, stop at T<1e-4;
`D += d w e^{β(w-0.3)}`, `Z += w e^{β(w-0.3)}` → softmax depth `log(D/Z+1e-8)`;
alpha depth `Σwd/Σw`; mode depth/index; first-contributor index (paper and bug-compatible
official index). Backward: exact derivative (supp. §A) computed front-to-back with stored
totals Σ w·e(1+βw)·d and Σ w·e(1+βw) (suffix = total − prefix), gradients to means2d, conics,
opacities, depths (→ means via gsplat projection backward).

## C2 Pearson (Eq. 6, official `loss_utils.py`)
`1 - mean(z(a)·z(b))` with `z(x)=(x-mean)/(std+1e-6)` (unbiased std); global on full maps;
local: `n=int(p_corr·⌊H/S⌋·⌊W/S⌋)` random S×S patches (`randint(0,H-S)`), mean. Skipped on
warp iterations. Target = sidecar depth of the frame (lookup by viewmat).

## C3 SDS
Ellipse pose (port of `setup_ellipse_sampling`, `getRandEllipsePose(idx, 0, 0)`), rotation
orthonormalised. Render RGB, bilinear 512², VAE encode (sample×0.18215, fp16, grads), fixed t,
CFG 100 with empty-prompt embeddings, loss `0.5·Σ(lat − (lat − w(ε̂−ε)).detach())²`.
Loaded lazily from `~/models/sd21base` with diffusers 0.30.3 from `~/models/sd21base/pylib`.

## C4 warp
Cameras: `R_c2w' = M(Ȳ,θ) R_c2w`, T kept (official `rotate_camera_poses`); GT: at step
`start` (4999) render alpha depth of each training view → min/max; forward-warp the training
image with depth `d_mono·(max−min)+min`, nearest (banker's) rounding, last-write-wins, mask.
Loss on warp iterations (step > start, p=0.4, stack without replacement):
`w·(0.8·Σ|I−Î|M/ΣM + 0.2·(1−SSIM_inria(I,Î)))`. On warp iterations the density uses the warp
render's screen gradients/radii instead of the training render's (official).

## C5 floater pruning
At `prune_step` after the optimizer step: per training view, alpha/mode depth + mode range;
`diff = (dα − dmode)/(dα + max(min dα, mean dα − 3·std dα))`; `D̄ = mean dipstat(diff[diff>0])`
(diptest); `perc = max(80, 100a·e^{−bD̄})`; mask `diff > percentile(diff[diff>0], perc)`,
pixels with mode −1 dropped; CUDA kernel marks every flatten-list entry in
[start, end] with power > −4 at the pixel; union over views → `gsplat.strategy.ops.remove`.

## Probes
| probe | contribution | measures / fails when |
|---|---|---|
| softmax-depth (CUDA) | C1 | kernel vs independent per-pixel torch reference (softmax, alpha, mode depth; grads w.r.t. means2d/conics/opacities/depths via autograd); β→0 equals log alpha depth; fails if depth = alpha depth or grads wrong |
| pearson | C2 | 0 for affine-related maps, 2 for negated, equals 1−torch.corrcoef, patch count formula |
| warp | C4 | identity warp reproduces image/mask; known shift for a pure camera translation; rotation equals Rodrigues (scipy) with official sign; masked loss formula |
| sds | C3 | stub UNet/VAE: d loss/d latents = w(ε̂−ε), fixed t from step_ratio, ellipse views look at centre within z bounds |
| floater-prune (CUDA) | C5 | synthetic floater in front of a wall: floater removed, wall kept (paper range); official bug index formula; percentile formula |
| state-transitions, photometric-gradient | base | built-ins |

## Unchanged from 3DGS
Photometric loss (0.8 L1 + 0.2 D-SSIM), Adam lrs, SH schedule, densification thresholds.

## Fidelity gaps
1. Split/test: authors' released 12 views (6 scenes, no room) ≠ paper's "7 scenes, evenly
   sampled"; test = llffhold8 minus train; official metric set (all non-train) = test ∪ val.
2. RGB rasterizer is gsplat (alpha clamp 0.999 + gsplat backward drops opacity/geometry
   gradients where opac·G > 0.999, see TODO-runtime.md; Inria 0.99) while depth uses the
   official 0.99; small effect.
3. Runtime samples views with replacement (Inria: without-replacement stack); runtime SSIM uses
   valid convolution (photometric) — shared with the gsplat baseline.
4. Optimizer step happens before density ops (Inria: after); negligible.
5. Ellipse SDS camera rotation orthonormalised (official keeps the PCA scale, only changing the
   near-plane culling depth).
6. SD 2.1-base from the `sd2-community` mirror (stabilityai repo removed; sha256 identical to a
   second mirror).
7. Paper vs README hyper-parameters (see paper.md §3); paper values used.
8. BMD 16-bit PNG output instead of float npy (min-max normalised afterwards).


## Review-1 resolution
- Major (prune step): default restored to the paper (after training, step 10000). `sparsegs-tuned6667` = TUNED variant, reported only with that label.
- Headline metric: per-view mean pooled over test ∪ validation (the official metrics.py set); test-only also reported.
- Density RNG now mixes the config seed with the run seed.
- Resolution ambiguity (paper 1/4 vs README -r 2 / 1600-px render) recorded as a fidelity gap.
