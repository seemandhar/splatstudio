# SteepGS — native SplatStudio plan

Method `steepgs`, module `ss_steepgs`, package `splatstudio-method-steepgs`.

## Architecture
The splitting statistic needs Σ_pixels dL/dG per Gaussian and view (dL/dG = o·dL/dα). gsplat's
backward exposes only G-weighted sums (dL/do = Σ G·dL/dα, dL/dconic, dL/dmeans2d) and drops
gradients where oG > 0.999, so the value cannot be recovered from gsplat. We therefore vendor
the official SteepGS rasterizer (`submodules/diff-gaussian-rasterization`, Inria 3DGS + splitting
matrix accumulation in `backward.cu`) **unmodified** in `ss_steepgs/csrc/` (Gaussian-Splatting
licence, `csrc/LICENSE.md`), glm from `reproductions/gof` (same upstream glm), JIT-built with
`torch.utils.cpp_extension.load` via `tools/ssenv`. This also gives the exact Inria
NDC-space screen gradient (no W/2,H/2 rescaling) and the 0.99 alpha clamp with full backward.

## Components
| kind | name / factory | requires | provides | config (paper/code value) |
|---|---|---|---|---|
| renderer | `steepgs-renderer` `ss_steepgs.renderer:create_renderer` | state.gaussian3d, camera.pinhole | render.rgb, gradient.screen.signed, gradient.splitting.inv_cov | background [0,0,0] (official default), S_estimator `inv_cov` |
| density | `steepgs-density` `ss_steepgs.density:create_density` | gradient.screen.signed, gradient.splitting.inv_cov | – | densify_from 500, densify_until 15000, every 100, grad_threshold 2e-4, S_threshold −1e-6, percent_dense 0.01, min_opacity 0.005, big_world 0.1 after 3000, reset_every 3000 (value 0.01), options [] (`stationary`,`no_uncertain`,`no_eig_cond`,`no_saddle`,`no_eig_upd`,`zero_upd`,`no_div_opacity`), min_grad 1e-4 |
| loss | `photometric` (base) | render.rgb | – | 0.8 L1 + 0.2 D-SSIM (unchanged) |

No extra FieldSpecs (statistics are density-internal, reset at every densification as in
`densification_postfix`).

## Implementation notes (equations → code)
- renderer: Inria raster settings from K/viewmat (as `reproductions/gof/plugin/ss_gof/raster.py`,
  znear 0.01 / zfar 100, centred principal point check). Inputs: means, means2D (zeros, retain
  grad), sh=cat(sh0,shN), sigmoid(opacity)[:,None], exp(scales), normalize(quats), and a zero
  `splitting_mats [N,3,3]` leaf with requires_grad whose `.grad` is the CUDA buffer; returns
  rgb HWC, `info = {means2d, splitting_mats, radii}`. Depth is not produced (method needs none).
- density.before_backward: retain grads; store Σ^{-1} = R diag(1/(s+1e-8))² R^T of the
  *pre-step* params (official computes it in `add_densification_stats` before optimizer.step).
- density.after_backward (runtime calls it after the optimizer step; official order is
  stats → densify → reset → step): if step < densify_until: visible = radii>0;
  vgn += ‖grad(means2D)[:, :2]‖; xyz-grad stat only for `stationary` (captured with a hook on
  means' gradient in before_backward); S += grad(splitting)[:,0,0] · Σ^{-1}; denom += 1.
  If step > 500 and step % 100 == 0: `densify_and_split_steepest` exactly
  (gaussian_model.py:530-644): eigh (ascending) of S/denom → λ_min, v_min; selected = vgn/denom
  ≥ 2e-4 ∧ λ_min < −1e-6; split = selected ∧ max s > 0.01·extent; clone = selected ∧ ¬big;
  clone: copy at old + lr·v and in-place shift original by lr·v (lr = current means LR, the
  runtime has already set the LR for step+1 = official lr(step)); split: offsets =
  v.repeat(2,1) · ‖N(0,diag s)‖ with sign + on rows 0::2 and − on rows 1::2 of the repeated
  array (official quirk), scale s/1.6, opacity copied (halved only with `no_div_opacity`),
  originals removed; then prune o<0.005 ∨ (step>3000 ∧ max s>0.1·extent); reset stats.
  If step % 3000 == 0: `reset_opa(0.01)` (moments zeroed, as `replace_tensor_to_optimizer`).
- Adam state: appended Gaussians get zero moments (official `cat_tensors_to_optimizer`),
  removal by mask; implemented with `gsplat.strategy.ops._update_param_with_optimizer` / `remove`.

## Probes
| probe | contribution | check (fails if mechanism removed) |
|---|---|---|
| `steepgs-dLdG` (CUDA) | C1 | vendored kernel's splitting[:,0,0] equals an independent PyTorch alpha-compositing re-implementation of Σ_pix o·dL/dα on a tiny scene (brute-force per-pixel, Inria conventions) |
| `steepgs-splitting-matrix` | C1 | density accumulates c·Σ^{-1} averaged over views (independent formula); eigen pair matches analytic c/s_min² and shortest axis |
| `steepgs-saddle-criterion` | C2 | Gaussians with negative / positive c and high / low screen grad: only (neg, high) densified; disabling eig condition changes selection |
| `steepgs-split-direction` | C3 | split offspring lie on the line ±v_min through the parent at distance ‖N(0,s)‖-like scale, sign pattern per official repeat quirk; clone shifted by lr·v_min |
| `steepgs-schedule` | cadence | densify only at >500, %100, <15000; opacity reset at 3000; screen prune inert / world prune after 3000 |

## Fidelity gaps
1. Runtime order: optimizer step precedes density (official densifies before the step and the
   step at densification iterations skips new tensors). Same as every native port; small.
2. Random-with-replacement view sampling; SSIM valid conv; SH degree switches one step later
   (runtime `(step-1)//1000`). Small.
3. LPIPS from `lpips` package; evaluation via ss-eval. Small.
4. Initialization: runtime 3DGS recipe (3-NN scales, opacity 0.1) ≈ official.
5. Background black (official) vs runtime convention white — only uncovered pixels.
6. Paper-vs-code: opacity halving off (code default), clone without displacement asymmetry,
   split sign quirk — all follow code, recorded here.

## Benchmark choice
Per-scene numbers exist (supp. Tables 6-9). Protocol = official `full_eval.py`: images_4
outdoor / images_2 indoor → native `mipnerf360` dataset, baseline
`baseline3dgs-mipnerf360-30000`. Scenes: garden (outdoor, paper 27.159 / 2.195M pts) and
counter (indoor, 28.737 / 0.592M). Compare PSNR per scene and Gaussian count vs baseline
(paper: SteepGS − 3DGS = −0.26 dB garden, −0.27 dB counter; counts ×0.38, ×0.49).
