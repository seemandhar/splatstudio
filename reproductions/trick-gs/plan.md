# Trick-GS native plan

Method `trick-gs`, module `ss_trick_gs`, package `splatstudio-method-trick-gs`
(`plugin/`). Headline configuration = all eight tricks (Table 3 caption). No official code;
each trick follows the official code of its source paper (ports reused/cited below).

## Shared run state
`ss_trick_gs/shared.py: STATE` (pattern of the LightGaussian port): the density activates it in
`initialize` and sets `next_step` after each `after_backward`; the renderer reads it for
gradient-enabled (training) renders. Inactive (ss-eval / probes without a run) ⇒ full
resolution, eps2d 0.3, masks applied — i.e. the final model.

## Components
| kind | name / factory | requires → provides | config (paper values) |
|---|---|---|---|
| renderer | `trick-renderer` `ss_trick_gs.renderer:create_renderer` | state.gaussian3d, camera.pinhole → render.rgb, render.depth.expected_z, gradient.screen.signed, render.trick_schedule | `gs_threshold 0.05`, `sh_threshold 0.1` |
| density | `trick-density` `ss_trick_gs.density:create_density` | gradient.screen.signed | schedule (below) |
| loss | `trick-photometric` `ss_trick_gs.losses:create_photometric` | render.rgb, render.trick_schedule | `ssim_weight 0.2`, separable same-padded SSIM (AT) |
| loss | `gaussian-mask` `ss_trick_gs.losses:create_gaussian_mask` | render.trick_schedule | `weight 0.05` (λ_m) |
| loss | `sh-mask` `ss_trick_gs.losses:create_sh_mask` | render.trick_schedule | `weight 0.05` (λ_sh) |

Density config: `total_steps 30000, densify_from 500, densify_until 15000, densify_every 100,
grad_threshold 2e-4, percent_dense 0.01, split_divisor 1.6, prune_opacity 0.005,
prune_big_world 0.1, prune_big_after 3000, reset_every 3000, reset_value 0.01, abe_until 10000,
abe_factor 0.3, lowpass_until 10000, lowpass_every 1000, lowpass_min 0.3, lowpass_max 300,
init_fraction 0.2, sfm_error_dir <workspace>/sfm_errors, ds_start 0.125, ds_until 19500,
ds_mode linear, blur_radius 4, blur_sigma 2.4, blur_every 100, blur_until 19500,
mask_prune_every 500, sh_mask_start 3500, sg_steps [5000,10000,15000,20000,22000,23000],
sg_first 0.6, sg_decay 0.7, sg_v_pow 0.1, redensify [20000, 20500], sh_update_every 16,
fp16_final true`.

## Fields
| name | shape | lr | init | density ops |
|---|---|---|---|---|
| `trick_mask` | () | 0.5 | 1.0 (Compact-3DGS) | copied on clone/split/ABE |
| `trick_sh_mask` | (3,) | 0.05 | 0.0 (RDO) | copied |

## Contribution → code
- C1 GM: renderer Eq. 3 STE on σ(m) > 0.05, multiplies activated scale and opacity; loss
  `mean σ(m)`; density prunes σ(m) ≤ ε at densify steps and every 500 its after 15000.
- C2 SG: `significance.py` (LightGaussian CUDA count kernel, copied from
  `reproductions/lightgaussian`, renamed extension), all training cameras at the current DS scale
  and low-pass, masked opacity, `(V/V_90)^0.1`, prune `v ≤ sorted(v)[int(p(N−1))]`,
  p_k = 0.6·0.7^k at `sg_steps`.
- C3 SHM: renderer Eq. 5 STE per band (3/5/7 coeffs) from `sh_mask_start`; loss RDO Eq. 8;
  final step zeroes masked bands (post-training band removal).
- C4 BL: `schedule.blur_params(step, scale)`; loss blurs the (downsampled) target, separable
  Gaussian, reflect padding (torchvision GaussianBlur numerics).
- C5 DS: `schedule.ds_size(step, H, W)` Eq. 6 on width; renderer scales K; loss
  bilinear-downsamples the target (EAGLES `downsample_image`).
- C6 SC: `schedule.lowpass` → gsplat `eps2d`; ABE split ≤ 10000 (RAIN-GS code); 20 % lowest-error
  SfM init in `set_training_cameras` (sidecar `sfm_errors/<sha256(points)>.npy` made by
  `plugin/make_sfm_errors.py` from COLMAP points3D.bin) + 3-NN scales on the subset.
- C7 AT: `IntervalAdam` (from the Taming port) for `shN` (Adam step on summed grads every 16);
  separable SSIM with the 3DGS same-padding numerics.
- C8 DE: grad stats restarted at 20000; densify at 20100…20500.
- post: FP16 round-trip of all params at `total_steps`.

## Probes (each fails if its mechanism is removed)
| probe | checks |
|---|---|
| `trick-gaussian-mask` | renderer output equals a hand-masked render (scales·M, α·M); STE grad = σ'(m)·…; prune rule removes exactly σ(m) ≤ ε |
| `trick-sh-mask` | masked band ⇒ render equals render with zeroed band coefficients; loss = independent (2l+1)/15 formula |
| `trick-significance` | CUDA count/score equal a brute-force per-pixel python loop; v-score/prune rule; schedule rates 0.6·0.7^k |
| `trick-progressive` | Eq. 6 sizes at 1, τ/2, τ; blur kernel/σ schedule, update only at multiples of 100, scaled by DS; blurred target equals torchvision-formula reference; render at scheduled size (CUDA) |
| `trick-lowpass-abe` | low-pass value/recompute steps; ABE emits mean·0.3·extent copies only ≤ 10000; init keeps the 20 % lowest-error points |
| `trick-accelerated` | IntervalAdam = Adam on 16-summed grads; separable SSIM equals 2-D same-padded SSIM |
| `trick-redensify` | schedule: densify steps include 20100..20500 and exclude 15000..20000; mask prune every 500 after 15000 |
| `trick-cycle` | short native run (compressed schedule) on a synthetic scene: all mechanisms fire, leaves stay trainable, FP16 at the end (CUDA) |

## Unchanged from 3DGS
Representation, L1+0.2 D-SSIM, LRs (runtime), means LR log decay, SH degree +1/1000, clone/split
rule, opacity reset every 3000 before 15000, opacity prune 0.005, world-size prune.

## Fidelity gaps
1. No official code: SG steps read off Fig. 3, BL decay law, mask inits, SH-mask start (3500),
   DS linear vs "logarithmic" are choices (paper.md §7). Effect: tenths of dB / size; closable
   only with author code.
2. Runtime: white background, random-with-replacement views (3DGS: shuffled stack) — shared with
   the baseline.
3. Evaluation SSIM is the runtime's valid SSIM; training SSIM is 3DGS same-padded.
4. Timing/FPS/storage are not the paper's RTX 3090 numbers; storage reported as FP16 bytes.
5. Significance kernel counts with atomics (official non-atomic); negligible.
