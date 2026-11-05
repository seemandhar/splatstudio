# splatstudio-method-dashgaussian

Native SplatStudio port of **DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds**
(Chen et al., CVPR 2025, [arXiv 2503.18402](https://arxiv.org/abs/2503.18402)), 3DGS-backbone variant,
following the official code [YouyuChen0207/DashGaussian](https://github.com/YouyuChen0207/DashGaussian)
(`train_dash.py`, `utils/schedule_utils.py`, `scene/gaussian_model.py:prune_and_densify`, CC BY-NC-SA 4.0).
Paper fidelity unverified.

Method name: `dashgaussian`.

## Mechanism → code
| contribution | code |
|---|---|
| Frequency-guided resolution schedule (Sec. 3.2, supp. Eq. 6–7) | `schedule.py:build_schedule,res_scale`; density owns it |
| Low-res training render with scaled intrinsics | `renderer.py` (render size `floor(H/r) × floor(W/r)`, K scaled by w/W, h/H) |
| Antialiased GT downsampling (supp. 7.1) | `losses.py` (`photometric` loss replaced: L1 + 0.2 D-SSIM vs. resampled target) |
| Resolution-guided primitive target (Eq. 4) + top-k prune-first densification (supp. 7.4) | `schedule.py:densify_rate`, `density.py:prune_and_densify` |
| Momentum budget (Eq. 5) | `schedule.py:update_momentum` |
| Position LR held during the LR stage (supp. 7.2) | `density.py:update_lr` (overrides the runtime's means LR for the next step) |

Evaluation always renders at full resolution: the renderer only lowers the resolution when
grad is enabled **and** a density with an active schedule is linked; `measure`/`ss-eval` run under
`torch.no_grad()` (and `ss-eval` builds fresh, schedule-less components).

## Runtime workarounds (and the hooks that would remove them)
- The runtime does not expose training images. `dash-density.set_training_cameras` re-loads the dataset
  with `splatstudio.data.load_dataset` from density config `data`/`downscale`, else from the `ss-train`
  command line (`--data`, `--downscale`), and verifies every training camera (K, viewmat, size) against
  the cameras the runtime passed; any mismatch or missing path raises. Hook wanted: images in
  `set_training_cameras` or a `set_training_images` callback.
- The renderer gets no step. `ss_dashgaussian.link.LINK` connects it to the density built in the same
  process; `before_backward` checks the rendered size against the schedule. Hook wanted: a step-aware
  renderer call or a runtime resolution-schedule hook.

## Loss (`photometric`, iteration 1)
L1 + 0.2 D-SSIM. `ssim_padding: "same"` (default): 11×11 Gaussian window, σ 1.5, zero padding 5, as in 3DGS
`loss_utils.ssim` / `fused_ssim` used by the official code. `"valid"` reproduces iteration 0 (runtime valid SSIM).
This changes the **training loss only**; evaluation SSIM is still the runtime's unchanged valid SSIM.

## Config (density `dash-density`, defaults = official code)
grow_grad2d 2e-4, grow_scale3d 0.01, prune_opa 0.005, refine_start 500, refine_every 100,
densify_until 27000 (`full_eval.py --dash`), reset_every 3000 (resets until 27000), reset_value 0.01,
total_steps 30000, lr_final_factor 0.01, start_significance_factor 4, max_reso_scale 8,
reso_sample_num 32, max_densify_rate 0.2, momentum_init_factor 5 (P_fin⁽⁰⁾ = 6·P_init as in code),
integrate_factor 0.98, momentum_step_cap 1e6, max_n_gaussian −1 (momentum on),
resolution_mode `freq`|`const`, densify_mode `freq`|`free` (official ablation switches).

## Fidelity gaps
gsplat rasterizer instead of 3DGS `3dgs-accel` (absolute time not comparable; compare speed-up vs our
gsplat baseline); white background, valid-SSIM, sampling with replacement (runtime-wide); optimizer step
happens before densification (official: densify, then step); dataset reload and schedule construction
happen before the runtime timer, as the official timer also excludes scheduler init.

## Commands
```bash
pip install -e .
ss-inspect dashgaussian
ss-verify dashgaussian --device cuda
ss-train dashgaussian --data ~/datasets/mipnerf360/native/garden --output runs/x --steps 30000 \
  --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
ss-eval --run runs/x --split test
pytest -q tests
```
