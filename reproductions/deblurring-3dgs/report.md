# Deblurring 3D Gaussian Splatting (plan #88) — build report (pre-benchmark)
**Status: built, benchmark pending.**
- Plugin `plugin/` (`ss_deblurring_3dgs`), with methods `deblurring-3dgs-defocus` and `deblurring-3dgs-motion` (20k steps, `--recipe 3dgs`).
- Official code (benhenryL/Deblurring-3D-Gaussian-Splatting) was read and ported, not run.
- No custom CUDA: the transformed covariance is built in torch and passed to gsplat as `covars`.

## Contributions → code
- **Defocus blur MLP:** an exact port of the official network. It outputs per-Gaussian scale and rotation factors clamped to [1, 1.1]. The covariance uses the unnormalised rotated quaternion, as the Inria rasterizer does.
- **Camera-motion blur:** 5 renders in one batched gsplat call (1 unshifted, 4 with MLP position offsets), averaged.
- **Extra points at step 2500:** follows the official code, including re-initialising all Gaussians, resetting every Adam (MLP included) and distance-proportional colour weights.
- **Depth-based pruning:** the opacity threshold is divided by a depth weight in [1, 3]. There is no opacity reset and no size pruning.
- **Sharp test renders:** ss-eval renders the original Gaussians, with no MLP.
- **Training recipe:** 0.8 L1 + 0.2 D-SSIM (zero-padded SSIM). Position LR decays over 30k steps although training stops at 20k. Black background.
- **MLP state:** the MLP and its optimizer live in a density-owned bank, saved in the checkpoint under `density_state["bank"]`.

## Data
- The authors' re-release (`deblur_dataset.zip`) has denser COLMAP points than our older deblurnerf copies:

  | scene | release points | older copy |
  |---|---|---|
  | cake | 14465 | 6655 |
  | ball | 9944 | 1867 |
  | girl | 12532 | 3947 |

- It is stored at /bigdata/sejain/datasets/deblurring3dgs and built by `tools/prepare_official.py`: 600×400, official split (sorted index % hold == 0).
- Registered as `deblur3dgs-defocus` and `deblur3dgs-motion`.
- `baseline-gsplat-deblurnerf-40000` does not fit, because it uses the old points and 40k steps.

## Verification
ss-inspect is clean, 6 unit tests pass, ruff is clean, and ss-verify passes 8/8 probes for both methods.

## Smoke test (2k steps, PSNR / SSIM / LPIPS)
| scene | ours | gsplat |
|---|---|---|
| cake (defocus) | 23.97 / 0.810 / 0.318 | 23.70 / 0.799 / 0.327 |
| ball (motion) | 25.20 / 0.755 / 0.397 | 24.06 / 0.687 / 0.450 |

An extra-points diagnostic, with the step moved to 1000, kept 2937 of 200k candidates; validation PSNR 23.58.

## Benchmark request
| dataset | scenes | paper PSNR | paper 3D-GS |
|---|---|---|---|
| defocus | cake, caps | 26.88, 24.50 | 20.16, 19.08 |
| motion | ball, girl | 28.27, 23.26 | 22.99, 19.72 |

- Each method runs on its own dataset at 20k steps.
- A gsplat control runs on all 4 scenes at 30k.
- Compare PSNR to the paper, and our gain over gsplat with the paper's gain over 3D-GS.

## Fidelity gaps
- gsplat is used instead of the Inria rasterizer.
- Views are sampled with replacement.
- Optimizers step before densification; at step 2500 the official code skips that step's update.
- The SH degree increases one step later.
- SSIM/LPIPS implementations differ.
- Single seed.
- Where the paper text and code disagree, the code is followed:
  - clamp [1, 1.1];
  - 4+1 renders;
  - no π in the encoding;
  - camera centre as the view input;
  - distance-proportional colour weights.

## Review-1 resolution (orchestrator)
Approved, with 0 critical, 0 major and 6 minor issues. Every builder claim was checked against the official code: the MLP, the 5 motion renders, the extra points, depth pruning, the densify schedule, the sharp test path, the per-scene hold values (cake 9, caps 5, ball 7, girl 7) and paper_results.json.
Unaddressed minors:
- No probe covers the densification statistics or the step-2500 trigger, although a GPU run confirmed both fire.
- Exact 3-NN is used instead of distCUDA2.
- The official code skips the optimizer step at the final step.
- The gsplat control is not the paper's 3D-GS protocol, so the gain comparison is approximate.
Benchmark: d3dgs-{defocus,motion} at 20k, with gsplat controls d3dgs-ctl-{defocus,motion} at 30k.

## Interim results (20k; ss-eval test PSNR / SSIM / LPIPS-VGG)
| scene | ours | paper | Δ | gsplat control (30k) | paper 3D-GS | gain ours / paper |
|---|---|---|---|---|---|---|
| ball (motion) | 28.20 / 0.857 / 0.257 | 28.27 | −0.07 | 21.17 | 22.99 | +7.03 / +5.28 |
| girl (motion) | 22.47 / 0.884 / 0.190 | 23.26 | **−0.79** | pending | 19.72 | — |
| cake (defocus) | 24.92 | 26.88 | **−1.96** | 19.83 | 20.16 | +5.10 / +6.72 |
| caps (defocus) | pending | 24.50 | | pending | 19.08 | |

Ball matches. Girl and cake are outside ±0.5 dB. The gap will be analysed once caps and the remaining controls finish.

## Defocus results complete
- **caps:** 22.63 vs paper 24.50 (Δ −1.87).
- **cake:** 24.92 vs paper 26.88 (Δ −1.96).
- Both defocus scenes are about 1.9 dB short while motion is close, so a gap analysis was requested from the builder, focused on the defocus path.

## Gap analysis (builder) and fix iteration 1
- **Not the cause:** data, split, background and eval. The authors' released models, rendered with our renderer on our test splits, reproduce the paper: cake 26.885, caps 24.467, ball 28.239, girl 23.257.
- **Symptom:** our sharp Gaussians absorb the blur. On blurry training views the sharp render scores cake 26.9 / caps 29.4 vs the official 25.4 / 25.0.
- **Root cause 1 (caps, confirmed and fixed):** the defocus MLP was dead for the whole run.
  - Its scale factors stayed at 1.0 and its Adam moments were about 1e-45.
  - Our CPU, forked-RNG init left 0% of the gain-0.1 scale-head outputs above the clamp at 1, so the head got no gradient.
  - The official code draws xavier weights on the GPU right after manual_seed(0), which gives 13–23% live outputs.
  - Fix in bank.py: the MLP is built on the training device with the official draw order and a fresh seed-0 generator, which is bit-identical to the official draw. The mlp-architecture probe now fails on the old init.
  - In a 5k-step diagnostic, caps improved from 22.44 to 23.73.
- **Root cause 2 (cake, unresolved).** The MLP is alive there, but the sharp render over-fits the blur and test PSNR falls from 5k to 20k. Candidates, unconfirmed:
  - saturated factors at the clamp limits;
  - runtime order (the official code skips the Gaussian update on densify iterations and at step 2500) and stack-shuffle sampling;
  - 327k vs the official 265k Gaussians, with floaters.
- **Rerun:** d3dgs-defocus-v2 and d3dgs-motion-v2 (20k). The gsplat controls are unchanged; the first runs are kept.

## Results after fix 1 (d3dgs-defocus-v2, 20k) — defocus within tolerance
| scene | v1 (dead-MLP init) | v2 (official init) | paper | Δ v2 |
|---|---|---|---|---|
| cake | 24.92 | **26.65** / 0.889 / 0.185 | 26.88 | −0.23 |
| caps | 22.63 | **24.42** / 0.854 / 0.206 | 24.50 | −0.08 |

- The init fix closed both defocus gaps, including cake, which the 5k diagnostic had not predicted.
- Cake's "unresolved root cause 2" therefore also depended on the init, although the builder found cake's MLP alive under the old init. The likely mechanism is that the old draw left most factors pinned at the clamp limits (candidate 1). It is not verified, and the 5k diagnostic did not show the effect.
- Motion v2 (ball, girl) is still pending.
- **gsplat control on girl (30k):** 19.88, vs the paper's 3D-GS 19.72. The caps control is still pending.
- **gsplat control on caps (30k):** 19.79, vs the paper's 3D-GS 19.08.

## Final verdict after fix 1: MATCHED (4/4 scenes within ±0.5 dB)
| scene | v1 | v2 (official MLP init) | paper | Δ v2 | gsplat 30k | paper 3D-GS | gain ours / paper |
|---|---|---|---|---|---|---|---|
| ball (motion) | 28.20 | **28.27** | 28.27 | 0.00 | 21.17 | 22.99 | +7.10 / +5.28 |
| girl (motion) | 22.47 | **23.03** | 23.26 | −0.23 | 19.88 | 19.72 | +3.15 / +3.54 |
| cake (defocus) | 24.92 | **26.65** | 26.88 | −0.23 | 19.83 | 20.16 | +6.82 / +6.72 |
| caps (defocus) | 22.63 | **24.42** | 24.50 | −0.08 | 19.79 | 19.08 | +4.63 / +5.42 |

- **The fix:** one grounded change, the official on-device xavier draw for the blur MLP (seed-0 order).
- **Effect:** it lifted all four scenes (+0.07 to +1.79 dB) and closed every gap.
- **v1 runs:** kept in runs/d3dgs-{defocus,motion}/.
