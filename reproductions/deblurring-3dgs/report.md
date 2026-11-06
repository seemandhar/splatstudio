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
