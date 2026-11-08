# CRiM-GS gap analysis 1: cozy2room

**Scope.** Every diagnostic here uses training views only. Nothing was tuned or selected on test images. Diagnostic script: `tools/diag_train_blur.py`. Output: `evidence/gap/train_blur_{cozy2room,ball}.json`.

## Gap
| scene | ours | paper | gap |
|---|---|---|---|
| cozy2room (synthetic) | 31.91 | 33.01 | **−1.10** |
| ball (real) | 27.80 | 28.25 | −0.45 (within ±0.5) |

- **Prior methods on cozy2room, from CRiM-GS's own table:** Deblurring-3DGS 31.45, BAGS 32.21, DP-NeRF 32.65.
- **No baseline decomposition:** there is no gsplat or Mip-Splatting baseline on `deblur3dgs-synmotion`, and the paper gives baseline averages only (Mip-Splatting on blurry images: synthetic 21.09). So `delta_gap` cannot be computed per scene.
- **Per-view test PSNR** is uniform: 31.0, 31.9, 33.0, 32.6, 31.0. No outlier view drives the gap.

## Evidence
1. **Densification stall (review m3): it recovers.**

   | step | 1k | 3k | 5k | 10k | 20k (end of densification) |
   |---|---|---|---|---|---|
   | cozy2room | 19.9k | 26.9k | 80.2k | 95.5k | **98.4k** |
   | ball | 49.7k | 50.1k | 75.3k | 89.8k | **104.8k** |

   - The 1k–3k stall is the 1/(N+1) scaling of the sharp-render gradient, the same as the CoMo code.
   - Growth resumes once (1 − M) ≈ 0.38 weights the sharp render.
   - Neither CRiM-GS nor CoMoGaussian reports Gaussian counts, so there is no reference to compare against.
   - Ball, with the same dynamics, matches the paper, so this is not a strong explanation for cozy2room.
2. **Blur reconstruction on training views is near-perfect.**

   | quantity | cozy2room | ball |
   |---|---|---|
   | I_out vs blurry GT, 29 train views | **41.5** (39.0–43.1) | 41.4 |
   | sharp render vs blurry GT | 26.9 | 26.7 |
   | plain mean of the 9 renders vs blurry GT | 35.1 | 36.5 |

   - The trajectory model explains the blur. The CNN weights plus the mask add about 6 dB on top of the plain average.
   - The mask is about 0.62 everywhere, nearly constant, so the sharp render keeps roughly 38% weight.
   - A 41 dB fit with a 31.9 dB sharp test suggests the blur model is flexible enough to absorb residual detail. Ball shows the same 41 dB fit and still matches the paper, so this is not specific to cozy2room.
3. **Trajectories are plausible and the ODE is not stiff.**
   - Path length is 0.35, against a camera spread of 5.1 (about 7%). Rotation is at most 0.36°.
   - The decoded τ0 pose sits 0.066 from the input pose.
   - ODE: 992k accepted / 118k rejected steps over 39k steps × 2 solves, about 12.7 steps per solve, with no blow-up. Review m1's stiffness concern did not materialise.
4. **Time grid (m1), time input (m2), embedding init (m5).**
   - CoMo integrates ±4 from the middle pose, with a time embedding and a zero embedding. The CRiM text says "from τ0", and uses f(z, τ).
   - None of the evidence above shows a failure these choices would explain: the blur fit is near-perfect, the trajectories are smooth and small, and the solver is well-conditioned.
   - The expected effect is small and its sign is unknown. I do not expect a ≥ 0.5 dB gain.
5. **Protocol candidates (unverified, likely the larger term).**
   - *COLMAP poses:* our data is the Deblurring-3DGS release's COLMAP. The paper's poses come from its own COLMAP run (CoMo distributes its own dataset with SfM). At 33 dB, test-pose error of a fraction of a pixel costs about 1 dB. The paper's post-optimisation gain on synthetic is only +0.2 dB, so their test poses are well aligned; ours may not be.
   - *Metric:* SSIM differs (ours 0.951, valid convolution, versus their skimage 0.941). PSNR is the same formula.

## Classification
- **Method fidelity is not demonstrated as the cause.** ball matches, and the train-view diagnostics show nothing broken.
- **Most likely:** a protocol/data offset (test-pose alignment of the COLMAP release on high-PSNR synthetic scenes), plus single-seed noise. This is not proven: no baseline and no test-pose diagnostic were run.

## Recommendations
- **No code change for iteration 1.** None of the CoMo-grounded fixes (centred ±4 integration, zero embedding init, time embedding) is supported by the evidence as a ≥ 0.5 dB fix.
- **Optional diagnostics, if budget allows:**
  - (a) gsplat or Mip-Splatting baseline on cozy2room at 40k, to get `baseline_offset`;
  - (b) the paper's own appendix protocol of test-pose post-optimisation (reuse `ss_deblurgs.evaluate`), reported only as a diagnostic of pose alignment. A gain well above +0.2 dB would point to the COLMAP data offset;
  - (c) CoMo's dataset/SfM release (Drive id in the CoMo README) for cozy2room, to test the pose hypothesis directly.
