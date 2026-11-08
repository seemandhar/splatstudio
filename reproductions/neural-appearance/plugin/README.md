# splatstudio-method-neural-appearance

Native SplatStudio port of **Compact Neural Appearance Models for Efficient Gaussian Splatting**
(Hahlbohm et al., arXiv 2609.05255), following the official code
github.com/nerficg-project/efficient-gaussian-appearance (Apache-2.0).

## Methods
- `neural-appearance-<scene>`: the paper's Neural model (C1). Each Gaussian has a base colour
  c0 = Y0·sh0 and 8 latent features `na_features`. A shared bias-free MLP (32→16→16→3, ReLU, 816
  weights) decodes A(d) = tanh(MLP([sin/cos f, SH₀₋₃(d)])), and the colour is c = ReLU(0.5 + c0 + A(d)).
- `neural-appearance-sh-<scene>`: the paper's paired SH row, degree-3 SH in the same pipeline.
- The scenes are garden, room, bicycle and truck. The suffix only selects the official MCMC
  primitive cap (5.0M, 1.5M, 6.0M, 2.5M).

Both methods use the paper's shared pipeline:
- the Faster-GS rasterizer, FusedAdam and Morton ordering, imported from `ss_faster_gs`
  (reproductions/faster-gs);
- MCMC densification, ported to PyTorch from the official `kernels_mcmc.cuh`;
- PPISP, vendored from nv-tlabs/ppisp @a247368 (Apache-2.0) and JIT-built, with its 5000-step
  controller distillation after 30k steps.

**Train with `--steps 35000 --recipe 3dgs --num-gaussians 10000000`.**

## State outside the per-Gaussian fields
The MLP and PPISP are component state (`bank.Bank`):
- The density creates the Bank in `set_training_cameras` and steps it in `after_backward`.
- After every step the Bank is packed into the per-Gaussian field `na_blob` (4 floats per
  Gaussian). Because of this, plain `ss-eval` rebuilds the Bank from the checkpoint. The
  `na-ppisp` probe checks this round trip.
- At evaluation every view is a novel view. PPISP then uses the distilled controller
  (`frame_idx=-1`), as in the official inference path.

## Config (paper value → key)
| Setting | Value | Key |
|---|---|---|
| MCMC schedule | 600..24900 every 100; min opacity 0.005; noise 5e5·lr_means | `densify_*`, `noise_lr` |
| MCMC init | opacity 0.5; scale 0.1 × RMS-3NN | `init_*` |
| Means lr | 1.6e-4 → 1.6e-6 (× extent) | `lr_means_*` |
| Scales / quats / opacities lr | 0.005 / 0.001 / 0.05 | `lr_scales`, `lr_quats`, `lr_opacities` |
| Neural lrs | base 0.00025, features 0.008, cosine 7k→37k; MLP 0.003 with warm-up 3k and decay to 0.1× at 30k, β₂ 0.99 | |
| SH lrs | base 0.0025·Y0, coefficients 0.000125, constant | |
| Regularizers | 0.01·mean(opacity) + 0.01·mean(scale) + PPISP regularizer, off during distillation | `na-regularizers` |
| Background / near / far | black / 0.2 / 1e4 | |

## Fidelity gaps
- **MLP evaluation:** fp32 PyTorch over all Gaussians, i.e. the paper's unfused reference path,
  instead of fused fp16 tcnn. The paper reports the same quality for both paths.
- **MLP init:** tcnn's Xavier-uniform init is reproduced in distribution only, not bit for bit.
- **MCMC noise RNG:** torch RNG instead of Philox.
- **Runtime differences:** runtime SSIM (valid convolution) instead of fused-ssim; views sampled
  with replacement; a single seed (the paper averages 5 runs).
- **PPISP camera:** a single PPISP camera is assumed (true for M360, T&T and DB).
