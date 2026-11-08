# splatstudio-method-crim-gs

A native SplatStudio port of **CRiM-GS**: Continuous Rigid Motion-Aware Gaussian Splatting from Motion-Blurred Images (Lee et al., arXiv 2407.03923v2). Paper fidelity is unverified.

There is no official CRiM-GS code; its repositories return 404. Numerics the paper omits follow the authors' successor **CoMoGaussian**, `github.com/Jho-Yonsei/CoMoGaussian` @cd6d536 (MIT). Where the two differ, the paper's text wins. The Mip-Splatting filters are vendored from `reproductions/mip-splatting`.

## Mechanism

### Training views (`crimgs-renderer` + the kernel owned by `crimgs-density`)
- **Sharp render:** I^{τ0} at the input pose, using:
  - the Mip-Splatting 3D filter;
  - gsplat antialiased (eps2d 0.1);
  - a black background.
  This render's `info` drives densification.
- **Trajectory (step > 1000):** per-image embedding (64) feeds two single-layer encoders, E_r and E_a.
  - Two latent neural ODEs run on our own torch dopri5, with rtol 1e-4 and atol 1e-5, over τ = 0..8.
  - The derivative is two parallel ReLU → Linear halves (rotation and translation).
- **Decoders:**
  - D_r gives (ω̂ normalised, θ, v), mapped through the screw exponential (Eq. 9–10) to T_r.
  - D_a gives (A, t_a), with R_a = A + I.
  - Decoder weights are initialised from U(±1e-5).
- **Poses:** the N = 9 poses are inv(c2w · T_r · T_a), rendered in one batched gsplat call.
- **Blend:**
  - steps 1001–3000: the mean of the N renders and the sharp render (code);
  - after step 3000: I_out = (1 − M)·I^{τ0} + M·Σ P_i I_i, where P = softmax over N of a 1×1 conv on the 3-layer CNN F (64 ch, 5/3/3). M = sigmoid of the mean over N of a 1×1 conv. F sees the detached renders.

### Losses
- `photometric` with ssim_weight 0.3 (λc).
- `crimgs-regularizers`: 1e-4 · mean ‖R_aᵀR_a − I‖_F + 1e-3 · mean(M).

### Density
- 3DGS densification: from 500, every 100, until 20000, gradient threshold 2e-4, prune opacity < 0.005.
- No opacity reset and no size prune (CoMo train.py).
- The 3D filter is recomputed after each densification, and every 100 steps afterwards.

### Kernel optimiser
Adam with lr 1e-4, decaying linearly by (1e-4 − 1e-6)/40000 per step (CoMo).

### Test / ss-eval
Sharp render at the given test pose. No kernel is used and no pose optimisation is applied, matching the paper's Table 1.

## Data
- **`deblur3dgs-synmotion`**: Deblur-NeRF synthetic motion blur at 600×400, hold 8. Build it with `python3 reproductions/crim-gs/tools/prepare_synthetic.py`.
- **`deblur3dgs-motion`**: real motion blur at 600×400, using the release's hold files.

## Commands
```bash
tools/ssenv ss-train crim-gs --data ~/datasets/deblurring3dgs/native-synmotion/cozy2room \
  --output runs/x --steps 40000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run runs/x --split test
```

## Fidelity gaps
Full list in `../plan.md`. The main ones:
- **CoMo-code substitutes:** kernel lr, ODE tolerances, CNN norm layers, the warm-up mean, no opacity reset, densify until 20k.
- **ODE time grid:** τ = 0..8, forward, autonomous derivative. This is assumed, not stated. The latent norm grows about 10–20× over the exposure at init.
- **Solver details:** the dopri5 hits output times exactly, with detached step control. Poses use a c2w right-multiplication convention.
- **Runtime differences:** views sampled with replacement, valid-convolution SSIM, LPIPS-VGG.
