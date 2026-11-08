# CRiM-GS: implementation spec

**Paper:** CRiM-GS: Continuous Rigid Motion-Aware Gaussian Splatting from Motion-Blurred Images.
Lee, Kim, Lee, Cho, Lee and Lee (Yonsei). arXiv 2407.03923 v2, Dec 2024, CVPR-2025 template.

**Sources:**
- LaTeX source in `source/latex/` (sec/4_method.tex, sec/main_appendix.tex).
- The PDF and its text are in `source/paper.pdf` and `source/paper.txt`.

**Official code: none.** As of 2026-11-07, `github.com/Jho-Yonsei/CRiM-GS` and `/CRiM-Gaussian` return 404, and the author's repository list has no CRiM-GS.

**Reference code:** the successor paper **CoMoGaussian** (ICCV 2025, same first author, arXiv 2503.05332). Its code is cloned at `source/comogaussian` (MIT, commit cd6d536). It is the same pipeline with renamed parts: "adaptive distortion-aware" becomes "CMR". We use it for the details the CRiM-GS paper omits, and cite it as `code:comogaussian/<file>`. Where the two disagree, we follow CRiM-GS's text.

## 1. Method summary
- **Base:** Mip-Splatting trained on motion-blurred images, 40k iterations.
- **Trajectory:** each training image i has a camera trajectory during exposure, made of N = 9 poses T^{τ_s} = T^{τ_0} · T̃_r^{τ_s} · T̃_a^{τ_s}.
  - T̃_r is a rigid SE(3) screw motion.
  - T̃_a is a near-SE(3) affine correction.
  - Both are decoded from latent states that evolve under a neural ODE, starting from a per-image embedding.
- **Rendering:** the 9 sharp images are rendered with 3DGS.
  - A shallow CNN on the 9 renders predicts softmax pixel-wise weights P, which blend them into I_blur.
  - A per-pixel mask M blends I_blur with the sharp render at the input pose: I_out = (1 − M)·I^{τ0} + M·I_blur.
- **Loss:** (1 − λc)·L1 + λc·D-SSIM of I_out against the blurry GT, plus an orthogonality regulariser on R̃_a and a mask sparsity term.
- **Test:** a sharp render at the given test pose. No test-pose optimisation in the main table.

**Contributions**
- **C1 Continuous rigid body motion** (Sec. 4.2): latent neural ODE, screw-axis decoder with independent ω̂ and θ, and the SE(3) exponential (Eq. 7, 9, 10).
- **C2 Adaptive distortion-aware transformation** (Sec. 4.3): a second latent neural ODE decodes A and t̃, with R̃_a = A + I. The decoder init is U(−1e-5, 1e-5), and L_o is the orthogonality loss.
- **C3 Pixel-wise weighted sum + scalar mask** (Sec. 4.4): CNN F, softmax weights, mask M, mask sparsity loss L_M.
- **C4 Objective and staged schedule** (Sec. 4.4, App. A): sharp only for 1k steps, then trajectory with averaging, then P and M from 3k.

## 2. Equations (verbatim, sec/4_method.tex)

**Eq. 1, pose composition:**
`\mathbf{T}^{\tau_s} = \mathbf{T}^{\tau_0} \tilde{\mathbf{T}}_{r}^{\tau_s} \tilde{\mathbf{T}}_{a}^{\tau_s}`, with T = [R|t; 0 1].

**Eq. 5, rigid latent ODE:**
`\mathbf{z}_{r}(\tau_{s})=\mathbf{z}_{r}(\tau_0)+\int_{\tau_0}^{\tau_s} f(\mathbf{z}_{r}(\tau), \tau; \phi) d\tau`
- z_r(τ0) = E_r(embed(i)), where E_r is a single linear layer.

**Decoder:**
`\mathcal{D}_{r}(\mathbf{z}_{r}(\tau)) = (\hat{\omega}^{\tau}, \theta^{\tau}, v^{\tau}),\ ||\hat{\omega}^{\tau}|| = 1`
- ω̂ is normalised and θ is predicted independently.

**Eq. 7:** `[\mathcal{S}^{\tau}] = [[\hat\omega^\tau], v^\tau; 0, 0] \in se(3)`

**Eq. 9:** `e^{[\hat{\omega}] \theta} = I + \sin\theta [\hat{\omega}] + (1 - \cos\theta) [\hat{\omega}]^2`

**Eq. 10:** `G(\theta) = I\theta + (1 - \cos\theta) [\hat{\omega}] + (\theta - \sin\theta) [\hat{\omega}]^2`

**Rigid transform:** T̃_r = [e^{[ω̂]θ} | G(θ) v].

**Adaptive latent ODE:**
`\mathbf{z}_{a}(\tau_s) = \mathbf{z}_{a}(\tau_0) + \int g(\mathbf{z}_{a}, \tau; \psi) d\tau`
- z_a(τ0) = E_a(embed(i)).
- Decoder: D_a(z_a) = (A, t̃), with R̃_a = A + I and T̃_a = [R̃_a | t̃_a; 0 1].
- D_a weights are initialised from U(−1e-5, 1e-5).

**Orthogonality loss:** `\mathcal{L}_{o} = \|\tilde{\mathbf{R}}_{a}^{\top}\tilde{\mathbf{R}}_{a} - \mathbf{I}\|_2`

**Pixel weights:** `\mathcal{I}_{blur} = \sum_{i=1}^{N}\mathcal{I}^{\tau_i}\cdot\mathcal{P}^{\tau_i},\ \mathcal{P} = softmax(\mathcal{F}(\mathcal{I}))`
- P is N×H×W×3.
- Mask: `\mathcal{I}_{out}=(1-\mathcal{M})\mathcal{I}^{\tau_0} + \mathcal{M}\mathcal{I}_{blur}`.
- L_M = mean(M), following BAGS.

**Objective:** `\mathcal{L} = (1-\lambda_c)\mathcal{L}_1 + \lambda_c \mathcal{L}_{D-SSIM} + \lambda_o\mathcal{L}_o + \lambda_M\mathcal{L}_M`

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 40000 | App. A; code:comogaussian/arguments OptimizationParams.iterations |
| N poses | 9 | App. A, Table N-ablation |
| embedding / hidden size | 64 (nn.Embedding) | App. A |
| E_r, E_a, D_r, D_a | single linear layers, hidden 64 | App. A, Sec. 4.2–4.3 |
| f, g | two parallel single linear layers (rotation half and translation half, 32 each) with ReLU | App. A; code: relu(x) → Linear on each half (NeuralDerivative) |
| ODE solver | Dormand–Prince 5(4) (dopri5 / RK45) | Sec. 3.3. CoMo code uses rk4 (`method='rk4'`); CRiM-GS text wins |
| rtol / atol | 1e-4 / 1e-5 | code:comogaussian/scene/como_kernel.py DiffEqSolver (not in paper) |
| CNN F | 3 conv layers, 64 ch, kernels 5×5, 3×3, 3×3 | App. A. CoMo: 2 layers, 32 ch, ReLU + InstanceNorm after each |
| weight head | 1×1 conv → softmax over N | App. A; code: `mlp_weight` Conv2d(c,1,1,bias=False), softmax dim 0 |
| mask head | 1×1 conv on F's output, averaged over the N axis (sigmoid) | App. A; code: sigmoid(Conv1x1 over the concatenation of all N features) |
| CNN input | the N rendered images, detached | code:comogaussian/train.py `get_weight_and_mask(rendered_images.detach())` |
| λc, λo, λM | 0.3, 1e-4, 1e-3 | App. A; code:train.py |
| start trajectory | step > 1000 (averaging of N warped + sharp until 3000) | App. A; code:train.py `start_warp`, `torch.mean` of N + 1 images |
| start P and M | step > 3000 | App. A; code `start_pixel_weight` |
| kernel optimiser | Adam lr 1e-4, linear decay by (1e-4 − 1e-6)/40000 per step after 1000 | code:como_kernel.py (not in paper) |
| decoder init | U(−1e-5, 1e-5) (paper: D_a); code: xavier with gain 1e-5/√((32+3)/6) for all decoders, bias 0 | Sec. 4.3; code |
| Mip-Splatting | 3D filter coef 0.2 (recomputed after every densify and every 100 steps after densification), 2D filter kernel 0.1, black background | code:comogaussian/gaussian_model.py, arguments `kernel_size=0.1`, `white_background=False` |
| densification | from 500, every 100, until 20000; grad 2e-4; prune opacity < 0.005; **no opacity reset, no size prune** | code:train.py (size_threshold None, no reset_opacity call) |
| Gaussian lrs | 3DGS defaults; position 1.6e-4 → 1.6e-6 over 40k | code:arguments |
| SH degree | 3, raised every 1000 | code |

## 4. Schedules
- **Steps 1–1000:** sharp render only. The kernel is not trained.
- **Steps 1001–3000:**
  - I_out = mean(I^{τ_1..τ_N}, I^{τ_0}), i.e. N + 1 images (code).
  - Kernel: Adam step, linear lr decay.
- **Steps > 3000:** P and M are on, and the L_M term is active. L_o is active whenever the trajectory is on.
- **Densification:** 500–20000. The 3D filter is recomputed after each densification, and every 100 steps after 20000 until 39900.

## 5. Builds on
- Mip-Splatting: 3D smoothing filter and 2D Mip filter.
- Kept from 3DGS: everything else (densification, Adam lrs, SH schedule, L1 + D-SSIM with weight 0.3 in place of 0.2).
- Gaussian densification statistics come only from the sharp render at the input pose (code).

## 6. Protocol
- **Data:** Deblur-NeRF camera-motion blur, 5 synthetic scenes (factory, cozy2room, pool, tanabata, trolley = `wine`) and 10 real scenes.
- **Poses and SfM:** COLMAP run on the blurry images.
- **Resolution:**
  - synthetic: full res 600×400 (`-r 1`);
  - real: `-r 4` from 2400×1600, i.e. 600×400.
- **Split:**
  - synthetic: llffhold 8 (idx % 8 == 0 are the sharp test views);
  - real: the release's hold=N files (Deblurring-3DGS convention).
- **Evaluation:** sharp test views rendered at the COLMAP test pose, no pose optimisation (Table 1). Post-optimised numbers are an appendix extra (30.65 / 28.34).
- **Metrics:** PSNR, SSIM and LPIPS. The CoMo code uses skimage SSIM on [−1, 1] images and LPIPS-Alex.

**Per-scene PSNR** (appendix tab:synthetic, tab:real):
- **synthetic:** factory 28.72, cozy2room 33.01, pool 32.05, tanabata 29.04, trolley 29.36;
- **real:** ball 28.25, basket 30.85, buick 26.51, coffee 32.26, decoration 26.34, girl 26.20, heron 22.84, parterre 26.09, puppet 26.66, stair 27.30.

## 7. Ambiguities and our choices
1. **Time points:** "N uniformly sampled time points within the fixed exposure time", integrated from τ0. The time unit is not given.
   - CoMo uses integer times 0..8, integrated outward from the middle.
   - **Choice:** τ_i = i (i = 0..N−1), integrated forward from τ0 = 0 (CRiM-GS text).
2. **Whether f depends on τ:** unspecified. The appendix describes only two linear layers. CoMo adds a learned per-step time embedding (a CoMo-only change).
   - **Choice:** autonomous f(z).
3. **The N poses vs the sharp pose:** the N poses are all decoded (including τ0, which is not forced to identity), and I^{τ0} is rendered at the unmodified input pose. This matches the CoMo code.
4. **Pose convention:** T is taken as camera-to-world, with right multiplication (perturbation in the camera frame); viewmat = (T^{τ})^{-1}.
   - CoMo multiplies a mixed matrix [R_c2w | t_w2c]. That is the same family, reparametrised.
5. **CNN:** we follow CRiM-GS: 3 layers, 64 ch, 5/3/3 kernels, with ReLU + InstanceNorm after each (from CoMo/BAGS).
   - Weights have 1 channel, broadcast over RGB (code). The paper writes N×H×W×3.
   - Mask: per-image 1×1 conv, mean over N, then sigmoid (CRiM text: "averaging it to the batch axis").
6. **L_o:** L2 (Frobenius) norm of R̃ᵀR̃ − I, averaged over the N poses (paper). CoMo uses mean |R Rᵀ − I|.
7. **Kernel lr:** 1e-4 with linear decay (CoMo code). Not given in the paper.
8. **Opacity reset:** not in the CoMo code. Mip-Splatting resets every 3000. **Choice:** follow the code (no reset); a config option is kept.
