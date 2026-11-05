# DeblurGS — implementation spec

Oh, Chung, Lee, Lee. *DeblurGS: Gaussian Splatting for Camera Motion Blur*, ECCV 2024, arXiv
2404.11358 (v2 LaTeX in `source/latex`; no supplementary in the e-print). Official code:
github.com/taekkii/deblurgs @e0587475 (2024-05-12), cloned to `source/code`. README: "Under
Construction: Please wait for the cleanup", TO-DO "Updated Code for reproduction". So the
released defaults may not be the ones behind the paper's tables. Code is the Inria 3DGS
(Gaussian-Splatting licence) with a lightly modified rasterizer: depth output, a ReLU/sigmoid
colour switch, and a near plane of 0.2.

## 1. Summary and contributions
- Each blurry training image i gets a latent camera trajectory during the exposure: a 9th-order
  Bézier curve in se(3), stored as 10 control points (6-D each).
- The blurry image is synthesised as the mean of N = 21 sharp sub-frame renders at poses
  P̂(ν_j) on that curve. The ν_j are learnable sub-frame alignment parameters (Sec. 3.2).
- A gamma tone curve γ(x) = x^{1/2.2} links the linear-irradiance average to the sRGB
  observation.
- Loss: L1 reconstruction plus a temporal smoothness loss between adjacent sub-frames, with an
  annealed weight (Sec. 3.4).
- Gaussian Densification Annealing (Sec. 3.3): the clone/split gradient threshold θ starts high
  and decays, so the scene does not densify at wrong positions while the poses are still bad.
- Test: the camera motion is discarded and the sharp scene is rendered. The test poses are
  aligned by test-time pose optimisation with the scene frozen (Sec. 4.2, "post-optimization").

Contributions:
- C1 Per-image Bézier-se(3) camera trajectory + sub-frame averaging blur synthesis (Eq. 4–7).
- C2 Learnable sub-frame alignment ν (Eq. 5–6).
- C3 Gamma tone mapping of the synthesised blur (Eq. 7).
- C4 Temporal smoothness loss with annealed λ (Eq. 9–10).
- C5 Gaussian Densification Annealing of θ (Sec. 3.3).
- C6 Evaluation protocol: sharp render plus test-pose post-optimisation (Sec. 4.2; code `test.py`).

## 2. Equations
Eq. 4: $\mathbf{B}=\int_{\tau_o}^{\tau_c}\mathbf{I}(\projmat_\tau)d\tau\approx\frac1N\sum_{i=1}^N\mathbf{I}(\projmat_{\tau_i})$, with $\tau_i=\tau_o+\frac{i-1}{N-1}(\tau_c-\tau_o)$.
Eq. 5: $\hat{\projmat}(\nu_i)\triangleq\projmat_{\tau_i}$.
Eq. 7: $\hat{\mathbf{B}}(G)=\gamma\left(\frac1N\sum_{i=1}^N\rendermat_G(\hat{\projmat}_i)\right)$, $\gamma(x)=x^{1/2.2}$.
Eq. 8: $\mathcal{L}_{rgb}=\sum_i\|\mathbf{B}_i-\hat{\mathbf{B}}_i\|_1$.
Eq. 9: $\mathcal{L}_{smooth}=\frac1N\sum_{i,j}\|\rendermat_G(\hat{\projmat}^{(i)}_{j+1})-\rendermat_G(\hat{\projmat}^{(i)}_j)\|_2$.
Eq. 10: $\mathcal{L}=\mathcal{L}_{rgb}+\lambda\mathcal{L}_{smooth}$. Paper: λ = 0.05 annealed to 0.01.

**What the code actually does** (train.py, scene/motion.py, scene/bezier.py, utils/loss_utils.py):
- Bézier curve (bezier.py): $p(t)=\sum_{k=0}^{C}\binom{C}{k}t^{C-k}(1-t)^k c_k$, C = 9.
  - Control-point index k pairs with $t^{C-k}$, so t = 1 gives c_0.
  - Init: all 10 points equal the pose's se(3) log, plus N(0, 0.001²) noise (`initial_noise`
    default; applied to both the rotation and translation curves).
- se(3) (motion.py `_set_initial_parameters`, curve_type "se3"):
  - pytorch3d row-vector convention: $M[:3,:3]=R_{c2w}^T$, $M[3,:3]=C$ (camera centre).
  - `params = se3_log_map(M)`. The translation curve holds `params[:, :3]` and the rotation curve
    holds `params[:, 3:]`.
  - Sub-frame: `se3_exp_map(cat[trans(ν), rot(ν)])`; then $R_{c2w}=M[:3,:3]^T$, $C=M[3,:3]$.
  - w2c = [R_c2w^T | −R_c2w^T C].
  - The vendored pytorch3d functions are in `utils/pytorch3d_functions.py` (BSD).
- ν (motion.py): raw $\_nu\in\mathbb R^{n\times(N-2)}$, initialised to
  logit(linspace(1/(N−1), 1−1/(N−1), N−2)).
  - Sub-frame times: sort(clamp(cat[0, σ(\_nu), 1], 0, 1)), 21 values. The end points are fixed.
  - `curve_random_sample` (jitter) defaults to False.
- Render (motion.py `query`): every sub-frame uses the same **random background** `torch.rand(3)`
  (one per step). blurred = mean_j(render_j). **No gamma on the render.**
- Loss target is the linearised GT: `gt = inverse_tonemap(gt) = clamp_min(gt, 1e-8)^2.2`
  (tonemapping.py). The loss is therefore computed in linear space, and γ is applied only at test.
  This is the code's realisation of Eq. 7.
- Loss (train.py:139-153):
  - `loss = L1(blur, gt_lin) + λ_t·L1(sub[1:], sub[:-1]) + λ_hinge·hinge_l2(opacity_raw)`.
  - L1 is the mean |·|.
  - Smoothness is the mean absolute difference of adjacent sub-frames, not the paper's L2 sum.
  - hinge_l2 = mean(x² for x≤0, (x−1)² for x≥1, else 0).
  - No D-SSIM term.
- Opacity activation: `Clamp()` = identity clamped to [0,1] (gaussian_model.py:47), not sigmoid.
  - Init 0.1; reset to min(o, 0.1); prune o < 0.005.
  - The hinge loss (λ_hinge = 0.1) keeps the raw value in range.
- Colour: SH + 0.5, ReLU (`activation relu`, same as Inria clamp). Max SH degree **2**.
- Scale activation: exp (lower bound 0).
- Densification (gaussian_model.py, train.py:180-191):
  - For each sub-frame: max_radii2D = max(...), grad_accum += ‖∇means2D[:2]‖, denom += 1/N
    (`add_densification_stats(..., 1.0/len(render_pkgs))`).
  - Densify when step > 500 and step % 200 == 0, while step < 75 000. Clone/split use θ(step);
    split N = 2, scale / (0.8·2).
  - Prune **only** opacity < 0.005 (no screen/world size prune).
  - Opacity reset every 3000 steps while step < 75 000.
- θ annealing: `get_expon_lr_func(4e-4, 2e-4, max_steps=25000)`, log-linear 4e-4 → 2e-4 over the
  first 25k steps, then 2e-4.
- λ_t: `get_expon_lr_func(1e-3, 1e-5, max_steps=150000)`, log-linear.
- Curve optimisation phases:
  - Steps < 1000 (curve_start_iter): the curve is frozen (requires_grad False) and a single
    sub-frame is rendered. `subframe_indice = 1` → `linspace(0, N−1, 1).long() = [0]`, i.e. ν = 0
    = control point c_9 (≈ the COLMAP pose).
  - 1000 ≤ step < 100 000: all 21 sub-frames, curve trainable.
  - Step 100 000 (curve_end_iter): the curve is frozen again and 21 sub-frames are still rendered.
- Curve LR: rotation 1e-3, translation (control points) 1e-2. From step 1000, each step
  multiplies the lr by 0.5^{1/15000} (halved every 15k).
- ν LR (`get_scheduler`): 0 until step 30 000, then exponential decay from `curve_alignment_lr`
  to 1e-7 at 150k. **Code default `curve_alignment_lr = 0.0` (comment "# 3e-3")**, so ν is never
  updated with the released defaults. Paper: 3e-3.
- Optimiser: a single Adam (eps 1e-15) holds the Gaussians and the curve params. The curve params
  are skipped while requires_grad is False (their grad is None).
- Scene extent (dataset_readers.getNerfppNorm, modified):
  `radius = min(percentile_10(‖C_cam − mean(SfM points)‖), 1.1·max‖C_cam − mean C‖)`.
- View sampling: shuffled stack (each training view once per epoch).

Test (test.py):
- Background white; render → γ (`clamp_min(1e-8)^(1/2.2)`).
- `optimize_test_pose`, 2000 epochs; each epoch visits every test view once, in shuffled order.
  - Parameters: a quaternion of R_c2w (lr 5e-5) and the w2c translation T (lr 5e-4).
  - Adam eps 1e-15; StepLR step 100 epochs, γ 0.9.
  - Loss: L1(clamp(γ(render), 0, 1), gt_sharp).
- Metrics on the optimised poses: PSNR (mean over views), SSIM (3DGS same-padded Gaussian SSIM),
  LPIPS **AlexNet** (lpipsPyTorch). The evaluate() image is not clamped.
- The paper reports post-optimised numbers ("we perform the post-optimization process to align
  the test image poses"). The scene is frozen during this step.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 150 000 | paper §4.1, arguments `iterations` |
| N sub-frames | 21 | paper §4.1, `num_subframes` |
| Bézier order | 9 | paper §4.1, `curve_order` |
| curve trans / rot lr | 1e-2 / 1e-3, halved every 15k | paper §4.1, args |
| ν lr | paper 3e-3; code default 0.0 (from 30k, exp decay to 1e-7 at 150k) | paper §4.1 / args:122,123 + train.py:79 |
| curve_start / end | 1000 / 100 000 | args |
| λ_smooth | paper 0.05 → 0.01; code 1e-3 → 1e-5 (expon over 150k, on mean-L1 form) | paper §3.4 / args |
| λ_hinge | 0.1 (code only) | args |
| θ densify | 4e-4 → 2e-4 over 25k (expon) | args (paper gives no numbers) |
| densify from / interval / until | 500 / 200 / 75 000 | args |
| opacity reset | every 3000 (< 75 000), to min(o, 0.1) | args, gaussian_model.reset_opacity |
| prune | opacity < 0.005 only | gaussian_model.densify_and_prune |
| percent_dense | 0.01 | args |
| position lr | 1.6e-4 → 1.6e-6 ×extent, over 150k | args |
| feature / opacity / scaling / rotation lr | 0.0025 (rest /20) / 0.05 / 0.005 / 0.001 | args |
| SH degree | 2 (+1 every 1000) | args `sh_degree` |
| opacity activation | clamp(0,1) identity | gaussian_model |
| tone mapping | gamma 2.2 | args `tone_mapping_type` |
| near plane | 0.2 | rasterizer |
| training background | random per step | motion.query |
| test pose opt | 2000 epochs, rot 5e-5, trans 5e-4, StepLR(100, 0.9) | test.py |
| LPIPS | AlexNet | test.py |

## 4. Schedules
- 1–999: single sub-frame at ν = 0, curve frozen.
- 1000: the curve unfreezes; 21 sub-frames from here on; the curve lr decay starts.
- 501–74 999: densify every 200 steps; θ anneals until 25k.
- Opacity reset every 3000 steps (< 75k).
- 30 000: the ν lr switches on (when nonzero) and decays to 1e-7 at 150k.
- 100 000: the curve freezes.
- λ_t and the xyz lr decay over 150k.

## 5. Builds on
Inria 3DGS: clone/split rules, SfM init, 3-NN init scales, Adam eps 1e-15, xyz LR schedule.
Changes:
- opacity activation (clamp + hinge);
- max SH degree 2;
- no D-SSIM;
- densify interval 200 until 75k, and θ annealing;
- reset value 0.1;
- no size pruning;
- scene-extent formula;
- random background.

## 6. Protocol
Real-motion-blur (Deblur-NeRF, Ma et al. 2022): 10 scenes.
- Scenes: ball, basket, buick, coffee, decoration, girl, heron, parterre, puppet, stair.
- Images: 2400×1600 JPGs, `-r 4` → 600×400 (PIL bicubic).
- COLMAP poses from all images (train + test); `hold=N` files give the split:
  int(name) % hold == 0 → sharp test.
  - Holds: ball 7, basket 7, buick 7, coffee 6, decoration 6, girl 7, heron 8, parterre 6,
    puppet 6, stair 6.
- Other datasets: Synthetic Extreme Blur and ExBlur (from ExBluRF). The noisy-pose ExBlur-NP
  variant is not released (README TO-DO).
- Metrics: PSNR / SSIM / LPIPS-Alex after test-pose optimisation.
- Only averages are reported (Table 1); there are no per-scene tables.
- The ablation (Table 2) is per scene but on ExBlur-NP (Camellia, Stone Lantern), which is
  unreleased.

## 7. Ambiguities
- **Paper vs code λ_smooth and loss form** (0.05→0.01 L2-sum vs 1e-3→1e-5 mean-L1). We follow the
  code.
- **ν lr**: the code default is 0 (ν frozen); the paper says 3e-3.
  - We use 3e-3 with the code's 30k-start / decay-to-1e-7 schedule, because the paper's ablation
    says ν is learned. This is a labelled choice.
  - Variant `deblurgs-nu0` = literal code default.
- **Gamma placement**: the paper applies γ to the render; the code applies γ⁻¹ to the GT and
  takes L1 in linear space. We follow the code.
- The released code is marked under construction, so its defaults may differ from the paper runs.
- The iteration count is 150k with 21 renders per step (≈ 3.1M renders per scene).
- The official evaluate() does not clamp the γ image; ss-eval clamps. The difference is
  negligible.
- The test-pose optimisation uses the test GT, as every method in Table 1 does. It is part of the
  protocol, not tuning.
