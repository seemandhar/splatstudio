# CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians — implementation spec

Paliwal, Ye, Xiong, Kotovenko, Ranjan, Chandra, Kalantari — ECCV 2024, arXiv 2403.19495 (v2).
Sources: `source/latex/` (arXiv e-print: main.tex, sec/*.tex incl. supp.tex), `source/paper.pdf`,
official code `source/code/` (github.com/avinashpaliwal/CoherentGS, depth 1 clone, Inria
non-commercial license). Author-processed LLFF data (Depth Anything `depth_rel/*.pt`,
FlowFormer++ `flow/*.npy` + consistency weights `*w.npy`) from the README's Google-Drive link,
`~/datasets/llff/coherentgs-official/llff_sparse.zip` sha256 `84caaa59…0f673d`.
**Code defaults override the paper where they disagree; both are recorded.**

## 1. Method summary
- One Gaussian per pixel of every input view ("pixel/ray-structured"); its centre is
  constrained to the pixel ray: x = g(D_init[p] + ΔD[p], p). Colour DC = pixel colour.
- ΔD (and a residual opacity) are produced per view by an implicit convolutional decoder
  f_φ(n) of the normalised view index, combined with a C=5 channel depth-segmentation mask,
  so updates are coherent (smooth) within depth layers.
- D_init = per-view scale/offset aligned monocular depth (Depth Anything), aligned with a
  flow-correspondence loss (FlowFormer++ flows, fwd-bwd consistency mask) for 1000 its.
- Regularisers: TV / masked TV on rendered disparity (blend λ_s scheduled), flow loss between
  corresponding Gaussian positions, and multi-sample (supersampled) rendering late in training.
- No densification / pruning; regions unseen by all inputs stay empty (masked in evaluation).

Contributions
- **C1** Pixel-structured, ray-constrained Gaussians with residual-depth parameterisation (§4.1 Eq. 3).
- **C2** Single-view constraint: implicit decoder for residual depth and opacity + C-channel depth masks (§4.1, supp. §1).
- **C3** Multiview constraint: L_multi = (1-λs) L_TV + λs L_MTV on rendered disparity (§4.1 Eq. 4-5).
- **C4** Flow-based regularisation of corresponding Gaussian positions (§4.2 Eq. 6).
- **C5** Multisampling (several samples per pixel, averaged) (§4.2).
- **C6** Initialisation: flow-based scale/offset coarse alignment of monocular depth (§4.3 Eq. 8) and depth-dependent isotropic Gaussian radius (Eq. 9), view-count-dependent initial opacity.

## 2. Equations (verbatim from sec/4_algorithm.tex)
Eq. 3 (pixel Gaussians): `\mathbf{x} = g(D^{\text{init}}_n [\mathbf{p}] + \Delta D_n [\mathbf{p}], \mathbf{p})`,
g(d,p) back-projects pixel p at depth d of camera n.

Decoder: `\Delta D_n = f_{\phi}(n)`; with masks `\Delta D_n = \text{chsum}(S \odot f_{\phi}(n))`.
Code (scene/gaussian_model.py get_residual / get_learned_opac):
`ΔD = 50 * Σ_c tanh(f_φ(n))_c * S_c / (Σ_c S_c + 1e-10)`;
`α = sigmoid(logit(α0) + 100 * Σ_c tanh(f_ψ(n))_c * S_c / (Σ_c S_c + 1e-10))`.

Eq. 4: `\mathcal{L}_\mathrm{TV} = \Vert \nabla (\frac{1}{1 + R_{\Sigma, \alpha, \mathbf{x}, d}}) \Vert_1, \quad \mathcal{L}_\mathrm{MTV} = \Vert \nabla (\mathbf{S}\odot(\frac{1}{1 + R_{\Sigma, \alpha, \mathbf{x}, d}})) \Vert_1`
Eq. 5: `\mathcal{L}_\mathrm{multi} = (1 - \lambda_{s})\mathcal{L}_\mathrm{TV} + \lambda_{s}\mathcal{L}_\mathrm{MTV}`, λs 0→1.
Code (train.py tv / second_order_tv): R_d = rendered_depth / (alpha + 1e-6) (accumulated camera-z
over accumulated alpha); disp = 1/(R_d+1); tv(x) = mean|x[:,:,:-1]-x[:,:,1:]| + mean|x[:,:-1]-x[:,1:]|;
L_multi = γ tv(S ⊙ disp) + (1-γ) tv(disp), **γ = iteration / 20000 with iteration ∈ [7001, 20000]**
(i.e. γ ramps 0.35 → 1, not 0 → 1 as the paper says).

Eq. 6: `\mathcal{L}_\mathrm{flow} = \sum_{(i, j)}\sum_{\mathbf{p}} \Vert M_{i \rightarrow j} \odot ( g(D_i [\mathbf{p}], \mathbf{p}) - g(D_j [\mathbf{q}], \mathbf{q}) ) \Vert_1`.
Code (train.py warping_loss): each call shuffles the view indices, takes idx1 = first,
for every other idx2: `L1Loss_mean(backwarp(X_idx2, F_{idx1→idx2}) * M, X_idx1 * M)` (mean over
3·H·W incl. masked zeros), summed over idx2. X_n = world xyz of view n's Gaussians as a
[3,H,W] image. backwarp = grid_sample(bilinear, zeros, align_corners=False) at
grid = 2((x+u)/W − 0.5), 2((y+v)/H − 0.5). M = consistency weight w > 0.5 (binary).
Consistency weight (data_preparation_scripts/gen_flow.py): w = erode5x5(exp(−0.5 |backwarp(−F_{j→i}, F_{i→j}) − F_{i→j}|_1)).

Eq. 7 (objective): `\mathcal{L}(R, R) + \beta_m \mathcal{L_\text{multi}} + \beta_f \mathcal{L_\text{flow}}`, βm = 5, βf = 0.1.
Code: `loss = 0.8·ML1 + 0.2·(1−SSIM) + 0.05·VGG + 5·L_multi + 0.1·(0.1·L_flow)` — the flow term is
multiplied by 0.1 twice (`wrploss = warping_loss(...) * warp_coeff; loss += warp_coeff * wrploss`),
**effective βf = 0.01**. ML1 = Σ_k w_k L1(pool^k(I), pool^k(I_gt)), w = [1, 1/4, 1/8, 1/16, 1/32],
pool = AvgPool2d(2). VGG = (1/32)·L1(relu1_1(VGG19)(I), relu1_1(I_gt)) — only the first slice
(conv1_1+ReLU) is used (`break` after i=0), raw [0,1] input, no ImageNet normalisation.
SSIM = 3DGS ssim (11×11 Gaussian σ 1.5, zero "same" padding, mean).

Eq. 8 (coarse alignment): `\mathbf{s}^*, \mathbf{o}^* = \arg \min_{\mathbf{s}, \mathbf{o}} \sum_{(i,j)}\sum_{\mathbf{p}}\Vert M_{i \rightarrow j} \odot ( g(s_i D^{\text{m}}_i[\mathbf{p}] + o_i, \mathbf{p}) - g(s_j D^{\text{m}}_j[\mathbf{q}] + o_j, \mathbf{q}) ) \Vert_1`,
D_init = s·D^m + o. Same warping_loss as Eq. 6 on xyz built from s·D^m+o.

Eq. 9 (radius): `r = {f \cdot D^{\text{init}}}/{H}`. Code: r = tan(FoVy/2)·D/H·radius_mult
(= D/(2 f_y)·1.2), isotropic log-scale, identity rotation.

Multisampling (§4.2, train.py): for iteration > 12000: s = randint(1,4); render at s× resolution
(same FoV) and AvgPool2d(s) the image, depth and alpha before the losses.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| views | 3 (also 2, 4) | Table 1; code `--num_cameras` |
| image dir / resolution | images_8, 504×378 | code dataset_readers.py `factor = 8` |
| mask channels C | 5 | §4.1, supp. Tab. 5; code `num_mask_channels` |
| init opacity | 0.6 / 0.5 / 0.35 (2/3/4 views) | §4.3; gaussian_model.py create_from_pcd |
| radius_mult | 1.2 | arguments/__init__.py |
| depth decoder capacity ngf | 10 / 15 / 18 | supp. §1; gaussian_model.py set_ref_camera |
| opacity decoder ngf | 6 / 10 / 12 | same |
| decoder layer widths | [16,16,16,8,8,8,4,4,…]·ngf, 1×1 first conv then 3×3, LeakyReLU 0.2, bilinear ×2 (align_corners) + reflect pad, coordconv concat after layer 1 | scene/decoder.py |
| decoder input | posenc(linspace(0,1,N)), 10 log freqs, sin/cos, no identity → 20 ch | posenc.py, gaussian_model.py |
| residual scale | 50·tanh (depth), 100·tanh (opacity logit) | gaussian_model.py |
| depth decoder opt | AdamW lr 1e-6 (wd 0.01 default), CyclicLR 1e-6..1e-4 triangular2, step_size_up 1000 | set_ref_camera |
| opacity decoder opt | AdamW, CyclicLR 1e-6..1e-5 triangular2, step_size_up 2000 | set_ref_camera |
| alignment opt | Adam scale lr 1, offset lr 5, CosineAnnealing T_max 1000, s=1, o=0 init | gaussian_model.py |
| alignment iterations | 1000 | §5; train.py 6001..7000 |
| training iterations | 13000 (7001..20000) | §5; train.py |
| feature_lr (shN) | 0.0005/10 = 5e-5; **DC not optimised** | arguments, training_setup (f_dc is a non-leaf-grad tensor) |
| scaling_lr | 1e-4 | arguments |
| rotation_lr | 1e-5 | arguments |
| λ_dssim | 0.2 (L1 term 0.8 on multiscale L1) | arguments, train.py |
| VGG weight | 0.05 | train.py |
| βm (smooth_coeff) | 5 | §4.2, train.py |
| βf | paper 0.1; code effective 0.01 | §4.2 vs train.py |
| flow mask threshold | 0.5 | train.py |
| scale reset | every 100 its while it < 15000, from current depth | train.py, set_scaling(use_decoder=True) |
| SH degree | 0 until it 15000, +1 at 15000/16000/17000 (max 3) | train.py |
| multisampling | it > 12000, randint(1,4) | train.py |
| background | white | arguments `_white_background = True` |
| bilateral filter | 3D-Photo-Inpainting sparse_bilateral_filtering, iters 5, filter_size [7,7,5,5,5], σs 4, σr 0.5, depth_threshold 0.1 | argument.yaml, bilateral_filtering.py |

## 4. Schedules (official iteration numbers; our step s = it − 7000)
- it 6001–7000: alignment only (no rendering).
- it 7001–15000 (s 1–8000): radius reset every 100 its; SH 0; decoders + shN/scale/rot optimised.
- it 12001+ (s 5001+): multisampling.
- it 15000/16000/17000 (s 8000/9000/10000): SH degree +1.
- γ = it/20000 continuously; CyclicLR schedulers stepped every iteration; no optimiser step on the
  final iteration.
- Shuffled-stack camera sampling (each view once per 3 iterations).

## 5. What it builds on
3DGS (Inria) code base with FSGS's depth-rendering rasterizer (diff-gaussian-rasterization-
confidence; confidence = 1). Kept: rasterisation, SH colour, Adam for SH/scale/rotation.
Removed: densification, pruning, opacity reset, position optimisation (positions only via ΔD),
per-Gaussian opacity optimisation (only via the opacity decoder), DC optimisation.

## 6. Protocol
- LLFF (8 scenes), images_8 (504×378), poses from poses_bounds.npy (fix_rotation to OpenCV,
  translations scaled by 1/(0.75·min bound)), principal point at the image centre.
- Split: test = every 8th image (name-sorted), train = `round(linspace(0, n_train−1, N))` of the
  rest (RegNeRF/FSGS split; same names as `llff-fewview3-r8`).
- Metrics: PSNR/SSIM/LPIPS, **masked**: "regions with rendered opacity below 1e-3 are considered
  occluded and masked out in numerical evaluation. We use the same mask for all approaches." (§5.1).
  The eval script is not in the repo (README says it is FSGS's); loss_utils.ssim's `mask`
  argument fills masked pixels with 1 (white) in both images, which we adopt for the masked metrics.
- NVS-RGBD (ZED2, iPhone) also reported — dataset not available here.
- GPU V100; seeds not stated (3DGS safe_state seed 0).

## 7. Ambiguities / paper-vs-code differences
1. βf: paper 0.1, code 0.01 effective (double multiplication). Use code.
2. λs: paper 0 → 1, code it/20000 (0.35 → 1). Use code.
3. Paper eq. 9 r = f·D/H; code r = tan(FoVy/2)·D/H·1.2 = 0.6·D/f. Use code.
4. Loss: paper "L1 + SSIM as 3DGS"; code multiscale L1 + 0.05 VGG relu1_1 term. Use code.
5. DC colour fixed to the pixel colour in code (not in paper). Use code.
6. Masked-eval details (fill value, LPIPS net) are not given; we compute both unmasked (native
   ss-eval) and masked (white-fill in both, as loss_utils.ssim(mask=)) metrics; LPIPS VGG.
7. Rotation: paper "identity for the first 8000 its"; code optimises rotation at lr 1e-5 from
   it 7001 but resets the isotropic scale every 100 its until 15000 (rotation of an isotropic
   Gaussian is irrelevant). Equivalent in effect.
8. Pixel-to-ray convention: kornia depth_to_3d uses integer pixel coordinates with cx = W/2;
   flow warping uses grid_sample align_corners=False (a half-pixel offset). Replicated verbatim.
