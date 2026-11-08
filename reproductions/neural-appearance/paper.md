# Compact Neural Appearance Models for Efficient Gaussian Splatting (arXiv 2609.05255)

Hahlbohm, Condor, Franke, Eisemann, Magnor, arXiv Sep 2026.
Official code: github.com/nerficg-project/efficient-gaussian-appearance (Apache-2.0), cloned to
`source/code/` (NeRFICG method "FasterGSVDA"). PPISP dependency: github.com/nv-tlabs/ppisp
(Apache-2.0, @a247368), cloned to `source/ppisp/`. LaTeX in `source/latex/`.

## 1. Method summary
- A controlled comparison of view-dependent appearance models (SH, Spherical Voronoi, NASG, NASGabor,
  Neural) inside one pipeline: Faster-GS rasterizer + MCMC densification + PPISP, only the
  appearance swapped.
- Common interface (Eq. 2): `c(d) = phi(c0 + A_psi(d))`, phi = ReLU(x + 0.5) by default.
- **Neural** (the paper's own model, §4): per-Gaussian latent `f in R^8` + base colour `c0 in R^3`;
  a tiny shared bias-free MLP (32 -> 16 -> 16 -> 3, ReLU, 816 params) decodes
  `A(d) = tanh(MLP([gamma_feat(f), gamma_dir(d)]))`.
- Fused into the CUDA rasterizer preprocess (tcnn FullyFusedMLP, fp16); the unfused reference path
  gives identical quality (supp. Table "Default w/o fusion": 28.56 vs 28.53 M360 avg).

Contributions
- C1 Neural appearance model (§4, Eq. 5-7, supp. §A.4): latent features + shared MLP + tanh residual,
  SH-schedule gating of the direction encoding, zero init, own LR schedules.
- C2 Common appearance interface / SH-as-baseline (§3.1 Eq. 1-2, supp. §A.1): base colour kept, SH
  residual over degrees 1..3 — used as the paired baseline row.
- C3 Shared pipeline (§5.1, supp. §B.1): Faster-GS + MCMC (scene caps) + PPISP (+controller
  distillation). Not a contribution of this paper but required for its numbers.
- (Not reproduced: fused CUDA/NVRTC kernels, WebGL viewer, SV/NASG/NASGabor models — the paper's
  claims for those are speed/footprint, not quality of the Neural row.)

## 2. Equations
Eq. 1 (SH): `c(d) = ReLU(0.5 + Y0 c0 + sum_{l=1}^{L} sum_m Y_l^m(d) c_l^m)`.
Eq. 2: `c(d) = phi(c0 + A_psi(d))`; Y0 folded into the base colour (lr scaled by Y0 instead).
Eq. 5: `x = (gamma_dir^B(d), gamma_feat^k(f)) in R^D` — code order is **[features, direction]**
  (`Neural.residual_colors`: `cat([features, sh_values])`).
Eq. 6: `gamma_feat^k(f) = (sin(2^j f), cos(2^j f))_{j=0}^{k-1}`, k = 1, F = 8 -> 16 values; code
  layout per feature: (sin, cos) interleaved (`frequency_encode`, stack(-1).flatten).
gamma_dir: real SH basis degrees {0,1,2,3} (16 values, tcnn sh_enc polynomials,
  `Appearance/utils.py:compute_sh_degrees`); degree 0 = constant 0.28209 acts as the bias.
Eq. 7: `A(d) = tanh(MLP_theta(x))`, MLP = 2 hidden layers x 16, ReLU, no biases, 3 outputs.
d = normalize(mu - camera_position) (gradient flows to mu, `DIRECTION_GRADIENT: true`).

## 3. Hyper-parameters (Neural, M360 garden/room config)
| name | value | source |
|---|---|---|
| F, k, SH degrees | 8, 1, [0,1,2,3] | config NEURAL |
| MLP | 32-16-16-3, ReLU, no bias | config / §4 |
| hidden init | tcnn default (Xavier-uniform) | supp. §A.4 |
| output layer init, features init | 0, 0 | supp. §A.4, `Neural.__init__` |
| base colour init | rgb_sfm - 0.5 (phi^{-1}) | Base.initialize_parameters |
| lr base colour | 0.00025, cosine 7k->37k to 0 | supp. Tab. lr, config |
| lr features | 0.008, cosine 7k->37k to 0 | idem |
| lr MLP weights | 0.003, Adam betas (0.9, 0.99), warm-up: LRDecayPolicy(init 3e-3, final 3e-4, delay 3000, delay_mult 1e-8, max 30000) | Neural.lr_schedulers |
| SH variant: lr base / shN | 0.0025*Y0 (= 0.0025 on sh0 units), 0.000125, constant | config sh |
| lr means | 1.6e-4 -> 1.6e-6 (x extent), log-linear over 30k | config |
| lr scales / rotations / opacities | 0.005 / 0.001 / 0.05 | config (MCMC) |
| Adam eps | 1e-15 (FusedAdam) | Base.setup_optimizer |
| MCMC | relocate+add every 100 in [600, 24900], min_opacity 0.005, +5%/step up to cap; noise 5e5*lr_means | Trainer/Model |
| caps | garden 5.0M, room 1.5M, bicycle 6M, truck 2.5M, ... | paper_configs/default/*/<scene>.yaml |
| MCMC init | opacity 0.5, scale 0.1 x RMS 3-NN distance | initialize_from_point_cloud |
| regularizers | 0.01 mean(opacity) + 0.01 mean(scale) | config LOSS |
| loss | 0.8 L1 + 0.2 D-SSIM (fused-ssim) | config LOSS |
| Morton | iterations 0, 5000, ..., 25000 | config |
| background | black | DATASET.BACKGROUND_COLOR |
| near / far | 0.2 / 10000 | DATASET |
| PPISP | default PPISPConfig, controller 5000 distillation steps after 30k, scene frozen, opacity/scale reg off | Model PPISP, Trainer.finalize |
| final prune | opacity < 1/255, then Morton sort | training_cleanup |

## 4. Schedules
- Appearance degree: +1 at iterations 1000/2000/3000 (residual off before 1000; inactive SH inputs = 0).
- Base colour and features: constant until 7000, cosine to 0 at 37000 (≈13% left at 30k).
- MLP: sinusoidal warm-up over 3000 (from 1e-8 x), log-linear decay to 0.1x at 30000.
- MCMC 600..24900 every 100; Morton until 25000; noise every step.
- PPISP: main params lr 0.002 linear warm-up 500 then exp decay to 0.01x over 30000; controller
  lr 0.001 fixed, only trained in the 5000 distillation steps (input detached, ISP params frozen).

## 5. What it builds on
Faster-GS (our `reproductions/faster-gs`, matched) **with MCMC and PPISP enabled** — i.e. not the
Faster-GS main configuration we reproduced (that one is 3DGS ADC, opacity lr 0.025, no PPISP).
Kept from 3DGS: L1+D-SSIM, SH degree schedule, means LR schedule, llffhold8.

## 6. Protocol
M360 (9), T&T (truck, train), DB (2), NeRF-Synthetic; 3DGS resolutions (outdoor /4, indoor /2) and
splits (every 8th test); 30k iterations (+5k PPISP distillation); black bg; PSNR/SSIM/LPIPS(VGG
assumed, NeRFICG default) — averaged over 5 runs; RTX 4090.

## 7. Ambiguities
- PPISP at test time: novel views use camera 0's vignetting/CRF and controller-predicted exposure /
  colour from the rendered radiance (PPISP `frame_idx=-1`).
- tcnn weight init seed/scale: tcnn default (Xavier uniform, seed 1337 in the Python bindings).
- fp16 MLP evaluation with loss scale 128 vs our fp32 — the paper's unfused path shows equal quality.
- NeRFICG M360 loader at IMAGE_SCALE_FACTOR 0.25/0.5 vs 3DGS images_4/images_2 (assumed equal, as in
  the faster-gs reproduction).
