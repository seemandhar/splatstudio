# FastGS: Training 3D Gaussian Splatting in 100 Seconds — implementation spec

arXiv 2511.04283 (CVPR 2026). Official code: https://github.com/fastgs/FastGS (MIT; the vendored
rasterizer `submodules/diff-gaussian-rasterization_fastgs` carries the Inria non-commercial
research licence), cloned at `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f` (2026-03-03) into
`source/code/`. LaTeX in `source/latex/` (main + supplementary). **Code values override the
paper where they disagree; both are recorded.**

## 1. Summary
- 3DGS trained with the vanilla loss (L1 + 0.2 D-SSIM), on top of "3DGS-accel" (Taming-3DGS
  per-splat backward + batched SH/optimizer step schedule).
- **C1 VCD — multi-view consistent densification** (§3.2): candidate Gaussians (3DGS gradient
  for clone, AbsGS absolute gradient for split) are densified only if, averaged over K=10
  random training views, more than τ_d=5 high-error pixels lie in their 2D footprint.
- **C2 VCP — multi-view consistent pruning** (§3.3): pruning score = min-max normalised
  Σ_views (high-error pixel count in footprint × view photometric loss). Before 15k: 3DGS
  prune candidates sub-sampled by score (budget 50%); after 15k every 3k: remove opacity < 0.1
  or score > 0.9.
- **C3 Compact Box (CB)** (§3.4, supp. §7): Speedy-Splat SnugBox/AccuTile tile assignment with
  the Mahalanobis threshold scaled by β (`mult`, 0.5 default, 0.7 T&T/DB) — mainly a speed
  device, but it truncates each splat at tile granularity in forward/backward, so it changes
  the image and the VCD/VCP footprints.
- Supporting (not claimed as contributions): AbsGS absolute gradient for splitting; opacity
  clamp to ≤ 0.8 after each densification; optimizer step schedule (supp. §8.1).

## 2. Equations (verbatim from sec/4_method.tex, supp.)
Per-pixel error, view j:  $e^j_{u,v}=\frac{1}{C'}\sum_{c'}|r^{j,c'}_{u,v}-g^{j,c'}_{u,v}|$  (Eq. 1)
Loss map: $\mathcal{M}^j=\mathcal{N}(\{e^j_{u,v}\})$, $\mathcal{N}$ = min–max normalisation (Eq. 2)
Mask: $\mathcal{M}^j_{mask}=\mathbb{I}(\mathcal{M}^j>\tau)$ (Eq. 3)
Importance: $s_d^i=\frac{1}{K}\sum_{j=1}^K\sum_p^{\Omega_i}\mathbb{I}(\mathcal{M}^j_{mask}(p)=1)$ (Eq. 4, `eq:s+`)
Densify iff $s_d^i>\tau_d$ (and the gradient criterion).
View photometric loss: $E^j_{photo}=(1-\lambda)L_1^j+\lambda(1-L^j_{SSIM})$ (Eq. 5)
Pruning score: $s_p^i=\mathcal{N}\big(\sum_j(\sum_p^{\Omega_i}\mathbb{I}(\mathcal{M}^j_{mask}(p)=1))\cdot E^j_{photo}\big)$ (Eq. 6, `eq:s-`)
CB threshold: $(p-\mu)\Sigma^{-1}(p-\mu)^T=\beta(2\ln\frac{\sigma_i}{\tau_\alpha})$, $\tau_\alpha=1/255$ (supp. Eq. 15)
Training loss: $\mathcal{L}=(1-\lambda)\mathcal{L}_1+\lambda(1-\mathcal{L}_{SSIM})$, λ = 0.2 (Eq. 7)

Code specifics (`utils/fast_utils.py`, `scene/gaussian_model.py`, rasterizer `forward.cu`):
- Footprint Ω_i = pixels where Gaussian i contributes in the forward α-blend (power ≤ 0,
  α=min(0.99, o·e^power) ≥ 1/255, before T would drop below 1e-4), restricted to tiles that CB
  assigns it (`forward.cu:403` atomicAdd when `metric_map[pix]==1`).
- e: mean over channels of |render − gt| (render unclamped), min–max over the image; mask
  `> loss_thresh`.
- s_d = floor(Σ_j count_j / K) (`torch.div(..., rounding_mode='floor')`), densify mask `> 5`.
- s_p over the *current* population with min/max over all Gaussians (no NaN guard).
- The 10 views are drawn without replacement from all training cameras at each call.
- E_photo uses `fused_ssim` (same/zero padding), λ=0.2, of the unclamped render.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | paper §4.1, arguments |
| K (views) | 10 | paper §4.1, fast_utils.sampling_cameras |
| τ (loss_thresh) | 0.1 (garden 0.06) | arguments; train_base.sh |
| τ_d | 5 (strict `>`) | supp. §8.1; gaussian_model.py:497 |
| τ_p (late prune) | 0.9; opacity < 0.1 | supp. §8.1; train.py |
| clone grad (grad_thresh) | 0.0002 | arguments |
| split abs grad (grad_abs_thresh) | 0.0012 default, per scene | arguments; scripts |
| clone/split scale boundary (`dense`·extent) | 0.001 default, per scene | arguments; scripts |
| densify interval | 500 (FastGS), 100 (FastGS-Big) | paper §4.1; scripts |
| densify from / until | >500 / <15000 | arguments |
| opacity reset | every 3000 while <15000, to min(o,0.01) (Adam moments zeroed) | train.py |
| post-densify opacity clamp | min(o, 0.8) (Adam moments zeroed) | gaussian_model.py:520 |
| prune before 15k | opacity < 0.005; after 3000 also max scale > 0.1·extent (screen-size test is a no-op: max_radii2D zeroed by densification_postfix) | train.py, gaussian_model.py |
| prune budget | int(0.5·#candidates) multinomial draws, weights 1/(1e-6+1−s_p) | gaussian_model.py:503 |
| late prune | at 18000, 21000, 24000, 27000 | train.py (`>15000`, `<30000`, `%3000`) |
| lr means | 1.6e-4·extent → 1.6e-6·extent, exp, delay_mult 0.01, 30k | arguments |
| lr sh0 (lowfeature_lr) | 0.0025 (drjohnson-big 0.0005) | arguments |
| lr shN | highfeature_lr/20; highfeature_lr 0.005 default, per scene | gaussian_model.py:205 |
| lr opacity / scales / quats | 0.025 / 0.005 / 0.001 | arguments |
| Adam eps | 1e-15 | gaussian_model.py |
| optimizer schedule | ≤15000: main Adam every step, shN Adam every 16 steps; 15001–20000: both every 32 steps; >20000: every 64 steps (gradients summed between steps) | gaussian_model.optimizer_step |
| CB β (`mult`) | 0.5 (m360), 0.7 (T&T, DB) | arguments; scripts |
| SH degree | +1 every 1000 to 3 | train.py |
| init | SfM points, opacity 0.1, 3-NN scales | 3DGS |
| background | black | arguments (`white_background` False) |

Per-scene overrides (train_base.sh / train_big.sh):
| scene | base: grad_abs, dense, hf_lr, other | big: grad_abs, dense, hf_lr |
|---|---|---|
| bicycle | 0.0012, –, – | 0.0008 |
| flowers | 0.0015, 0.005 | 0.001, 0.005 |
| garden | 0.0008, –, 0.02, loss_thresh 0.06 | 0.0003, –, 0.02, loss_thresh 0.06 |
| stump | 0.0015, 0.004 | 0.001, 0.004 |
| treehill | 0.002, 0.01 | 0.0018, 0.01 |
| room | 0.0008, –, 0.02 | 0.0004, –, 0.02 |
| counter | 0.0008, –, 0.02 | 0.0004, –, 0.02 |
| kitchen | 0.0006, –, 0.02 | 0.0002, –, 0.02 |
| bonsai | 0.0006, –, 0.02 | 0.0002, –, 0.02 |
| truck | 0.0009, –, 0.04, mult 0.7 | 0.0004, –, 0.04, mult 0.7 |
| train | 0.0015, 0.01, 0.042, mult 0.7 | 0.0004, 0.015, 0.042, mult 0.7 |
| playroom | (0.0012), 0.003, 0.0015, mult 0.7 | 0.0005, 0.003, 0.0015, mult 0.7 |
| drjohnson | 0.0012, 0.013, 0.0025, mult 0.7 | 0.0005, 0.005, 0.0025, lowfeature 0.0005, mult 0.7 |

## 4. Schedules
- Steps 1–15000: accumulate ‖∇μ2D‖ (signed) and ‖Σ|∇μ2D|‖ (abs) for visible Gaussians.
- s>500, s%interval==0, s<15000: score (VCD+VCP, 10 views) → clone → split → prune
  (score-sampled) → opacity clamp 0.8. Stats reset after clone/split.
- s%3000==0, s<15000: opacity reset 0.01 (after the densification of the same step).
- s ∈ {18000, 21000, 24000, 27000}: VCP late prune (10 views, pruning score only).
- Optimizer step schedule above; SH degree every 1000.

## 5. Builds on
3DGS (Inria code) + Taming-3DGS (per-splat backward, SH batching → generalised optimizer
schedule) + AbsGS (abs gradient for split) + Speedy-Splat (SnugBox/AccuTile → CB). Loss,
initialisation, LR schedules, SH schedule and opacity reset are 3DGS.

## 6. Protocol
Mip-NeRF 360 (9 scenes), T&T (truck, train), DB (drjohnson, playroom); `--eval` llffhold8;
`-i images` with `--resolution -1` → images wider than 1.6K are PIL-resized to width 1600
(Mip-NeRF 360 all scenes) — our `mipnerf360-1600` dataset. 30k iterations, black background,
RTX 4090. Metrics: official metrics.py (PSNR, SSIM, LPIPS-VGG via lpipsPyTorch). Per-scene
tables in supp. Tables 17–19 for both FastGS and FastGS-Big.

## 7. Ambiguities / code-vs-paper
1. Paper "sampling half of the candidate Gaussians according to their pruning scores": code
   draws int(0.5·#candidates) samples from **all** pre-densification Gaussians (weights
   positional, computed before clone/split shifted the indices) and prunes only candidates that
   were drawn — so far fewer than half of the candidates are pruned. Reproduced as coded.
2. s_p in densification is computed before clone/split; its indices are reused positionally on
   the post-densification arrays (as in Taming-3DGS).
3. Paper τ_d "exceeds" = code `> 5` on floor-averaged counts.
4. Clone/split boundary uses `dense` (default 0.001), not 3DGS percent_dense 0.01.
5. Two variants; both have per-scene numbers. FastGS (interval 500) is the paper's main row.
6. CB described as speed-only ("comparable quality", Table cs: 27.46 → 27.44 dB), but it is part
   of both training and evaluation rendering in the code.
7. 3DGS-accel's per-splat backward is a speed-only kernel; numerically the same gradients.
