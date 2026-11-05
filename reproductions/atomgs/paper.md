# AtomGS: Atomizing Gaussian Splatting for High-Fidelity Radiance Field — implementation spec

arXiv 2405.12369 (BMVC 2024). Official code: github.com/RongLiu-Leo/AtomicGS (cloned to
`source/code`, commit 5b9ce904). Built on Inria 3DGS + GS-Monitor; vendors a modified Inria
rasterizer (`submodules/diff-gaussian-rasterization`, slothfulxtx "diff-gauss" variant: extra
mean depth / median depth / alpha outputs, depth and alpha backward). Inria/MPII Gaussian-Splatting
licence (non-commercial research).

**Code overrides paper where they disagree (recorded below).**

## 1. Method summary
- 3DGS training with a re-designed adaptive density control ("Atomized Proliferation", §3.1,
  supp. Alg. 1): an *atom scale* S (1st percentile of SfM NN distances) that decays
  geometrically; Gaussians whose smallest axis ≤ S are reset to isotropic atoms of radius S;
  only Gaussians larger than S may split; split threshold warm-up (linear 0→τs over 7k).
- Clone has no size condition (any high-gradient Gaussian is cloned) and split happens before
  clone; opacity reset every 300 iterations (to ≤0.1) with prune threshold 0.005.
- Geometry-Guided Optimization (§3.2): Edge-Aware Normal Loss on normals derived from the
  rendered (accumulated) mean depth, weighted by (|∇I|-1)^500 so it vanishes at image edges;
  active only during atomized proliferation (iter < 7000).
- Photometric loss L = 0.9 L1 + 0.1 (1 − "MS-SSIM") (3-scale averaged SSIM, not D-SSIM 0.2).
- Feature renderer: alpha-blended mean depth, median depth, alpha (supp. §"Gaussian Splatting
  and Additional Feature Rendering").

Contributions
- C1 Atomized proliferation (atom scale init + decay, atomize op, atom-constrained split,
  warm-up split threshold, unconstrained clone, 300-step opacity reset) — §3.1, supp. Alg. 1.
- C2 Edge-aware normal loss with depth-derived normals — §3.2, Eq. (normal loss), supp. unprojection.
- C3 Mean/median depth + alpha feature rasterizer (enabler for C2) — supp. "Additional Feature Rendering".
- C4 MS-SSIM photometric term (λ=0.1) — supp. hyper-parameters.

## 2. Equations
Rendering weight / maps (supp.):
$w_i(x)=\alpha_i\mathcal G_i(x)\prod_{j<i}(1-\alpha_j\mathcal G_j(x))$,
$C=\sum c_i w_i$, $A=\sum w_i$, $D_{mean}=\sum z_i w_i$ (NOT normalised by A),
$D_{median}=z_i$ for the Gaussian where T crosses 0.5.

Unprojection (supp.; `utils/image_utils.py:unproject_depth_map`):
$x_{norm}=2j/(W-1)-1$, $y_{norm}=2i/(H-1)-1$, $s_d=(f_1 d+f_2)/(d+10^{-8})$ with
$f_1=P^\top[2,2]=z_f/(z_f-z_n)$, $f_2=P^\top[3,2]=-z_f z_n/(z_f-z_n)$ ($z_n=0.01,z_f=100$),
$p_w=[x_{norm},y_{norm},s_d,1]\,T^{-1}$ (row vector, $T$ = full_proj_transform), divide by w.
Points padded with a zero last row/column (H+1,W+1), $p_1=P[:-1,:-1],p_2=P[1:,:-1],p_3=P[:-1,1:]$,
$N=\frac{(p_2-p_1)\times(p_3-p_1)}{\|\cdot\|+10^{-8}}$ (supp. Normal Map Calculation).

Edge-aware normal loss (§3.2; `utils/loss_utils.py:edge_aware_normal_loss`):
Sobel kernels /4, zero padding 1; $\partial N_{x,y}$ per normal channel;
$\partial I_{x,y}=\mathrm{mean}_{rgb}|\mathrm{sobel}(I_{gt})|$;
$\omega_{x,y}=(\partial I_{x,y}-1)^{500}$;
$\mathcal L_{normal}=\mathrm{mean}_{pixels}\big\|\,|\partial_x N|\omega_x+|\partial_y N|\omega_y\big\|_{2,\text{over xyz}}$.

Total loss (train.py): $\mathcal L=(1-\lambda_{ms})\mathcal L_1+\lambda_{ms}(1-\mathrm{MSSSIM})
+[i<t_a]\lambda_n\mathcal L_{normal}$, MSSSIM $=\frac13\sum_{k=0}^{2}\mathrm{SSIM}(\downarrow^{k}x,\downarrow^{k}y)$
with 2×2 avg-pool downsampling and the Inria SSIM (11×11 Gaussian σ=1.5, zero "same" padding).

Atomized proliferation (supp. Alg. 1; code `scene/gaussian_model.py`), at iteration i:
- prune: $\alpha<\epsilon$ or $\max s>0.1\cdot$extent (always, not only after the first reset)
- split: $\bar g\ge\min(i\tau_s/t_w,\tau_s)$ and $\max s>\mathcal S$; 2 children sampled
  $\mathcal N(\mu,\Sigma)$, $s/1.6$, same opacity/colour; parent removed
- clone: $\bar g\ge\tau_c$ (no scale condition); exact copy
- atomize (every 100 it, i<t_a): rows with $\min s\le\mathcal S$ set to $s=(\mathcal S,\mathcal S,\mathcal S)$;
  then $\mathcal S\leftarrow0.99\mathcal S$ ("geometric progression ensures the atom scale will be
  optimized into 1/2 at 7k": 0.99^69≈0.50)
- $\mathcal S_0=P_{q}(d)$, $d_k=\sqrt{\max(\text{mean 3-NN sq. dist},10^{-7})}$, q=0.01;
  initial scales $\max(d_k,\mathcal S_0)$ isotropic.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 30000 | arguments/__init__.py:73 |
| position lr | 1.6e-4→1.6e-6 ×extent, max_steps **7000**, delay_mult 0.01 (delay_steps 0 ⇒ inert) | args:74-77 |
| feature lr / rest | 0.0025 / 0.0025/20 | args:78, gaussian_model |
| opacity / scaling / rotation lr | 0.05 / 0.005 / 0.001 | args:79-81 |
| λ_ms_ssim | 0.1 | args:82, supp. |
| λ_normal | 0.1 | args:83, supp. |
| densification_interval | 100 | args |
| densify_from_iter | 100 (first densify at 200, `>`) | args, train.py |
| opacity_reset_interval | **300** | args |
| atom_proliferation_until t_a | 7000 | args, supp. |
| warm_up_until t_w | 7000 | args, supp. |
| densify_until_iter | 15000 | args |
| prune_threshold ε | 0.005 | args, supp. |
| clone / split threshold τc, τs | **0.0002** (paper text says 0.002 "following 3DGS" — typo; 3DGS is 0.0002) | args |
| atom_init_quantile | 0.01 | args, supp. P1 |
| opacity reset value | min(α, **0.1**) (3DGS uses 0.01) | gaussian_model.reset_opacity |
| init opacity | 0.1 | create_from_pcd |
| split N, scale divisor | 2, 1.6 | split_points |
| world prune | max s > 0.1·extent (every densify) | densify_and_prune |
| SH degree | 3, +1 every 1000 | train.py |
| background | black | white_background False |
| Adam eps | 1e-15 | training_setup |

## 4. Schedules
- iter 1..14999: densification stats (‖∇μ2D[:2]‖ NDC, visible only).
- densify at 200, 300, …, 14900; split threshold ramps `min(2e-4·i/7000, 2e-4)`.
- atomize at 100, 200, …, 6900 (69 calls; S halves by 7k); S stays fixed afterwards (still gates splits until 15k).
- opacity reset at 300, 600, …, 14700 (Adam moments of opacity zeroed).
- normal loss on for iter < 7000.
- xyz LR exponential decay over the first 7000 iterations, then constant 1.6e-6·extent.
- Order inside an official iteration: render → loss → backward → stats → densify → atomize →
  reset → optimizer.step (replaced tensors have no grad ⇒ skip that step).

## 5. Builds on
Inria 3DGS (rasterizer projection, EWA 0.3 dilation, alpha clamp 0.99, 3-NN init, Adam, SH
schedule, opacity activation, world-size prune). Changed: loss, density control, opacity-reset
interval/value, xyz-LR horizon, init-scale clamp, feature rasterizer.

Quirks of the official densify_and_prune (replicated; `official_quirks`):
- `grads` computed once; split first, then `clone_mask` is built from the *pre-split* gradient
  vector zero-padded to the post-split length although split removed parents and appended
  children ⇒ clone mask is index-shifted relative to the Gaussians it is applied to.
- `densification_postfix` resets grad accumulators after every densify (even if nothing is added).
- No `max_radii2D` / screen-size prune at all.
- `atomize` replaces the scaling tensor and zeroes Adam moments of **all** Gaussians every 100 it.

## 6. Protocol
Mip-NeRF 360 (9 scenes) + Tanks&Temples (truck, train), PSNR/SSIM/LPIPS; DTU chamfer (geometry,
not reproduced here). No eval script in the repo; paper's per-scene 3DGS row (retrained) has
LPIPS ≈ 3DGS-paper values ⇒ 3DGS full_eval protocol: images_4 (outdoor) / images_2 (indoor),
llffhold 8 (`--eval`), 30k iterations, black background, LPIPS-VGG (lpipsPyTorch). Single RTX 4090.
Per-scene numbers: supp. Table "breakdown" (M360 9 scenes + truck/train).

## 7. Ambiguities
- τ = 0.002 in paper vs 0.0002 in code → code.
- Resolution protocol not stated → 3DGS full_eval (images_4/2), as implied by LPIPS.
- The clone/split index misalignment is presumably unintended but is what produced the numbers.
- Mean depth is unnormalised (background → 0 depth → s_d≈−1e6 → far points); normals there
  are arbitrary; border row/column normals use a zero point. Replicated.
- Paper Alg. 1 order (prune, clone, split, atomize) differs from code (split, clone, prune; atomize after) → code.
