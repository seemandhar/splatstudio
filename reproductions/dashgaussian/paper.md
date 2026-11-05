# DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds — implementation spec

arXiv 2503.18402 (CVPR 2025 highlight). Chen, Jiang, Jiang, Tang, Li, Liu, Nie.
Official code: https://github.com/YouyuChen0207/DashGaussian (cloned to `source/code/`, CC BY-NC-SA 4.0,
includes the 2025-08-16 "reproduction bug fix"). DashGaussian is a plug-in schedule for 3DGS backbones;
this spec targets the **3DGS (Adam) backbone** variant (`train_dash.py`, `full_eval.py --dash`).

## 1. Method summary
- Train with a **rendering resolution that rises from low to full** over the densification period;
  the schedule is derived per scene from the Fourier spectra of the training images.
- LR supervision = antialiased bilinear downsampling of the GT (PyTorch `interpolate(antialias=True)`);
  the rasterizer renders at the reduced size with the same field of view (= scaled intrinsics).
- **Primitive count grows in sync with resolution**: target count P_i is a function of the current
  render scale; densification is capped to the top-k gradient candidates to hit that target.
- **Momentum-based budget**: the final count P_fin is not preset but integrated from the number of
  primitives that naturally pass the densification threshold.
- Position LR is held at its initial value during the low-resolution stage and decays afterwards.

Contributions
- **C1 Frequency-guided resolution scheduler** (Sec. 3.2, Eq. 2–3, supp. Sec. 7.3 Eq. 6–7, code `utils/schedule_utils.py:init_reso_scheduler,get_res_scale`).
- **C2 Resolution-guided primitive scheduler / top-k densification** (Sec. 3.3 Eq. 4, supp. Sec. 7.4, code `get_densify_rate`, `scene/gaussian_model.py:prune_and_densify`).
- **C3 Momentum-based primitive budgeting** (Sec. 3.3 Eq. 5, code `update_momentum`).
- **C4 Position-LR schedule tied to the resolution schedule** (supp. Sec. 7.2, code `utils/general_utils.py:helper_dashgaussian`, `lr_decay_from_iter`).
- (Implementation) GT downsampling with antialiased bilinear (supp. Sec. 7.1, `train_dash.py:113`).

## 2. Equations
Significance (Eq. 2):
$$\mathcal{X}(\mathbf{F})=\frac{1}{N}\sum_{n=1}^N\sum_{i=1}^h\sum_{j=1}^w{||\mathbf{F}^n(i,j)||_2}$$
In code, $\mathcal{X}(\mathbf{F}_r)$ is the sum of the **centred** |DFT| of the full-resolution image
inside a centred window of size (int(H/r), int(W/r)) (i.e. the frequencies an ideal r× downsample keeps),
summed over the 3 colour channels, averaged over training views (`compute_win_significance`).

Fraction (Eq. 3): $f(\mathbf{F},\mathbf{F}_r)=\frac{\mathcal{X}(\mathbf{F})-\mathcal{X}(\mathbf{F}_r)}{\mathcal{X}(\mathbf{F})}$, $s_r=S\cdot\mathcal{X}(\mathbf{F}_r)/\mathcal{X}(\mathbf{F})$.

Actually used (supp. Eq. 6–7, log modulation):
$$s_{r_i}=S\cdot\frac{\ln(\mathcal{X}(\mathbf{F}_{r_i})/\mathcal{X}(\mathbf{F}_{r_m}))}{\ln(\mathcal{X}(\mathbf{F})/\mathcal{X}(\mathbf{F}_{r_m}))}$$
with S = `increase_reso_until` = densify_until_iter = **27000** in code (paper: S = total steps).

Max factor $r_m$: $\mathcal{X}(\mathbf{F}_{r_m})=\frac1a\mathcal{X}(\mathbf{F})$, a = 4. Code: for each training
image solve the per-image r with window energy = E_total/4 (bisection on 1/r in (0,1), 64 iters,
window significance monotone), take the **minimum over images**, capped at 8. Intermediate levels: m = 32
sample points; significance targets linearly spaced between $\mathcal{X}(\mathbf{F}_{r_m})$ (=E_min of the
mean spectrum at r_m) and E_total: $E_i = E_{min} + (E_{total}-E_{min})\, i/31$, i = 1..30, each solved for r_i;
level 31 is r = 1. Begin step of level i: int(27000 · ln(E_i/E_min)/ln(E_total/E_min)); level 0 begins at 0;
an extra sentinel 27000 is appended.

Resolution between levels (paper: "interpolated with the inverse square"; code `get_res_scale`): for
k in [s_i, s_{i+1}) with neighbouring scales r_{i-1} (at s_i) → r_i (at s_{i+1}):
$$r^{(k)}=\Big(\tfrac{k-s_i}{s_{i+1}-s_i}(r_i^{-2}-r_{i-1}^{-2})+r_{i-1}^{-2}\Big)^{-1/2},\quad r^{(k)}\leftarrow\lfloor r^{(k)}\rfloor$$
Code detail: begin[1] = int(S·ln(E_0/E_min)/L) = 0, so the `k < s_1` branch (unfloored r_m) is
dead code; every step uses the floored interpolation (level i interpolates r_{i-1} → r_i over
[begin[i], begin[i+1]), begin[i+1] = int(S·ln(E_i/E_min)/ln(E_total/E_min))). For k ≥ 27000: r = 1.
The render scale is only re-evaluated **at densification steps** (k > 500, k % 100 == 0, k < 27000),
after densification, with `get_res_scale(k)`; the initial value is `get_res_scale(1)` = r_m.
LR image size = floor(H/r), floor(W/r) (`interpolate(scale_factor=1/r, recompute_scale_factor=True)`).

Primitive target (Eq. 4):
$$P_i = P_\mathrm{init} + (P_\mathrm{fin}-P_\mathrm{init})/(r^{(i)})^{2-i/S}$$
Code: `next_n = int((max_n - init_n) / cur_scale**(2 - it/densify_until_iter)) + init_n` if
`it + 100 < 27000`, else `next_n = max_n`; `densify_rate = min(max((next_n - cur_n)/cur_n, 0), 0.2)`.
`cur_scale` = the render scale used at this step (before re-evaluation).

Densification (supp. 7.4, `prune_and_densify`): (1) prune opacity < 0.005 (no screen-size or
world-size pruning); (2) n_densify = min(int(cur_n·(1+rate) − n_after_prune), n_after_prune);
(3) grads = accum/denom (NaN→0); top-n_densify by grad; clone those with grad ≥ 2e-4 and
max scale ≤ 0.01·extent; split (N=2, scale/1.6, samples ~ N(0, Σ)) those with grad ≥ 2e-4 and max
scale > 0.01·extent; (4) return P_add = #(grads ≥ 2e-4) (after prune, before densify).

Momentum (Eq. 5): $P_\mathrm{fin}^{(i)}=\max(P_\mathrm{fin}^{(i-1)},\gamma P_\mathrm{fin}^{(i-1)}+\eta P_\mathrm{add}^{(i)})$, γ = 0.98, η = 1.
Code applies it to the **momentum** M (not P_fin): M = max(M, int(0.98 M + min(1e6, P_add))), P_fin = P_init + M,
M_0 = 5·P_init (so P_fin^{(0)} = 6·P_init; paper says 5·P_init).

Position LR (supp. 7.2, `helper_dashgaussian`): lr(k) = exp((1−t) ln lr_0 + t ln lr_f),
t = clip((k − d)/(30000 − d), 0, 1), d = first level begin with r < 2. lr_0 = 1.6e-4·extent,
lr_f = 1.6e-6·extent (no delay multiplier).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| a (start_significance_factor) | 4 | paper §3.2; code schedule_utils.py:26 |
| max_reso_scale cap | 8 | code schedule_utils.py:27 |
| reso_sample_num (m+1) | 32 | code :28 |
| max_densify_rate_per_step | 0.2 | code :29 |
| increase_reso_until (S) | = densify_until_iter = 27000 | code :32 + full_eval.py:84 |
| densify_until_iter | 27000 (3DGS: 15000) | full_eval.py:84 `--densify_until_iter 27000` |
| densify_from_iter / interval | 500 / 100 | arguments |
| densify_grad_threshold | 2e-4 | arguments |
| percent_dense | 0.01 | arguments |
| prune min opacity | 0.005 | train_dash.py:183 |
| opacity reset | every 3000 while it < 27000 (3000..24000), to min(σ, 0.01) | train_dash.py:191 |
| γ (integrate_factor) / η | 0.98 / 1 | paper Eq. 5; code :41 |
| momentum init | 5·P_init (P_fin0 = 6·P_init) | code :39-40 |
| momentum_step_cap | 1e6 | code :42 |
| position lr | 1.6e-4 → 1.6e-6 (×extent), hold until d | arguments, general_utils.py:65 |
| other LRs, λ_dssim 0.2, SH every 1000 | unchanged 3DGS | arguments |
| GT downsample | bilinear, antialias=True | train_dash.py:114 |
| iterations | 30000 | arguments |

## 4. Schedules
- Step 1: r = floor(r_m) (r_m ≤ 8). Then integer r from the inverse-square interpolation, re-evaluated every 100 steps (600..26900).
- Densification: steps 600..26900 every 100 (500 excluded: `iteration > 500`). Stats accumulate for it < 27000.
- Opacity reset every 3000 while < 27000.
- Position LR constant until d (≈ first step where floor(r) = 1), then log-linear to 30000.
- P_fin grows by momentum at each densification; after 26900 densify target = P_fin.

## 5. What it builds on
Official 3DGS (graphdeco, recent version with `3dgs-accel` rasterizer backward). Kept unchanged: L1 +
0.2 D-SSIM (fused SSIM), Adam LRs for all non-position fields, SH schedule, clone/split rules,
grad threshold, opacity reset rule, random view without replacement. Changed: resolution,
densification (top-k, prune-first, no size pruning, until 27000), position LR schedule.

## 6. Protocol
Mip-NeRF 360 (images_4 outdoor, images_2 indoor), T&T (truck, train), Deep Blending (drjohnson,
playroom); `--eval` llffhold 8; 30k iterations; black background (default); LPIPS VGG (3DGS metrics.py);
time = sum of CUDA-event iteration+optimizer times (excludes scheduler init, data load, eval).
Per-scene numbers in supp. Tables 7–9.

## 7. Ambiguities
- Paper Eq. 4 exponent 2 − i/S with S = 30000; code uses densify_until_iter = 27000. Code used.
- Paper P_fin^(0) = 5·P_init; code P_fin^(0) = 6·P_init (momentum 5·P_init). Code used.
- Paper does not state densify_until 27000 or opacity resets continuing to 24000; both from `full_eval.py`/train_dash.py.
- All scales are floored (the unfloored first-level branch is unreachable because begin[1] = 0).
- The paper's Table 2 numbers predate the 2025-08 bug fix in the official code; the README's post-fix
  numbers are lower on Mip-NeRF 360 (27.70 vs 27.81) but are averages only and with a different 3DGS baseline (27.51).
- "Frequency energy" uses |DFT| (not squared) summed over RGB channels on [0,1] images.
