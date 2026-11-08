# 3D Student Splatting and Scooping (SSS) — implementation spec

arXiv 2503.10148 (CVPR 2025, best paper honorable mention). Zhu, Yue, He, Wang.
Official code: https://github.com/realcrane/3D-student-splating-and-scooping (cloned to
`source/code/`, GPL-2.0 repo; the `diff-t-rasterization` submodule is under the Inria
Gaussian-Splatting research licence). Built on vanilla 3DGS + 3DGS-MCMC code.

## 1. Summary
- Replace the Gaussian primitive by an unnormalised Student's t with a learnable per-primitive
  degree of freedom nu in [1, 10000] (Sec. 3.2).
- Signed opacity o = tanh(raw) in (-1, 1): negative components "scoop" (subtract colour) (Sec. 3.3).
- Means are sampled with an SGHMC-style sampler (Adam-preconditioned gradient, sigmoid-gated
  friction and noise, burn-in without friction and with covariance-shaped noise); every other
  parameter uses Adam (Sec. 3.4, supp. "SGHMC Sampling").
- No ADC: 3DGS-MCMC-style recycling of near-zero |o| components to sampled components with a
  t-distribution-specific relocation formula, plus +5 % growth every 100 steps up to cap_max
  (Sec. 3.4.2, supp. "Component Recycling").
- Loss = 0.8 L1 + 0.2 D-SSIM + 0.01 mean|o| + 0.01 mean(scale) (MCMC regularisers).

Contributions
- C1 Student's t kernel with learnable nu (Sec. 3.2, Eq. t-dist / t_2D; supp. forward/backward).
- C2 Signed (negative) opacity via tanh — splatting and scooping (Sec. 3.3).
- C3 SGHMC sampler for means with friction/noise scheduling and burn-in (Sec. 3.4.1, Eq. sigmoidUpdate).
- C4 Recycling/adding components with the t relocation formula (Sec. 3.4.2, Eq. relocation).

## 2. Equations
Eq. (t_2D):
$$T^{2D}(u) = [1+\frac{1}{\nu}(u-\mu^{2D})^T(\Sigma^{2D})^{-1}(u-\mu^{2D})]^{-\frac{\nu+2}{2}},\quad
\Sigma^{2D} = (JW\Sigma W^TJ^T)_{1:2,1:2}$$
Eq. (SSS_color): $C(u) = \sum_i c_i o_i T^{2D}_i(u)\prod_{j<i}(1-o_jT^{2D}_j(u))$, $o\in[-1,1]$.

Code (`forward.cu renderCUDA`): base = d^T conic d; skip if base <= 0; alpha = o * G,
clamped to [-0.99, 0.99]; skip if |alpha| < 1/255; stop the pixel when T(1-alpha) < 1e-4
(that primitive excluded). No low-pass filter (the `+0.3` dilation is commented out in fwd and
bwd). Bounding radius = ceil(e(nu) sqrt(lambda_max)), e(nu) piecewise-linear interpolation of
t critical values (nu=1: 63.657 ... 12: 3.055, else 3.0) [forward.cu:225-270].

Backward (`backward.cu renderCUDA`): gradients ignore the ±0.99 clamp (straight through); with
p = -(nu+2)/2, dG/dbase = p/nu (1+base/nu)^{p-1};
**dL/dnu uses only the base term**: (nu+2)/2 · base/nu² · (1+base/nu)^{p-1} — the
$-\tfrac12\ln(1+base/nu)\,G$ term of the exact derivative is missing in the official kernel
(= derivative with the exponent's nu held constant). Conic backward uses Inria's
`denom2inv = 1/(det^2 + 1e-7)`.

Loss (Sec. 3.4): $L=(1-\epsilon)L_1+\epsilon L_{D-SSIM}+\epsilon_o\sum|o_i|+\epsilon_\Sigma\sum\sum|\sqrt{\lambda}|$.
Code: `0.01*|tanh(o)|.mean() + 0.01*exp(s).mean()` every step (train.py:131-132).

Eq. (sigmoidUpdate), as implemented in `utils/sghmc.py` (AdamSGHMC, group 'xyz'):
```
g_adam = sqrt(1-b2^t)/(1-b1^t) * m/(sqrt(v)+1e-15)        (Adam b=(0.9,0.999))
sig    = 1/(1+exp(-100((1-|tanh o|) - 0.995)))             (op_sigmoid, before the step)
burn-in (iter <= B):  noise = cov @ (N(0, sqrt(2 lr C_b)) * sig);   cov = R S S^T R^T
                      r = r*sig - lr g_adam - r_old*sig + noise   (friction cancels: r = -lr g + noise)
after (iter > B):     r reset to 0 at iter B+1;  noise = N(0, sqrt(2 lr C)) * sig
                      r = r*sig - lr g_adam - lr C r_old sig + noise
mu += lr * r
```
lr (epsilon) = exp-decay schedule from sqrt(0.0012264*s/7.4518) to sqrt(1.22663e-5*s/7.4518),
s = 3DGS camera extent, over 30k steps (clamped after; position_lr_max_steps=30000) — i.e. the
effective step lr² equals the 3DGS lr range of T&T Train scaled by s/7.45 (train.py:56-62).

Eq. (relocation) — code `utils.cu compute_relocation_student_t` (same nu for old and new):
```
o_new = 1 - (1 - o_old)^(1/N)                 (o_old signed = tanh(o))
term1 = B(1/2, (nu+2)/2)
K     = sum_{i=1..N} sum_{k=0..i-1} C(i-1,k) (-1)^k o_new^{k+1} B(1/2, ((k+1)(nu+3)-1)/2)
scale_new = (o_old term1 / K)^2 * scale_old     (paper: Sigma_new = (o beta/K)^2 Sigma_old;
                                                 code squares the factor on the *scales*)
o_new clamped to ±(1-eps), |o_new|<0.005 pushed to ±0.005; N = count+1 clamped to [1, 51].
```

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| feature lr (sh0) / shN | 0.0025 / 0.000125 | arguments/__init__.py |
| opacity lr (tanh pre-activation) | 0.005 | arguments |
| scaling lr / rotation lr | 0.005 / 0.001 | arguments |
| degree (nu) lr | 10, x0.5 every 5000 its, min 0.01 | arguments degree_lr, nt_model get_step_lr_func |
| nu activation | hardtanh(1, 10000) | nt_model |
| nu init | 100 (all configs) | configs/*.json |
| opacity init | tanh^-1(0.5) | nt_model.create_from_pcd |
| scale init | log(0.1 · sqrt(mean 3-NN d²)) | nt_model.create_from_pcd |
| opacity_reg / scale_reg | 0.01 / 0.01 | arguments |
| densify | 500 < it < 25000, every 100 | train.py |
| dead threshold | \|o\| < 0.005 (opacity_threshold) | arguments |
| recycle cap | ≤ 5 % of N (lowest \|o\| if more dead) | nt_model.recycle_components |
| growth | min(cap_max, 1.05 N) | nt_model.add_components |
| sampling probs | \|o\| (signed opacity abs) | _sample_alives |
| Adam eps | 1e-15 | nt_model.training_setup |
| C_burnin / C / burn-in iters | bicycle 5e8/120/15000; room, truck, garden 5e5/120/7000 | configs |
| cap_max | bicycle 3M, room 1.5M, truck 2.6M, garden 3M | configs |
| iterations | bicycle 30k; room, truck, garden 40k | configs |
| background | black | arguments white_background False |

## 4. Schedules
SH degree +1 every 1000 its; xyz lr exp decay to 30k; nu lr step decay every 5000; densify
500–25000; burn-in until B (7000 or 15000) then momentum reset and friction on.

## 5. Builds on
3DGS (Inria rasteriser with t kernel, no low-pass) + 3DGS-MCMC (relocation/growth, regularisers,
Adam for non-position params). No opacity reset, no ADC, no screen-gradient statistics.

## 6. Protocol
Mip-NeRF 360 (7 scenes: bicycle, bonsai, counter, garden, kitchen, room, stump; outdoor -r 4,
indoor -r 2 PIL-resized from full res), T&T truck/train (auto 1600 width), DB drjohnson/playroom;
llffhold 8 (`eval=True`); metrics from `metrics.py` on saved 8-bit PNGs (3DGS evaluation, LPIPS VGG);
average of three runs. Per-scene iterations follow the configs (30k bicycle, 40k others).

## 7. Ambiguities
- dL/dnu partial gradient (above) — follow code.
- Relocation squares the scale factor (code) vs. covariance factor (paper) — follow code.
- Paper Eq. sigmoidUpdate writes N ~ N(0, 2 eps^{3/2} C); code samples N(0, sqrt(2 lr C)) for r
  (std, not variance) and multiplies by lr on update.
- `negative` attribute is a constant 1 (requires_grad False) — sign only from tanh.
- Recycled components keep their old Adam moments (only SGHMC momentum is zeroed); sources get
  Adam moments zeroed (momentum kept).
- sig/cov are computed from parameters before the optimiser step.

## 8. Notes for the benchmark report (review 1)
- **dν is the official partial derivative**: the kernel drops the −½ ln(1+base/ν)·G term of the
  exact derivative (`exact_nu_grad: false` default; `true` gives the exact form). The missing
  term would always pull ν down, so the official gradient biases ν upwards; this may explain
  ν drifting into the thousands (towards the 1e4 hardtanh cap) in trained models.
- **Radius table**: e(ν) is ported verbatim from forward.cu:246-281, including its ordering
  (case k: (T[k-1]-T[k])(ν-k)+T[k], so 9.925 at ν=1 rising towards 63.657 just below ν=2).
- **room** follows `configs/room.json` (40k iterations) while the shared gsplat baseline is 30k;
  compare deltas with that in mind.
- **Method labels**: `student-splatting` is a generic label (the common 5e5/120/7000, 3M-cap
  config) and is used only for the smoke test; benchmark rows must use the per-scene methods
  `student-splatting-bicycle` / `-room` (/ `-truck`), which carry the official per-scene configs.
- **Known gaps**: runtime SSIM valid convolution; view sampling with replacement; SH degree step
  at 1001 vs 1000; gsplat tile rectangle vs Inria getRect; float vs 8-bit PNG evaluation; Adam
  still steps on the final iteration (official skips it); 1 seed vs paper 3-run mean; relocation
  squares the scale factor and dν is partial (both official-code behaviour, kept).
