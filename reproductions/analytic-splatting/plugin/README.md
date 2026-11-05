# splatstudio-method-analytic-splatting

Native SplatStudio port of **Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via
Analytic Integration** (Liang et al., ECCV 2024, arXiv 2403.11056), official code
github.com/lzhnb/Analytic-Splatting commit `a905939`. Paper fidelity unverified.

## Mechanism
Each pixel's response to a projected Gaussian is the integral of the Gaussian over the 1×1
pixel window instead of the value at the pixel centre (Eq. 10). The window is rotated onto the
eigen-axes of Σ̂ so the integral factorises (Eq. 11–12), and each 1D integral uses the
conditioned logistic CDF S(x)=1/(1+exp(−1.6x−0.07x³)) (Eq. 7):
I = 2πσ1σ2 [S((ũx+½)/σ1) − S((ũx−½)/σ1)] [S((ũy+½)/σ2) − S((ũy−½)/σ2)] (Eq. 13). The 3DGS
0.3 px dilation is removed. Forward and backward are in the official CUDA rasterizer,
vendored **unmodified** in `ss_analytic_splatting/csrc/` (Inria Gaussian-Splatting licence —
non-commercial research use — plus the repository's MIT licence) and JIT-built on first use
(run through `tools/ssenv`, ~2 min).

## Components
| kind | name | notes |
|---|---|---|
| renderer | `analytic-renderer` | vendored kernel, black background, SH in kernel; `gradient.screen.signed` = Inria NDC means2D grad |
| density | `analytic-density` | official train.py ADC (`dense=False`): 500–15000 every 100, grad 2e-4, percent_dense 0.01, prune opacity < 0.005 and (after 3000) scale > 0.1·extent, no screen-size prune (official no-op), opacity reset every 3000 to `min(o, 0.01/coef)` with the Mip-Splatting 3D-filter coefficient (official quirk, filter not rendered) |
| loss | `photometric` | L1 + 0.2 D-SSIM (unchanged) |

## Probes
`analytic-forward-integral` (CUDA): kernel = independent float64 Eq. 13 (measured 3e-7),
2.5 % from brute-force pixel quadrature for a σ≈0.4 px Gaussian vs 42 % for the 3DGS centre
sample. `analytic-backward` (CUDA): kernel gradients = float64 autograd of the reference
(≤3e-6) and central finite differences (≤1.5e-4). `analytic-logistic-cdf`, `analytic-density-schedule` (CPU).

## Fidelity gaps
- The paper's headline multi-scale (MTMT) protocol is not runnable natively (no multi-resolution
  camera sampling / `sample_more_highres`); the STST per-scene supplementary table is the target.
- Densify/reset after the optimizer step (runtime order); random view sampling with replacement.
- Opacity-reset coefficient evaluated in log space (official fp32 form can give 0/0).
- "Ours + 3D filter" variant not ported.

## Commands
```bash
cd plugin && /home/sejain/splatify/tools/ssenv pip install -e .
tools/ssenv ss-verify analytic-splatting --device cuda
tools/ssenv ss-train analytic-splatting --data ~/datasets/mipnerf360/native/garden --recipe 3dgs \
  --num-gaussians 10000000 --steps 30000 --output runs/...
```
