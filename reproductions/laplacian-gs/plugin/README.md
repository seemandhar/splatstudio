# splatstudio-method-laplacian-gs

Native SplatStudio port of **Laplacian Frequency Hierarchies for Efficient 3D Gaussian
Splatting Training** (Laplacian-GS, arXiv 2609.03334), following the official code
https://github.com/SorenZhang574/Laplacian-GS (MIT) with its `--backbone taming` path
(DashGaussian ADC with both schedulers off = 3DGS-equivalent densification). Paper fidelity
unverified.

Methods: `laplacian-gs` (2 levels, 10K/20K) and `laplacian-gs-smoke` (1000/1000, densify until
900 — smoke tests only).

## Mechanism → code
- Field `lap_level` marks each Gaussian's pyramid level (1 = low-frequency base, 0 = residual).
- `renderer.py`: base field rendered at the coarse pyramid size with Eq. 5 colours, clamped
  [0,1]; residual fields with signed SH colours (no +0.5, no clamp; official `_pn`
  rasterizer), background 0, clamped [−1,1]. Training: only the active level is rendered with
  grad, the archived levels give the cache Up(Î^{ℓ+1}) under no_grad (Eq. 7). Evaluation:
  Laplacian reconstruction of all levels.
- `pyramid.py`: Down = reflect-pad 5×5 binomial stride 2; Up = bilinear + binomial blur.
- `density.py`: stage-local 3DGS ADC on the active field (prune σ<0.005 first, clone/split
  grad ≥ 2e-4, percent_dense 0.01, until 15000·T/30000, reset every 3000), position LR
  restart per stage, archive at stage end + xyz-only inheritance (ρ=1), fresh Adam.
- `losses.py`: L1 + 0.2 D-SSIM (same padding) of the composed image vs Down^ℓ(GT).

| config | value | source |
|---|---|---|
| levels / iter_plan | 2 / [10000, 20000] | paper §4.1, run_mipnerf360_1k_taming.sh |
| inherit_ratio | 1.0, xyz only | supp. A |
| LR sh0 / shN / opacity / scales / quats | 0.0025 / 2.5e-4 / 0.025 / 0.005 / 0.001 | arguments/__init__.py |
| grow_grad2d / percent_dense / prune_opa | 2e-4 / 0.01 / 0.005 | arguments, Taming script |

## Fidelity gaps
gsplat + dense Adam instead of Taming fast backward + Sparse Adam; archived field kept in
GPU memory and its cache re-rendered each step (speed/memory claims not reproduced, quality
identical); white base background (code: black); random view sampling with replacement;
optimizer step taken on each stage's last iteration; exact cKDTree 3-NN for new-field scales;
FastGS backbone not ported.

## Commands
```bash
pip install -e .
ss-inspect laplacian-gs && ss-verify laplacian-gs --device cuda
ss-train laplacian-gs --data ~/datasets/mipnerf360/w1600/garden --output runs/x --steps 30000 \
  --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```
`--steps` must equal sum(iter_plan) (30000) so that both stages run.
