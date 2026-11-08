# splatstudio-method-erank-gs

Native SplatStudio port of *Effective Rank Analysis and Regularization for Enhanced 3D Gaussian
Splatting* (Hyung et al., NeurIPS 2024, arXiv 2406.11672), following the official code
(github.com/junhahyung/erank_gs @99fd128).

| method | paper row | components |
|---|---|---|
| `erank-gs` | +a+b / 3DGS+e | gsplat classic renderer (black bg, absgrad), GOF-style fixed ADC, L1+0.2 D-SSIM, erank loss |
| `erank-gs-adc` | +a | same without erank loss |
| `erank-gs-3dgs` | 3DGS | Inria ADC (prune 0.005), black bg |

Config (sources in `ss_erank_gs/losses.py`, `density.py`): erank_lambda 0.01, thin_lambda 1,
eps 1e-7 (code; paper 1e-5), start 7000, mean reduction (code; paper Σ); densify thr 2e-4,
quantile-matched abs gradient, clone resample, min opacity 0.05, reset 3000 → 0.01.

Data: DTU from the 2DGS release converted by `../tools/prepare_dtu_2dgs.py` (777×581, RGB·mask on
black, llffhold8) — registered as `dtu-2dgs` in `tools/datasets.json`.

Fidelity gaps: see `../plan.md` (NVS split assumed llffhold8; opacity-mask loss of the WIP
release not ported; runtime sampling/order; no mesh/CD evaluation).

```bash
tools/ssenv ss-train erank-gs --data ~/datasets/dtu_2dgs/native/scan24 --output runs/x \
  --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```
