# splatstudio-method-gabsplat

Native SplatStudio port of **3DGabSplat: 3D Gabor Splatting for Frequency-adaptive Radiance
Field Rendering** (Zhou et al., ACM MM 2025, arXiv [2508.05343](https://arxiv.org/abs/2508.05343)).
Paper fidelity unverified. No official code exists (github.com/Zhjy1/3DGabSplat: README only).

## Mechanism
- Primitive (Eq. 7): G(x) [(1 − Σω) + Σ_i ω_i cos(2π f_iᵀ(x − μ))], F = 2, f_i ∈ R³ (field
  `gabor_freq [N,2,3]`, world cycles/unit), ω_i = sigmoid(`gabor_weight [N,2]`).
- Splatting (Eq. 9-10, supp. A): 2D frequency f2d = conic · (J2 Σ f) (`frequency.py`), equal to
  the paper's (JW)^{-T} projection + z-correction (probe `frequency-projection`), evaluated per
  view in torch (autograd to means/quats/scales/freq).
- Rasterizer (Eq. 12-13): `csrc/gabor_raster.cu`, forward + backward, adapted from
  `reproductions/linear-splatting` (← gsplat 1.5.3 RasterizeToPixels3DGS{Fwd,Bwd}.cu, Apache-2.0).
- Frequency-adaptive density (Sec. 3.4.2): children of clone/split get f = 0.001, ω = 0.01;
  all ω reset to 0.01 together with the opacity reset every 3000 iterations (< 15k).

## Config (Mip-NeRF 360, supp. B)
| key | value | source |
|---|---|---|
| F | 2 | Sec. 4.1 |
| init f / ω | 0.05 / 0.01 | supp. B (main text: 0.001 / 0.01) |
| lr f / ω | 0.02 / 0.01 | supp. B (main text: 0.01 / 0.02) |
| lr opacity | 0.025 | Sec. 4.1 / supp. B |
| prune_opa / opacity reset value | 0.01 / 0.02 | supp. B |
| child reset f / ω | 0.001 / 0.01 | supp. B |
| weight reset | 0.01 every 3000 | supp. B |

## Methods
`gabsplat` (paper), `gabsplat-noreset` (ablation: no frequency-adaptive resets).

## Fidelity gaps
See `../plan.md`: no official code; identical init of the two kernels keeps them identical
(literal paper); main-text vs supplementary conflicts (supp. chosen); dilated conic in f2d;
runtime protocol differences vs Inria 3DGS.

## Commands
```bash
tools/ssenv pip install -e reproductions/gabsplat/plugin
tools/ssenv ss-verify gabsplat --device cuda
tools/ssenv ss-train gabsplat --data ~/datasets/mipnerf360/native/garden --recipe 3dgs \
  --num-gaussians 10000000 --steps 30000 --output runs/...
```
