# splatstudio-method-confidence-densify

Native SplatStudio port of *Efficient Confidence-Aware Densification for 3D Gaussian Splatting*
(arXiv 2610.13541). No official code was released; implemented from the paper's LaTeX source
and CoMe's confidence code (github.com/r4dl/CoMe). Paper fidelity unverified.

Methods: `confidence-densify` (the paper, budget `cap_max` = 1M by default) and
`cad-mcmc-control` (3DGS-MCMC at the same budget: gsplat MCMCStrategy + λ_reg 0.01 + initial
opacity 0.5).

Mechanism (config keys in `density.DEFAULTS`, `losses.CONF_DEFAULTS`/`REG_DEFAULTS`):
- learned confidence exp(γ) (field `confidence`, lr 2.5e-4); Ĉ = Σbγ̄/max(1−T,1e-3) from a
  pass on detached geometry, so its loss gradient reaches only γ (CoMe rasteriser numerics); loss mean(L_rgb(p)·Ĉ − 0.075 log Ĉ), Ĉ ∈ [1e-3, 5] (Eq. confidence, replaces L_rgb);
- MCMC regulariser 0.01 (mean opacity + mean scale) until 25k;
- every 100 it in (500, 25000): +5 % up to the cap; every 400 it relocation of the lowest
  max-blend-weight primitives (w=0 only ≤10 % before 5k; 10 % until 10k; Gaussian fall-off);
  one source draw for both: 80 % Eq. pdf (rank|∇|_abs·((1−η)+η(1.2−rank γ̄))), 20 % SSIM term;
  gsplat `compute_relocation`, means sampled from the source Gaussian, no position noise;
- new primitives take the GT colour in the first view where their radius < 16 px (≤ 400 it).
- custom CUDA (`csrc/cad_stats.cu`, JIT): per-view max α·T and Σ(1−SSIM)·α·T per primitive.

Fidelity gaps: gsplat rasteriser instead of Faster-GS; runtime white background / view sampling;
rank ties, SSIM-term reduction and moment resets are our reading (see ../paper.md §7); initial
opacity 0.5 assumed from the MCMC code.

```bash
tools/ssenv pip install -e reproductions/confidence-densify/plugin
tools/ssenv ss-verify confidence-densify --device cuda
tools/ssenv ss-train confidence-densify --data ~/datasets/mipnerf360/native/garden --output runs/x --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```
