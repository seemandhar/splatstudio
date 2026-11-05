# splatstudio-method-improving-adc

Native SplatStudio port of **Improving Adaptive Density Control for 3D Gaussian Splatting**
(Grubert, Barthel, Hilsmann, Eisert; GRAPP 2025, arXiv 2503.14274), following the official code
https://github.com/fraunhoferhhi/Improving-ADC-3DGS (65901d8, a PixelGS fork). Paper fidelity
unverified until benchmarked.

Method `improving-adc`: stock gsplat renderer + stock photometric loss + density
`ss_improving_adc.density:ImprovedADC`, which implements
- C1 corrected scene extent e' = mean ||p_i - mean(C)|| over SfM points (Eq. 13); used for
  clone/split and as the means-LR scale (official `spatial_lr_scale`);
- C2 threshold 1e-4 * 4^(k/145) at densification call k (Eq. 14, official discrete form);
- C3 significance-aware pruning (Alg. 2): low-opacity Gaussians pruned only when their
  blend-weight sum sigma (Eq. 15) is below the N_prune/N_old quantile; no size pruning;
- base: PixelGS pixel-count-weighted gradient with squared depth scaling min(1,(z/(0.37 e_cam))^2),
  and Bulo et al. revised opacity 1-sqrt(1-o) for clones (parent too) and split children.

Per-Gaussian pixel counts and sigma come from a forward-only CUDA kernel (`csrc/adc_stats.cu`,
JIT-built on first use) over gsplat's tile lists; it equals the official fake-colour autograd
route (probe `adc-stats-kernel`). Config keys and defaults: `density.DEFAULTS` (all from the
official code); ablation switches `extent: camera`, `threshold: constant`, `pruning: opacity`,
`pixel_gradient: false`, `revised_opacity: false`.

Fidelity gaps: white background (official black), Adam step before density, gsplat vs Inria
rasterizer numerics, view sampling with replacement, quantile fraction clamped to <= 1.

```
tools/ssenv pip install -e reproductions/improving-adc/plugin
tools/ssenv ss-verify improving-adc --device cuda
tools/ssenv ss-train improving-adc --data ~/datasets/mipnerf360/native/garden --output runs/x \
  --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```
