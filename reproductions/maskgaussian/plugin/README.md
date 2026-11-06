# splatstudio-method-maskgaussian

Native SplatStudio port of **MaskGaussian: Adaptive 3D Gaussian Representation from
Probabilistic Masks** (Liu et al., CVPR 2025, arXiv 2412.20522), following the official code
github.com/kaikai23/MaskGaussian @ 2f581d0 (vanilla Inria 3D-GS host). The masked CUDA
rasterizer github.com/kaikai23/mask-diff-gaussian-rasterization @ 540dc11 is vendored in
`ss_maskgaussian/csrc/maskrast` (Gaussian-Splatting licence, Inria/MPII, non-commercial research
use) and JIT-built with `torch.utils.cpp_extension` on first use (~2 min, cached in
`~/.cache/torch_extensions`).

## Mechanism → code
| paper | code |
|---|---|
| 2 mask scores / Gaussian, init [10, 1] | `FieldSpec("mask_scores", (2,), lr 0.01, 0)` + renderer `score_offset` [10, 1] (Adam is shift-invariant, so scores+offset ≡ official init) |
| hard Gumbel-Softmax sample each step (τ=1, straight-through) | `renderer.sample_mask` = `F.gumbel_softmax(log_softmax(s), hard=True)[:, :1]` (gaussian_model.get_mask) |
| Eq. 3-4 masked blending, Eq. 7 mask gradient | vendored `forward.cu` / `backward.cu`, `raster._MaskedRasterize` returns dL/dM |
| L_m = (mean M)^2, λ window | `losses.create_mask_loss` (simple_scheduler, both ends inclusive) |
| prune never-sampled-in-10-draws at densification + every 1000 after | `density.MaskDensity` (densify_and_prune / mask_prune) |
| saved model = one mask sample, rendered with the stock rasterizer | final step keeps one sample; eval renders with M = 1 (= stock blending) |
| 3D-GS ADC, opacity reset, LRs, black background | unchanged (official values) |

## Methods
| name | setting |
|---|---|
| `maskgaussian` | Ours-α: λ=0.1 on 19000–20000 (paper per-scene tables, Table 1) |
| `maskgaussian-beta` | Ours-β: λ=0.0005 on 0–30000 |
| `maskgaussian-gamma` | Ours-γ: λ=0.001 on 0–30000 |
| `maskgaussian-opacitymask` | ablation "Mask × opacity" (β loss) — not MaskGaussian |
| `maskgaussian-exactgrad` | Ours-α with the exact supplementary mask gradient (see below) |

## Note on the released backward kernel
`backward.cu` updates the colour behind the current Gaussian as
`accum_rec = mask*last_alpha*last_color + (1-mask*last_alpha)*accum_rec`, using the *current*
Gaussian's mask although `last_alpha/last_color` belong to the Gaussian behind it. The exact
recursion (supplementary, b_i = M_i α_i c_i + (1 − M_i α_i) b_{i+1}) needs that Gaussian's own
mask. The probe `maskgs-raster-exact` shows the released kernel equals an independent replica of
its own recursion (rel. err ~6e-7) but deviates from autograd of Eq. 3-4 by up to ~32 % of
max|dL/dM| on a fixture with 13/40 masked Gaussians; with all masks present both agree. The
default build is the released kernel (it produced the paper numbers); `maskgaussian-exactgrad`
compiles the same sources with `-DMASKGS_EXACT_ACCUM` (only edit to the vendored code).

## Fidelity gaps
- Runtime order: optimizer step precedes density control (official densifies first).
- Views sampled uniformly with replacement (official: shuffled stack without replacement).
- Official skips the optimizer step at the final iteration; runtime does not.
- Different RNG → different Gumbel samples; the final model is a random sample (as official).
- Runtime SSIM/LPIPS implementations (runtime-wide).
- `total_steps` (30000) is a config key: with a different `--steps` the final single-sample keep
  does not run and mask_prune cadence still follows absolute steps.

## Commands
```bash
cd plugin && ../../../tools/ssenv pip install -e .
tools/ssenv ss-verify maskgaussian --device cuda
tools/ssenv ss-train maskgaussian --data ~/datasets/mipnerf360/w1600/garden --output runs/x \
  --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```
