# splatstudio-method-taming-3dgs

Native SplatStudio port of **Taming 3DGS: High-Quality Radiance Fields with Limited Resources**
(Mallick, Goel, Kerbl, Steinberger, Carrasco, De La Torre — SIGGRAPH Asia 2024,
[arXiv 2406.15643](https://arxiv.org/abs/2406.15643)), following the official code
[humansensinglab/taming-3dgs](https://github.com/humansensinglab/taming-3dgs) @ fd0f7d9.
Paper fidelity unverified.

| method | paper row | budget | densification interval |
|---|---|---|---|
| `taming-3dgs` | "Ours (#3DGS)" (`full_eval.py --mode big --sh_lower`) | 3DGS's final count per scene | 100 |
| `taming-3dgs-budget` | "Ours" (`--mode budget --sh_lower`) | 15× SfM (360 outdoor), 2× (360 indoor, T&T), 5× (DB) | 500 |

The scene's budget is looked up from its initial SfM point count (`core.SCENE_SFM_COUNTS`,
our native Mip-NeRF360 / T&T / DB conversions). Other scenes need density config `budget`;
otherwise training stops at the first densification.

## Mechanism → code
- **Growth curve** (Sec. 3.2, Eq. 2; code `get_count_array`): `core.count_array`, target of the
  j-th densification = `counts[j+1]`.
- **Score** (Eq. 3–4, Alg. 1; code `compute_gaussian_score`): `TamingDensity.score` over 10 random
  cached training views. Pixel accumulations (count, centre distance, saliency, blend weight) come
  from `csrc/taming_stats.cu`, a forward-only CUDA kernel over gsplat's tile lists
  (adapted from the SplatStudio Mini-Splatting port's kernel; statistics as in the official
  rasterizer's `pixel_weights` branch). Weights = official `score_coefficients`.
- **Sampled densification** (`densify_with_score`): multinomial without replacement on score ×
  {clone, split} candidates (grad ≥ 2e-4), proportional budgets, gsplat `duplicate`/`split`;
  score-guided 50 % pruning for the first 27 calls; world-size pruning after 3000.
- **High opacity** (Sec. 3.4): official `modify_functions` freezes the opacities at 15k —
  implemented as opacity lr = 0 from step 15000.
- **Batched SH** (Sec. 4.2): `optim.IntervalAdam` on `shN`, one step per 16 iterations on summed
  gradients; lr opacity 0.025, shN 2.5e-4 (code values).
- The renderer is gsplat's, plus `outputs["target_image"]`/`outputs["camera"]` so the density can
  cache training images for scoring.

## Fidelity gaps
Speed-only kernels (per-splat backward, fused SSIM, separate SH tensors, sparse Adam) are not
ported; the paper's trainable abs-opacity is replaced by the code's freeze; score views are drawn
from training views the trainer has already shown; gsplat backend numerics (0.999 alpha clamp, opacity-aware radii), white
background and with-replacement view sampling are shared with the baseline.
Multinomial sampling reproduces torch's zero-weight fill, so every budget is met as in the
official code. See `../plan.md`.

## Commands
```bash
tools/ssenv pip install -e reproductions/taming-3dgs/plugin
tools/ssenv ss-verify taming-3dgs --device cuda
tools/ssenv ss-train taming-3dgs --data ~/datasets/mipnerf360/native/garden --output runs/x \
    --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```
