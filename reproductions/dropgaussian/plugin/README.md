# splatstudio-method-dropgaussian

Native SplatStudio port of **DropGaussian: Structural Regularization for Sparse-view Gaussian
Splatting** (Park, Ryu, Kim — CVPR 2025, arXiv 2504.00773), following the official release
https://github.com/DCVL-3D/DropGaussian_release (Apache-2.0 for the DropGaussian parts).

Methods:
- `dropgaussian`: the paper method.
- `dropgaussian-nodrop`: identical pipeline with γ = 0 (paper Table 5 first row), used as the
  protocol-matched baseline.

## Mechanism
- **Eq. 5** (`renderer.py`): every training render (autograd on) multiplies the activated
  opacity by `M = F.dropout(ones(N), p=r_t)`, so kept Gaussians get `1/(1-r_t)` and dropped
  ones get 0. The mask is redrawn on every render and the population never changes.
  Renders without autograd (runtime validation, `ss-eval`) use all Gaussians with unscaled
  opacity, as the official `render.py` does.
- **Eq. 6**: `r_t = γ t / t_total`, γ = 0.2, t_total = 10000 (hard-coded in the official code),
  where t is the renderer's training-render count. The density component raises if that
  count differs from the runtime step.
- Official training setup: black background, renders clamped to [0,1], loss `L1 + 0.2(1-SSIM)`
  with L1 weight 1.0 and same-padding SSIM, 3DGS densification with τ = 5e-4, from step >500
  until <10000 every 100 steps, prune opacity < 0.005, **no** screen- or world-size pruning,
  opacity reset to 0.01 at 3000/6000/9000, 10000 iterations.
- Densification visibility: gsplat culls Gaussians with opacity < 1/255, which includes every
  dropped one. The Inria rasterizer does not, so the renderer also projects without the mask
  and the density counts dropped in-frustum Gaussians as visible with a zero gradient.

## Config (paper / code defaults)
| component | key | value | source |
|---|---|---|---|
| renderer | gamma | 0.2 | Table 5, renderer:98 |
| renderer | total_steps | 10000 | renderer:98 |
| renderer | background / clamp_rgb | black / true | arguments, renderer:117 |
| density | grad_threshold | 5e-4 | §4.1, arguments |
| density | densify_from/until/interval | 500/10000/100 | arguments |
| density | percent_dense / min_opacity | 0.01 / 0.005 | arguments, train.py:113 |
| density | opacity_reset_interval / value | 3000 / 0.01 | arguments, gaussian_model.reset_opacity |
| loss | l1_weight / ssim_weight | 1.0 / 0.2 | train.py:81 |

## Fidelity gaps
1. The runtime samples training views with replacement. The official code pops a shuffled
   stack, so each of the 3 views is used once every 3 iterations.
2. The runtime takes the Adam step before densification. The official code densifies first,
   so the densified iteration gets no optimizer update.
3. gsplat vs Inria rasterizer: alpha clamp 0.999 vs 0.99, opacity-aware radii for very
   low-opacity (non-dropped) Gaussians, and the SH degree goes up one step later.
4. LPIPS-VGG comes from the `lpips` package, not lpipsPyTorch (same weights).
5. The MVS init is built with COLMAP 3.9.1. The authors' COLMAP version is not stated.

## Data
LLFF 3-view at 1/8 resolution (504×378), which is the official `-r 8`:
`~/datasets/llff/fewview3-r8/<scene>`. It is built by
`reproductions/ad-gs/tools/prepare_llff_fewview.py --downscale 8` and uses the FSGS split and
an MVS init from the 3 training views.

## Commands
```bash
tools/ssenv pip install -e reproductions/dropgaussian/plugin
tools/ssenv ss-verify dropgaussian --device cuda
tools/ssenv ss-train dropgaussian --data ~/datasets/llff/fewview3-r8/horns --output runs/x \
    --steps 10000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run runs/x --split test
```
