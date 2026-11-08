# splatstudio-method-cor-gs

Native SplatStudio port of **CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization**
(Zhang et al., ECCV 2024, arXiv 2405.12110). Logic follows the official code
(https://github.com/jiaw-z/CoR-GS, Inria Gaussian-Splatting licence, read-only reference; no
code executed) and the FSGS plugin it shares the pose generator / densification with.

Methods: `cor-gs` (two fields, co-pruning, pseudo-view co-regularisation) and `cor-gs-single`
(same pipeline, one field = authors' 3DGS row).

## Mechanism → code
| paper | code |
|---|---|
| two fields Θ¹, Θ² (§4) | Θ¹ = runtime params (evaluated by `ss-eval`); Θ² = `CoRGSDensity.field2`, cloned at `initialize`, own Adams, own densification |
| co-pruning (Eq. 2-3, τ=5, every 500 its after 500) | `density.py: nonmatching_masks`, `CoRGSDensity.coprune` |
| pseudo views (Eq. 4) | `pseudo.py: random_poses_llff` (official LLFF bounds table), `next_pseudo_camera` |
| co-reg (Eq. 5, 7, λ_p=1, 500 ≤ t ≤ 10000) | `losses.py: create_coreg` (targets detached) |
| Θ² photometric (Eq. 6) | `losses.py: create_field2` |
| protocol | `renderer.py`: random train bg / black eval bg, SH every 500; density: FSGS-fork densify (5e-4, far split, no unpooling), resets 501/3501/6501/9501 |

## Config (official values)
renderer: train_background random, eval_background black, sh_interval 500.
density: n_fields 2, grad_threshold 5e-4, interval 100, from 500, until 10000, prune 0.005,
percent_dense 0.01, dist_thres 10, opacity reset 0.05 after 500 every 3000, coprune τ 5 every 500,
10 000 pseudo poses. coreg: weight 1, ssim_weight 0.2, start 500, end 10000.

## Fidelity gaps
See `../plan.md` (runtime optimizer/density order, view sampling with replacement, gsplat vs
Inria rasterizer, valid-conv SSIM / lpips package, COLMAP version, single seed).

## Commands
```bash
tools/ssenv pip install -e reproductions/cor-gs/plugin
tools/ssenv ss-verify cor-gs --device cuda
tools/ssenv ss-train cor-gs --data ~/datasets/llff/fewview3-r8/horns --output runs/x --steps 10000 \
    --recipe 3dgs --num-gaussians 10000000
```
