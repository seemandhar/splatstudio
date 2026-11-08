# splatstudio-method-stablegs

StableGS: A Floater-Free Framework for 3D Gaussian Splatting (arXiv 2503.18458), native
SplatStudio port of the official code github.com/MooreThreads/StableGS @ 397a9c00
(`run.sh` block 1 = Table 1 row "StableGS w/o L_prior"). Paper fidelity unverified.

Methods: `stablegs` (official one-sided depth consistency), `stablegs-paper-consis` (paper Eq. 4
two-sided L1, ablation).

## Mechanism → code
| contribution | code | values (source) |
|---|---|---|
| Dual opacity: α (geometric) and α' (aux, field `opacity_aux`), appearance opacity α·α' | `renderer.py` | α' init 0.99, lr 0.025 (gaussian_model.py:195,208) |
| Appearance path for the image from step 15001, geometric path renders for losses | `renderer.py`, `density.py:_make_batch` | `dual_from` 15000 (train_pair.py:121) |
| Two-view colour loss, duplicated view-0 SSIM | `losses.py:create_pair_color` | λ 0.2 (train_pair.py:126-129) |
| Geometric-path L1 | `create_geometric_color` | λ_s 1 (`--single_loss`) |
| RaDe-GS depth-normal loss on the geometric path | `create_depth_normal` | 0.05, ratio 0.6 (run.sh, not in paper) |
| Depth consistency on P | `create_depth_consistency` | λ 0.05, official form (train_pair.py:157-176) |
| Pair set P | `pairs.py` | covis > 30, 16–60°, H inlier < 0.8 @ 4 px (paper §3.4) |
| Pair stack | `pairs.py:refill_stack`, `density.py` | P + |P| inverse-frequency random pairs |
| Taming densification + GOF qualifier, α freeze at 15000, SH every 16 | `density.py` (subclass of `ss_taming_3dgs`) | run.sh budgets |
| RaDe-GS@ec9199f rasterizer (as shipped by StableGS) | `raster.py`, `csrc/` | kernel_size 0, black bg |

The official release lacks `gaussian_renderer/` and `arguments/`; their content is reconstructed
(RaDe-GS@ec9199f render, Taming learning rates). Not built: the DUSt3R prior (full StableGS row).

## Runtime notes
The density reloads the training images from `ss-train --data` and owns the pair schedule; with
gradients enabled the renderer renders the published pair and ignores the runtime's camera
(MVGS precedent). `ss-eval` renders the appearance path (α·α'). Each iteration renders 2 views
(4 after 15k) with the Inria-style kernel: ~5–10× the time of a gsplat baseline step.

## Fidelity gaps
See `../plan.md` and `../report.md`.

## Commands
```
tools/ssenv pip install -e reproductions/stablegs/plugin
tools/ssenv ss-verify stablegs --device cuda
tools/ssenv ss-train stablegs --data ~/datasets/tandt_db/native/truck --output runs/x --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```
