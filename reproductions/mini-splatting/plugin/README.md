# splatstudio-method-mini-splatting

Native SplatStudio port of **Mini-Splatting: Representing Scenes with a Constrained Number of
Gaussians** (Fang & Wang, ECCV 2024, [arXiv 2403.14166](https://arxiv.org/abs/2403.14166)),
written against the official code <https://github.com/fatPeter/mini-splatting>
(`ms/train.py`, `ms_d/train.py`, `scene/gaussian_model.py`,
`submodules/diff-gaussian-rasterization_ms`). Paper fidelity unverified.

| method | official script | importance metric |
|---|---|---|
| `mini-splatting` | `ms/train.py --imp_metric outdoor` | I² (M360 outdoor scenes, T&T) |
| `mini-splatting-indoor` | `ms/train.py --imp_metric indoor` | I¹ (M360 indoor scenes, Deep Blending) |
| `mini-splatting-d` | `ms_d/train.py` | – (no simplification) |

## Mechanism → code
- **Statistics (custom CUDA, `csrc/ms_raster.cu`)**: gsplat 1.5.3 returns the per-view tile
  intersection lists but no per-Gaussian blending-weight sums / max-weight pixel counts /
  per-pixel argmax. The kernel re-traverses gsplat's `isect_offsets`/`flatten_ids` with gsplat's
  alpha numerics and accumulates the official `render_imp` outputs (`accum_weights`,
  `area_proj`, `area_max`) and `render_depth` outputs (mid-point of the max-weight Gaussian,
  transmittance). Adapted from the official `forward.cu` (Inria 3DGS licence, research use,
  `csrc/LICENSE-3DGS.md`). `csrc/simple_knn.cu` is the vendored simple-knn `distCUDA2` used by
  `reinitial_pts`. Built JIT with `torch.utils.cpp_extension` on first use (run through
  `tools/ssenv`; ~1.5 min, cached in `~/.cache/torch_extensions`).
- **Blur split (Eq. 1)** `density._accumulate/_densify`: per step `S_i > 2e-4·H·W` OR-ed into
  the 3DGS split mask (`densify_and_split_mask`).
- **Depth reinit** `density._depth_reinit`: steps 5000, 10000; per view sample
  `int(3.5M/n_views)` pixels ∝ (1−T) without replacement (MS-D: with replacement + unique),
  mid-points + GT colours (the renderer forwards the training frame's image, capability
  `training.target_image`), 3-NN scales, opacity 0.1, fresh Adam.
- **Intersection preserving (Eq. 2) + importance sampling** `density._simplify_sample`
  (step 15000): importance I¹/I², zero for never-argmax Gaussians, keep `int(0.5·nnz)` by
  weighted sampling without replacement (exponential race ≡ numpy `choice(replace=False, p)`),
  reinit from the kept centres. `density._simplify_prune` (step 20000): 1 % CDF prune.
- **Schedules**: SH degree 0 until 15000 then +1 per 1000 (inactive shN coefficients and their
  Adam moments are zeroed after every step — equivalent to the official degree-0 model);
  means LR `lr(it)` before 15000 and `lr(it−10000)` after; no opacity reset (as in the code);
  densification stops at 15000 and is skipped at multiples of 5000 and when N ≥ 4.5M (MS).

All hyper-parameters are in `ss_mini_splatting.density.DEFAULTS` (official defaults).

## Fidelity gaps
White background (runtime) vs black; random view sampling with replacement vs shuffled stack;
optimizer step before density control; gsplat alpha numerics (0.999 clamp) vs 3DGS (0.99);
initial SfM scales from exact k-NN (reinit scales from simple-knn as official); importance
metric chosen by method name. The 3DGS screen-size prune is inert in the official code
(max_radii2D is zeroed before the test) and therefore omitted.

## Commands
```bash
pip install -e .            # inside tools/ssenv
ss-inspect mini-splatting && python -m pytest -q tests && ruff check .
ss-verify mini-splatting --device cuda
ss-train mini-splatting --data ~/datasets/mipnerf360/native/garden --output runs/x \
  --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
python smoke_schedule.py mini-splatting <data> <out>   # every event within 2000 steps
```
