# splatstudio-method-hierarchical-3dgs

Native SplatStudio port of the single-chunk path of **A Hierarchical 3D Gaussian Representation
for Real-Time Rendering of Very Large Datasets** (Kerbl, Meuleman, Kopanas, Wimmer, Lanvin,
Drettakis, SIGGRAPH 2024, arXiv 2406.12080), following the official
`scripts/full_train.py`: `train_single.py` (30k) → `GaussianHierarchyCreator` →
`train_post.py` (15k) → `render_hierarchy.py --taus`.

## Vendored code (unmodified)
- `vendor/hierarchy_rasterizer` — github.com/graphdeco-inria/hierarchy-rasterizer @63fa247
- `vendor/gaussian_hierarchy` — github.com/graphdeco-inria/gaussian-hierarchy @677c855
  (`dependencies/json.hpp` dropped, only used by the merger binary)
- `vendor/third_party` — glm @6f14f47 (MIT), Eigen 3.4.0 (MPL2)
- Licence of the Inria code: `vendor/LICENSE_hierarchical-3d-gaussians.md` (non-commercial
  research and evaluation use).
`csrc/creator_ext.cpp` is `mainHierarchyCreator.cpp::main` as a function (no step changed).
All three extensions are JIT-built by `torch.utils.cpp_extension.load` (run via `tools/ssenv`,
~2.5 min the first time).

## Usage
```
ss-train hierarchical-3dgs --data D --output R --steps 45000 --recipe 3dgs \
    --num-gaussians 10000000 --max-gaussians 40000000      # optimized hierarchy
ss-train hierarchical-3dgs ... --steps 30000                # unoptimized ("basic") hierarchy
ss-eval --run R --split test                                # all leaves (tau -> threshold 0)
python -m ss_hierarchical_3dgs.eval_taus --run R --taus -0.5 0 3 6 15   # + per-channel PSNR
python -m ss_hierarchical_3dgs.eval_taus --run R --basic --taus 0 3 6 15   # unoptimized hierarchy
python -m ss_hierarchical_3dgs.summary --ours runs/EXP --baseline B1 B2 --channel-baseline
```
Defaults follow the official single-chunk eval command (README.md:550 of the source):
`--skip_scale_big_gauss` (shrink off; `skip_scale_big_gauss: False` enables it) and
`--exposure_lr_init 0.0`; extent = 1.1 × 0.9-quantile camera distance (getNerfppNorm).
Benchmark on `mipnerf360-1600` (official default resolution; τ is in pixels).
`hierarchical-3dgs-quick` builds the hierarchy after 1500 steps (pipeline smoke test only).

## Mapping
| paper | code |
|---|---|
| Sec. 5.2 max-gradient densification, train_single schedule, sparse Adam, optional big-Gaussian shrink | `density.py`, `optim.py` |
| Sec. 3.1 / 4.2 hierarchy generation + compaction | vendored creator via `hier.build` |
| Sec. 3.2 cut + interpolation, Eq. alphap | `hier.Cut`, `raster.render_cut`, kernel |
| Sec. 4.1 hierarchy optimization | `renderer.py` (random target), `optim.FrozenAdam` |
| tree storage for ss-eval | field `h3d` (`hier.pack/unpack`) |

Config values (paper/code sources in `density.DEFAULTS`, `renderer.DEFAULTS` and
`../paper.md`).

## Fidelity gaps
No depth regularization (the official eval command passes `-d depths`), no exposure optimization, no scaffold/skybox/chunking/consolidation/
streaming viewer; runtime view sampling with replacement and densification after (not before)
the optimizer step; appearance filter fed with training cameras only; black evaluation
background (official) instead of the runtime's white convention. See `../plan.md`.
