# splatstudio-method-dngaussian

DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth
Normalization (Li et al., CVPR 2024, arXiv 2403.06912) — native SplatStudio port written
against the paper and the official code (github.com/Fictionarry/DNGaussian, LLFF settings of
`scripts/run_llff.sh`). Paper fidelity unverified.

| method | what |
|---|---|
| `dngaussian` | full method: hard + soft depth regularization with global-local depth normalization, neural colour renderer, official recipe |
| `dngaussian-3dgs` | same pipeline without the depth passes (paper's "3DGS†" row: same hyper-parameters and neural renderer) |

## Mechanism → code
- Hard depth (Eq. 5, τ = 0.95, frozen s/q, only μ learns) and soft depth (Eq. 6, frozen μ/s/q,
  only α learns, after it 1000): `renderer.hard_depth/soft_depth`; each pass has its own
  backward and Adam step inside `density.after_backward` (official: three optimizer steps per
  iteration).
- Global-local depth normalization (Eq. 7–9) with error tolerance 2.5e-4, random patch
  5–17, 0.1·local + 1.0·global (+0.1 edge-aware smoothness after 3000): `depth.py`.
- Neural colour renderer (hash grid 16 levels, 2^19, 16 → 512·bound; sigma MLP 3×64 → 1+64;
  colour MLP [SH4(d), geo] 2×64 → 3): `neural.py`, CUDA encoders vendored unchanged from the
  official repo (torch-ngp, MIT) in `csrc/` and JIT-compiled (first use ≈ 3 min; run via
  `tools/ssenv`).
- Recipe: random init (dataset), densify > 500 / < 6000 / every 100 with grad 0.0013 and an
  opacity > 0.1 gate on clone/split, prune < 0.01, no opacity reset, no screen-size pruning;
  prune Gaussians within 10 of the 180 spiral cameras every 25 its after 2000; prune Gaussians
  invisible from all training views at it 1 and 6000; xyz LR 0.016 → 0.00016 (× extent) over
  5500 its after a 500-it hold; scaling LR 0.003; penalties 0.002 shape / 0.001 scale / 0.001
  opacity; L1 + 0.2 D-SSIM; black background.

## Global learned state
The density owns the GridRenderer and its Adam (`neural.ACTIVE`). Every `bake_every` (1000)
steps and at `total_steps` (6000) it bakes the paper's inference cache into Gaussian fields:
`dng_geo` (geometry feature) and `dng_mlp` (packed colour-MLP weights + header with the
Gaussian count and a means fingerprint). No-grad renders after a bake (validation, `ss-eval`)
use only the checkpoint; a stale or missing bake raises. Run lengths must be multiples of 1000
(or 6000) for `ss-eval` to work.

## Data
`llff-fewview3-r8-dng` (built by `reproductions/dngaussian/tools/prepare_dng_llff.py`): the
llff-fewview3-r8 images/splits (hard links, byte-identical transforms.json) + official random
init (`points.npz`) + `dng_depth.npz` (DPT-Hybrid depth of the 3 training views, keyed by
viewmat, and the official spiral camera centres). The data root is read from `DNG_DATA` or
`ss-train --data`.

## Commands
```bash
tools/ssenv ss-train dngaussian --data ~/datasets/llff/fewview3-r8-dng/fern --output R --steps 6000 --recipe 3dgs --num-gaussians 10000000
tools/ssenv ss-eval --run R --split test
```

## Fidelity gaps
gsplat vs ashawkey rasterizer; per runtime step the photometric pass precedes the next
iteration's depth passes (hard pass of iteration 1 lost); runtime steps the photometric update
before densification/pruning (official drops it on those iterations); random view sampling with
replacement; eval near-cull per view (official: union over eval cameras); runtime SSIM/LPIPS;
NaN → 0 for empty selections (same gradients); one seed (official README: PSNR unstable);
MiDaS master + timm 0.6.12 for DPT; iteration 6000 not skipped (one extra photometric/neural step before the final clean); `dngaussian-3dgs` is a literal reading of 3DGS† (no official script). Depth-pass losses are printed (`{'dngaussian': ...}` rows)
and kept in `density_state["log"]`, not in `training.jsonl`.
