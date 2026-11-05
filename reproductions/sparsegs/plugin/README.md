# splatstudio-method-sparsegs

Native SplatStudio port of **SparseGS: Sparse View Synthesis using 3D Gaussian Splatting**
(Xiong et al., arXiv [2312.00206](https://arxiv.org/abs/2312.00206); official code
[ForMyCat/SparseGS](https://github.com/ForMyCat/SparseGS)). Paper fidelity unverified.

Methods: `sparsegs` (full), `sparsegs-paperprune` (paper pruning range) and `sparsegs-3dgs` (protocol-matched 3DGS control: same renderer
and density, photometric loss only, no warp/SDS/pruning).

## Mechanism → code
| paper | code |
|---|---|
| C1 softmax / mode / alpha depth (Eq. 1-5, supp. A) | `csrc/sgs_depth.cu` (JIT), `depth_raster.py`, `renderer.py` — gsplat RGB, then an official-rules depth pass on gsplat's projection (exact analytic backward) |
| C2 global + local Pearson (Eq. 6) | `losses.create_pearson` (official `loss_utils.py` numerics; skipped on warp iterations) |
| C3 SDS, SD 2.1-base, elliptic-cylinder views (Eq. 7-8, supp. B) | `losses.create_sds`, `sds.py`, `context.Ellipse` |
| C4 depth-warping loss (Eq. 9-10, supp. B) | `context.warp_viewmats/forward_warp`, `density.build_warp_targets`, `losses.create_warp` |
| C5 dip-test floater pruning (supp. C) | `density.prune_floaters` + `prune_mark` kernel |
| 3DGS training as configured by SparseGS | `density.SparseGSDensity` (Inria clone/split/prune, min opacity 0.05, reset 0.01/3000, xyz lr expon over 30k) |

## Defaults (paper §4.1 where stated, else official code / README)
β 5; Pearson global 0.15, local 0.15, patch 64 (360), p_corr 0.5; warp 0.05 from step 5000
(targets warped at 4999), p 0.4, angles ±2.5° (code; paper ±1.5°/±3°); SDS 5e-4 after 2/3 of
training, freq 0.1, step_ratio 0.99 (README), CFG 100, empty prompt; pruning once after the last step (paper text; variant `sparsegs-tuned6667` = TUNED prune at 6667 chosen on a test-split smoke, not the README recipe; README uses 20000 of 30000 in a different -r 2 recipe; paper text: after
training — pruning at the end without recovery lost 1.4–2.9 dB in the smoke test), a=0.97, b=7.5 (paper), min percentile 80, power threshold −4,
`mode_range: official` (bug-compatible start/mode indices of forward.cu; `paper` = intended
"all Gaussians up to the mode"); 10000 iterations (12 views). Full table: `../paper.md` §3.

## Data
`mipnerf360-sgs12` (`~/datasets/mipnerf360/sparsegs12/<scene>`), built by
`../tools/prepare_m360_sgs12.py`: authors' released 12 training views (6 scenes), 1/4
resolution, COLMAP MVS points of the training views, BoostingMonocularDepth (LeReS) priors in
`sgs_depth.npz`. The data root is read from `SGS_DATA` or `ss-train --data`.
External assets: `~/models/sd21base` (+ `pylib/` with diffusers 0.30.3), `~/models/bmd`
(BMD weights, upstream repo, `pylib/` with diptest + scikit-image).

## Fidelity gaps
See `../plan.md` "Fidelity gaps": released split ≠ paper text (6 scenes, no room); gsplat RGB
rasterizer (alpha clamp 0.999; its backward drops opacity/geometry gradients where o·G > 0.999,
TODO-runtime.md) vs Inria 0.99; runtime view sampling with replacement and valid-conv SSIM in
the photometric loss; optimizer step before density ops; orthonormalised SDS camera; SD weights
from the sd2-community mirror; paper vs README hyper-parameters.

## Commands
```bash
tools/ssenv pip install -e reproductions/sparsegs/plugin
tools/ssenv ss-verify sparsegs --device cuda
tools/ssenv ss-train sparsegs --data ~/datasets/mipnerf360/sparsegs12/garden --output runs/x \
    --steps 10000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run runs/x --split test      # test = llffhold8 minus train
tools/ssenv ss-eval --run runs/x --split validation  # other non-train images (official set = both)
```
First use JIT-compiles the CUDA kernel (~1 min, cached in ~/.cache/torch_extensions).
