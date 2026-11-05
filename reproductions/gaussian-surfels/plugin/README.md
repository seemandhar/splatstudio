# splatstudio-method-gaussian-surfels

Native SplatStudio port of **High-quality Surface Reconstruction using Gaussian Surfels**
(Dai et al., SIGGRAPH 2024, [arXiv 2404.17774](https://arxiv.org/abs/2404.17774)), following the
official code <https://github.com/turandai/gaussian_surfels> (commit `01d1249`). Paper fidelity
unverified until the DTU benchmark has run.

Methods: `gaussian-surfels` (full), `gaussian-surfels-nogeo` (L_p + L_m only; in-house control).

## Mechanism → code
| paper | code |
|---|---|
| Gaussian surfels, per-pixel depth (Eq. 2–8), ×10 normal gradient (§3.5) | `csrc/` official rasterizer, vendored unmodified, JIT (`raster.py`); config `[surface, normalize_depth, perpix_depth, dual_visible] = 1` |
| L_p (Eq. 9) | `losses.create_photometric` — GT = image·mask + bg·(1−mask), random bg per step |
| L_m | `losses.create_mask` — code: 0.1·mean(opac·(1−maxpool9(mask))) |
| L_c (Eq. 10) | `losses.create_consistency` — (0.01 + 0.1·min(2it/15000,1))·cos_loss(N, depth2normal(D)) |
| L_n (Eq. 11) | `losses.create_normal_prior` — (0.04 − 0.02·it/15000)·cos_loss(N, mono·mask) + 0.005·mean(curv) |
| L_o (Eq. 12) | `losses.create_opacity` — 0.01·mean(exp(−20(o−.5)²)·band) |
| densification / pruning / reset / LR (§3.5) | `density.py` (see module docstring) |
| coarse-to-fine, random bg | `renderer.py` (factor 4 up to training render 2016, 2 up to 5040) |
| volumetric cutting + Poisson (§3.4) | `mesh.py` (vendored `csrc/utils/cuda_utils.cu` occupancy grid, pymeshlab) |
| DTU CD | `dtu.py` (`gsurfels-dtu-eval`, official eval.py / DTUeval-python) |

All hyper-parameters are `ComponentSpec.config` keys with the official-code values as defaults
(code overrides the paper; differences listed in `../paper.md` §3).

## Data
DTU from the authors (HF `turandai/gaussian-surfels-dtu`, IDR layout with masks and Omnidata
normals). Convert with `python ../tools/prepare_dtu.py --raw ~/datasets/gsurfels_dtu/raw --out
~/datasets/gsurfels_dtu/native`. Native images are masked on black (the official PSNR protocol);
the sidecar `<scene>/gsurfels/` holds masks, priors and downscaled images for training and is
found from `--data` (or `GSURFELS_DATA`). No Omnidata weights are needed for DTU (priors are
precomputed by the authors).

## Commands
```bash
T=/home/sejain/splatify/tools/ssenv
$T ss-train gaussian-surfels --data ~/datasets/gsurfels_dtu/native/scan24 --output RUN \
   --steps 15000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
$T ss-eval --run RUN --split test            # PSNR on masked images, black background
PYTHONPATH=~/models/gsurfels-pylib:~/models/dtueval-pylib $T gsurfels-dtu-eval \
   --run RUN --data ~/datasets/gsurfels_dtu/native/scan24 --dtu ~/datasets/dtu_official/gt
```
`--num-gaussians` must keep all initial points: the sidecar init (scales, rotations) is only
applied when the runtime keeps the points.npz order.

## Fidelity gaps
1. Views sampled with replacement (runtime) vs the official per-epoch stack: the invisible-point
   prune uses a window that lasts until every training view was rendered (paper: "every N
   iterations"); resolution switches at fixed training-render counts.
2. Optimizer step precedes density control (official densifies before the step).
3. `mono *= mask` in-place compounding at soft mask edges not replicated.
4. Masked images stored as uint8.
5. SH degree rises one step later; seeds differ (official unseeded).
6. Final it-15000 densify excluded (official saves before it); optimizer step at 15000 not skipped.
7. pymeshlab threads=16 (default spins on 256 cores); NN distance with cKDTree (pytorch3d in
   the official code).
8. Licence: Inria Gaussian-Splatting licence (non-commercial research) for `csrc/`.
