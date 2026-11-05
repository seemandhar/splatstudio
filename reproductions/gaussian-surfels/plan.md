# Plan — Gaussian Surfels native port

Method name **`gaussian-surfels`**, module **`ss_gsurfels`**, representation `surfel2d` (state keeps
three scale columns; the third is held at −1e10 like the official code and ignored by the kernel).
Ablation/control method **`gaussian-surfels-nogeo`** (same pipeline without L_c, L_n, curvature,
L_o: rgb + mask only) for an in-house "3DGS-like surfel" control (not the paper's 3DGS row).

## Components
| kind | name | factory | requires → provides | config (defaults = official code) |
|---|---|---|---|---|
| renderer | `gsurfels-raster` | `ss_gsurfels.renderer:create_renderer` | `state.surfel2d`, `camera.pinhole` → `render.rgb`, `render.depth.expected_z`, `render.normal.camera`, `render.alpha`, `gradient.gsurfels` | `resolution_schedule [[2016,4],[5040,2]]`, `random_background true`, `eval_background [0,0,0]` |
| density | `gsurfels-density` | `ss_gsurfels.density:create_density` | `gradient.gsurfels` | 200 / interval 100 / thr 1e-4 / percent_dense 0.01 / min_opac 0.1 / big 0.5·extent / tiny 1e-8·extent² / reset ×0.12 every 1000 (at 1001…) / xyz LR 1.6e-4→1.6e-6 over 15000 / clone premask 1e-7 & logit≤2 / epoch-window visibility prune |
| loss | `gsurfels-photometric` | `losses:create_photometric` | `render.rgb` | 0.8 L1 + 0.2 (1−SSIM same-padding) vs mask-composited GT with the step's random bg |
| loss | `gsurfels-mask` | `losses:create_mask` | `render.alpha` | 0.1 · mean(opac·(1−maxpool9(mask))) |
| loss | `gsurfels-consistency` | `losses:create_consistency` | `render.depth.expected_z`, `render.normal.camera` | (0.01 + 0.1·min(2·it/15000,1))·cos_loss(N, depth2normal(D)) |
| loss | `gsurfels-normal-prior` | `losses:create_normal_prior` | `render.normal.camera` | (0.04 − 0.02·it/15000)·cos_loss(N, mono·mask, w=mask) + 0.005·mean(curv(N)) |
| loss | `gsurfels-opacity` | `losses:create_opacity` | (params) | 0.01·mean(exp(−20(o−.5)²)·w), w=0.01 on (0.01,0.5], 1 on (0.5,0.99] |

Renderer = the official CUDA rasterizer (`submodules/diff-gaussian-rasterization` of the
gaussian_surfels repo) vendored **unmodified** in `ss_gsurfels/csrc/` and JIT-built through
`torch.utils.cpp_extension.load` (Inria Gaussian-Splatting licence, non-commercial research).
Config tensor `[surface, normalize_depth, perpix_depth, dual_visible] = [1,1,1,1]`.
The renderer keeps a training-call counter (grad-enabled renders) to apply the official
coarse-to-fine schedule (/4 → /2 → /1; boundaries 2016/5040 = the epoch refill after it 2002/5002
with 42 or 56 training views and stack sampling). It looks up the per-view sidecar (mask,
Omnidata normal, PIL-bicubic downscaled image) by viewmat and returns it in the outputs; losses
are pure functions of outputs.

## Fields
None beyond the base fields. Init (points.npz + `gsurfels/init.npz`): density.initialize
overwrites scales (log √(knn3/4), z −1e10) and quats (`normal2rotation(random normals)`, not
normalised) when the sidecar init matches the means; opacity 0.1 and SH dc rand/255 come from the
runtime defaults / points colours.

## Probes
| probe | measures | fails if |
|---|---|---|
| `surfel-raster` (cuda) | slanted analytic plane of surfels: rendered depth vs ray–plane intersection, normal = plane normal, zero thickness (third scale has no effect), ×10 normal gradient vs finite difference ratio | per-pixel depth / normal / surfel flattening / ×10 removed |
| `consistency-formula` | depth2normal on an analytic plane depth → exact normal; L_c zero for consistent, >0 for tilted normals; schedule weights at 1, 7500, 15000 | formula or schedule wrong |
| `normal-prior-formula` | cos/curvature terms vs independent loops; weight schedule | wrong |
| `opacity-formula` | per-Gaussian weights at band edges, value vs exp(−(o−.5)²/0.05) | wrong |
| `density-recipe` (cuda) | prune rules (opacity, big, tiny, invisible-over-epoch), clone premask, split in-plane with s/1.6 and z kept −1e10, reset ×0.12 at 1001, xyz LR | rule removed |
| `cutting-mesh` (cuda) | occupancy grid of a single surfel: voxel sums vs analytic Gaussian·opacity at voxel centres; points in empty voxels removed | cutting removed / wrong |

## Mesh + CD tool
`gsurfels-dtu-eval --run RUN --data SCENE --dtu ~/datasets/dtu_official/gt`: official render.py
train-view rendering (black bg, full res), resample_points (opac>1e-5 · mask · 0<D<20), occ grid
512 (vendored `utils/cuda_utils.cu`, unmodified), `grid > 1` prune, pymeshlab screened Poisson
depth 10 (threads=16; default threads spin on this 256-core host), vertex prune on squared NN
distance > 3e-5 (cKDTree replaces pytorch3d knn), close holes 300, then eval.py DTU CD (vertices
/S + T). Extra deps (pymeshlab) in `~/models/gsurfels-pylib`; open3d/sklearn from
`~/models/dtueval-pylib`; both via PYTHONPATH (sys.path fallback in the tool).

## Fidelity gaps
1. View sampling with replacement (runtime `random.choice`) vs official per-epoch stack → the
   invisible-point prune uses an "all train views seen since last window" window (paper's
   "every N iterations") instead of the last 100 steps; resolution switches at fixed steps.
2. Optimizer step happens before density control (official: densify/prune before step, so the
   step on those iterations is lost for replaced tensors). Minor.
3. `mono *= mask` in-place compounding at soft mask edges not replicated.
4. Images stored as uint8 after masking (≤0.5/255 rounding); official int8 float-cast truncation.
5. SH degree increase one step later (runtime `(step-1)//1000`).
6. Seeds/random init differ (official unseeded).
7. Optimizer step at it 15000 not skipped; no it 15001.
8. Poisson threads=16 (meshlab default threads), knn via cKDTree.
