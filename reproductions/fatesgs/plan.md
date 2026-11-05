# FatesGS — native plan

Method `fatesgs` (module `ss_fatesgs`), control `fatesgs-2dgs` (same pipeline, L_r/L_s/L_f off
= paper ablation row "none", mean CD 2.47 little-overlap; and the paper's 2DGS row).

## Feasibility (decided in the read stage)
Primary metric = DTU Chamfer distance (per-scene, main Table 1, large-overlap set 23/24/33).
Data: authors' Google-Drive release (DTU_TEST 3.1 GB, DTU_EVAL 0.23 GB, Marigold depth 70 MB)
downloaded with gdown in < 2 min to `~/datasets/dtu_fatesgs/`. GT: official DTU SampleSet.zip
(6.9 GB, ObsMask/Plane) + Points.zip (7.0 GB, STL) from roboimagedata2.compute.dtu.dk (~6-20 MB/s).
Evaluator: official `eval/eval_dtu` (DTUeval-python) run with open3d 0.19 / sklearn / trimesh /
skimage installed to `~/models/dtueval-pylib` (PYTHONPATH only; env untouched).
=> benchmark target = CD (not PSNR); PSNR on 3 extra DTU views reported for information only.

## Components
| C | kind | factory | notes |
|---|---|---|---|
| backbone | renderer `fatesgs-surfel` | `ss_fatesgs.renderer:create_renderer` | vendored official diff-surfel-rasterization (2DGS CUDA, JIT); black bg; outputs rgb, depth (= surf_depth, expected/alpha), alpha, rend_normal (world), surf_normal (depth_to_normal·alpha.detach), distortion, info{means2d, radii}, camera |
| recipe | density `fatesgs-density` | `ss_fatesgs.density:create_density` | port of 2DGS/3DGS densify_and_prune (clone/split-in-plane/prune <0.1, size prune after 1000), reset to min(o,0.1) every 1000, xyz LR expon 1.6e-4→1.6e-6 over 30000 |
| L_c | loss `photometric` | `ss_fatesgs.losses:create_photometric` | 0.8 L1 + 0.2 (1-SSIM same-padding) |
| L_d,L_n | loss `surfel-reg` | `ss_fatesgs.losses:create_surfel_reg` | 1e4·dist (>3000) + 0.05·normal (>7000), official formulas |
| C1 L_r | loss `depth-rank` | `ss_fatesgs.losses:create_rank` | 10 × patch rank, 2 centre crops, M=32, m=1e-4 |
| C2 L_s | loss `depth-smooth` | `ss_fatesgs.losses:create_smooth` | 1 × TVLoss(m_e=1e-2, m_t=1e-4) |
| C3 L_f | loss `feature` | `ss_fatesgs.losses:create_feature` | 1.5 × multi-level Vis-MVSNet cosine with z-buffer visibility, 2 source views |

Sidecar `<data>/fatesgs.npz` (prepare script): train viewmats, Marigold depth [3,581,777], pair
indices, alpha masks; full-res RGB in `<data>/images_full/` for features (computed once, lazily,
with Vis-MVSNet weights `~/models/vismvsnet/vismvsnet.pt`, loaded weights_only).
Dataset `dtu-fatesgs-23_24_33` in `~/datasets/dtu_fatesgs/native/scanX`: train = 3 views
(RGB 777×581 PIL-bicubic, alpha dropped as official use_mask=False), validation = view 22,
test = views 25, 28 (set_22_25_28 images; same normalised frame), points.npz = dense/fused.ply.
Post-train tool `fatesgs-dtu-eval --run R --data D --dtu GT`: TSDF (official parameters) + cull +
DTUeval → `R/dtu_cd.json`.

## Fields: none (scales keep 3 columns; 3rd is pinned to log 1e-13 so gsplat split samples in-plane).

## Probes
rank-formula (vs independent loop; ordered depth → 0 loss, reversed > 0), smooth-formula,
feature-consistency (synthetic 2-view scene: correct depth gives lower L_f than shifted depth;
visibility z-buffer on an analytic case), surfel-renderer (official CUDA: depth of a plane at z,
normal, gradient to means2d), density-recipe (clone/split in plane/prune/reset/LR schedule).

## Fidelity gaps
1. Runtime samples views with replacement (official: shuffled stack) — minor.
2. Densify runs after the optimizer step (official before) — minor.
3. Random rotation init: runtime normalised randn vs official torch.rand (unnormalised) — minor.
4. Benchmark protocol = CD on 3 scans (scan24, scan37, scan106) — subset, per-scene comparison.
5. NVS (supp. Table 5, pixelNeRF split 25 views) not reproduced: images not in the release.
6. Little-overlap set: no Marigold depth released (would need Marigold run) — not attempted.


## Review-1 resolution
- Control label: `fatesgs-2dgs` is the ablation row 'none', not the paper's 2DGS row (review-1 minor 1).
- Scale column 3 is not pinned (no effect; nothing reads it).
- Densify order gap also skips the official Adam step on densify iterations / opacity step on reset iterations.
