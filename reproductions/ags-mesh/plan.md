# AGS-Mesh native plan

Method `ags-mesh` (Splatfacto + Ours row), module `ss_agsmesh`, plugin `reproductions/ags-mesh/plugin/`.
Control `ags-mesh-rgb` (same renderer/density/init, photometric only = in-house Splatfacto row).
**Reuse (recorded):** `density.py`, `sidecar.py` (extended), renderer core and the density /
sidecar-init / gaussian-normals probes are copied from `reproductions/dn-splatter/plugin/ss_dnsplatter`
(official `ags_mesh` config = `dn_splatter` config for model/optimizer/density/init). Data tool
`tools/prepare_mushroom.py` copied unchanged (to build the 5 extra rooms); new `tools/prepare_dnc.py`
builds the DNC masks into `<scene>/agsmesh/` with `ss_agsmesh.dnc.dnc_frame`.

## Components
| component | kind | factory | requires → provides | config (defaults = official code) |
|---|---|---|---|---|
| ags-render | renderer | `ss_agsmesh.renderer:create_renderer` | gaussian3d, pinhole → rgb, depth.expected_z, normal.camera, normal.surface, absgrad, prior.sensor_depth, prior.mono_normal, prior.depth_confidence | as DN + `surface_grad False` (official detaches depth) |
| splatfacto-density | density | `ss_agsmesh.density:create_density` | gradient.screen.absolute | DN-Splatter values |
| photometric | loss | base | rgb | 0.2 SSIM |
| ags-depth (C1 DNC) | loss | `losses:create_depth` | depth, sensor_depth, depth_confidence | depth_lambda 0.2, depth_mask_step 7000, tolerance 0.1, rgb_clamp 10/255, edge_aware True |
| ags-normal (C2 ANR) | loss | `losses:create_normal` | normal.camera, normal.surface, mono_normal | normal_lambda 0.1, normal_start 7000 (ns > start), normal_mask_step 15000, anr_threshold 0.1 rad, edge_threshold 0.01 |
| ags-scale | loss | `losses:create_scale` | gaussian3d | weight 1 |

Schedules use ns = step − 1 (runtime 1-based, Nerfstudio 0-based).
Frames: all normal terms evaluated in the official GL camera frame: pred_gl = D·n_cv, gt_gl = D·(2p−1),
surf_gl = D·pcd_to_normal(backproject(depth.detach())) with D = diag(1,−1,−1) — needed because
`find_edges` (1/(g+1e−6) Laplacian) is not sign-invariant.

## DNC (C1) offline prior filter
`dnc_frame`: back-project sensor depth (pixel +0.5, OpenCV), Open3D KNN-200 normals flipped
against the ray, Omnidata png rotated by R_c2w, angle between (n+1)/2 encodings > 10° → 255, JPEG
round trip; plugin confidence = 1 − m/255 > 0.

## Probes
| probe | checks | fails if |
|---|---|---|
| dnc-filter | synthetic room corner + corrupted depth patch + wrong-prior patch: flat consistent pixels kept, corrupted/wrong pixels rejected; Open3D KNN normals = independent numpy PCA | filter removed, wrong axis (max vs min eigvec), no flip, threshold changed |
| depth-schedule | ns<7000 uses raw depth (= independent edge-aware formula ×0.2), ns≥7000 zero gradient at rejected pixels and matches formula on D_f | switch/filter/weight removed |
| normal-anr | 0 before ns 7001; edge-masked surf+pred L1 vs independent loops (incl. independent find_edges) for 7000<ns<15000; ANR-confidence mask after 15000; surf term gradient-free under official setting | schedule, masks, threshold or λ changed |
| scale-loss, splatfacto-density, sidecar-init, gaussian-normals | as DN-Splatter | — |
| surface-normals (CUDA) | renderer surface normal of a tilted plane = analytic normal (GL frame), zero border | wrong frame/orientation |

## Fidelity gaps
1. Meshing (adaptive TSDF + IsoOctree) and mesh metrics not implemented (Splatfacto+Ours row uses
   Open3D TSDF; Faro GT needed). IsoOctree only matters for the 2DGS rows. 2DGS variant not ported.
2. Benchmark = 6-scene MuSHRoom average (only NVS number in the paper); our TSDF init/regenerated
   Omnidata normals differ from the authors' files.
3. gsplat 1.5.3 vs 1.0, LPIPS VGG vs AlexNet, view sampling with replacement (as DN-Splatter port).
4. Paper vs code (code used): EdgeAwareLogL1 vs L1, τ_N 0.1 rad vs 10°, ANR gradient-free.
5. ScanNet++ (licence) infeasible.
