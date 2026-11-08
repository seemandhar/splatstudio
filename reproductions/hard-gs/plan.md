# HGS native plan

Method `hard-gs`, module `ss_hard_gs`, plugin `reproductions/hard-gs/plugin/`.

HGS is Scaffold-GS + a modified growing candidate set. No official HGS code, so the Scaffold-GS backbone is ported
from the official Scaffold-GS code (`source/code/scaffold-gs`, commit 59c833b) using the already-matched HAC native
plugin as the engineering template (`reproductions/hac/plugin/ss_hac`: anchor population as native Gaussians,
renderer-owned MLP bank, `visible_anchors` = Scaffold fork `visible_filter`, gsplat-ops anchor growing/pruning, MLPs
packed into a per-anchor blob field for ss-eval). HAC-specific parts (hash grid, quantization, entropy, masks,
feat_dim 50, distance input to the MLPs, densify pause [3000,4000)) are removed. The max-weight-pixel statistics
kernel is reused from `reproductions/mini-splatting/plugin/ss_mini_splatting/csrc/ms_raster.cu` (re-traverses
gsplat's tile lists; same numerics) together with its vendored simple-knn (distCUDA2 for the anchor init scale).

## Components
| component | kind | factory | requires / provides | config |
|---|---|---|---|---|
| hgs-renderer | renderer | ss_hard_gs.renderer:create_renderer | state.gaussian3d, camera.pinhole → render.rgb, render.depth.expected_z, gradient.screen.signed | feat_dim 32, n_offsets 10 |
| hgs-density | density | ss_hard_gs.density:create_density | gradient.screen.signed | Scaffold defaults + `pghgs` {enabled, k 3, lambda 1.0} + `rehgs` {enabled, tau_large 2e-4, tau_ssim 0.7, min_views 2} |
| photometric | loss | splatstudio.losses:create_photometric | render.rgb | ssim 0.2 |
| hgs-scaling-reg | loss | ss_hard_gs.losses:create_scaling_reg | render.rgb | 0.01·mean(prod scale); also taps the target image into `outputs` (REHGS needs GT, density hooks only get `outputs`) |

## Fields
| field | shape | lr | init |
|---|---|---|---|
| means (anchor) | [3] | 0 | voxelized SfM (0.001 / 0.01 / 0.005) |
| hgs_feat | [32] | 0.0075 | 0 |
| hgs_offset | [10,3] | 0.01→1e-4 ×extent | 0 |
| hgs_scaling | [6] | 0.007 | log sqrt(distCUDA2) |
| hgs_blob | [8] | none | MLP weights repacked every step |

## C1 PGHGS (Eq. 5–6) → density
Per stats step, for every rasterized neural Gaussian (Scaffold `combined_mask`), the gradient norm (NDC-scaled like
3DGS) is pushed into a running top-k buffer `topk[N·K, k]` (descending). At growth: `kth = topk[:, k-1]`.
Level i candidate: `(avg ≥ τ_i & denom>40) | (kth ≥ λ τ_i)`, τ_i = τ·2^i. Buffers reset every growth step.
## C2 REHGS (Eq. 7–8) → density + loss tap
Per stats step: `ms_stats(meta)` → `area_max` per rasterized Gaussian (|S_i|). Over-large: `area_max > τ_large·W·H`.
SSIM map (3DGS window, same padding) of detached render vs target; sample at round(means2d). Hard if SSIM < 0.7.
Counter `re_count[N·K] += 1`. At growth: `re_hard = re_count ≥ 2` joins every level's candidates. Reset per growth.

## Probes
| probe | fails if |
|---|---|
| neural-gaussians | decode ≠ per-anchor loop reference (Scaffold: [feat, view dir] input, no distance), not view dependent |
| visibility-filter | anchor prefilter ≠ constructed Scaffold-fork cases (behind camera, off-screen) |
| anchor-density | growing does not place an anchor in the offset's voxel; low-opacity anchor not pruned; misalignment; fields not trainable leaves |
| pghgs-criterion | top-k buffer ≠ sort reference; a Gaussian with low average but 3 large gradients is not grown with PGHGS, or *is* grown with PGHGS disabled |
| rehgs-criterion | CUDA |S_i| ≠ torch argmax reference; large+low-SSIM Gaussian seen in 2 views not grown; same Gaussian with high SSIM or 1 view grown; disabled → not grown |
| blob-eval | fresh renderer (no bank, save/load round trip) ≠ live render |

## Fidelity gaps
1. **No official HGS code**: the way hard Gaussians enter Scaffold's 3-level growing, the reset period, and the
   ≥40-view success mask exemption are my choices (paper.md §7). Expected effect: Gaussian count and PSNR may differ.
2. Rasterizer gsplat (classic, eps2d 0.3) vs Scaffold diff-gaussian-rasterization fork; argmax/|S_i| from a kernel
   re-traversing gsplat tiles (Mini-Splatting plugin).
3. Background white (runtime) vs black (Scaffold default).
4. View sampling random-with-replacement vs shuffled stack; SSIM valid conv in the loss.
5. Update order: runtime steps field Adam before `after_backward`; Scaffold densifies before `optimizer.step()`.
6. Global MLP state transported in `hgs_blob` per-anchor field.
7. Per-dataset voxel size: method default mip360 (0.001); tandt/db need `voxel_size` override (variants `hard-gs-tandt`, `hard-gs-db`).
8. Paper baselines retrained by authors on their platform; compare improvement over our gsplat baseline vs paper's over 3DGS (+0.59 garden, +1.15 counter) and over Scaffold-GS (+0.50, +0.60).
9. Effi-HGS not implemented (not the reproduced row).
