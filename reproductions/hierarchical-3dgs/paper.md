# A Hierarchical 3D Gaussian Representation for Real-Time Rendering of Very Large Datasets

Kerbl*, Meuleman*, Kopanas, Wimmer, Lanvin, Drettakis — ACM TOG (SIGGRAPH) 2024, arXiv 2406.12080.
Official code: https://github.com/graphdeco-inria/hierarchical-3d-gaussians (cloned to `source/code/`,
plus submodules `hierarchy-rasterizer` @63fa247 and `gaussian-hierarchy` @677c855).
Licence: Inria/TUW non-commercial research & evaluation licence (same terms as 3DGS).

## 1. Method summary
- Very large captures are split into chunks; a coarse *scaffold* + *skybox* gives backdrop content.
- Each chunk is trained with a modified 3DGS optimization (max-based densification, monocular depth
  regularization, exposure compensation, EWA/Mip-Splatting 2D AA filter).
- A tree hierarchy (BVH median split + KL-minimizing merge, Eq. merge) is built over the chunk's
  Gaussians; interior nodes are Gaussians with "falloff" instead of opacity.
- A cut is selected per view at target granularity τ_ε (projected node AABB size in pixels);
  parent→children transitions interpolate attributes (Eq. t_interp, α′ Eq. alphap).
- The hierarchy's interior nodes are optimized at random granularities (leaves frozen), then
  chunks are consolidated into a scene hierarchy rendered with streaming LOD.

Contributions:
- **C1** hierarchy generation (§3.1): BVH median split, merge Eq.(merge) with w′_i = o_i·S_i,
  SH averaging, falloff, orientation matching; compaction (§4.2, in `AppearanceFilter`).
- **C2** cut selection + smooth level switching (§3.2): granularity ε(n), t_n Eq.(t_interp),
  α′ = 1−(1−α_p)^{1/K}, α(t) = t·α_i + (1−t)·α′.
- **C3** hierarchy optimization (§4.1): random τ ∈ [τ_min, τ_max) log-uniform, full resolution,
  gradients through interpolation and α′, leaves (and "anchors") frozen.
- **C4** chunk-scale training changes (§5.2): max (not mean) screen-gradient densification,
  depth regularization (App. C.1), exposure (App. C.2), skybox/scaffold, AA filter.
- **C5** large-scene pipeline (chunking, scaffold, consolidation, streaming viewer) (§5.1, 5.3).

## 2. Equations (verbatim from method.tex)
- Merge: $\mu^{(l+1)} = \sum_i^N w_i \mu_i^{(l)}$,
  $\Sigma^{(l+1)} = \sum_i^N w_i (\Sigma_i^{(l)} + (\mu_i^{(l)} - \mu^{(l+1)})(\mu_i^{(l)} - \mu^{(l+1)})^T)$,
  $w_i = w'_i/\sum w'_i$, $w'_i = \mathrm{o_i} \sqrt{|\mathbf{\Sigma}'_i|}$ (implemented with ellipsoid surface S_i).
- SH: $SH^{(l+1)} = SH_1^{(l)} w_1 + SH_2^{(l)} w_2$; falloff $\frac{\sum_i^N w_i'}{S_p}$ (may exceed 1; α clamped).
- Interpolation: $t_n = \frac{\tau_{\epsilon} - \epsilon(n)}{\epsilon(p) - \epsilon(n)}$ (Eq. t_interp).
- $\alpha' = 1 - (1 - \alpha_p)^{1/K}$ (Eq. alphap, K children); $\alpha(t) = t \alpha_i + (1-t) \alpha'$.
- Granularity sampling: $\tau_{\epsilon} = \tau_{max}^\xi~\tau_{min}^{1-\xi}$, ξ∈[0,1).
- Depth (App. C.1): $\mathcal{L}_D = |\hat D - D^*|$ on inverse depth, weight exp-decayed 1→0.01.
- Exposure: $C_c = E[C|1]^T$, E per image 3×4, init identity.
- Code (`gaussian_model.py:620-688`): densification statistic = running **max** of ‖∂L/∂μ′‖ (NDC
  units of the Inria rasterizer) between densification steps; split/clone selection
  `max_grad · max_radii2D · σ(o)^{1/5} ≥ 0.015` **and** `σ(o) > 0.15`; split if
  `max scale > percent_dense·extent` (percent_dense = 1e-4), new scale = s/(0.8·2), N = 2;
  prune σ(o) < 0.005 only (no screen/world-size prune).
- Code (`train_single.py:183-190`): every iteration, Gaussians with max scale > 0.02·extent are
  shrunk ×0.8 (`skip_scale_big_gauss` False by default).
- Code (`render_hierarchy.py:57`): pixel τ → world-ratio threshold
  `(2·(τ+0.5))·tan(fovx/2)/(0.5·W)`; `expand_to_size` compares node AABB max extent / distance.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations (leaves) | 30000 | code arguments/__init__.py:84 |
| position lr | 2e-5 → 2e-7 ×extent, max_steps 30000 | code arguments:85-88 (3DGS: 1.6e-4) |
| feature lr / shN | 0.0025 / 0.0025/20 | code arguments:89, gaussian_model:270 |
| opacity / scaling / rotation lr | 0.05 / 0.005 / 0.001 | code arguments:90-92 |
| Adam eps | 1e-15; "OurAdam" sparse update of rows with ∂opacity≠0 (global step count) | gaussian_model:281, OurAdam.py |
| percent_dense | 1e-4 | code arguments:97 |
| densify interval / from / until | 300 / 500 / 15000 | code arguments:99-102 |
| opacity reset interval | 3000 (reset to min(σ,0.01)) | code arguments:100, gaussian_model:510 |
| densify_grad_threshold | 0.015 (max-grad · radius · σ^{0.2}) | code arguments:103 |
| init opacity | 0.01 | gaussian_model:202 |
| lambda_dssim | 0.2 | code |
| depth weight | 1.0 → 0.01 exp | code arguments:104-105, App. C.1 |
| exposure lr | 1e-3→1e-4, delay 5000, mult 1e-3 (disabled for single-chunk comparisons) | App. C.2 / §6.2 |
| background | random per iteration (train_single), black (train_post, render_hierarchy) | code |
| big-Gaussian shrink | >0.02·extent → ×0.8 every step | train_single:183-190 |
| AA | Mip-Splatting 2D filter (h_var = 0.3, opacity × sqrt(det ratio)) | hierarchy-rasterizer forward/backward |
| post-opt iterations | 15000 | scripts/full_train.py:143 |
| post-opt lr | feature 5e-4, opacity 0.01, scaling 1e-3, rotation 1e-3, position 2e-5 sched. (max_steps 30000) | full_train.py:143-144 |
| post-opt τ range (world ratio) | limmin 0.005, limmax 0.1, log-uniform | train_post.py:45-50 |
| post-opt optimizer | torch Adam (dense), opacity activation = abs (falloff stored directly) | train_post:30, gaussian_model:393 |
| hierarchy filter limit | 0.0005, 2.0 | mainHierarchyCreator.cpp:150-157 |
| eval τ (pixels) | 0, 3, 6, 15 | render_hierarchy.py --taus |

## 4. Schedules
SH degree +1 every 1000 its; position lr exp decay over 30000; depth weight exp decay over
iterations; densification 500–15000 every 300; opacity reset every 3000 (< 15000); post-opt
15000 its with freshly-reset schedule (pos lr continues to 2e-5·0.01^{0.5} at its 15000).

## 5. Builds on
Inria 3DGS codebase with Mip-Splatting 2D filter; per-pixel loss unchanged (0.8 L1 + 0.2 D-SSIM).

## 6. Protocol
- Main tables (Tab. 1, 5): one chunk of each of their own datasets SmallCity / Wayve / Campus / BigCity,
  test = every 50th image alphabetically per rig camera, exposure disabled; which chunk is not stated.
  Tab. 2: Mill-19 Building/Rubble vs Mega-NeRF with train_test_exp.
- **Mip-NeRF 360 (text, §6.2)**: average over the entire dataset, single-chunk optimization:
  3DGS 29.11, ours (leaves) 28.96; LOD τ1/τ2/τ3 basic/optimized: 28.86/28.87, 28.05/28.52, 24.82/27.25.
  No per-scene numbers, no SSIM/LPIPS, resolution/split not stated (code default: llffhold 8 with
  `--eval`, images capped to 1600 px width with `-r -1`).
- Metrics: PSNR/SSIM/LPIPS(VGG) on black background, clamped renders (render_hierarchy.py).

## 7. Ambiguities
- Mip-NeRF 360 3DGS = 29.11 is far above the 3DGS paper (27.21 at images_4/images_2): resolution
  and split for this experiment are unknown → compare deltas, not absolutes.
- Whether depth regularization was used on Mip-NeRF 360 (needs DPT/Depth-Anything; no depth
  files ship with Mip-NeRF 360). Code applies depth only when `-d` is passed.
- Whether a scaffold/skybox was used on Mip-NeRF 360 (none needed for single scenes; we use none).
- Post-optimization hyper-parameters on Mip-NeRF 360 are taken from the chunk script.
- Compaction (§4.2) in code is `AppearanceFilter::filter(root, gaussians, 0.0005, 2.0)` using all
  COLMAP cameras of the source (official includes test cameras; we pass train cameras only).
