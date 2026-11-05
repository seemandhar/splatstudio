# Octree-GS — implementation spec

**Paper:** *Octree-GS: Towards Consistent Real-time Rendering with LOD-Structured 3D Gaussians*, Ren, Jiang, Lu, Yu,
Xu, Ni, Dai (arXiv 2403.17898, TPAMI 2025 version; LaTeX e-print of 2024-10-18 in `source/latex`).
**Official code:** github.com/city-super/Octree-GS @ `9db64fe` (2025-08-19), `source/code` (read only, never run).
Licence: Inria/MPII Gaussian-Splatting licence (non-commercial research). The repo vendors the Scaffold-GS fork of
`diff-gaussian-rasterization` (adds `visible_filter`) and `simple-knn` (Inria).

Variant reproduced: **Our-Scaffold-GS** (neural Gaussians decoded from anchors; the paper's headline rows). `train.py`
of the official repo implements exactly this variant (the 2D-GS/3D-GS variants live in Octree-AnyGS).

## 1. Method summary
- Scaffold-GS backbone: anchors (fixed positions) with a feature `f` (32-D), `k=10` learnable offsets, 6-D scaling;
  global MLPs decode per-view opacity / colour / covariance of k neural Gaussians per anchor.
- Anchors are organised in an octree of `K` LOD levels: voxelisation of the SfM cloud at voxel size `δ/2^L` for each
  level `L` (all levels are initialised from the same SfM points).
- Per view, an anchor is used iff its level `L_j ≤ L̂_ij`, where `L̂_ij` is predicted from the camera–anchor distance
  plus a per-anchor LOD bias `ΔL_j` (cumulative LOD: render level ≤ L̂).
- LOD-aware anchor growing (same level / "next level" by gradient thresholds that grow by `2^{βL}`), LOD-bias
  increase for high-gradient anchors, Scaffold opacity pruning and a view-frequency ("weed out") filter.
- Progressive training: start with levels `≤ ⌊K/2⌋`, unlock one more level after geometric intervals.

Contributions:
- **C1** LOD-structured anchors: octree construction (K from distance range, per-level voxelisation, §3.1.1–3.1.2).
- **C2** View-dependent anchor (LOD) selection with learnable-by-growth LOD bias ΔL (§3.1.3, Eq. 3).
- **C3** Adaptive anchor control: next-level growing with level thresholds, ΔL update, opacity pruning and
  view-frequency filtering (§3.2).
- **C4** Progressive (coarse-to-fine) training (§3.3).
- (C5 appearance embedding §3.4: disabled for Mip-NeRF360/T&T/DB in the official scripts, `appearance_dim=0`.)

## 2. Equations (verbatim from `sec/method.tex`)
- Levels: $K = \lfloor \log_{2}(\hat{d}_{max}/\hat{d}_{min}) \rceil + 1$; $d_{max}$/$d_{min}$ are the $r_d$-quantiles of
  camera–point distances, $r_d=0.999$.
- Voxelisation: $\mathbf{V}_L = \{ \lfloor\frac{\mathbf{P}}{\delta / 2^{L}}\rceil \cdot \delta / 2^{L} \}$.
- LOD selection (Eq. 3): $\hat{L_{ij}} = \lfloor L_{ij}^* \rfloor = \lfloor \Phi(\log_2(d_{max}/(d_{ij}*s))) + \Delta L_j\rfloor$, anchor used iff $L_j \le \hat L_{ij}$.
- Growing thresholds: $\tau_g^{L}=\tau_g 2^{\beta L}$, $\tau_g=0.0002$, $\beta=0.2$; promote to $L+1$ iff $\nabla_g>\tau_g^{L+1}$.
- LOD bias: anchors with $\nabla_v > 0.25\,\tau_g^{L}$ get $\Delta L \mathrel{+}= \epsilon$, $\epsilon = 0.01$.
- Progressive: $N_{i-1} = \omega N_i$, ω = 1.5, total progressive iterations 10k, start level $\lfloor K/2\rfloor$.
- Loss (Scaffold variant, §4 impl.): $0.8\,L_1 + 0.2\,L_{D\text{-}SSIM} + 0.01\,\mathcal{L}_{vol}$, $\mathcal{L}_{vol}=\text{mean}\prod s_i$
  (code: mean over the rendered neural Gaussians).

### Code-level definitions (code overrides the paper; all in `scene/gaussian_model.py` unless noted)
- `set_level`: for every training camera, `dist = ||P - c||`; per camera `q(dist, 0.999)` and `q(dist, 0.001)`
  (torch.quantile, linear); over all `[min,max]` pairs again `dist_max = q(all, 0.999)`, `dist_min = q(all, 0.001)`;
  `standard_dist = dist_max`; `levels = round(log2(dist_max/dist_min)/log2(fork)) + 1`; `init_level = int(levels/2)`.
- `create_from_pcd`: points = `unique(SfM)`; `box_min = min(P)*1.1`, `box_max = max(P)*1.1` (scalars over all
  coordinates); `voxel_size = (box_max-box_min) / fork^base_layer` (`base_layer=10` mip360/T&T, 12 DB);
  `init_pos = box_min·1`; level L positions `unique(round((P-init_pos)/(voxel/2^L)))·(voxel/2^L)+init_pos`.
  Then `weed_out` twice when `visible_threshold < 0` (mip360 script: -1): first with threshold 0 (drops never-selected
  anchors) returning the **mean view frequency of all initial anchors** as the new `visible_threshold`, then again
  with it. Scales `log sqrt(distCUDA2)` ×6, feat 0, offsets 0, opacity/rotation unused (no grad).
- `weed_out(pos, lev)`: view frequency = mean over training cams of `[lev ≤ clamp(round(log2(std/||pos-c||)), 0, K-1)]`
  (no ΔL, no half-voxel shift); keep iff `> visible_threshold`.
- `set_anchor_mask(c, it)`: `pos = anchor + (voxel/2)/fork^lev`; `pred = log2(std/||pos-c||)/log2(fork) + ΔL`;
  `coarse = searchsorted(coarse_intervals, it) + 1 + init_level` in training (progressive), `K` at eval;
  `int = clamp(round(pred), 0, coarse-1)` (`dist2level='round'`); mask `lev ≤ int`.
- `set_coarse_interval(10000, 1.5)`: `n = K-1-init_level`, `q=1/1.5`, `a1 = 10000(1-q)/(1-q^n)`, cumulative
  `a1 q^i` (sums to 10000).
- `prefilter_voxel`: `visible = mask & visible_filter(anchor[mask], exp(scaling)[mask,:3], identity rotation)`.
- `generate_neural_gaussians`: input `[f, view_dir]` (35-D; `add_*_dist=False`, no distance); opacity Tanh, keep
  `>0`; colour Sigmoid; cov 7 → `scale = scaling[:,3:]·sigmoid(.[:3])`, `rot = normalize(.[3:])`;
  `xyz = anchor + offset·scaling[:,:3]`. Black background.
- `training_statis` (500 < it < 25000): Scaffold statistics (opacity accum, visit count `anchor_demon` for visible
  anchors, NDC means2D gradient norm per neural Gaussian with radii > 0).
- `adjust_anchor` (it > 1500, it % 100 == 0, it < 25000): `grads = accum/denom`, `offset_mask = denom > 40`;
  `anchor_growing`; reset; prune `opacity_accum < 0.005·demon` & `demon > 80`; on prune `scaling[:,3:]` (raw log
  value) clamped at 0.05.
- `anchor_growing`, for each level `L` with anchors: `uv = 2^0.2`; `th = 0.0002·uv^L`, `th_ds = th·uv`,
  `th_extra = th·0.25`; same-level candidates `th ≤ g < th_ds` of level-L anchors' offsets → new anchors in empty
  level-L voxels (size `voxel/2^L`), feature = scatter_max of the parents; next-level candidates `g ≥ th_ds` → new
  level-(L+1) anchors in empty voxels of size `voxel/2^{L+1}` (only if L < K-1 and level L+1 already has anchors),
  **feature 0**; both sets filtered by `weed_out`; new scaling `log(voxel size)`×6, offsets 0, ΔL 0;
  `ΔL += 0.01·[anchor_grad ≥ th_extra]` for **all** anchors at every level iteration (not restricted to level L).
- **Quirk:** `if ~self.progressive or ...` — `~True == -2` is truthy, so the "disable next-level growing / ΔL during
  progressive training" condition of the paper is never active in the official code: next-level growing and ΔL
  updates run from iteration 1600 on. Port follows the code.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| iterations | 40000 | train.sh `iterations=40_000`; §4 impl. "40k" |
| feat_dim / n_offsets | 32 / 10 | arguments/__init__.py:47-48 |
| fork | 2 | scripts |
| base_layer | 10 (mip360, T&T), 12 (DB) | train_*.sh (paper: intermediate level voxel 0.02 → base_layer<0 path, unused) |
| visible_threshold τ_v | -1 → adaptive mean (mip360), 0.9 (T&T, DB) | scripts (paper says 0.7) |
| dist2level | round | scripts (paper Eq. 3 uses floor + blending) |
| update_ratio β | 0.2 | scripts / paper |
| extra_ratio / extra_up | 0.25 / 0.01 | scripts / paper ε |
| dist_ratio r_d | 0.999 | scripts / paper |
| levels / init_level | -1 / -1 → adaptive | scripts |
| progressive | True; coarse_iter 10000, coarse_factor 1.5 | scripts, arguments |
| appearance_dim | 0 | scripts |
| start_stat / update_from / interval / update_until | 500 / 1500 / 100 / 25000 | arguments (paper: densify to 20k) |
| min_opacity / success_threshold / grad threshold | 0.005 / 0.8 / 0.0002 | arguments |
| position lr | 0 | arguments |
| offset lr | 0.01 → 0.0001 (× spatial_lr_scale = cameras_extent) | arguments |
| feature / scaling / opacity / rotation lr | 0.0075 / 0.007 / 0.02 / 0.002 | arguments |
| mlp_opacity / mlp_cov / mlp_color lr | 0.002→2e-5 / 0.004→0.004 / 0.008→5e-5, exp decay over 40000 | arguments |
| Adam eps | 1e-15 | training_setup |
| λ_dssim / vol reg | 0.2 / 0.01 | arguments, train.py |
| resolution | `-r -1`: images/ PIL-resized to width 1600 when wider | camera_utils.py:25-39 |

## 4. Schedules
- LR: exp interpolation (`get_expon_lr_func`, no delay) over 40000 iterations for offset and MLPs; set at the start
  of every iteration.
- Progressive levels: coarse levels `0..init_level` up to `coarse_intervals[0]`, then +1 level per interval, all levels
  after 10000.
- Densification window (500, 25000); growing/pruning every 100 iterations from 1600.
- Optimiser not stepped at the final iteration.

## 5. Builds on
Scaffold-GS (anchor MLP decode, statistics, growing via offset voxels, opacity pruning, scaling clamp on prune,
vol. regulariser). Unchanged from 3DGS: L1 + D-SSIM loss, rasteriser (Scaffold fork).

## 6. Protocol
Mip-NeRF 360 (9 scenes), Tanks&Temples (truck, train), Deep Blending (drjohnson, playroom); llffhold 8 (`--eval`);
mip360 loaded from `images/` with `-r -1` → width 1600 (all 9 scenes are wider than 1600); black background;
40k iterations; metrics from `metrics.py` (PSNR, SSIM, LPIPS-VGG). #GS(k) = mean number of rendered neural
Gaussians per test view (`visibility_filter.sum()`); Mem = stored size. Per-scene tables in the supplementary.
Training on A100.

## 7. Ambiguities
1. Paper: floor + opacity blending (Eq. 3 + transition); code (scripts): `dist2level=round`, no blending. → code.
2. Paper: τ_v = 0.7; code: adaptive mean view frequency (mip360), 0.9 (T&T/DB). → code.
3. Paper: next-level growing disabled during progressive training; code: always enabled (`~bool` quirk). → code.
4. Paper: densification ends at 20k; code: `update_until=25000`. → code.
5. Paper: intermediate-level voxel 0.02; scripts use `base_layer=10` (voxel = box/1024). → scripts.
6. `set_anchor_mask` adds a half-voxel shift to the anchor for LOD distances; `weed_out` and the MLP view direction
   do not. → verbatim.
