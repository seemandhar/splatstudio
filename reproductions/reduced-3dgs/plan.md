# Plan — Reduced-3DGS native port

Method **`reduced-3dgs`**, module **`ss_reduced_3dgs`**. Base: `make_method("gsplat")` (3DGS
recipe, gsplat DefaultStrategy with the opacity-reset fix) — kept unchanged except as below.

## Components
| component | kind | factory | requires / provides | config (paper value) |
|---|---|---|---|---|
| `reduced-renderer` | renderer | `ss_reduced_3dgs.renderer:create_renderer` | provides base caps | – ; masks `shN` by per-Gaussian `sh_degree` (official `_degrees`) |
| `reduced-density` | density | `ss_reduced_3dgs.density:create_density` | `gradient.screen.signed` | base 3DGS config; `total_steps` 30000, `densify_until` 15000, `mercy_interval` 1000, `mercy_until` total−3000, `opacity_reset_every` 3000, `lambda_mercy` 1, `mercy_minimum` 3, `opacity_quantile` 0.03, `opacity_cap` 0.05, `knn` 30, `box_size` 1, `dead_opacity` 1/255, `dead_prune_every` 100, `save_steps` [7000], `cull_sh_step` 15000, `std_threshold` 0.04, `cdist_threshold` 6√3/255, `encode` true, `codebook_size` 256, `half_float` true |
| `photometric` | loss | base | – | L1 + 0.2 D-SSIM |
| `opacity-l1` | loss | `ss_reduced_3dgs.losses:create_opacity_l1` | `render.rgb` | λ_α 0.001 (mean σ(o) over visible) |
| `sh-sparsity` | loss | `ss_reduced_3dgs.losses:create_sh_sparsity` | `render.rgb` | λ_sh 0.1, normalised by 45·V (official CUDA) |

## Field
`sh_degree` shape (), lr 0, init 3.0 — per-Gaussian SH band count; copied by gsplat
duplicate/split/remove; set by SH culling; encoded as PLY group.

## Density events (after the runtime optimizer step, `step` 1-based)
1. base gsplat step (densify/prune/reset).
2. dead prune σ(o)<1/255 if step≥15000 and step%100==0, or step∈save_steps or step==total.
3. mercy (C1) if step%1000==0, step≤27000, and (step≥15000 or step%3000≠0).
4. SH culling (C2) at step 15000: two passes over all training cameras with the CUDA
   transmittance kernel on gsplat's tile lists.
5. step==total and encode: K-means codebooks + float16 → official PLY bytes → decode back
   into params (decoded model is what the checkpoint, `validation.json` and `ss-eval` see).
   Encoded size via `generate_results.py:memory_results` (MiB) and the actual PLY byte length,
   stored in `density_state["reduced3dgs"]` and printed.

## CUDA (JIT, `kernels.py`, `csrc/`)
- `transmittance_stats`: per-Gaussian ΣT and #pixels on gsplat 1.5.3 tile lists (traversal
  adapted from the mini-splatting plugin's `ms_raster.cu`, Inria-derived; gsplat alpha numerics).
- `knn_index`: exact 30-NN excluding self, `boxKnn2` from reduced-3dgs `simple-knn` (Inria).

## Probes
| probe | contribution | fails if |
|---|---|---|
| `sh-sparsity-gradient` | C2 | gradient ≠ λ/(45V)·sign on visible active coeffs |
| `opacity-l1-gradient` | C1 | gradient ≠ λ/V·σ(1−σ) on visible |
| `knn-exact` (cuda) | C1 | kernel kNN ≠ brute force |
| `redundancy-score` (cuda) | C1 | scores/mercy mask ≠ independent O(N²) reference |
| `transmittance-stats` (cuda) | C2 | ΣT / counts ≠ per-pixel torch compositing reference |
| `sh-band-culling` (cuda) | C2 | synthetic diffuse/deg-1/deg-3 Gaussians not assigned 0/1/3 |
| `codec-roundtrip` | C3 | decode(encode) ≠ codebook values / half positions, size ≠ formula, K-means tie rule |
| `schedule` | C1–C3 | event steps ≠ official enumeration |

## Fidelity gaps
1. Runtime order: optimizer step before density ops (official: prune/mercy before step) — tiny.
2. gsplat rasterizer numerics (alpha clamp 0.999, eps2d 0.3) vs Inria (0.99); per-Gaussian
   degree implemented as a coefficient mask (identical forward).
3. Baseline differences (white bg, sampling with replacement, no 20-px screen-size prune in
   gsplat DefaultStrategy) shared with `baseline3dgs-*`.
4. SH culling uses gsplat's transmittance (same definition, different rasteriser numerics).
5. After culling, Adam moments of dropped coefficients are zeroed (official keeps them; the
   dropped coefficients drift slightly and enter the K-means population) — negligible.
6. Decoded opacity/scale ±inf (codebook centre exactly 1.0 / 0) replaced by ±65504 to pass the
   runtime finiteness check (renders identically).
7. The encoded PLY is not written to the run directory (density has no output path); its exact
   byte length + sha256 are recorded.
