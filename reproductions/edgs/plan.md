# EDGS native plan

Method names: **`edgs`** (paper-full: correspondence init + pinv SH init + densification-free training)
and **`edgs-nosh`** (released-code equivalent: SH rest = 0; paper ablation "w/o SH init").
Module `ss_edgs`, package `splatstudio-method-edgs` in `plugin/`.

## Architecture
EDGS has two stages and SplatStudio has no initialization hook, so:

1. **Preprocessing (outside training)** — `ss-edgs-init` (console script in the plugin,
   `ss_edgs/preprocess.py`) reads a native scene, uses **training views only**
   (`splatstudio.data.load_dataset(...)["splits"]["train"]`), runs RoMa and writes a dataset copy
   `~/datasets/edgs-init/<dataset>/<scene>/`:
   - `transforms.json` (byte copy) and `images/` (hard links) → identical `dataset_sha256` / protocol id;
   - `points.npz` (`points` float32 [N,3], `colors` [N,3] in [0,1]) — the runtime's init cloud;
   - `edgs_init.npz` — per-point extras the runtime cannot carry: `points_sha256`, `log_scales [N]`,
     `opacity_logits [N]`, `shN [N,15,3]`;
   - `edgs_init.json` — config, train-view list, timings, counts, RoMa/weights hashes.
2. **Training** — gsplat renderer + photometric loss unchanged; a custom **`edgs-density`**
   component (no densification). In `initialize` it hashes `params["means"]`, finds the matching
   `edgs_init.npz` under `init_root` (error if none: no fallback), and overwrites
   scales/opacities/(shN if `sh_init`). Train with **`--recipe splatstudio`**; the density restores 3DGS's
   camera-extent means LR from `set_training_cameras` (`--recipe 3dgs` OOMs, see gap 10).

## Components
| component | kind | factory | requires/provides | config (paper/code value) |
|---|---|---|---|---|
| gsplat | renderer | splatstudio.backends.gsplat:create_renderer | as built-in | classic, no absgrad |
| edgs-density | density | ss_edgs.density:create_density | requires () | `init_root ~/datasets/edgs-init`, `sh_init` True/False, `prune_opacity 0.005`, `prune_until 15000`, `opacity_decay 0.99`, `decay_every 10`, `decay_until 15000`, `means_lr_floor_step 8000`, `total_steps 30000`, `opacity_lr 0.025` |
| photometric | loss | splatstudio.losses:create_photometric | render.rgb | 0.8 L1 + 0.2 D-SSIM |

No new FieldSpecs.

## Preprocessing (maps to official `init_gaussians_with_corr`)
- refs: scipy k-means (K=min(num_refs, #train)) on flattened W2C, nearest camera per cluster, seed 228.
- NNs: `nns_per_ref` closest train cameras by Frobenius distance of W2C (excluding self).
- RoMa outdoor, `upsample_preds=False`, `symmetric=False`, `match()` at 560², warp/certainty
  bilinearly resized to image resolution; `c^i = max_j c^{ij}`.
- weights `w=c^i; w[w>0.05]=1`; `multinomial(w, M, replacement=False)` (M = matches_per_ref).
- For every NN j: triangulate sampled pixel (A = pixel-centre NDC, B = warp^{ij}) with the
  DLT/lstsq of §3.3 using NDC projection rows built from true K; L1 reprojection error in NDC for
  both views; keep the NN with minimal max-error; drop points with error > 0.01 (official sets opacity
  logit −10 and prunes them at step 1 — equivalent).
- colour = reference pixel; scale = log(0.001·0.5·‖x − c_ref‖) (isotropic); opacity logit 0; quat identity.
- SH (C4, paper; revised after review-1): DC = RGB2SH(reference colour) (runtime: RGB2SH(colors)).
  Bands 1–3 fit the residual R = O_nn − O_ref with rows projected off the reference direction
  (P = I − y_ref y_refᵀ/|y_ref|²) and ridge λ = 1/16: H = P Y_oᵀ(Y_o P Y_oᵀ + λI)⁻¹R. The reference
  colour is exact at every degree; ‖H‖ ≤ ‖R‖/(2√λ) = 2‖R‖ (λ chosen from this conditioning bound,
  not from metrics). The literal pinv-over-16-then-overwrite-DC reading double counts colour
  (garden: max |coef| 217, 0.7% > 2) and was dropped. Observations, directions and view indices are
  cached in `edgs_init.npz` (`ss-edgs-init --refit-sh` refits without RoMa).

## Density (`edgs-density`, official `trainer.py`)
- initialize: apply sidecar; opacity LR → 0.025; means LR base captured (already × scene scale),
  set to base·0.01^(8000/30000).
- after_backward(step): if step < 15000: remove opacity<0.005 (gsplat `ops.remove`); if step < 15000 and
  step % 10 == 0: logit += log(0.99). Means LR for next step = base·0.01^(max(step+1, 8000)/total_steps).
- No densification, no opacity reset (the built-in Density's 3DGS reset is not used).

## Probes
| probe | contribution | fails if |
|---|---|---|
| triangulation (CPU) | C2 | DLT wrong/removed: synthetic cameras, recovered points vs ground truth <1e-4, error filter keeps clean & rejects perturbed matches |
| sampling (CPU) | C3 | weights not thresholded/uniform, NN choice not argmin of max error, max-aggregation of confidences broken |
| sh-fit (CUDA) | C4 | fitted SH evaluated by gsplat's CUDA `spherical_harmonics` does not reproduce both observations; DC not the ref colour |
| no-densify-density (CPU) | C5 | count increases, pruning outside schedule, opacity decay step wrong, means LR not floored at lr(8000), opacity LR ≠ 0.025, sidecar not applied |
| roma-matching (CUDA) | C1 | RoMa missing (ProbeUnavailable if weights absent) or warp convention wrong: a horizontally shifted crop must yield flow shift ≈ shift in x, ≈0 in y |
| base state/photometric/renderer probes | runtime | as built-ins |

## Benchmark
Per-scene numbers exist. Scenes: mipnerf360 bicycle, garden (outdoor), counter (indoor);
tandt truck, train; db drjohnson, playroom. Matches per ref: 25000 (M360), 15000 (T&T/DB), 3 NNs,
180 refs (code full_eval). 30000 steps, `--recipe splatstudio`. Data roots `~/datasets/edgs-init/<dataset>`.

## Fidelity gaps
1. SH-init composition (C4) is a documented deviation: paper gives no DC/rest consistency rule or
   regularizer; we use the constrained ridge residual fit (λ = 1/16) — `edgs-nosh` bounds it.
2. White background + random-with-replacement view sampling (runtime) vs black bg + shuffled stack.
3. RoMa `match()` zeroes certainty of out-of-image flow and clamps; official custom forward does not. Minor.
4. Pixel-centre A coordinates instead of the bilinearly-resized grid; colours from the exact pixel
   (official rounds r+0.5). Sub-pixel.
5. Camera proximity uses our native W2C (same world frame as the COLMAP conversion; if the converter
   rescaled the world the rotation/translation weighting of the Frobenius norm changes). Minor.
6. True principal point used in NDC (3DGS assumes centred). Mip-NeRF360 cx=W/2 so no effect there.
7. Means LR: runtime applies its own decay after each step; the density overrides it in after_backward
   (same step granularity). Step 1 uses lr(8000) set at initialize — matches official.
8. Pruning every step triggers a runtime `check_state` + log row on each count change (speed only).
9. Training time excludes the preprocessing (recorded separately in `edgs_init.json`).
10. **Recipe** (found in smoke test): `--recipe 3dgs` OOMs (runtime `knn_log_scales` cdist [4096,N],
    61 GiB for N=4M). EDGS trains with `--recipe splatstudio`; `edgs-density` (`camera_extent_lr`)
    rescales the means LR to 3DGS's camera extent from `set_training_cameras`. Scales are overwritten
    by the sidecar either way, so the recipes are otherwise equivalent for EDGS.
