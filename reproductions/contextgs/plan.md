# ContextGS native plan

Method `contextgs`, module `ss_contextgs`, plugin `reproductions/contextgs/plugin/`.

ContextGS = Scaffold-GS + HAC's quantization/entropy scheme, with HAC's hash grid replaced by (a) an anchor-level
autoregressive context over 3 voxel levels and (b) a per-anchor hyperprior coded with a factorized density. The HAC
native port (`reproductions/hac/plugin`) already solved the shared infrastructure; the parts that apply are **copied
with attribution** (no import across plugins): anchor population as native fields, MLP-owning bank shared through
`model.ACTIVE`, Scaffold `visible_filter` prefilter, anchor growing/pruning, encode → decode → bake, `*_blob` field
carrying the render MLPs into ss-eval, the CUDA arithmetic coder (vendored from official HAC).

## Feasibility
`feasible_natively: true` with the same documented workaround as HAC (global render-MLP weights reach ss-eval through
a per-anchor field `cgs_blob`, because `runtime.evaluate()` does not restore component state).

## Components

| component | kind | factory | requires → provides | config |
|---|---|---|---|---|
| cgs-renderer | renderer | ss_contextgs.renderer:create_renderer | state.gaussian3d, camera.pinhole → render.rgb, render.depth.expected_z, gradient.screen.signed | feat_dim 50, n_offsets 10 |
| cgs-density | density | ss_contextgs.density:create_density | gradient.screen.signed | `density.DEFAULTS` (paper.md §3) |
| photometric | loss | splatstudio.losses:create_photometric | render.rgb | 0.8 L1 + 0.2 D-SSIM (unchanged) |
| cgs-scaling-reg | loss | ss_contextgs.losses:create_scaling_reg | render.rgb | 0.01·mean(prod s) |
| cgs-rate | loss | ss_contextgs.losses:create_rate | render.rgb | λ_e 0.004 (low rate), λ_m 5e-4 |

## Fields (anchors are the native population)

| field | shape | lr (set by density) | init |
|---|---|---|---|
| means | [3] | 0 | voxelized SfM (0.001) |
| cgs_feat | [50] | 0.0075 | 0 |
| cgs_offset | [10,3] | 0.01·s → 1e-4·s | 0 |
| cgs_scaling | [6] | 0.007 | log sqrt(3-NN mean d²) |
| cgs_mask | [10,1] | 0.01·s → 1e-4·s | 1 |
| cgs_hyper | [12] | 0.0075 | 0 (grown anchors: scatter_max of parents, like feat) |
| cgs_blob | [8] | never stepped | 0; render MLPs after the bake |

Bank (`model.ContextBank`, own Adam, eps 1e-15): Scaffold mlp_opacity/cov/color, `FactorizedPrior(12)` (faithful
torch port of compressai `EntropyBottleneck`: filters (3,3,3,3), init_scale 10, likelihood bound 1e-9; quantiles fixed
at init because ContextGS never calls the aux loss, so medians = 0), `mlp_grid` = 3 level MLPs
(in 15 / 71 / 71 → 100 → 175), anchor bounds, `level_scale` (searched once at the first entropy step).

## Mechanism → code
- `levels.py`: verbatim ports of `torch_unique_with_indices`, `find_divide_scale`, `divide_levels`,
  `mapping_to_orign`, `index_of_level_L_in_orign`, `extract_context_feat`, and `multi_scale(...)` = official
  `multi_scale_generating` (training noise / eval STE, per-level prediction, 15% sampled bits, bit_per_param incl.
  hyper bits). Including the official parent-position context and level-1 gather order (paper.md §7.1–2).
- `renderer.py`: phases exactly as `generate_neural_gaussians` (≤3000 plain, (3000,10000] fixed-Q noise on visible
  anchors, 10000 bound update, >10000 multi-level context on all anchors then visible selection); gsplat
  rasterization (near 0.2); decoded/blob path for ss-eval (copied from HAC port).
- `density.py`: HAC port's init / stats / growing (+ hyper latent scatter_max) / pruning / LR schedules; bake =
  `conduct_encoding` + `conduct_decoding` order (hyper first, levels 2→1→0 with decoded context), lossless check,
  size accounting (meta + hyper + anchor 16 bit + feat + scaling + offsets + masks + 32-bit `mlp*` params).
- `codec.py`: HAC CUDA coder (Gaussian, Bernoulli) + factorized-prior coding of ẑ with the same coder (CDF from the
  factorized logits over the data range).

## Probes

| probe | fails if… |
|---|---|
| entropy-model | Gaussian bits ≠ erf formula; Q not in (0, 2Q0); STE rounding wrong |
| factorized-prior | likelihood ≠ an independent re-derivation of compressai's cumulative logits; noise / dequantize modes or the 1e-9 bound wrong; parameter count ≠ compressai's |
| level-partition | levels not disjoint / not covering; representative ≠ min index of its voxel (brute force); `find_divide_scale` ratio not ≈ τ; masked anchors not moved to the origin |
| context-model | level-k predictions depend on finer-level attributes (autoregressive order broken); context rows ≠ an independent gather of the official order; hyperprior removal changes nothing |
| codec-roundtrip | Gaussian / Bernoulli / factorized streams not lossless or >2% off entropy |
| neural-gaussians | decode ≠ per-anchor loop reference; not view dependent |
| visibility-filter | prefilter ≠ Scaffold visible_filter on a constructed case (z-only filter fails) |
| anchor-density | growing does not place an anchor at the offset voxel / prune wrong / fields+Adam misaligned / hyper not grown |
| training-phases | phase boundaries 3000/3001/10000/10001; ~15% sample; rate-loss value λ_e·bpp + λ_m·mean σ(mask); gradient reaches feat, hyper, mlp_grid, prior; LR schedules |
| bake-eval | fresh bank-free render of the checkpoint ≠ baked render; blob ≠ trained weights; un-baked state renders |

## Fidelity gaps
1. **Rasterizer**: gsplat instead of the Scaffold fork; `visible_filter` re-implemented (copied from HAC port).
2. **Background** white (runtime) vs black (official mip360).
3. **Sampling / SSIM**: runtime random-with-replacement views, valid-conv SSIM in the loss (official same-padding).
4. **Update order**: runtime steps field Adam before `after_backward`; official densifies before the step and skips
   the optimizer at the last iteration (bank step skipped at 30000 here).
5. **Global-state transport** via `cgs_blob` (as HAC).
6. **Codec**: HAC CUDA arithmetic coder for all streams instead of torchac (Gaussian, masks) and compressai rANS
   (hyper). Same probability models; per-level streams instead of 1000-anchor batches; meta size counted from our
   own metadata. Expected size difference ≪ 1%.
7. **Per-dataset voxel size**: registered config is mip360's (0.001).
8. **Run guards / total steps**: as HAC — needs `--recipe 3dgs`, `--num-gaussians` > SfM size, `--steps 30000`.
9. **Data**: benchmark on `mipnerf360-1600` (1.6K-width PIL resize = official loader); paper's 3DGS/Scaffold rows
   are copied from other papers, so judge against ContextGS rows (and our HAC runs on the same images).
