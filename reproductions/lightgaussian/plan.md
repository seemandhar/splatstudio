# LightGaussian native plan

Method `lightgaussian`, module `ss_lightgaussian`, package `plugin/` (`splatstudio-method-lightgaussian`).
Single native run of **45000 steps** = 3DGS 30000 + co-adaptation 5000 + SH distillation 5000 +
VQ fine-tune 5000 (pattern of pup-3dgs / gaussianpop / c3dgs: ss-train cannot resume). Variant
`lightgaussian-released` = the released code path (40000 steps; VQ + encode at 40000, no VQ
fine-tune). `lightgaussian-smoke` = 2000-step compressed schedule (diagnostic only).

## Components
| component | kind | factory | notes |
|---|---|---|---|
| `lightgaussian-renderer` | renderer | `ss_lightgaussian.renderer:create_renderer` | wraps the gsplat `Renderer` (same config/provides); during distillation on steps with `step % 3 != 0` renders the student at a pseudo view (camera centre + N(0, 0.05²) per axis, world units); stores the camera used in `outputs["lg_camera"]` |
| `lightgaussian-density` | density | `ss_lightgaussian.density:create_density` | orchestrates the stages (below); requires = base gsplat density requires |
| `photometric` | loss | `ss_lightgaussian.losses:create_photometric` | L1 + 0.2 D-SSIM vs GT; returns an attached 0 during distillation (official distill loss never sees GT) |
| `lightgaussian-distill` | loss | `ss_lightgaussian.losses:create_distill` | distillation only: L1 + 0.2 D-SSIM(student, teacher render at `lg_camera`, teacher = frozen SH-3 copy of the 35k model, no grad); attached 0 otherwise |

Shared run state (stage, next step, teacher params, pseudo-view RNG) lives in
`ss_lightgaussian.shared.STATE`, set by the density's `initialize` (one run per process, as the
c3dgs bank registry). At `ss-eval` time STATE is inactive → plain rendering of the decoded model.

No new FieldSpecs. SH degree 2 is expressed by keeping `shN[:, 8:] == 0` (re-zeroed after every
optimizer step from 35000 on); a degree-3 render with zero degree-3 coefficients equals the
degree-2 render, so checkpoints/eval need no degree change.

## Stage schedule (density `after_backward`, after the runtime optimizer step of `step`)
1. `step <= 30000`: unchanged SplatStudio 3DGS density (gsplat DefaultStrategy + opacity-reset
   fix); means LR of a 30000-step run (identical to `baseline3dgs-*-30000`).
2. `step == 30000`: global significance (CUDA kernel adapted from the official
   `renderCUDA_count`, on gsplat 1.5.3 tile lists, over all training views):
   `score_j = Σ_pixels σ_j T` with the official skip/termination rules; `v = score·(V/V_90)^0.1`,
   `V = Π exp(scales)`, `V_90 = sort(V, desc)[int(0.9N)]`; remove `v <= sort(v)[int(0.66(N-1))]`
   with `gsplat.strategy.ops.remove` (Adam moments kept: official restores the 3DGS optimizer).
3. `30001..35000` co-adaptation: no densification; Adam (weight decay 0, restored Adam groups);
   non-means LRs × 0.95^(⌊k/400⌋−75); means LR = 3DGS exp schedule with max_steps 35000.
4. `step == 35000`: teacher = detached copy of all params (SH-3); fresh optimizer state (official
   `training_setup` after `restore`); zero `shN[:, 8:]`; snapshot opacities (frozen).
5. `35001..40000` distillation: AdamW semantics (decoupled weight decay 0.01 applied after the
   step with the LR used), non-means LRs × 0.90^(⌊k/500⌋−70), means max_steps 40000, opacity
   restored after each step (official `requires_grad=False`; the Parameter itself stays a
   trainable leaf — requires_grad pitfall avoided), renderer pseudo views on `k % 3 != 0`.
6. `step == 40000`: v-score recomputed on the student (β 0.1); non-VQ = top int(0.4N);
   official EuclideanCodebook port (kaiming-uniform init, significance-weighted EMA 0.8 with
   Laplace smoothing, 1000 iterations × 80000 samples with replacement, 10 dead codes replaced by
   the batch's 10 most significant vectors each iteration) on the 27-d [f_dc, f_rest(deg ≤ 2)]
   vectors in the official PLY order; nearest-code assignment; model set to the official
   dequantised values (codebook fp16 → VQ rows, other SH fp16, opacity/scale/quat fp16, xyz fp32).
7. `40001..45000` VQ fine-tune (not in released code; choice): codebook (DC part and rest part)
   trainable via AdamW (wd 0.01, LRs 0.0025 / 0.000125), gradient = Σ per-Gaussian SH gradients
   of its members (post-accumulate-grad hooks zero those rows for the runtime optimizer and
   accumulate into the codebook); assignment fixed; all other attributes trainable (opacity
   unfrozen); fresh optimizer state; non-means LR ×0.95 every 400 its, means max_steps 45000,
   weight decay 0.01 (fresh `GaussianModel.training_setup` = AdamW default); photometric loss.
8. `step == total_steps`: encode to the official `extreme_saving` files (npz each, zipped in
   memory, size = bytes / 2²⁰), decode them back into the params → checkpoint, validation and
   `ss-eval` measure the decoded model. The zip bytes are stored in the density state.
A run-length guard (means LR after step 1) raises if `--steps != total_steps`.

## Probes
| probe | checks (fails if mechanism removed) | CUDA |
|---|---|---|
| `significance-oracle` | kernel score/count vs brute-force per-pixel PyTorch composite (σ·T, 0.99 clamp, 1/255 skip, 1e-4 termination) incl. an opaque stack; differs from Σ alpha·T | yes |
| `prune-rule` | v-score vs independent formula (β, V_90 index, no clip), prune mask `<=` 66th percentile | no |
| `distill-views` | pseudo-view centre shift ~N(0,0.05²) only on k%3≠0; distill loss = L1+0.2DSSIM vs teacher at that camera, zero for identical models; photometric gated off; degree-3 render with zeroed coeffs == degree-2 | yes |
| `vq-official` | one EMA step equals the one-hot official formula; kaiming bound; dead-code replacement; learned codebook beats init | yes |
| `codec-roundtrip` | official layout (13-bit indices, packbits mask, fp16), decode == expected, size = zip length | no |
| `pipeline` | tiny end-to-end run through all stages: prune count, teacher copy, frozen opacity, shN[:,8:]=0, VQ rows == codebook during fine-tune and the codebook moves, final decode, all fields trainable leaves | yes |
| `schedule` | stage boundaries, LR factors, weight decay per stage, run-length guard | no |

## Fidelity gaps
1. VQ fine-tune optimiser/LRs/schedule unspecified (no released code) — chosen as above; effect
   ±0.1–0.3 dB on the final stage (paper room #7→#8: +0.31 dB). `lightgaussian-released` gives
   the released-code number.
2. Teacher = pruned 35k model (paper Alg. 1) not the unpruned 30k model (script placeholder).
3. Runtime: white background (official black), random-with-replacement view sampling (official
   shuffled stack, matters for distill/pseudo-view cadence only statistically), SSIM valid conv,
   LPIPS-VGG `lpips` package. 3DGS stage is our gsplat baseline (calibration in `_calibration`).
4. Score kernel uses gsplat projection (pixel centres +0.5, opacity-aware radii) instead of the
   Inria rasterizer; skip/termination rules are the official ones. Atomics instead of the
   official racy `+=` (official scores are slightly under-counted on contended Gaussians).
5. Prune happens after step 30000's optimizer step (official: after backward of 30001, before its
   step) — one-step offset.
6. AdamW weight decay is applied after the Adam update (official before): O(lr²) difference.
7. FPS not measured with the paper's rasterizer; sizes are computed exactly as the official zip.
