# WildGaussians — native SplatStudio plan

Methods (module `ss_wildgaussians`, plugin `reproductions/wildgaussians/plugin/`):
| method | config | use |
|---|---|---|
| `wildgaussians` | On-the-go schedule (30k), appearance **on** (paper Tab. 1/7 "Ours") | headline On-the-go |
| `wildgaussians-noapp` | On-the-go, appearance off (released `nerfonthego.yml`) | code-default variant |
| `wildgaussians-pt` | Photo Tourism schedule (200k, `phototourism.yml`), appearance on | headline Photo Tourism |

Port target = official code (`source/code/wildgaussians/method.py`), whose numerics override the
paper where they differ (paper.md §7).

## Components
| name | kind | factory | requires | provides | config (paper/code) |
|---|---|---|---|---|---|
| wg-renderer | renderer | `renderer:create_renderer` | state.gaussian3d, camera.pinhole | render.rgb, render.rgb_raw, render.appearance, gradient.screen.signed, gradient.screen.absolute, render.antialiased_2d, render.filter_3d | kernel_size 0.1, near 0.2, alpha_cap 0.999, background black, appearance on/off |
| wg-density | density | `density:create_density` | gradient.screen.signed, gradient.screen.absolute | – | schedule (iterations, densify from/until/every, reset interval), grad 2e-4, abs-grad quantile, percent_dense 0.01, min_opacity 0.005, big-ws 0.1, lrs, sky 100k, Fourier freqs 4, bank cfg (emb dim 32, lrs, DINO path) |
| wg-uncertainty | loss | `losses:create_uncertainty` | render.rgb | – | λ_prior 0.5, clip 0.1, dropout 0.5, dino max 350, warmup start/iters, protected iters |
| wg-color | loss | `losses:create_color` | render.rgb, render.rgb_raw | – | λ_dssim 0.2 |

Base `photometric` loss is **replaced** by `wg-color` (Eq. 6/11); gsplat DefaultStrategy density
replaced by the official densification (GOF abs-grad union, never-reset statistics).

## Shared learned state (no runtime support → component-owned bank)
`ss_wildgaussians.bank.ACTIVE` (WildBank, created by the density in `set_training_cameras`):
image embeddings [N_train, 32] (N(0, 0.01), lr 1e-3), appearance MLP 59→128→128→6 (lr 5e-4),
uncertainty head (BN2d(384) + Conv1×1, lr 1e-3), frozen DINOv2 ViT-S/14-reg (own implementation of
the Meta architecture, weights `~/models/dinov2/dinov2_vits14_reg4_pretrain.pth`, sha256
f433177089a6…641fb), GT-feature cache. Own Adam (eps 1e-15), stepped in `after_backward`.
Exported in `density.state["bank"]` (a property, so the checkpoint always has the final values).
View → image index by viewmat match (as DeblurGS/SpotLessSplats). Evaluation loads the bank
from the checkpoint (`python -m ss_wildgaussians.evaluate --run RUN`).

## Fields
| field | shape | lr | init |
|---|---|---|---|
| wg_embedding (appearance variants) | (24,) | 0.005 | Fourier features of SfM+sky xyz + N(0,1e-4) |
| filter_3d | () | 1e-8 (no grad) | Mip-Splatting 3D filter std, recomputed by density |

## Implementation notes (equation → code)
1. Initialisation (`density.set_training_cameras`): sh0 := RGB (official features_dc = rgb quirk),
   opacity 0.1, add 100k Fibonacci sky points (r = 10·q97, kept if v < 2H/3 in some camera with
   the official projection quirk), colour (237,247,252)/255, opacity logit 20 (= sigmoid 1.0 in
   fp32; official logit(1)=inf, which the runtime's finiteness check rejects — same behaviour:
   zero opacity gradient until the first reset), log-sqrt mean 3-NN dist² scales over all points,
   identity quats, Fourier embeddings, 3D filter, bank.
2. Renderer: F = clamp_max(cat(sh0, shN), 1); raw colours = relu(SH(F, μ − campos) + 0.5);
   toned F' = clamp_max(F · mul + [offset/C0, 0…], 1) with (offset, mul) = 0.01·MLP(F_dc, g_i, e_j)
   (Eq. 4–5, code); two gsplat calls (antialiased, eps2d 0.1, absgrad, near 0.2, black bg) with
   3D-filtered scales/opacity and a straight-through 0.999 opacity cap. SH degree min(step//1000, 3)
   (official +1 at every 1000th iteration before rendering).
3. wg-uncertainty (Eq. 8–10): σ from GT DINO (full res, cached), cosine at ≤350 px between GT and
   detached toned prediction, L = mean(dino_part · down(1/(2σ²))) + 0.5 mean(log σ); mask
   M = (min(1/(2σ²),3) > 1), warm-up; detached (head grad None) in the protected window after
   resets. M is passed to wg-color through `outputs["wg_mask"]` (wg-uncertainty runs first).
4. wg-color (Eq. 6/11): 0.8·mean(M·|toned − gt|) + 0.2·mean(M·(1 − SSIM_map(raw, gt))), 3DGS
   same-padded SSIM map averaged over channels.
5. Density: statistics accumulate ‖Σ_calls ∂L/∂μ'‖ (NDC units) and ‖Σ_calls |∂L/∂μ'|‖ on visible
   Gaussians, never reset (new Gaussians start at 0); at densify steps: clone (≤ 0.01·extent) and
   split (> 0.01·extent) where g ≥ 2e-4 or a ≥ quantile(a, 1 − ratio); prune σ(o) < 0.005 and,
   after the first reset interval, max scale > 0.1·extent; recompute 3D filter. Filter-aware
   opacity reset with the official doubled-filter quirk (paper.md §7.7); 3D filter every 100 it after densification until iterations − 100; xyz lr
   log-linear 1.6e-4→1.6e-6 (On-the-go) / 1.6e-5→1.6e-7 (PT) × extent.
6. Evaluation (`ss_wildgaussians.evaluate`): official NerfBaselines metrics (uint8 truncation,
   dm_pix SSIM, LPIPS-Alex; LPIPS-VGG and runtime SSIM also reported). On-the-go: zero embedding,
   full image. Photo Tourism (`nerfw`): embedding := 0, 128 Adam steps (lr 0.1) on the left
   W − W//2 columns with L = 0.8 L1 + 0.2 (1 − SSIM) of the toned render, gradient masked by the
   eval-mode uncertainty mask of the left GT; metrics on columns [W//2:] (official) and
   [W − W//2:] (strict, no shared column for odd W).

## Probes (each fails if its mechanism is removed)
| probe | measures |
|---|---|
| appearance-toning | toned SH = independent formula (manual MLP, offset/C0 on DC, mul on all 16); Fourier features = paper formula sin/cos(π p 2^m); embedding change changes toned colours, not raw |
| split-objective | wg-color = independent formula; SSIM gradient reaches only the raw image and L1 only the toned image; masked pixels get zero gradient |
| dino-backbone | own ViT-S/14-reg block-5 features == timm `vit_small_patch14_reg4_dinov2` with the same checkpoint (518², and a non-square padded input) |
| uncertainty-loss | L_unc and M vs NumPy-style reference on synthetic features; head gets gradient, prediction does not; protected window ⇒ head grad None; warm-up ramp |
| sky-init | Fibonacci sphere (unit norm, golden-angle spacing), radius 10·q97, official visibility quirk, opacity/colour |
| densify-official | union of abs-grad quantile and grad threshold; clone keeps original stats and zeroes clone's; split/prune rules; big-ws prune gated by step |
| filter-3d | 3D filter = official formula (W/2 principal point, max focal); filter-aware reset |
| nerfw-protocol (CUDA) | test-time embedding optimisation lowers the left-half loss on a synthetic scene; metric is invariant to the left half of the GT |
| render-two-pass (CUDA) | renderer's two gsplat passes: raw = gsplat(SH colours), toned = gsplat(toned colours); means2d grads of both reach the density statistics |
| state-transitions | built-in alignment probe (with wg_embedding, filter_3d) |

## Data
`tools/prepare_wild.py` → `~/datasets/onthego-undist/native/{patio,patio-high}` (HF undistorted,
NerfBaselines intrinsics rescale) and `~/datasets/phototourism/native/brandenburg-gate` (IMC-2020
dense, NeRF-W split, full res), both on /bigdata via symlink. Validation = hard-linked copy of
test. Not registered in `tools/datasets.json` (benchmarker registers / passes `--data`).

## Fidelity gaps
1. View sampling with replacement (runtime) vs shuffled stack (official). Small.
2. Runtime steps the Gaussian Adam before density; official densifies/resets first and the new
   nn.Parameters have no grad, so on densify and reset iterations the official code effectively
   skips the Gaussian Adam step (for the replaced tensors). One step in 100 (On-the-go) /
   400 (PT); negligible.
3. Exact 3-NN (runtime) vs distCUDA2 approximate KNN for initial scales. Small.
4. gsplat rasterizer (antialiased mode = Mip 2D filter, near 0.2 cull, tile numerics) vs the
   official Mip/GOF CUDA rasterizer: gsplat culls Gaussians with (filtered, compensated)
   opacity < 1/255 (radii 0) and uses an opacity-aware extent (≤ 3.33σ) where the official code
   counts every projected Gaussian visible with radius 3σ, so `denom` differs for faint /
   sub-filter Gaussians; alpha clamp 0.999 (gsplat, with straight-through cap) vs 0.99. Small.
5. Sky opacity logit 20 instead of +inf (identical in fp32 until the first reset).
6. Separate Adams for bank/Gaussians (identical math to one Adam with param groups).
7. Single seed; paper on RTX 4090.
8. Photo Tourism opacity reset 15 000 (code) vs 3000 (paper); On-the-go appearance on (paper) vs off
   (code); γ without +1 (code). Variants cover the On-the-go choice.
9. `ss-eval` cannot see the bank: its numbers are raw renders (no toning, no test-time
   optimisation); the headline is `ss_wildgaussians.evaluate`.
