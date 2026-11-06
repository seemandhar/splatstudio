# Splatfacto-W — native SplatStudio plan

Module `ss_splatfactow` (plugin `reproductions/splatfacto-w/plugin/`), written fresh against the
paper and the official Nerfstudio code (`source/code/splatfactow/`, Apache-2.0). Code defaults
override the paper (paper.md §7).

| method | what (paper Tab. 1) |
|---|---|
| `splatfacto-w` | full: appearance MLP + robust mask + SH background model + alpha loss |
| `splatfacto-w-a` | appearance model only (random training background, no mask, no alpha loss) |
| `splatfacto-w-t` | appearance model + robust mask |

## Components
| name | kind | factory | requires | provides | key config |
|---|---|---|---|---|---|
| sfw-renderer | renderer | `ss_splatfactow.renderer:create_renderer` | state.gaussian3d, camera.pinhole | render.rgb, gradient.screen.absolute, render.antialiased_2d | bg_model, sh_degree 3 / interval 2000, bg_sh_degree 4 / interval 1000, num_downscales 2, resolution_schedule 3000, eval_background (0.149,0.1647,0.2157) |
| sfw-density | density | `ss_splatfactow.density:create_density` | gradient.screen.absolute | – | splatfacto refinement (warmup 1000, every 100, reset 25×100, grad 8e-4, size 0.01, cull α 0.1 / scale 0.5 / screen 0.15, split screen 0.05, stop screen 15000, stop split 20000), nerfstudio scene normalisation (orient up, center poses, ×3), all LR schedules, bank (embed 48, features 72, MLP 256×2, bg MLP 128×2) |
| sfw-photometric | loss | `ss_splatfactow.losses:create_photometric` | render.rgb | – | 0.8 L1 + 0.2 D-SSIM with robust mask (0–40 %, start 6000, reset 6000, upper 0.4, box 5 / 0.4) |
| sfw-alpha | loss | `ss_splatfactow.losses:create_alpha_loss` | render.rgb | – | λ 0.15, thr 0.003, box 3, select > 0.6 (full method only) |

## Fields
`appearance_features` [72], init 0, lr 0.02 → 1e-3 over 40000 (density overrides the LR every step).
`sh0/shN` remain (runtime base fields) but are unused: they never get a gradient, so Adam never
allocates moments for them.

## Global learned state (runtime cannot hold it) → component-owned bank
`ss_splatfactow.bank.ACTIVE` (`SfwBank`, built in `density.set_training_cameras`): image embeddings
[N_train, 48] ~ N(0,1); colour MLP 120→256→256 (ReLU, ReLU) + heads 3 / 45 (rest zero-init);
background MLP 48→128→128 + heads 3 / 72; nerfstudio normalisation (R, centre, scale s); one Adam
(eps 1e-15) with the 7 nerfstudio param groups and their exponential schedules; step counter.
Exported to `density.state["bank"]`; `evaluate.py` rebuilds it from the checkpoint.

## Renderer (equation → code)
ns-step k = bank.step (0-based, = runtime step − 1). Training (grad on, bank.training):
downscale d = 2^max(2 − k//3000, 0); K[:2] /= d, W,H = floor(W/d), floor(H/d).
1. ℓ = embedding of the matching training view (viewmat lookup); eval: override > matching
   training view (code protocol: test views are trained on their left halves) > embedding 0 (code
   `use_avg_appearance=False`).
2. b = MLP(ℓ, f) [N,16,3]; c = max(SH_k'(b, R(μ − o)) + 0.5, 0), k' = min(k//2000, 3) (code;
   paper writes a sigmoid). R = nerfstudio orientation so SH lives in the same frame as the code.
3. gsplat `rasterization` antialiased, absgrad, near 0.01, explicit colours.
4. Background: b_bg = MLP_bg(ℓ) [25,3]; C_bg(r) = SH_kb(b_bg, R·d_ray(r)), kb = min(k//1000, 4),
   d_ray through pixel centres; rgb = clamp(C + (1 − α) C_bg, 0, 1). Without bg model: random
   colour per training render, fixed colour at eval.

## Losses
`sfw-photometric`: target area-downscaled by d (= `resize_image`); masked (test-left-half) views
are zero-padded on the right to the full width (exactly the code's `gt·mask, pred·mask`
on a full-size image); robust mask per code (paper.md §2); stores the masked gt for the alpha loss.
`sfw-alpha`: 0.15 · mean(α[box3(mean_c|gt_m − bg| < 0.003) > 0.6]).

## Density
Own implementation of splatfacto `after_train` / `refinement_after` using gsplat ops
(`duplicate`, `split`, `remove`, `reset_opa`), thresholds in nerfstudio units converted by s.
Re-initialises scales to log(mean 3-NN distance) and random quaternions (code init). Steps the
bank Adam and sets every LR for the next step (means lr_ns/s; nerfstudio f(k)).

## Data / protocol
Native PT data (`~/datasets/phototourism/native/<scene>`, NeRF-W tsv split with NaN rows dropped,
full resolution) matches the code's dataparser (verified: 763/10 BG, 830/21 SC). Code protocol
needs test images in training with the right half masked: `tools/prepare_sfw.py` builds
`~/datasets/phototourism/sfw/<scene>` (on /bigdata): hard links of all images + PNG crops of the
left W//2 columns of each test image added to the train split (same pose/intrinsics). Masked views
are recognised by cx > 0.75·width (all normal PT frames have |cx/W − 0.5| < 0.1; checked by the
prep tool and by the density). Headline evaluation `python -m ss_splatfactow.evaluate --run RUN`:
right half (cols ≥ W//2) of test views, PSNR / SSIM (valid Gaussian, = pytorch_msssim) / LPIPS-Alex
(+VGG). Views without a trained embedding (native data) fall back to embedding-only left-half
optimisation (paper-text protocol).

## Probes
| probe | contribution | fails if |
|---|---|---|
| appearance-colour (CUDA) | C1 | renderer colours ≠ independent MLP + hand-written real SH (deg ≤ 3) + 0.5 clamp; ℓ has no effect; MLP sees view direction |
| robust-mask | C2 | mask ≠ independent loop implementation; upper 40 % masked; transient block not masked; k interpolation wrong |
| masked-view-padding | protocol | crop+pad loss ≠ full image × left-half mask loss (L1, SSIM, robust mask) |
| background-model (CUDA) | C3 | bg depends on camera translation; composite ≠ C + (1−α)bg; degree-0 bg ≠ C0·b0; ℓ has no effect |
| alpha-loss | C3 | value ≠ 0.15·mean α over independently selected pixels; no gradient; boundary pixels selected |
| splatfacto-schedule | base | reset / densify / cull windows, resolution and SH schedules, LR values ≠ independent formulas |
| splatfacto-refine | base | refinement on a fixture gives wrong split/dup/cull counts or misaligned optimizer state |
| ns-normalisation | base | scale/rotation ≠ independent numpy nerfstudio auto_orient computation |
| state-transitions | base | built-in |

## Fidelity gaps
1. View sampling with replacement (runtime) vs nerfstudio's per-epoch permutation. Small.
2. Masked views: crop width W//2 padded to 2·(W//2) (≠ W for odd W: one column); alpha loss sees
   only the left half of masked views (code also scores the zero right half, which can only be
   selected when the background is black). Small.
3. Masked-view detection by principal point heuristic (validated on the data).
4. Runtime losses are summed per step identically; mixed precision off in both.
5. 55000 iterations (code) vs 65000 (paper text).
6. LPIPS from `lpips` package (Alex) = torchmetrics' backend; PSNR float; SSIM same formula.
7. No shared gsplat baseline on Photo Tourism; compare to the paper's absolute numbers and the
   -A/-T ablation deltas.
