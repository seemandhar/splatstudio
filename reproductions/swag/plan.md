# SWAG — native SplatStudio plan

No official code exists (project page "Code (TBA)"), so this is a **paper-only** build. Every
choice the paper leaves open is listed under *Assumptions* (A1…), and each one is a fidelity risk.

Methods (module `ss_swag`, plugin `reproductions/swag/plugin/`):
| method | what |
|---|---|
| `swag` | full method (C1 + C2), Photo Tourism schedule (60k it, densify every 2000 until 30k) |
| `swag-a` | ablation SWAG-A: transient opacity variation off (Tab. 3) |
| `swag-t` | ablation SWAG-T: colour dependency off; the MLP still predicts Δα from (c, emb(x), l_I) (Tab. 3) |

## Components
| name | kind | factory | requires | provides | config |
|---|---|---|---|---|---|
| swag-renderer | renderer | `ss_swag.renderer:create_renderer` | state.gaussian3d, camera.pinhole | render.rgb, render.alpha, gradient.screen.signed | appearance, transient, temperature 0.1, eval_u 0.5, log_eps 1e-10, background 0 |
| swag-density | density | `ss_swag.density:create_density` | gradient.screen.signed | – | refine_every 2000, refine_start_iter 500, refine_stop_iter 30000, reset_every 3000, grow_grad2d 2e-4, position_lr_max_steps 30000, bank_* (emb dim 24, hash T 2^19 / L 12 / F 2 / N_min 16 / N_max 2048, MLP 3×64, lrs 0.0025, aabb quantile 0.005) |
| photometric | loss | built-in | render.rgb | – | 0.8 L1 + 0.2 D-SSIM (unchanged 3DGS loss, §5.1) |

## Learned global state (no runtime support → component-owned bank)
`ss_swag.bank.ACTIVE` (`SwagBank`, built by the density in `set_training_cameras`): image embeddings
[N_train, 24], `HashGrid` (pure PyTorch, tcnn-style: dense indexing where (res+1)^3 ≤ T, otherwise
spatial hash with primes (1, 2654435761, 805459861), trilinear interpolation), MLP 51→64→64→64→4,
AABB of the SfM points. Its own Adam (lr 0.0025, eps 1e-15) is stepped in `density.after_backward`;
exported as `density.state["bank"]` (checkpoint). View → embedding index by viewmat match.

## Renderer (equation → code)
1. `c` = SH colour: `clamp_min(SH(sh, μ − campos) + 0.5, 0)` (3DGS), degree from the runtime.
2. `h = HashGrid(normalise(μ))` [N, 24]; `x = cat(c, h, l_I)` [N, 51]; `o = MLP(x)` [N, 4].
3. Eq. 7/8: `c^I = sigmoid(o[:, :3])` (A3), `Δα^I = o[:, 3]`.
4. Eq. 9/10: `Δα̃ = sigmoid((log(|Δα^I| + ε) + log U − log(1 − U)) / T)`, U ~ U(0,1) per Gaussian
   per render while training (training view, grad on), U = 0.5 otherwise; T = 0.1.
5. Eq. 11: `α^I = clamp_min(sigmoid(o_logit) − Δα̃, 0)`.
6. gsplat `rasterization(colors=c^I [N,3], opacities=α^I, sh_degree=None)`, black background (A9).
SWAG-A: Δα̃ ≡ 0. SWAG-T: `c^I = c`. Non-training views use the bank's evaluation override, else the
mean training embedding (A7); with no bank (plain `ss-eval`) the renderer renders plain 3DGS
(c, α) — the headline is `ss_swag.evaluate`.

## Density
Wraps `splatstudio.backends.gsplat.Density` (gsplat DefaultStrategy + the 3DGS opacity-reset fix)
with the paper schedule; adds the bank, steps its Adam each step and overrides the means LR to the
3DGS schedule `lr0 · 0.01^(min(step, 30000)/30000)` (3DGS `position_lr_max_steps` default; the
runtime would decay over all 60k steps).

## Evaluation (`python -m ss_swag.evaluate --run RUN`)
NeRF-W protocol (§5.3): for each test view, initialise a test embedding at the mean training
embedding (A7), optimise only it for 128 Adam steps at lr 0.1 (A8) with the training loss on the
left W − W//2 columns (eval mode, U = 0.5); render the full view with it; metrics on columns
[W//2:]. Metrics: PSNR (float [0,1]), runtime SSIM, LPIPS-Alex and LPIPS-VGG (A10). Also reports
the no-optimisation (mean-embedding) numbers and the transient fraction Var_I[Δα̃] > 0 (C3) over
the training images.

## Probes
| probe | contribution | fails if |
|---|---|---|
| hash-grid | C1 | HashGrid ≠ an independent per-point loop implementation (dense + hashed levels), or the encoding is constant |
| appearance-mlp (CUDA) | C1 | rendered colours/opacities ≠ independent formula; changing l_I does not change c^I; SWAG-T colour ≠ SH colour |
| concrete-opacity | C2 | Eq. 9/11 ≠ independent formula with fixed U; eval U=0.5 ≠ closed form |Δα|^{1/T}/(1+|Δα|^{1/T}); training samples not near-binary at T=0.1; SWAG-A does not give α^I = α |
| transient-variance | C3 | Var over images not zero for static (Δα=0) Gaussians or not > 0 for varying ones |
| nerfw-protocol (CUDA) | protocol | optimising the embedding does not lower the left-half loss; right-half metric depends on the left half |
| bank-training (CUDA) | C1/C2 | a training step does not update embeddings, hash table and MLP; LR override wrong |
| state-transitions | base | built-in |

## Assumptions (unspecified in the paper; no code to check)
- A1 MLP "3 layers of 64 hidden units" = 3 hidden ReLU layers (51→64→64→64→4), PyTorch default init.
- A2 Hash grid: 2 features/level (forced by input dim 51), growth b = exp((ln 2048 − ln 16)/11),
  level res = floor(16·b^l), init U(−1e-4, 1e-4) (tcnn), AABB = per-axis 0.5–99.5 % quantiles of
  the SfM points (+5 % margin), positions clamped into the box.
- A3 c^I = sigmoid; Δα^I linear; ε = 1e-10 inside log|Δα|.
- A4 `c` (SH colour) is the view-dependent 3DGS colour, not detached (gradients reach the SH).
- A5 Image embeddings init N(0, 1) (nn.Embedding default, as NeRF-W); all bank lrs constant 0.0025,
  Adam eps 1e-15.
- A6 3DGS defaults for the rest: densify from 500, grad 2e-4, opacity reset every 3000 until
  densify end (30000), SH +1 every 1000, position LR decays over 30000 steps then stays at final.
- A7 Embedding for unseen views: mean of training embeddings (start point of test-time optimisation).
- A8 Test-time optimisation: 128 Adam steps, lr 0.1 (as WildGaussians' NeRF-W implementation), L1 + 0.2
  D-SSIM on the left half, embedding only. lr 0.01 was the first choice; on the 2k smoke checkpoint
  it had not converged the **left-half** loss after 128 steps (0.137 vs 0.125 at lr 0.1, 0.123 at
  lr 0.1 x 512 / lr 1 x 256), so lr was raised using only the optimised left-half objective, never
  right-half metrics.
- A9 Black background (3DGS default); full resolution (3DGS default; all PT images ≤ 1600 px).
- A10 Metrics: float PSNR, runtime SSIM, LPIPS-Alex headline (NeRF-W-line papers), VGG also given.
- A11 U is sampled independently per Gaussian per training render.

## Data
Photo Tourism, NeRF-W splits, scenes **brandenburg-gate** (prepared by the WildGaussians build:
`~/datasets/phototourism/native/brandenburg-gate`) and **sacre-coeur** (IMC-2020 `dense/`, NeRF-W
`sacre.tsv`, converted with the WildGaussians `prepare_wild.py`; on /bigdata via symlink). Trevi
(11.5 GB) skipped per the 2-scene rule.

## Fidelity gaps
1. All assumptions A1–A11 (no code). Largest risks: test-time optimisation (A7/A8), AABB/hash
   normalisation for unbounded PT scenes (A2), embedding init (A5).
2. Own JIT CUDA hash-encoding kernel (`hash_cuda.py`, checked against the PyTorch reference by the
   `hash-cuda` probe) instead of tiny-cuda-nn: fp32 tables (tcnn uses fp16 params/half MLP); gradients
   reach the Gaussian centres through emb(x) (as tcnn would if xyz requires grad — unspecified).
3. Runtime SSIM (valid conv) in the loss vs 3DGS same-padded SSIM; random view sampling with
   replacement. Small.
4. gsplat rasteriser vs Inria rasteriser. Small.
5. Paper's 3DGS baseline row (19.99 / 17.57) is the authors' run; our gsplat baseline on PT is not
   available → compare absolute numbers and the SWAG-A / SWAG-T ablation deltas.
6. `ss-eval` cannot do the NeRF-W protocol; the headline is `ss_swag.evaluate`.
