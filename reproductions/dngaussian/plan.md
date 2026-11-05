# DNGaussian — native plan

Method `dngaussian` (module `ss_dngaussian`), ablation/baseline `dngaussian-3dgs` (same pipeline,
no depth passes = paper's 3DGS† row, 16.46 dB). Plugin: `plugin/`. Dataset: `llff-fewview3-r8-dng`.

## Components
| component | kind | factory | role |
|---|---|---|---|
| dngaussian-renderer | renderer | `renderer:create_renderer` | photometric render with neural colours (black bg, near 0.2); `hard_depth` / `soft_depth` helpers; baked inference path + eval `--near 10` cull; Inria-style visibility radii |
| dngaussian-density | density | `density:create_density` | owns GridRenderer + Adam (`neural.ACTIVE`), hard/soft depth passes with their own Adam steps, opacity-gated densification, near/invisible pruning, LR schedule, bake |
| photometric | loss | `losses:create_photometric` | L1 + 0.2 (1 − SSIM_same) (Eq. 10, L1 weight 1) |
| dngaussian-regularization | loss | `losses:create_regularization` | shape 0.002 / scale 0.001 / opacity 0.001 penalties (code) |

Capabilities: renderer provides `render.rgb`, `gradient.screen.signed`; density requires
`gradient.screen.signed`.

## Fields
| field | shape | lr | purpose |
|---|---|---|---|
| dng_geo | (64,) | 1e-3 (never receives gradients) | cached geometry feature (paper's inference cache) |
| dng_mlp | (8,) | 1e-3 (never receives gradients) | packed colour-MLP weights (5312 floats + header; needs N ≥ 665) |
Base `sh0/shN` exist but are unused (official: SH replaced by the neural renderer).

## Mapping of equations to code
- Eq. 5 hard depth → `renderer.hard_depth` (opacity 0.95, colours ones, detached s/q, gsplat `D`).
- Eq. 6 soft depth → `renderer.soft_depth` (detached μ/s/q, sigmoid(o)).
- Eq. 7/8/9 + tolerance + smoothness → `depth.py` (`local_loss`, `global_loss`, `margin_l2`,
  `smoothness`, `depth_regularization`), patch sizes from a private `random.Random(0)`.
- Three optimizer steps per iteration → `density.after_backward(s)`: neural Adam step,
  clean/densify/near-prune of iteration s, hard (means Adam) and soft (opacity Adam, > 1000)
  passes of iteration s+1 on the camera of step s, xyz LR of s+1.
- Neural renderer → `neural.py` (vendored torch-ngp CUDA hash grid + SH encoder, JIT-built).
- DPT depth / random init / spiral centres → `tools/prepare_dng_llff.py` (sidecar `dng_depth.npz`,
  looked up by viewmat; data root from `DNG_DATA` or `--data`).

## Probes
| probe | measures | fails if |
|---|---|---|
| glnorm-formula | local/global loss vs independent per-patch loop, affine invariance, tolerance, pass weights, smoothness switch | normalisation, ε, tolerance or weights change |
| hard-depth | centre depth = τ·z on an analytic splat; grads only on means | τ ≠ 0.95, shape not frozen |
| soft-depth | centre depth = σ(o)·z; grads only on opacities | centre not frozen |
| depth-pass-step | density passes change only means / only opacities; schedule 0/1000/3000 | separate steps removed |
| neural-color | CUDA hash grid vs torch reference, SH4 vs formula, colour grad → means, bake = live, stale bake rejected | encoder/head/bake wrong |
| official-density | LR schedule, opacity-gated clone/split counts, prune, near & invisible pruning | gating/thresholds removed |
| photo-pass-losses | L1 + 0.2 D-SSIM and penalties vs independent formulas | weights / SSIM padding wrong |
| dng-renderer-gradient | screen gradients, net trained, SH unused, black bg | colours not neural |
| state-transitions (base) | field/optimizer alignment through ops | — |

## Fidelity gaps
1. **Rasterizer**: gsplat instead of ashawkey's Inria variant (radius 3.33σ vs 3σ in visibility,
   opacity-aware tile radius in rendering). Small effect.
2. **Pass ordering**: per runtime step the photometric pass comes first and the depth passes of
   the next iteration follow on the same camera (official: depth passes then photometric on the
   next camera). Same distribution; the hard pass of iteration 1 is lost. Cannot be closed
   without a runtime hook before rendering.
3. **Optimizer step vs population changes**: the runtime steps the photometric update before
   densification / near pruning / cleaning; the official code prunes first, which drops that
   iteration's photometric update of the Gaussian fields (every 100 its while densifying, every
   25 after 2000). Also the official code skips all optimizer steps at iteration 6000.
4. **View sampling**: random with replacement (runtime) vs shuffled stack (official).
5. **Eval near cull**: per-view (Gaussians within 10 of the *rendered* camera) instead of the
   union over all eval cameras. Training already prunes within 10 of the 180 spiral centres
   (which lie ≤ 2.5 units from the real cameras), so few Gaussians differ.
6. **Metrics**: runtime SSIM (valid conv) ≠ paper's (skimage or 3DGS); LPIPS-VGG from `lpips`
   (official lpipsPyTorch); float renders vs 8-bit PNGs.
7. **NaN handling**: empty selections in the penalties / tolerance mask give 0 instead of NaN
   (gradients identical).
8. **Randomness**: official README warns PSNR is unstable (random init + densification); one
   seed per scene.
9. **DPT environment**: MiDaS master (2024-08) + timm 0.6.12 + torch 2.4 vs the authors'
   2023 torch.hub snapshot / torch 1.12; same `dpt_hybrid_384` weights (sha256 501f0c75…).
10. **Final iteration** (review-1 minor): the official code skips every optimizer step at
    iteration 6000 and runs clean_views before any step at 1 and 6000; the runtime applies the
    photometric (and our neural) Adam step at 6000 before the final clean, and the depth passes of
    iteration 6000 (run from step 5999) are stepped. One extra update; the bake is the last
    operation, so it equals the evaluated model.
11. **3DGS† interpretation** (review-1 minor): `dngaussian-3dgs` keeps penalties, opacity-gated
    densification, cleaning and near pruning and removes only the hard/soft depth passes — the
    literal reading of "same hyperparameters and neural color renderer"; no official script
    exists. It still reads the sidecar for the spiral centres. State this in the benchmark report.
(Gaps 2, 4, 5 were confirmed by review-1 as minors; logic unchanged.)
