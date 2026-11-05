# Plan — Trim3DGS native SplatStudio port

Methods: `trimgs` (outdoor, normal weight 0.1) and `trimgs-indoor` (indoor, 0.01); module `ss_trimgs`.
One 37000-step run = 30000 steps flat 3DGS pre-training (official `train.py` with
`FlattenGaussianModel`) + 7000 steps Trim fine-tune (official `tune.py`, Mip-NeRF 360 command line).
Fine-tune iteration `t = step - 30000`.

## Components
| component | kind | factory | requires | provides | config (paper/code value) |
|---|---|---|---|---|---|
| `trim-flat` | renderer | `ss_trimgs.renderer:create_renderer` | state.gaussian3d, camera.pinhole | render.rgb, render.depth.expected_z, gradient.screen.signed, render.flat_normal, training.camera | `flat_eps 1e-8` |
| `trim-density` | density | `ss_trimgs.density:create_density` | gradient.screen.signed | – | DefaultStrategy defaults (= baseline) for t≤0; `pretrain_steps 30000`, `finetune_steps 7000`, `densify_interval 1000`, `percent_dense 0.01`, `scale_factor 1.0`, `max_screen_size 100`, `split_n 3`, `min_opacity 0.005`, `cull_expansion 2.0`, `prune_ratio 0.1`, `top_k 5`, `gamma 0.5`, `trim_interval 1000`, `final_lr_ratio 0.01`, `reset_optimizer true`, `flat_eps 1e-8` |
| `photometric` | loss | splatstudio.losses | render.rgb | – | ssim 0.2 (unchanged 3DGS) |
| `trim-normal` | loss | `ss_trimgs.losses:create_normal_loss` | render.flat_normal, training.camera | – | `weight 0.1`/`0.01`, `start_step 30000`, `depth_grad_thresh 0.05`, `mask_dilation 1`, `dilation 2`, `close_thresh 1.0`, `knn 16`, `knn_interval 500`, `densify_interval 1000` |

No new FieldSpecs. The flat axis is scales[:,0], pinned to log(flat_eps) by the density at
initialize and after every population change; the renderer ignores that column anyway and
uses the constant eps (official `get_scaling`). Gaussian normal = R(q)[:,0].

## Implementation notes
- Pre-training: gsplat `DefaultStrategy` with the runtime's opacity-reset fix (identical to the
  `baseline3dgs` runs) + flat scales. Means LR overridden to lr0·0.01^(min(s,30000)/30000)
  (3DGS 30k schedule; afterwards constant = final LR, official `--position_lr_init 1.6e-6`).
- Transition (after step 30000): clear all Adam state (official fresh optimizer); reset
  max-radii; DefaultStrategy no longer called.
- Fine-tune: track max 2D radii (info["radii"]) every step; at t%1000==0, t<7000: culling
  box (camera span ×2, z uses x span) → scale split (max scale > 0.01·extent or radii>100,
  inside box) into N=3 with s/(0.8·3), Adam moments of children zero (official cat zeros) →
  prune opacity<0.005 → contribution trimming: for every training camera run the custom CUDA
  kernel on gsplat's tile lists (sum α^(2γ)T^(2−2γ) and covered-pixel count over pixels with
  exponent ≤0, before the 1/255 skip), C_k = sum/(cnt+1e-6), running top-5, remove
  C < quantile(C,0.1) → radii reset.
- Normal loss (active step>30000): one extra gsplat pass with colors [z_cam, n_gs] and black
  background (flat scales), Sobel mask, depth back-projection with K (pixel centres +0.5),
  two-cross-product normals with dilation 2, L1 on (n+1)/2; smoothness term on visible
  Gaussians close to the depth surface vs. sign-aligned mean of 16-NN normals (scipy cKDTree,
  exact, self included), kNN refreshed when (t−1)%500==0 or N changed. NaN → 0 (attached).

## Probes
| probe | contribution | fails if |
|---|---|---|
| `contribution-kernel` (CUDA) | C1 | kernel sums ≠ brute-force per-pixel loop (independent python blend) for γ=0.5 and γ=0.25 |
| `trimming` (CUDA) | C1 | a Gaussian fully occluded behind an opaque one is not trimmed / visible high-contribution one is trimmed; top-5 averaging differs from independent topk |
| `scale-split` (CUDA) | C2 | large in-box Gaussian not split into 3 with s/2.4, out-of-box one split, small one split |
| `normal-loss` (CUDA) | C3 | depth normal of a rendered plane ≠ analytic plane normal; loss ≈0 for a plane of aligned flat Gaussians and >0 when tilted; gradient zero |
| `flat-schedule` (CPU) | C0 | scales[:,0] not pinned, rendered scale differs, means LR not 3DGS-30k then constant, Adam state not cleared at 30000, loss active before 30000 |

## Fidelity gaps
1. White background (runtime) vs black (official) — shared with baseline; small.
2. gsplat rasterizer (alpha clamp 0.999, eps2d 0.3, tile radius) vs official 0.99 — small.
3. Pre-training uses gsplat DefaultStrategy (same as our baseline) instead of 3DGS's exact densify
   (no screen-size prune) — shared with baseline, so compare deltas.
4. Random-with-replacement view sampling (runtime) vs official stack — small.
5. Densification/kNN happen after the optimizer step (runtime order) vs before — negligible.
6. Our gsplat baseline is non-flat 3DGS; the paper's 3DGS* is flat 3DGS. Δ(Trim − flat 3DGS) is
   measurable inside our run: the step-30000 model is not saved separately; optional
   `trimgs-pretrain` method (30000 steps) gives that reference.
7. Geometry (DTU Chamfer, TnT F-score) out of scope: no DTU data, no TSDF/Chamfer pipeline.

## Documentation notes (post-review)
- **Assumption, not established:** we treat our 30k flat pre-train (`trimgs --steps 30000`) as a stand-in for the paper's "3DGS*". The paper calls 3DGS* "our re-implementation" and says it fine-tunes from 3DGS pretrained models. The official scripts pre-train with `FlattenGaussianModel` (flat 3DGS). Our pre-train uses gsplat `DefaultStrategy` (as in our baseline), not the 3DGS densifier, so it may differ from the authors' 3DGS*.
- **Contribution weight deviation:** paper Eq. 5 uses α^γ·T^(1−γ). The official kernel computes α^(2γ)·T^(2−2γ) (the paper's term squared; at γ=0.5 this is α·T). We follow the code.
- **Top-5 views quirk depends on view order:** the official list only inserts a view that beats the current maximum and is initialised with K copies of the first view. It therefore holds the last K record-breaking values, not a true top-K, and the result depends on the order the training views are visited. The official order comes from Scene's seeded `random.shuffle`. Ours is a `random.Random(0)` shuffle of SplatStudio's train list, so the per-Gaussian values (and which Gaussians get trimmed) can differ from the official run.
