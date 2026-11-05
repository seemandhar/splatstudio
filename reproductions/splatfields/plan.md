# SplatFields native plan

Package `splatstudio-method-splatfields`, module `ss_splatfields`, **two** entry points:

| method | role |
|---|---|
| `splatfields-3dgs` | stage 1 = the paper's own 3DGS row (Tab. 2) and the source of SplatFields' point cloud: 3DGS + mask loss, densification over the whole run, no opacity reset, official stage-1 LRs, 40k steps, visual-hull init |
| `splatfields` | stage 2 = SplatFields3D: networks predict all splat attributes from anchors initialised from the stage-1 cloud; baked into plain Gaussians at the last step |

Copied/adapted (with attribution in comments) from the official repo (MIT): `GeneralMLP`,
NeRF embedder, `SplatFields.forward` (static branch), `VarTriPlaneEncoder` + `Tensorial2D`,
`TimeVAEDecoder` (re-written without diffusers/mmgen/mmcv: ResnetBlock2D, Attention with
`_from_deprecated_attn_block`, Upsample2D(use_conv), kaiming_init/constant_init/zero-init).
Precedent patterns (copied, not imported): eagles' process-local `ACTIVE` slot + bake at
`total_steps` (decoder bank owned by the density).

## Components

| component | kind | factory | requires / provides | config (paper/code value) |
|---|---|---|---|---|
| `splatfields-renderer` | renderer | `ss_splatfields.renderer:create_renderer` | state.gaussian3d, camera.pinhole / render.rgb, render.depth.expected_z, render.alpha, gradient.screen.signed | — |
| `splatfields-density` | density | `ss_splatfields.density:create_density` | gradient.screen.signed | total_steps 40000, lr_init 8e-4, lr_final 1.6e-6, lr_max_steps 40000, adam_eps 1e-15, net_seed 0, base {refine_start_iter 500, refine_stop_iter 40000 (official saves before the 40000 densify), refine_every 100, reset_every 3000, grow_grad2d 2e-4, grow_scale3d 0.01, prune_scale3d 0.1, prune_opa 0.005}, net {deform 6×128 skip3 PE6, rgb 6×128 skip3 PE6, scale 4×64 skip2 PE4, opacity 4×64 skip2 PE3, rotation 3×64 PE3, triplane in 8/out 16/noise 20, deform_weight 1.0, use_triplane true} |
| `photometric` | loss | splatstudio.losses:create_photometric | render.rgb | ssim_weight 0.2 |
| `mask` | loss | `ss_splatfields.losses:create_mask` | render.alpha | weight 0.1, sidecar masks.npz |
| `splat-norm` | loss | `ss_splatfields.losses:create_norm` | render.rgb (uses outputs["sf_means"]) | weight 0.01 |
| (stage 1) `sf3dgs-renderer` | renderer | `ss_splatfields.renderer:create_plain_renderer` | as gsplat + render.alpha | — |
| (stage 1) `sf3dgs-density` | density | `ss_splatfields.density:create_stage1_density` | gradient.screen.signed | base as above, means_lr_init 8e-4, means_lr_final 8e-6, means_lr_max_steps 30000 |

No extra per-Gaussian fields. Base fields in stage 2: `means` = anchors (no gradient; the
renderer detaches them, official `get_xyz.detach()`), `scales` = base log scale (3-NN init,
detached), `quats`/`opacities` constant (identity / logit 0.1) so gsplat's split and opacity
prune behave like the official stored attributes; `sh0`/`shN` unused until the bake.

## Mechanism → code
1. Renderer (training, network active, not baked): `net(means.detach(), exp(scales).detach())`
   → p̂ = x + D(PE6(x) ⊕ f); s = S(PE4(p̂) ⊕ f) + exp(base); α = sigmoid(lrelu(·));
   q = normalize(lrelu(·)); c = sigmoid(lrelu(·)) (view-independent, code default) →
   `gsplat.rasterization(colors=[N,3], sh_degree=None)`; white bg via `rgb + (1-alpha)`;
   `outputs["sf_means"] = p̂`, `outputs["viewmat"]`.
2. Density: owns `SplatFieldsNet` + Adam(eps 1e-15) and registers it in `ACTIVE`.
   after_backward: net Adam step + zero_grad; LR(step) = exp-lerp 8e-4→1.6e-6 over 40k
   (official `update_learning_rate(iteration)` after the step); DefaultStrategy on the anchors
   (screen-space grads of the *network* means, as official); at `total_steps` bake p̂, log|s|,
   q, logit(α), (c−0.5)/C0, shN = 0 into the params and mark baked.
3. Mask loss: L1(clamp(alpha), GT alpha) with GT alpha from `<data>/masks.npz`, looked up by
   the frame viewmat; data root from `SPLATFIELDS_DATA` or ss-train's `--data` argv.
4. Norm loss: λ3 · mean ||p̂||₂.
5. Stage 1: plain gsplat render + photometric + mask; DefaultStrategy (no reset), means LR
   8e-4·(0.01)^(min(step,30000)/30000) overriding the runtime schedule each step.

## Probes
| probe | measures; fails if |
|---|---|
| `triplane-generator` | decoder output [3,16,160,160], fixed noise buffers, zero-init residual/attn-out, attention vs independent softmax formula, grid_sample vs independent bilinear formula; fails if a plane/CNN is missing or sampling differs |
| `splat-fields` | GeneralMLP vs independent re-implementation (skip concat, act on last layer), attribute formulas (residual deform, scale = MLP + base, normalize, sigmoid), gradients reach the networks and not the base fields |
| `renderer-uses-fields` (CUDA) | render changes when network weights change and does not depend on base sh0/opacities/quats; fails if the renderer falls back to base fields |
| `bake` (CUDA) | network render == bank-free render of the baked params (max abs err < 1e-4) |
| `norm-loss` | value = mean ‖p̂‖, gradient on the deform MLP |
| `mask-loss` | L1 vs independent formula with a sidecar lookup by viewmat; raises without masks |
| `schedule` | network LR at 0/20000/40000, refine 500/100/45000, no opacity reset, stage-1 means LR |

## Data / protocol (tools in `reproductions/splatfields/tools/`)
- `select_views.py` (scikit-learn 1.0.2 venv `~/models/venv-sklearn102`, the newest release for
  the official py3.7 env; defaults KMeans(N, random_state=0)) -> `views.json`;
  `prepare_blender_fewview.py --views views.json --n-views N`: RGB images = trunc(255·(rgb·a+1−a)), world scale
  2/model_scale, explicit splits train=N, val=8 of transforms_val, test=200;
  points.npz = visual-hull grid samples (official projection incl. clip-z divide), 100k,
  colours U/255; masks.npz = train viewmats + alpha.
- `make_stage2.py --run <stage-1 run> --out <dir>`: hard-links images/json/masks, writes
  points.npz = stage-1 means (from the checkpoint) filtered by the hull, ≤100k random.
- Datasets: `blender-fewview6`, `blender-fewview6-sf` (and `-12`), scenes chair, hotdog, ficus.

## Fidelity gaps
1. gsplat rasterizer vs ingra14m diff-gaussian-rasterization (same math; 2D blur 0.3, near plane).
2. Float-render metrics vs 8-bit PNG metrics in the official eval (~0.0x dB).
3. KMeans views from scikit-learn 1.0.2 defaults; the authors' exact version is unknown.
4. Runtime steps all per-Gaussian optimizers before density.after_backward; official densifies
   then steps. Irrelevant in stage 2 (anchors get no gradient); stage 1 order differs (as in
   all native ports).
5. Runtime SH schedule (degree=(step−1)//1000) vs official step//1000 (one-step offset).
6. Mask lookup through sidecar + argv (`--data`) — infrastructure workaround, no numerical effect.
7. Last step: official skips the optimizer step at the final iteration; runtime steps.
8. Stage-1 stochasticity (hull subsample, densification) differs from the authors' run; their
   exact 3DGS clouds are not released.
9. 4D branch, DTU protocol: not implemented.
10. Runtime photometric SSIM uses valid convolution; official 3DGS SSIM zero-pads (same conv).
11. No opacity reset via NoResetStrategy (gsplat step_post_backward without reset_opa);
    reset_every 3000 kept only to gate the world-size prune (official iteration > 3000).
12. Both densities raise at step 1 if ss-train --steps != total_steps (40000).
