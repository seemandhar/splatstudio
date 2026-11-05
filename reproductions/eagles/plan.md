# EAGLES native plan

Method `eagles`, module `ss_eagles`, package `splatstudio-method-eagles`.

## Architecture: where the global learned decoders live
The runtime has only per-Gaussian fields (one Adam each) and `ss-eval` rebuilds just the
renderer and renders `checkpoint["params"]`; density state is saved but never handed to the
renderer at evaluation. Therefore:

- **Latents** are Gaussian `FieldSpec`s (`q_shN [16]`, `q_rot [8]`, `q_opa [1]`), so densification
  copies/splits/prunes them with gsplat ops exactly like the official `cat_tensors_to_optimizer`.
- **Decoders** (`DecoderBank`, an `nn.Module` holding the three linear decoders + their own Adam,
  eps 1e-15, per-decoder LRs) are created and initialised by the density in `initialize()` and
  registered in a process-local slot that the renderer reads (`shared.ACTIVE`). The density steps
  the decoder Adam in `after_backward` (decoder grads come from the same backward as the
  latents; the runtime's `zero_grad` does not touch them).
- While training, the renderer decodes `D(STE(q))` and renders those values; the base fields
  `quats / opacities / shN` are **mirrors**: after each decoder step the density writes the
  decoded values into them (no grad), so gsplat's split (rotation), opacity pruning and every
  probe see the decoded attributes. They receive no gradient → Adam never updates them.
- **Final step (= total_steps)**: round latents → entropy-encode (bitstream stored in density
  state) → decode the bitstream → run decoders → write decoded attributes into the base fields
  and mark the bank `baked`. From then on (and in any fresh process, where no bank exists) the
  renderer renders the base fields as-is, so the runtime's validation, the checkpoint and
  `ss-eval` all measure the decoded model. Decoder weights are also kept in density state for
  the size report. Pattern copied from the Reduced-3DGS plugin (encode inside density at final
  step, decode, replace params).

## Components
| contribution | component | factory | config (paper/code value) |
|---|---|---|---|
| C1 quantized latents + decoders + entropy coding | renderer `eagles-renderer` (decode + render), density (bank init/step/LR/encode) | `ss_eagles.renderer:create_renderer`, `ss_eagles.density:create_density` | latent dims 16/8/1, decoder std 0.0005/0.01/0.5, decoder LR 1e-4 (effective official; yaml values are dead keys), latent LR scale 1.0, div norm shN/rot |
| C2 progressive training | renderer (render at floor(s·H), K·s) + loss `eagles-photometric` (bilinear downsample target with the same scale factor) | `ss_eagles.losses:create_photometric` | resize_scale 0.3, resize_period 0.7, total_steps 30000 |
| C3 controlled densification + influence pruning | density `eagles-density` (DefaultStrategy subclass; CUDA influence kernel) | `ss_eagles.density:create_density` | refine every 125, until 18000, reset every 2500, prune every 5100 until 27000, quantile 0.15 |

Renderer capabilities = gsplat's (`render.rgb`, `render.depth.expected_z`,
`gradient.screen.signed`); density requires `gradient.screen.signed`.

## Fields
| name | shape | init | lr (initial; density overwrites) |
|---|---|---|---|
| q_shN | (16,) | 0 (official zeros) | 0.000125/‖s_shN‖ |
| q_rot | (8,) | lstsq D⁻¹(identity quat) | 0.001/‖S_rot‖_F |
| q_opa | (1,) | (logit 0.1 − b)/(S+1e-6) | 0.05 |

## Density after_backward order (official iteration order adapted to "runtime steps first")
1. decoder Adam step + zero grad; 2. mirror decoded attributes into base fields;
3. accumulate influence of this render (step < 27000, before densification as in official);
4. gsplat DefaultStrategy post-backward (refine/prune; new Gaussians get influence 0);
5. opacity reset on the opacity latent at multiples of 2500 below 18000 (moments zeroed);
6. influence prune at multiples of 5100 below 27000 (≤ 15% quantile), reset influence;
7. latent LR update from decoder norms; 8. at total_steps: encode → decode → bake.

## Probes
| probe | checks | fails if |
|---|---|---|
| decoder-ste | decoder forward = independent round(q)F·s+b / round(q)S+b; DCT matrix formula; STE gradient = straight-through | rounding/STE/DCT removed |
| latent-init-lr | inverted latents decode (unrounded) to identity quat / logit 0.1; LR = base·0.1/‖S‖ | init or LR rule wrong |
| entropy-size | `size()` bits vs independent Shannon sum on synthetic latents; range-coder roundtrip exact, length ≈ entropy | size formula / coder wrong |
| progressive-schedule | s(t) at boundaries vs cosine formula; renderer size floor(sH); loss target = F.interpolate | schedule removed |
| influence-kernel (CUDA) | CUDA Σ alpha·T vs per-pixel torch compositing on gsplat projection | kernel wrong |
| influence-prune | prunes exactly the ≤ quantile set, zeros W; grow gives new Gaussians W=0 | pruning wrong |
| opacity-reset | latent = (logit(min(σ,0.01)) − b)/(S+1e-6), moments zeroed | reset applied to mirror only |
| codec-bake (CUDA) | encode→decode→bake: bank-free render of baked params == training render | baking skipped |
| schedule | event steps (refine, reset, infl prune, encode) equal official loop | schedule wrong |

## Fidelity gaps
1. Runtime: random-with-replacement view sampling, white background, optimizer step before
   densification, xyz LR without delay, gsplat rasterizer (alpha clamp 0.999) — shared with the
   gsplat baseline; compare deltas.
2. gsplat screen-size (radii > 20 px) pruning disabled (gsplat default, same as baseline).
3. Influence accumulated with atomicAdd (official non-atomic `+=` undercounts under races).
4. Downsampled render uses K·s (exactly the interpolate pixel mapping) instead of the official
   same-FoV W'/W focal; sub-pixel difference.
5. Actual bitstream: per-latent-dimension categorical range coding (constriction) instead of
   torchac with one flattened per-attribute CDF; the paper's reported number (`size()` Shannon
   estimate) is reproduced exactly, the real bytes are reported alongside.
6. Decoders are process-local renderer state, not runtime-checkpointable parameters; the
   decoded model is baked into base fields at the final step (`ss-eval` cannot re-decode).
   Training runs with `--steps != total_steps` never encode/bake (quality then = un-baked
   mirrors, which equal the decoded model anyway).
