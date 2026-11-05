# HAC native plan

Method `hac`, module `ss_hac`, plugin `reproductions/hac/plugin/`.

## Feasibility: can HAC be expressed with components only?

Yes, with one explicit workaround. Below, each item the caller asked about.

1. **Population = anchors.** The native `ParameterDict` holds anchors as its "Gaussians":
   - `means` is the anchor xyz (lr 0, as in the official `position_lr_init=0`).
   - Per-anchor `FieldSpec`s: `hac_feat [50]`, `hac_offset [10,3]`, `hac_scaling [6]`,
     `hac_mask [10,1]`.
   - The base fields `scales/quats/opacities/sh0/shN` are carried but unused. They get no gradient, so
     Adam skips them.
   - `hac-density.initialize` voxelizes the runtime's SfM initialisation (voxel 0.001) and swaps the population
     and optimizers in place. The runtime reads `optimizers[...]` through the dict, so this works.
   - Field lrs and schedules for offset/mask decay are set in the density component. The runtime overrides only the
     means lr (`initial_lr * 0.01^(t/T)`). Because the density sets the means lr to 0 before `initial_lr` is captured,
     the means lr stays 0.
2. **Renderer-owned MLPs + hash grid.** `model.HACBank` (nn.Module with its own Adam) holds:
   - the Scaffold opacity/cov/color MLPs;
   - the mixed 3D+2D binary hash grid (official CUDA `gridencoder`, JIT);
   - the context MLP.

   It follows the EAGLES `DecoderBank` pattern: the density component creates it, `model.ACTIVE` shares it with the
   renderer, the density steps it, and `bank.step` carries the step number into the next render. The renderer decodes
   neural Gaussians per view on every render and rasterizes them with gsplat.
3. **Anchor growing / pruning.** `hac-density` ports `training_statis` / `anchor_growing` / `adjust_anchor`.
   It appends and removes rows through `gsplat.strategy.ops._update_param_with_optimizer`, so fields and Adam moments
   stay aligned.
4. **Learned modules in ss-eval.** `density.state` *is* persisted: the runtime saves `density.state` into
   `checkpoint.pt`. But `runtime.evaluate()` never hands it back. It rebuilds the components with `build(method)` and
   calls `renderer.render(checkpoint["params"], ...)`, so a renderer only ever sees the checkpoint's params. Restoring
   from `density.state` would mean sniffing `sys.argv` (`--run`) or the call stack. That is fragile and CLI-only, so
   it was rejected.

   **The public channel that does reach ss-eval is the params dict.** At the final step the density runs encode →
   decode (the real arithmetic coder), bakes the decoded anchors into the fields, and packs the three render MLPs
   (13,860 floats) into a Gaussian-owned field `hac_blob [N,8]` (header + flat weights in the first rows). This needs
   ≥1,733 anchors and raises otherwise. In ss-eval the renderer finds no live bank and decodes the MLPs from
   `params["hac_blob"]`.
   - No change to SplatStudio core.
   - Decoding at eval is lossless with respect to the trained weights (probe `bake-eval`, exact equality after a
     `torch.save`/`load(weights_only=True)` round trip).
   - The hash grid and context MLP are not needed after decoding, because the decoded anchor attributes are baked
     (official `decoded_version=True`).

   This is a representation abuse (global state inside a per-Gaussian field). It is documented here and in the
   report as a fidelity note, not hidden. The clean runtime extension would be "component state persisted in the
   checkpoint and handed to components at evaluate()". A design note for it is below.

`feasible_natively: true` (with the blob workaround).

## Components

| component | kind | factory | requires / provides | config (paper/code values) |
|---|---|---|---|---|
| hac-renderer | renderer | ss_hac.renderer:create_renderer | state.gaussian3d, camera.pinhole → render.rgb, render.depth.expected_z, gradient.screen.signed | feat_dim 50, n_offsets 10 |
| hac-density | density | ss_hac.density:create_density | gradient.screen.signed | `density.DEFAULTS` (arguments/__init__.py, run_shell_mip360.py: voxel 0.001, update_init_factor 16, start_stat 500, update_from 1500, interval 100, until 15000, pause [3000,4000), min_opacity 0.005, success 0.8, grad thr 2e-4; Q0 1/0.001/0.2; noise >3000, entropy >10000, 5% sample; all lr schedules; hash grid 12×3D levels 18..514 2^13, 4×2D levels 130..1026 2^15, 4 feats) |
| photometric | loss | splatstudio.losses:create_photometric | render.rgb | ssim weight 0.2 (= lambda_dssim) |
| hac-scaling-reg | loss | ss_hac.losses:create_scaling_reg | render.rgb | 0.01·mean(prod scale) |
| hac-rate | loss | ss_hac.losses:create_rate | render.rgb | λ_e 0.004, λ_m 5e-4 |

## Fields

| field | shape | lr (set by density) | init |
|---|---|---|---|
| hac_feat | [50] | 0.0075 | 0 |
| hac_offset | [10,3] | 0.01·s → 1e-4·s (exp) | 0 |
| hac_scaling | [6] | 0.007 | log sqrt(3-NN mean d²) |
| hac_mask | [10,1] | 0.01·s → 1e-4·s | 1 |
| hac_blob | [8] | never stepped | 0 (decoder weights after bake) |

## Probes

| probe | measures; fails if… |
|---|---|
| hash-grid | the CUDA grid differs from an independent torch hash/dense-index multilinear reference (values and gradients), or the embeddings are not binary |
| entropy-model | bits ≠ −log2(Φ(x+q/2)−Φ(x−q/2)) computed with erf; Q is outside (0, 2Q0); STE rounding or binary entropy is wrong |
| codec-roundtrip | the arithmetic coder is not lossless, or its size is >2% from the entropy |
| neural-gaussians | the decode differs from a per-anchor loop reference, or the decode is not view dependent |
| visibility-filter | the anchor set differs from the fork's rule on a constructed case (off-screen anchor, off-screen-but-large anchor, behind camera); fails with the z-only filter |
| anchor-density | growing does not place an anchor in the offset's voxel, the low-opacity anchor is not pruned, or anything is misaligned |
| training-phases | phase at exact boundaries 3000/3001/10000/10001 (deterministic / noise / bound update / rate), 5% sample fraction, the rate loss value for fixed bits (λ_e(b+hash/denom)+λ_m mean σ), gradient reach, lr schedules |
| bake-eval | a fresh, bank-free render of the checkpoint ≠ the baked render, the blob ≠ the trained weights, or an un-baked state renders |

## Fidelity gaps

1. **Resolution.** The official code uses Mip-NeRF360 at 1.6K width (PIL resize of full-res images to width 1600).
   Closed for the benchmark by the coordinator's `mipnerf360-1600` conversion (garden/bicycle/counter); the plugin
   settings (voxel 0.001, update_init_factor 16) are the official mip360 ones and need no change there. On the
   images_4/images_2 data absolute PSNR/size differ. That the paper's 3DGS/Scaffold-GS rows share HAC's protocol is
   an **unverified assumption**; judge fidelity against HAC's own rows and our baselines on identical images.
2. **Rasterizer.** gsplat (classic, eps2d 0.3, near plane 0.2) is used instead of the Scaffold
   diff-gaussian-rasterization fork. The anchor prefilter re-implements the fork's `visible_filter`
   (filter_preprocessCUDA: z>0.2, EWA 2D covariance of exp(scaling[:3]) with identity rotation, +0.3 low-pass,
   3σ radius, non-empty 16-px tile rect); it is used for the render and for the growing/pruning statistics.
3. **Background.** White (runtime convention) vs black in the official mip360 runs. This matters only in uncovered
   pixels.
4. **View sampling.** Random with replacement vs shuffled stack. SSIM is valid-conv vs same-padding (runtime).
5. **Update order.** The runtime steps field Adam before `after_backward`. The official code densifies before
   `optimizer.step()`, and on the last iteration it skips the step (we skip only the bank step).
6. **Global-state transport.** The `hac_blob` field (above) holds global weights as a Gaussian-owned field.
7. **Per-dataset voxel size.** The method registers mip360's voxel 0.001. tandt (0.01) and db (0.005) need a config
   variant.
8. **Codec.** Official CUDA arithmetic coder, in memory (same bit formula as the .b files). The paper used torchac.
9. **Run guards.** `hac-density` refuses to train when the SfM cloud was subsampled (`--num-gaussians` ≤ SfM size under ss-train) or when the scene scale is not the 3DGS camera extent (`--recipe 3dgs`).
10. **Total-steps dependency.** The bake happens at `total_steps` (30000). A run with fewer `--steps` never bakes and
   its validation raises. For short runs, use `smoke_schedule.py` (compressed schedule).

## Runtime extension design note (what would remove gap 6)

"Component state checkpoint protocol":
- Components may implement `state_dict()` / `load_state_dict(d)`.
- `runtime.train` saves `{"components": {name: obj.state_dict()}}` in `checkpoint.pt`.
- `runtime.evaluate` calls `load_state_dict` on the freshly built components before `measure`.
- It must be weights_only-safe (tensors / primitives) and included in `specification_id` checks.
- With it, HAC would store the bank `state_dict` (MLPs, binary hash grid, context MLP, bounds) and could even decode
  the bitstream inside ss-eval. No per-Gaussian field abuse would be needed.
