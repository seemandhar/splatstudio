# OMG native plan

- Methods:
  - `omg`: OMG-XS, τ 0.96, outdoor metric.
  - `omg-indoor`: OMG-XS, indoor metric.
  - `omg-m`: τ 0.99.
- Module: `ss_omg`.
- Base: our matched Mini-Splatting port (`reproductions/mini-splatting`, freeze a6428eb7). Its
  density, core, kernels, CUDA sources and probes are copied into the plugin (`ms_*`). Copying,
  rather than importing, keeps the freeze independent.
- Official OMG code: it has no licence, so it is read and re-implemented (no verbatim copy).
- Reused from our Compact-3DGS port (same group): the global-bank pattern, the dahuffman port,
  the closed-form ChainedScheduler and the encode → decode → replace bake.

## Components
| kind | factory | role |
|---|---|---|
| renderer | `ss_omg.renderer:create_renderer` | gsplat. On base fields (MS phase, decoded model, fresh process); on the OMG attributes from the live bank (15k → bake), with SVQ geometry/features from 29k and the official SH degree. Forwards the target image (MS depth reinit). |
| density | `ss_omg.density:create_density` | `OMGDensity(MiniSplattingDensity)`. Handles the MS events, the LD prune at 20k (replacing the MS 1 % prune), construct_net at 15k, SVQ at 28999 (the state used by render 29000), the MLP and codebook Adam steps, and encode/decode/bake at 30k. |
| loss | photometric (L1 + 0.2 D-SSIM) | unchanged |

## Fields
`omg_static` (T) [N,3] and `omg_view` (V) [N,3], lr 0.0025 each.
- They are zero until construct_net.
- At construct_net T = sh0 and V = 0.
- From 15k the base `opacities`, `sh0` and `shN` get no gradient. They stay aligned and are
  overwritten by the decoded model at 30k.

## Global state (`model.OMGBank`, registered in `model.ACTIVE`)
- MLPs: MLP_s, MLP_t (dc), MLP_o and MLP_v in tcnn layouts (no bias, padded), fp32. The net Adam
  uses lr 0.01 with warm-up and step decay in closed form.
- SVQ: codebooks are Parameters with Adam at lr 1e-8; the indices are frozen.
- The decoded model is a plain 3DGS SH3 state. ss-eval (a fresh process with no bank) renders it
  with the stock path, so no blob field is needed.

## Probes
| probe | what fails it |
|---|---|
| omg-ld-scoring | wrong LD formula, neighbours, λ, τ-CDF; LD removed (kept set equals the plain CDF); optimiser not reset |
| omg-space-feature | wrong frequency layout, contraction, MLP layout/activations, or feature routing |
| omg-train-render | live render differs from an independent rasterisation; wrong gradient flow; wrong SH schedule |
| omg-net-schedule | net LR differs from a torch scheduler replay; wrong event steps |
| omg-svq | wrong layout, non-nearest indices, per-Gaussian geometry still used, indices change |
| omg-codec | lossy Huffman/fp16/G-PCC, sub-optimal Huffman, or decoded model differs from the live SVQ model |
| ms-* (6) | Mini-Splatting mechanisms (copied probes) |

## Benchmark choice
- Variant: OMG-XS (headline; τ 0.96, the official default).
- Scenes (supp. Table tab:perscene360):
  - bicycle (outdoor, `omg`): 24.95 dB, 4.61 MB, 480,772 G;
  - counter (indoor, `omg-indoor`): 28.40 dB, 2.95 MB, 310,056 G.
- Our MS port is matched on both: bicycle −0.20 and counter +0.09 dB.
- Report the PSNR of the decoded model (the bake at 30k feeds both ss-eval and validation.json),
  the official-formula size and the actual container size.

## Fidelity gaps
1. **Runtime.** White background, gsplat rasteriser numerics and random-with-replacement view
   sampling, all inherited from the MS port. The MS port is still within ±0.2 dB.
2. **fp32 MLPs instead of tcnn fp16.**
   - Training precision differs.
   - The high-frequency (2^15 π) features use torch's exact sin, while tcnn uses `__sinf`.
   - The weights are rounded to fp16 only at encode.
   - Expected effect: small, and either sign.
3. **K-means.** We use torch k-means++ plus Lloyd instead of cuML's scalable-k-means++. Cluster
   quality is the same kind; the effect is within noise.
4. **G-PCC.** The container holds the bypass-flag-0 stream because TMC13 v14 cannot decode
   official-flag streams. Both sizes are reported, and the geometry is identical (lossless).
5. **Container.** Our container (canonical Huffman tables + JSON + LZMA) replaces the official
   LZMA pickle of dahuffman tables. The paper size is reported by the official formula
   (write_storage) and also as the actual container size.
6. **Update order.** The per-Gaussian Adam steps before the density hook, so the final step 30000
   moves positions once (lr ~1e-6·extent). The MS port has the same order at the 15k/20k
   replacements.
7. **LD neighbours.** We follow the official code (the permutation applied twice), not the
   paper's text; `ld_neighbors: morton` is available.
8. **Absolute step constants.** Runs must be 30k steps; a different run length is refused at
   step 1.
