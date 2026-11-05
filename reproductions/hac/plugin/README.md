# splatstudio-method-hac

This is a native SplatStudio port of **HAC: Hash-grid Assisted Context for 3D Gaussian Splatting
Compression** (Chen et al., ECCV 2024, arXiv 2403.14530). The official code is at
https://github.com/YihangChen-ee/HAC (commit 110dcf0a); this port was read from it and its CUDA
sources are vendored here. Paper fidelity is unverified.

## Mechanism and how it maps to components

- **Scaffold-GS anchors as the native population.**
  - Per-anchor fields: `means` (anchor xyz, lr 0), `hac_feat [50]`, `hac_offset [10,3]`,
    `hac_scaling [6]` and `hac_mask [10,1]`.
  - `hac-renderer` decodes K=10 neural Gaussians per visible anchor on every render. It uses the
    opacity/cov/color MLPs with input `[f, view dir, dist]` and rasterizes them with gsplat.
- **Global learned state.** `model.HACBank` holds the MLPs, the binary mixed 3D+2D hash grid
  (official CUDA `gridencoder`) and the context MLP, and has its own Adam.
  - `hac-density` creates and steps it.
  - The renderer reads it through `model.ACTIVE`, following the EAGLES DecoderBank pattern.
- **Hash-grid context and adaptive quantization (Eq. 2–7).**
  - Steps 3001–10000: uniform-noise quantization with a fixed Q0.
  - Step 10000: the anchor bounds are updated.
  - From step 10001: Q = Q0(1+tanh r) and the Gaussian entropy bits are estimated on a 5% anchor
    sample. `hac-rate` adds λ_e(bits + L_hash/denom) + 5e-4·mean σ(mask).
- **Anchor growing and pruning.** These follow Scaffold-GS: every 100 steps in (1500, 15000),
  skipping steps 3000–3999.
- **Bitstream.** At `total_steps`, `hac-density` encodes the model with the official CUDA
  arithmetic coder and decodes it again.
  - Encoded: anchors at 16 bit; feat/scaling/offsets under the predicted Gaussians; hash and masks
    as Bernoulli; MLPs at fp32.
  - The decoded attributes are baked into the fields.
  - The render MLPs are packed into the `hac_blob [8]` field, so `ss-eval` (a fresh process
    holding only the checkpoint's params) renders the decoded model.
  - Sizes are stored in `checkpoint["density_state"]["hac"]["encoded"]`. Read them with
    `python report_size.py <run>`.

## Commands

```bash
tools/ssenv pip install -e reproductions/hac/plugin
tools/ssenv ss-verify hac --device cuda
# full protocol (30k steps; bakes at step 30000)
tools/ssenv ss-train hac --data ~/datasets/mipnerf360/native/garden --output runs/x --steps 30000 \
  --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run runs/x --split test
# short diagnostic with every phase compressed into N steps
tools/ssenv python smoke_schedule.py <data> <output> 2000
```

`--steps` must equal `total_steps` (30000). A shorter run never bakes, and the final validation
then refuses to render an un-baked state.

## Fidelity gaps

See `../plan.md`. In brief:
- resolution 1.6K (official) vs images_4/images_2 (here);
- gsplat vs the Scaffold rasterizer;
- white vs black background;
- the runtime's view sampling and SSIM;
- the update order differs (the official code densifies before the optimizer step);
- global MLP weights travel to ss-eval in a per-anchor field;
- the voxel size is mip360's (0.001).
