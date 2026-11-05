# splatstudio-method-eagles

Native SplatStudio port of **EAGLES: Efficient Accelerated 3D Gaussians with Lightweight
EncodingS** (Girish, Gupta, Shrivastava, ECCV 2024, arXiv:2312.04564), written against the paper
and the official code (https://github.com/Sharath-girish/efficientgaussian @ b9b50db,
`configs/efficient-3dgs.yaml`). Paper fidelity unverified.

## Mechanism → code
| contribution | code |
|---|---|
| C1 quantized latents (rotation 8, opacity 1, SH-rest 16 dims) decoded by learned linear decoders with straight-through rounding (Eq. 1); latent init by decoder inversion (supp. Eq. 1); latent LR / ‖S‖ | `decoders.py` (`DecoderBank`), `renderer.py` (decode while training), `density.py` (decoder Adam, LRs, opacity reset in latent space) |
| C1 entropy-coded size + decoded evaluation | `codec.py` (`paper_size_bits` = official `size()`; constriction range coder), `density.py:_encode` (final step: round → encode → decode → bake) |
| C2 progressive training (cosine 0.3 → 1 over 70 %) | `density.resize_scale`, `renderer.scaled_camera`, `losses.py` (bilinear target downsample) |
| C3 densification every 125 to 18000, influence pruning (15 % lowest Σ αT every 5100 to 27000) | `density.py` (`EaglesStrategy`, `_influence_prune`), `csrc/influence.cu` (adapted from the Reduced-3DGS plugin's transmittance kernel) |

## Global learned state
The runtime has no parameter type for global modules and `ss-eval` only renders checkpoint
params. The density owns the decoders (`DecoderBank`: three `nn.Module` decoders + their own
Adam, stepped in `after_backward`) and registers them in `decoders.ACTIVE`, which the renderer
reads. Base fields `quats/opacities/shN` mirror the decoded values every step (so gsplat's split
and opacity pruning see them). At step `total_steps` (30000) the latents are rounded,
entropy-coded, decoded from the bitstream and pushed through the decoders; the decoded
attributes are **baked** into the base fields and the bank is marked baked, so the runtime's
validation, the checkpoint and `ss-eval` all measure the decoded model. Decoder weights, the
bitstream and the size report are stored in `density_state["eagles"]`.

Training with `--steps` ≠ `total_steps` never encodes (mirrors already equal the decoded model,
but no size report is produced).

## Config (density component; defaults = official config)
total_steps 30000; base DefaultStrategy: grow_grad2d 2e-4, refine 500..18000 every 125,
reset_every 2500; opacity reset every 2500 below 18000 to 0.01; influence prune every 5100
below 27000 at quantile 0.15; resize_scale 0.3, resize_period 0.7; decoders: SH (dft, std 5e-4),
rotation (std 0.01), opacity (std 0.5), all decoder LR 1e-4; latent LR = 3DGS LR (scaling 1.0),
divided by the decoder-scale norm for SH and rotation. The yaml's `*_ldecs_lr` / `*_lr_scaling`
entries sit under `quantize_params` and never reach the official optimizer (only `opt_params`
feeds OptimizationParams), so the effective official values are the defaults = supp. Table 1.

## Commands
```bash
tools/ssenv ss-train eagles --data ~/datasets/mipnerf360/native/garden --output runs/x \
  --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run runs/x --split test
tools/ssenv python reproductions/eagles/plugin/report_size.py runs/x   # paper_size_MB
tools/ssenv python reproductions/eagles/plugin/smoke_schedule.py <data> <out>  # all events in 2k steps
```
The first run JIT-builds the CUDA influence kernel (cached in `~/.cache/torch_extensions`).

## Fidelity gaps
- Runtime: random-with-replacement views, white background, optimizer step before densification,
  xyz LR without delay, gsplat rasterizer (shared with the gsplat baseline).
- No screen-size (radii > 20 px) pruning (gsplat default; baseline same).
- Influence accumulated with atomicAdd (official CUDA uses a racy non-atomic `+=`).
- Downsampled render uses K·s (matches `F.interpolate`'s pixel mapping) instead of the official
  same-FoV W'/W focal (sub-pixel difference).
- Real bitstream: per-dimension categorical range coding (constriction) vs torchac with a
  flattened per-attribute CDF; the paper's `size()` number is reproduced exactly.
- Decoders cannot be runtime-checkpointed parameters; the decoded model is baked at the end.
