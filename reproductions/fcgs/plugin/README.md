# splatstudio-method-fcgs

Native SplatStudio port of **FCGS — Fast Feedforward 3D Gaussian Splatting Compression**
(Chen et al., ICLR 2025, arXiv 2410.08017, code github.com/YihangChen-ee/FCGS @ 31e59a46).
Paper fidelity unverified.

## What it does
FCGS compresses a finished vanilla 3DGS in one feed-forward pass with a **pretrained** codec
(no per-scene optimisation). Methods:

| method | λ | paper label |
|---|---|---|
| `fcgs` | 1e-4 | Ours-highrate |
| `fcgs-lowrate` | 16e-4 | Ours-lowrate |

A run trains the unchanged gsplat 3DGS recipe for 30 K steps; right after the last optimizer
step the `fcgs-density` component encodes all Gaussians to real bitstreams with the official
FCGS network/coders, decodes them from disk and replaces the parameters, so the checkpoint,
`validation.json` and `ss-eval` measure the **decoded** model. The size report is in the
checkpoint (`report_size.py`) and in the training log (`{"fcgs": "encode", ...}` line).

Mechanisms (all official code, vendored): MEM mask (Eq. 3), autoencoder / learned-step
quantisation, inter-Gaussian grid contexts (Eqs. 4–5, CUDA grid creator/encoder),
intra-Gaussian channel contexts (Eq. 6), GMM + hyperprior + factorised prior (Eq. 7),
CUDA arithmetic coding, Bernoulli mask coding, 16-bit voxelised positions.

## Requirements
- Weights: `~/models/fcgs/checkpoint_{0.0001,...,0.0016}.pkl` (official release, copied from
  the repository's `checkpoints/`; SHA256 checked by the `official-weights` probe). Override
  with `$FCGS_WEIGHTS` or density config `weights_dir`.
- CUDA extensions are JIT-built on first use (several minutes; cached).
- Positions: official G-PCC needs MPEG TMC13 `tmc3` (`$FCGS_TMC3` or PATH). Without it,
  `xyz_coder: auto` uses a lossless substitute (Morton octree + LZMA): decoded model
  identical, xyz bitstream larger than GPCC. `xyz_coder_used` is recorded.

## Size
`size_MB` = Σ bits returned by the official `compress` / (8·1024·1024), exactly what the
official encoder prints and the paper reports ("MB" = MiB; Mip-NeRF 360 high-rate mean of the
official per-scene CSV bytes 67.16e6 B = 64.05 "MB" in the paper's table).

## Commands
```bash
T=/home/sejain/splatify/tools/ssenv
$T ss-train fcgs --data ~/datasets/mipnerf360/native/garden --output runs/x --steps 30000 \
   --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
$T ss-eval --run runs/x --split test
$T python report_size.py runs/x
# compress an existing gsplat run (no training), any lambda:
$T python apply_fcgs.py --run runs/baseline3dgs-mipnerf360-30000/gsplat/mipnerf360-garden \
   --lmd 1e-4 --lmd 16e-4 --out garden.json
```
The run length must equal `total_steps` (checked at step 1), otherwise the encode never runs.

## Fidelity gaps
See `../plan.md`: GPCC substitute (size only), gsplat-trained base 3DGS, SplatStudio metrics
(white background, no 8-bit rounding), codec training not reproduced (released weights).

## Licence
Vendored FCGS code: Apache-2.0 with a non-commercial clause (`ss_fcgs/csrc/LICENSE-FCGS.md`);
arithmetic coder derived from torchac. Research use only.
