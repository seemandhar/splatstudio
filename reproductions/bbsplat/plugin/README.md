# splatstudio-method-bbsplat

Native SplatStudio port of **BillBoard Splatting (BBSplat): Learnable Textured Primitives for
Novel View Synthesis** (Svitov et al., arXiv [2411.08508](https://arxiv.org/abs/2411.08508)),
following the official code [david-svitov/BBSplat](https://github.com/david-svitov/BBSplat)
@ cff8d6a. The official CUDA rasterizer
([diff-bbsplat-rasterization](https://github.com/david-svitov/diff-bbsplat-rasterization) @ 2dbc336)
is vendored unmodified in `ss_bbsplat/csrc/` (BSD-3 + Inria GS licence) and JIT-built at first
use. Paper fidelity unverified.

## Mechanism
* Planar billboards {mu, 2D scale, quaternion, SH} with per-billboard 16x16 alpha map and RGB
  texture (fields `bb_alpha` [N,16,16], `bb_color` [N,3,16,16], sigmoid-activated); no scalar
  opacity (the runtime's `opacities` field is unused, lr 0).
* Colour = texture + SH colour at the centre direction; alpha = min(0.99, bilinear alpha map).
* MCMC densification: relocate dead (mean alpha <= 0.005), add 5 % per 100 its to `cap_max`,
  rescale only alpha 1-(1-a)^(1/(n+1)); no position noise (zero in the official code).
* Losses: 0.8 L1 + 0.2 D-SSIM; impact-weighted texture regulariser (sigma 100, lambdas 1e-4);
  0.01 mean-alpha regulariser; 2DGS normal/distortion (weights 0, as the published scripts).
* Init: FPS to `max_read_points` + 10K Fibonacci sky points; textures frozen until 500; all but
  SH rest frozen after 30000; 32000 iterations.
* Evaluation renders 8-bit quantised textures (official save/load round trip).

## Methods
| name | budget | use |
|---|---|---|
| `bbsplat` | cap 60K = 50K SfM + 10K sky | Mip-NeRF 360, paper Tab. 1 |
| `bbsplat-tnt` | cap 30K = 20K SfM + 10K sky | Tanks&Temples, paper Tab. 1 |
| `bbsplat-300k` | cap 300K = 290K + 10K | official `scripts/train_all.sh` (Tab. 2 regime) |

## Commands
```bash
tools/ssenv pip install -e reproductions/bbsplat/plugin
tools/ssenv ss-verify bbsplat --device cuda
tools/ssenv ss-train bbsplat --data ~/datasets/mipnerf360/native/bicycle --output runs/x \
  --steps 32000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run runs/x --split test
```

## Fidelity gaps
See `reproductions/bbsplat/plan.md` (Adam step on relocation iterations, densify after the
optimizer step, runtime view sampling/SSIM, unpublished Tab. 1 init split, depth-normal reg off,
+-inf logits stored as +-1e4, rendered images not 8-bit quantised before metrics).
