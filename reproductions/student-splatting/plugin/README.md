# splatstudio-method-student-splatting

Native SplatStudio port of **3D Student Splatting and Scooping** (Zhu, Yue, He, Wang; CVPR 2025,
arXiv [2503.10148](https://arxiv.org/abs/2503.10148)), from the official code
<https://github.com/realcrane/3D-student-splating-and-scooping> (GPL-2.0; rasterizer under the
Inria Gaussian-Splatting research licence). Paper fidelity unverified.

## Mechanism → code
| contribution | code |
|---|---|
| Student's t kernel, learnable nu = hardtanh(raw, 1, 1e4), no low-pass, critical-value radius | `projection.py`, `csrc/t_raster.cu` (official renderCUDA fwd/bwd incl. straight-through clamp and partial dnu) |
| Scooping: signed opacity tanh | `renderer.py`, kernel signed alpha ±0.99 |
| SGHMC for means (Adam moments from the runtime's lr-0 Adam), Adam for the rest | `density.py:sghmc_update` (official `utils/sghmc.py`) |
| Recycling (≤5 %) + 5 % growth with the t relocation formula | `density.py:recycle/add_components/relocation_student_t` (official `nt_model.py`, `utils.cu`) |
| Regularisers 0.01 mean\|o\| + 0.01 mean scale | `losses.py` |

## Methods (official `configs/<scene>.json`)
`student-splatting` (3M, C_b 5e5, B 7000), `-bicycle` (3M, 5e8, 15000; 30k steps, -r 4),
`-room` (1.5M, 5e5, 7000; 40k steps, -r 2), `-truck` (2.6M, 5e5, 7000; 40k steps). C = 120, nu0 = 100.

## Fidelity gaps
Official-code behaviours kept: partial dν (drops −½ln(q)·G; may push ν to thousands), verbatim
radius-table ordering (9.925 at ν=1), relocation squaring the scale factor. Room runs 40k steps
(official) vs the 30k gsplat baseline; `student-splatting` is a generic label — benchmark the
per-scene methods.
Runtime SSIM (valid conv), view sampling with replacement, SH-degree step offset (1001 vs 1000),
gsplat tile rectangle instead of Inria getRect, float evaluation instead of 8-bit PNG round trip,
the final iteration's Adam step (official skips it), single seed vs the paper's 3-run mean.

## Commands
```
tools/ssenv pip install -e reproductions/student-splatting/plugin
tools/ssenv ss-verify student-splatting --device cuda
tools/ssenv ss-train student-splatting-bicycle --data ~/datasets/mipnerf360/pilr/bicycle --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000 ...
```
