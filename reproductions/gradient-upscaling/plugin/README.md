# splatstudio-method-gradient-upscaling

Native SplatStudio port of **Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting
Images** (Niedermayr, Neuhauser, Westermann, ICCV 2025, [arXiv 2503.14171](https://arxiv.org/abs/2503.14171)).
Paper fidelity unverified.

The official repo (github.com/KeKsBoTer/upscale3dgs) has no code ("training code coming soon").
The CUDA kernel `ss_gradient_upscaling/csrc/jet_raster.cu` was written for this port. Its tile
traversal and alpha numerics are adapted from gsplat 1.5.3 (Apache-2.0). Its forward derivative
recurrences match the authors' brush viewer shader (`rasterize.wgsl`, Apache-2.0). Its backward
pass is reverse mode over the derivative jets, using the paper's inversion trick. The kernel is
JIT-built with `torch.utils.cpp_extension` (run it through `tools/ssenv`).

| method | paper row |
|---|---|
| `gradient-upscaling-2x`, `-4x` | Spline (Ours): renders at W//s with I, I_x, I_y, I_xy; spline-upscales; loss at W |
| `bicubic-upscaling-2x`, `-4x` | Bicubic: gsplat at W//s, then PyTorch bicubic, trained the same way |
| `gradient-upscaling-3dgs` | 3DGS control: stock renderer; train with `--downscale s`, evaluate at W with `apply_upscaling.py` |

All five use the Taming-3DGS densification (plugin #30): final count 1 M, interval 100. Its
other Taming changes are switched off.

## Mechanism → code
- Analytic image gradients: `jet_fwd_kernel`. Each per-pixel quantity is a jet (v, ∂x, ∂y, ∂xy),
  with w = α⊗T, C += c·w and T ← T⊗(1−α). The white background enters through bg·T.
- Backward: `jet_bwd_kernel`. It recovers T_i = T_{i+1}⊘(1−α_i) and computes
  M = cT − (buffer + bg·T_N)⊘(1−α), then chains through α ↦ opacity, conic and mean2d.
- Spline: `upscale.spline_upscale`, a separable tensor-product Hermite. This is identical to the
  paper's A = C⁻¹F C⁻ᵀ (checked by a probe).
- Renderer: `renderer.UpscaleRenderer` uses gsplat's projection, SH and tile lists, then the jets
  kernel, then the spline.

## Commands
```bash
tools/ssenv pip install -e reproductions/gradient-upscaling/plugin
tools/ssenv ss-verify gradient-upscaling-4x --device cuda
tools/ssenv ss-train gradient-upscaling-4x --data ~/datasets/tandt_db/native/truck --output runs/x --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv python reproductions/gradient-upscaling/plugin/apply_upscaling.py --run runs/x --out r.json --data ~/datasets/tandt_db/native/truck
```

## Fidelity gaps
See `../plan.md`. In short:
- There is no official code to check against.
- Numerics are gsplat's (0.999 alpha clamp, with zero derivative where it is clamped).
- The background is white.
- The upscaler border is an edge-cell extrapolation.
- Only the Taming densification is used.
- Mip-NeRF360 at full resolution (5187×3361) is not benchmarked.
- The upscaler is PyTorch ops; the paper uses a SLANG.D kernel.
