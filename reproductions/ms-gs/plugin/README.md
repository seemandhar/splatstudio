# splatstudio-method-ms-gs

Native SplatStudio port of **Multi-Scale 3D Gaussian Splatting for Anti-Aliased Rendering**
(Yan, Low, Chen, Lee, CVPR 2024, arXiv 2311.17089). The port follows the official code
github.com/JokerYan/MS-GS @ 50e5040 (`train.py --ms_train --filter_small --insert_large`) and its
rasterizer fork github.com/JokerYan/MS-GS-rasterizer @ e7a2b4f (Inria 3DGS licence, research use).
Paper fidelity is unverified.

## Mechanism → code
| contribution | code |
|---|---|
| C1 pixel coverage S_k: min half-extent of the undilated 2D Gaussian at the 1/255 level set | `coverage.pixel_coverage` (forward.cu:240-259) |
| C2 aggregation: at it 1010/1020/1030 small level-0 Gaussians (S<1 px at 4×/16×/64× on any train view where visible at 1×) are average-pooled per voxel (size 0.02·r/4 after 2−1/x contraction) and enlarged by 2/clip(S̄,0.25,2) | `aggregate.voxel_aggregate`, `density.Density.insert_large` |
| C3 selective rendering: cull if (S/S_min<0.5 ∧ S<2) or S/S_max>2; S_max/S_min updated at the Gaussian's own level (0.95/1.05 decay) | `coverage.select`, `coverage.update_ranges`, `renderer.Renderer` (gsplat on the selected subset) |
| C4 multi-scale training: per-epoch level, 1× w.p. 0.75 else least-trained of {2,3},{4,5},{6,7}; loss ×0.1 off 1×; only level-0 densify/prune/reset; position LR over 30k then hold | `schedule.MultiScaleSampler`, `losses`, `density` |

Per-Gaussian metadata are FieldSpecs `ms_level`, `ms_max_px`, `ms_min_px` (never in the autograd
graph; nominal lr 1e-8 because the contract needs > 0). The density controller owns the sampler
and tells the renderer (process-local `link.LINK`) which resolution to draw; evaluation renders
(no_grad) draw the camera they get. Coverage + selection run in a small fused CUDA kernel
(`csrc/coverage.cu`, JIT via `torch.utils.cpp_extension.load_inline`; torch reference in
`coverage.py` for CPU, probe `msgs-cuda-kernel` checks agreement). Culled Gaussians are given
opacity 0, which gsplat's projection culls (radius 0) exactly like the official kernel's early
return; the overhead over plain gsplat is ~1 ms per 1600-px render at 300k Gaussians.

## Config (defaults = official values)
renderer: `filter_small` true, `filter_large` true, `min_rel` 0.5, `max_rel` 2.0 (paper 1.5),
`abs_threshold` 2.0, `opacity_level` 1/255. density: `max_scale` 7 (Mip-NeRF 360; TnT/DB 6),
`insert_threshold` 1.0, `voxel_base` 0.02, `target_px` 2, `scale_clip` [0.25, 2], 3DGS ADC values,
`position_lr_max_steps` 30000; its `render` block must equal the renderer config.
loss: `ssim_weight` 0.2 (same-padded 3DGS SSIM), `low_res_weight` 0.1.

## Data / commands
Dataset `mipnerf360-msgs` (tools/make_multiscale_msgs.py): train = 1× (width 1600) frames,
test = 1–64× PIL-from-full-res. Train 40000 steps:
```
tools/ssenv ss-train ms-gs --data ~/datasets/mipnerf360/msgs/garden --output <run> --steps 40000 \
  --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run <run> --split test
python3 tools/multiscale_summary.py <run> --scales 1,4,16,64
```

## Fidelity gaps
1. Training GT at 2×–128× is an antialiased bicubic resize (CPU) of the 1600-px frame (official: PIL from
   the full-res JPEG); measured 53–84 dB PSNR apart (evidence/gt_resample.json).
2. Views drawn with replacement inside the official per-epoch resolution sequence; densify after
   the optimizer step; white background (official black); gsplat rasterizer (near plane 0.01)
   instead of the MS-GS rasterizer fork; SH degree step one iteration later.
3. 128× test images (12×8 px) are below the native loader minimum (11 px), so only 1–64× are
   evaluated; 128× is still used for training.
4. Evaluation SSIM/LPIPS are the runtime's (valid SSIM, `lpips` VGG).
