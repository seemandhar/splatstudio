# splatstudio-method-stopthepop

StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Rendering (Radl, Steiner
et al., ACM TOG 43(4) 2024, arXiv 2402.00525) — native SplatStudio port.

## What is ported
- **Renderer**: the official sorted CUDA rasterizer, vendored *unmodified* in `ss_stopthepop/csrc/`
  from github.com/r4dl/StopThePop-Rasterization @ d8cad09 (submodule of r4dl/StopThePop @ 859b11b),
  JIT-built with `torch.utils.cpp_extension.load` (all template variants, ~70 s on first use;
  run through `tools/ssenv`). Used for **training and evaluation** with `configs/hierarchical.json`:
  HIER sort mode, per-tile depth at max-contribution position, queues 64/8/4, hierarchical 4x4
  culling, rect + tight-opacity bounding, load balancing, `tile_based_culling` false. Black
  background (official default).
- **Density**: unchanged 3DGS ADC of the official `train.py`/`gaussian_model.py` (grad 2e-4,
  500–15000 every 100, reset every 3000, prune o<0.005, world prune >0.1·extent after 3000;
  the 20-px screen prune is inert because `densification_postfix` zeroes `max_radii2D`).
  `opacity_decay: 0.9995` reproduces the paper's "Opacity Decay" variant.
- **Loss**: base L1 + 0.2 D-SSIM.

Methods: `stopthepop`; `stopthepop-global` = same code with `configs/vanilla.json`
(3DGS global z sort) as a control.

Licences: Gaussian-Splatting licence (non-commercial research) for the Inria-derived parts
(`csrc/LICENSE.md`); StopThePop headers MIT (`csrc/cuda_rasterizer/stopthepop/LICENSE.txt`);
nlohmann json MIT; pybind11_json BSD-3; glm MIT.

## Probes (`ss-verify stopthepop --device cuda`)
`stopthepop-topt` (Eq. 1 blend order vs d·(μ−o) and global z ablations), `-perpixel-order`
(HIER/PPX_FULL = independent float64 per-pixel sorted compositor, GLOBAL = global reference),
`-backward` (kernel gradients = autograd of the sorted reference), `-popping` (yaw sweep:
rotation-warped consistency error pops for GLOBAL only), `-culling` (exactness + fewer sort
entries), `-schedule` (density cadence / opacity decay).

## Fidelity gaps
Optimizer step precedes density control in the runtime; runtime sampling with replacement,
SSIM valid convolution, LPIPS package; FLIP+RAFT popping metric replaced by an analytic probe;
timings on A6000 (paper: RTX 4090); `tile_based_culling` follows the released config.

## Commands
```
tools/ssenv pip install -e reproductions/stopthepop/plugin
tools/ssenv ss-verify stopthepop --device cuda
tools/ssenv ss-train stopthepop --data ~/datasets/mipnerf360/native/garden --recipe 3dgs --num-gaussians 10000000 --steps 30000 --output runs/x
```
