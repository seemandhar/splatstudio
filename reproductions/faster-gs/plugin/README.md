# splatstudio-method-faster-gs

Native SplatStudio port of **Faster-GS: Analyzing and Improving Gaussian Splatting Optimization**
(Hahlbohm, Franke, Eisemann, Magnor — CVPR 2026, arXiv 2602.09999), wrapping the official CUDA
backend (`FasterGSCudaBackend`, github.com/nerficg-project/faster-gaussian-splatting `main` @
090e8ad, Apache-2.0, see `ss_faster_gs/csrc/LICENSE`). Paper fidelity unverified.

## What it is
Faster-GS keeps 3DGS optimization unchanged and makes it fast. This plugin pins the paper's main
configuration ("Ours" in Tab. 1 / "Full" in Tab. 2):

| piece | code | origin |
|---|---|---|
| rasterizer forward/backward (fused activations, separate SH buffers, opacity-aware tight bounds, load-balanced exact tile culling, separate depth/tile sort, per-Gaussian backward with shared-memory state loading, fused densification stats) | `csrc/rasterization` (vendored, unmodified), `raster.py`, `renderer.py` | consolidated prior work + own (§3.2) |
| warp-level sub-tile culling in the blend kernel | inside the vendored kernel, always on | **borrowed** (Yang et al., HPG 2026); lossless speed feature, not a Faster-GS contribution |
| CUDA Adam | `csrc/adam`, `optim.py:FusedAdam` (replaces the runtime Adam in `density.initialize`) | own (§3.3) |
| Morton ordering every 5000 ≤ 15000 | `morton.py`, `density.py` | own (§3.3) |
| 3DGS ADC / opacity reset / schedule | `density.py` (official `Model.py`) | 3DGS, unchanged |

Not included (optional, off in the paper's headline row): fused backward+optimizer
(`FasterGSFused` branch), skip-invisible updates, Mip-Splatting filters, 3DGS-MCMC, Speedy-Splat
pruning, AbsGS, PPISP, random background (the filter3d/MCMC CUDA modules are not even built).

## Configuration (defaults = official `Trainer.py`)
Renderer: background black, near 0.2, far 10000. Density: densify 600..14900 every 100, grad
threshold 2e-4, percent dense 0.01, prune opacity < 0.005, prune scale > 0.1·extent after 3000,
reset to 0.01 every 3000 ≤ 14900, Morton every 5000 ≤ 15000 (and at step 0), opacity lr 0.025,
FusedAdam on, final prune opacity < 1/255 at `total_steps` (30000). Loss: runtime photometric
(0.8 L1 + 0.2 D-SSIM).

Optimizer state alignment: `FusedAdam` subclasses `torch.optim.Adam` with the same state keys, and
every population change (ADC append/prune, Morton permutation, opacity reset) goes through
`gsplat.strategy.ops._update_param_with_optimizer` / `remove` / `reset_opa`; probes
`fastergs-adc` and `fastergs-morton` check moments move with the parameters.

## Build
First use JIT-compiles the backend with `torch.utils.cpp_extension.load` (`-O3 -use_fast_math
-std=c++17`, ~75 s on sm_86 with CUDA 12.1; cached in `~/.cache/torch_extensions`). Run through
`tools/ssenv` so nvcc uses gcc 11.

## Commands
```bash
tools/ssenv pip install -e reproductions/faster-gs/plugin
tools/ssenv ss-verify faster-gs --device cuda
tools/ssenv ss-train faster-gs --data ~/datasets/tandt_db/native/truck --output runs/x --steps 30000 \
    --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv python -m ss_faster_gs.profile --run runs/x --views 50   # fwd / bwd / update vs gsplat
```

## Fidelity gaps
Runtime SSIM (valid conv) instead of fused-ssim; view sampling with replacement; one-step means-LR
offset; one seed (paper: mean of 5); main-branch kernels (16×12 tiles + borrowed warp culling)
instead of the paper testbed; timing on shared A6000 / CUDA 12.1 (paper: RTX 4090 / 12.8).
