# splatstudio-method-pgsr

Native SplatStudio port of **PGSR: Planar-based Gaussian Splatting for Efficient and
High-Fidelity Surface Reconstruction** (Chen et al., IEEE TVCG 2024,
[arXiv 2406.06521](https://arxiv.org/abs/2406.06521)), following the official code
[zju3dv/PGSR](https://github.com/zju3dv/PGSR) at commit `de24f1a`.

`ss_pgsr/csrc/` is the official `submodules/diff-plane-rasterization`, **unmodified**
(Inria/MPII Gaussian-Splatting licence and ZJU 3DV non-commercial research licence, see
`csrc/LICENSE-*.md`). It is JIT-built with `torch.utils.cpp_extension.load` on first use
(~1 min, cached in `~/.cache/torch_extensions`); run through `tools/ssenv`.

## Methods
| name | settings |
|---|---|
| `pgsr` | official `scripts/run_mip360.py`: exposure off, `densify_abs_grad_threshold` 0.0002 |
| `pgsr-tnt` | official `scripts/run_tnt.py`: exposure on, abs 0.00015, opacity cull 0.05 (ncc_scale 0.5 not supported → 1.0) |

## Mechanism → code
| paper | code | values (source) |
|---|---|---|
| L_s = ‖min(s)‖₁ (Sec. 3.1) | `losses.create_flatten` | λ₁ = 100, Gaussians with radii > 0 (train.py) |
| plane normal / distance / depth (Sec. 3.1) | `raster.render`, kernel `forward.cu:404` | normal = min-scale axis facing camera |
| L_svgeom (Sec. 3.2.1) | `losses.create_single_view` | λ₂ = 0.015, step > 7000 |
| L_mvgeom (Sec. 3.2.2) | `geometry.forward_backward_error`, `losses.create_mv_geo` | λ₄ = 0.03, φ < 1 px, w = exp(−φ) detached, step > 7000 |
| L_mvrgb (Sec. 3.2.2) | `geometry.ncc_terms`, `losses.create_mv_ncc` | λ₃ = 0.15, 7×7 grey patches, ≤ 102400 pixels, ncc < 0.9 |
| neighbour graph (supp. A) | `geometry.neighbour_graph` | ≤ 8, angle < 30°, 0.01 < d < 1.5, sorted by distance (code) |
| exposure (Sec. 3.3) | `losses.create_rgb` | exp(a)I + b in L1 if step > 1000 and 1−SSIM < 0.5; Adam lr 1e-3 (0.9, 0.99) |
| densification (Sec. 4, code) | `density.Density` | AbsGS split radius > 20, ≤ 50k, cap 6M, percent_dense 0.001, cull 0.005, trim < 2 views every 1000, reset 3000 |

The density controller reloads the **training split** images from `--data` (grey images for
NCC, cameras for the neighbour graph and the trim); test images are never loaded. The
multi-view losses render one random neighbour per step (gradients through its plane depth).

## Fidelity gaps
Runtime optimizer step precedes densification; view sampling with replacement; `mipnerf360`
uses the dataset's images_4/images_2 instead of PIL `-r` resizing; no TSDF mesh / DTU / TnT
geometry evaluation; ncc_scale ≠ 1 unsupported; seed and SSIM/LPIPS evaluation
implementations are SplatStudio's.

## Commands
```bash
tools/ssenv pip install -e reproductions/pgsr/plugin
tools/ssenv ss-verify pgsr --device cuda
tools/ssenv ss-train pgsr --data ~/datasets/mipnerf360/native/garden --output runs/x \
    --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv python reproductions/pgsr/plugin/smoke_schedule.py --data <scene> --output <run>
```
