# splatstudio-method-rain-gs

Native SplatStudio port of **RAIN-GS: Relaxing Accurate Initialization Constraint for 3D
Gaussian Splatting** (Jung et al., arXiv [2403.09413](https://arxiv.org/abs/2403.09413)),
following the official code [cvlab-kaist/RAIN-GS](https://github.com/cvlab-kaist/RAIN-GS)
(MIT License, Copyright (c) 2024 KU-CVLAB), path `train.py --ours_new --train_from random`.
Paper fidelity is unverified.

## Mechanism → code
| contribution | code | values (source) |
|---|---|---|
| SLV random init | `density.py: RainDensity.set_training_cameras` | N=10 points in a cube of side 3(max−min) of the camera `tvec` coordinates, 3-NN scales, opacity 0.1, sh0~U[0,1) (dataset_readers.py) |
| Progressive low-pass | `renderer.py` | eps2d = min(max(HW/(9πN), 0.3), 300) at step 1 and every 1000 below 25000; test 0.3 (train.py) |
| ABE-Split | `density.py: abe_append` | during step ≤ 10000 each split adds a copy at 0.3·extent·μ, unchanged scale (gaussian_model.py) |
| Warm-up | `density.py: means_lr`, `renderer.py: sh_degree_at` | means LR fixed for 10k then expon(step−10k, max 30k); densify until 25k; SH +1 per 1000 from 5000; split scale /1.4 |

Unchanged 3DGS: L1 + 0.2 D-SSIM, LRs, grad threshold 2e-4, clone/split at 0.01·extent,
opacity reset every 3000 (to ≤0.01), prune α<0.005 and world size >0.1·extent after 3000.
Black background (official default).

## Fidelity gaps
- SLV cube from training-camera tvecs only (official uses all images).
- Native runtime: optimizer step before densification, random view sampling with
  replacement, valid-conv SSIM.
- gsplat per-axis 3σ radius vs Inria max-eigenvalue circle; near plane 0.01 vs 0.2.
- Random-init variance: one seed per scene.

## Commands
```bash
pip install -e .
ss-inspect rain-gs && ss-verify rain-gs --device cuda
ss-train rain-gs --data ~/datasets/mipnerf360/native/garden --output runs/x --steps 30000 \
  --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```
(`--num-gaussians` only affects the SfM points the runtime loads; SLV replaces them.)
