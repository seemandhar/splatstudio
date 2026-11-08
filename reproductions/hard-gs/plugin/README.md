# splatstudio-method-hard-gs

Native SplatStudio port of **HGS — Pushing Rendering Boundaries: Hard Gaussian Splatting** (Xu et al., CVPR 2025,
https://arxiv.org/abs/2412.04826). There is no official HGS code (github.com/GhiXu/HGS is 404). HGS is built on
**Scaffold-GS**, so this plugin ports the Scaffold-GS backbone from the official code
(https://github.com/city-super/Scaffold-GS @59c833b) and adds the two HGS growth criteria.

Methods: `hard-gs` (Mip-NeRF 360, voxel 0.001), `hard-gs-tandt` (0.01), `hard-gs-db` (0.005),
`scaffold-gs-hgsbase` (the same backbone with PGHGS and REHGS disabled).

## Mechanism
- Backbone: anchors (native Gaussians) with 32-D feature, 10 offsets, 6-D log scaling; opacity/cov/color MLPs
  (input [feat, view dir]) decode neural Gaussians per view; gsplat rasterizes them. Growing every 100 its in
  (1500, 15000) from neural-Gaussian average view-space gradients (3 voxel levels), pruning by accumulated opacity.
  Loss: 0.8 L1 + 0.2 D-SSIM + 0.01 mean(prod scale).
- **PGHGS** (Eq. 5–6): running top-k (k=3) per-view gradient norms per neural Gaussian; k-th ≥ λτ (λ=1) → hard.
- **REHGS** (Eq. 7–8): CUDA |S_i| (max-weight pixel count, kernel reused from the Mini-Splatting native plugin);
  |S_i| > 2e-4·|P| and 3DGS SSIM map < 0.7 at the projected centre → detection; ≥2 detections in the interval → hard.
- Hard Gaussians join every growth level's candidates (OG ∪ PGHGS ∪ REHGS); hard stats reset every interval.
  This integration is a reproduction choice (see ../paper.md §7).

## Config (hgs-density) with sources
voxel_size 0.001/0.01/0.005 (train_*.sh), update_init_factor 16, update_hierachy_factor 4, update_depth 3,
start_stat 500, update_from 1500, update_interval 100, update_until 15000, min_opacity 0.005, success 0.8,
densify_grad_threshold 2e-4 (arguments/__init__.py); pghgs {k 3, lambda 1.0}, rehgs {tau_large 2e-4, tau_ssim 0.7,
min_views 2} (HGS §6.1, §5.2).

## Fidelity gaps
See ../plan.md: integration of hard Gaussians into growing (no official code), gsplat rasterizer, white background,
random view sampling, update order, MLP weights transported through the `hgs_blob` per-anchor field (needs ≥ 887
anchors), per-dataset voxel size variants, Effi-HGS not implemented.

## Commands
```
tools/ssenv ss-verify hard-gs --device cuda
tools/ssenv ss-train hard-gs --data ~/datasets/mipnerf360/pilr/garden --output runs/x --steps 30000 \
    --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```
`--recipe 3dgs` and `--num-gaussians` above the SfM size are required (guards raise otherwise).
