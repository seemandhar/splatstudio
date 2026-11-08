# splatstudio-method-scaffold-gs

Native SplatStudio port of **Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering** (Lu et al.,
CVPR 2024, https://arxiv.org/abs/2312.00109). Official code: https://github.com/city-super/Scaffold-GS @59c833b.

The backbone is extracted from the HGS native plugin (`reproductions/hard-gs/plugin/ss_hard_gs`, control method
`scaffold-gs-hgsbase`), which was written against the official code; all HGS code is removed. Differences from that
control: field names (`sgs_*`), and a straight-through opacity cap at 0.999 (renderer config `opacity_cap`, `null`
disables) to avoid gsplat's dropped gradients for saturated alpha.

Methods: `scaffold-gs` (Mip-NeRF 360, voxel 0.001), `scaffold-gs-tandt` (0.01), `scaffold-gs-db` (0.005).

## Mechanism
Anchors (native Gaussians: `means`, `sgs_feat` [32], `sgs_offset` [10,3], `sgs_scaling` [6]) from the voxelized
SfM cloud; per view, frustum-visible anchors decode 10 neural Gaussians each by opacity (tanh) / cov / color
(sigmoid) MLPs from [feature, view direction]; neural Gaussians with opacity > 0 are rasterized by gsplat. Growing
every 100 steps in (1500, 15000) at 3 voxel levels from average NDC view-space gradients, pruning by accumulated
opacity. Loss 0.8 L1 + 0.2 D-SSIM + 0.01 mean(prod scale). The MLPs live in a density-owned bank and are packed into
`sgs_blob` each step so `ss-eval` can render.

## Config sources
`arguments/__init__.py` and `train_{mip360,tnt,db}.sh` of the official repo (see ../paper.md §3).

## Commands
```
tools/ssenv ss-verify scaffold-gs --device cuda
tools/ssenv ss-train scaffold-gs --data ~/datasets/mipnerf360/w1600/garden --output runs/x --steps 30000 \
    --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```
`--recipe 3dgs` and `--num-gaussians` above the SfM size are required (guards raise otherwise).
Fidelity gaps: ../plan.md.
