# splatstudio-method-dn-splatter

Native SplatStudio port of **DN-Splatter: Depth and Normal Priors for Gaussian Splatting and
Meshing** (Turkulainen et al., WACV 2025, arXiv 2403.17822), following the official Nerfstudio
plugin github.com/maturk/dn-splatter @ 97588b4 (Splatfacto 1.1.3 + gsplat 1.0.0).

Methods: `dn-splatter` (full), `dn-splatter-splatfacto` (paper baseline: vanilla Splatfacto init + resolution schedule on the same points) and `dn-splatter-rgb` (priors-off ablation: same pipeline, photometric loss only; not the paper baseline).

| contribution | code | value (source) |
|---|---|---|
| C1 edge-aware log-L1 sensor-depth loss (Eq. 3) | `losses.create_depth` | effective weight 1.2 = (1+0.2), regularization_strategy.py:179 (paper λd 0.2); D>0.1 m; edge image = gt.clamp(≥10/255) |
| C2 normals from the shortest axis, flipped to the camera, composited with detached means2d over a white background (Eq. 4,7) | `renderer.Renderer` | dn_model.py:470-520; gsplat 1.0 legacy background ones |
| C3 Omnidata normal L1 + TV (Eq. 8,9) | `losses.create_normal` | weights 1 / 1 (code; paper 0.1/0.1) |
| C4 min-scale loss (Eq. 6) | `losses.create_scale` | weight 1 |
| C5 1M TSDF sensor-depth init, Open3D normals → quats, z-scale ÷10 | `tools/prepare_mushroom.py` + `density.initialize` | pointcloud_utils.py, mushroom_dataparser.py, dn_model.py |
| base: Splatfacto ADC (absgrad 0.0008, size 0.01, cull α 0.1, reset 3000, unscaled means LR) | `density.Density` | splatfacto.py v1.1.3 |

Train/eval background: random / (0.149, 0.1647, 0.2157). Normals compared in the OpenCV camera
frame with the raw Omnidata png (identical loss to the official GL-frame comparison with a
diag(1,−1,−1)-converted png).

## Data
```bash
# raw: zenodo 10230733 vr_room_iphone.tar.gz -> ~/datasets/mushroom/raw/vr_room_x
# weights: zenodo 10447888 omnidata_dpt_normal_v2.ckpt -> ~/models/omnidata (sha256 in sidecar manifest)
PYTHONPATH=$HOME/models/dtueval-pylib:$HOME/models/dpt/pylib:$HOME/models/omnidata/pylib \
CUDA_VISIBLE_DEVICES=<gpu> tools/ssenv python reproductions/dn-splatter/tools/prepare_mushroom.py --room vr_room
```
Splits: train 376 long frames, test = 42 `test.txt` frames (within), validation = 186
short_capture frames (cross trajectory).

## Train / eval
```bash
tools/ssenv ss-train dn-splatter --data ~/datasets/mushroom/native/vr_room --output RUN \
  --steps 30000 --recipe splatstudio --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run RUN --split test         # within
tools/ssenv ss-eval --run RUN --split validation   # cross trajectory
```
`--num-gaussians` must keep all 1M sidecar points (init check). `--recipe splatstudio` avoids
the runtime's 1M×1M kNN (scales/quats come from the sidecar; the means LR is reset to 1.6e-4).

## Fidelity gaps
gsplat 1.5.3 instead of 1.0.0; re-generated (seeded) TSDF init — 35 % duplicate samples with
zero 3-NN distance get log-scale −1e4 instead of −inf (same behaviour); LPIPS-VGG instead of
torchmetrics AlexNet; view sampling with replacement; no Poisson meshing / mesh metrics; Table 4
eval split not stated.
