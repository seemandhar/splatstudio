# splatstudio-method-lpm

Native SplatStudio port of **LPM — Improving Gaussian Splatting with Localized Points Management**
(Yang et al., CVPR 2025 Highlight, arXiv 2406.04251) on vanilla 3DGS, following the official code
https://github.com/Surrey-UP-Lab/GS-LPM (@7c06026). Paper fidelity unverified.

## Mechanism (official `train.py`, `lpm/*.py`)
- **Error map (C1)**: brightness-normalised L1 of render vs GT, >0.4-quantile pixels, 16×16 patches
  kept if >85% marked.
- **Region mapping (C2)**: SuperPoint + LightGlue matches to ≤3 random neighbours within `angle`
  (centre-ray angle); per error component, bbox of matched keypoints in both views.
- **Zones (C3)**: closest points of paired perimeter rays → AABB zone.
- **Localized addition (C4)**, every 100 steps in (500, 15000), replacing 3DGS densify_and_prune:
  clone/split threshold ×0.5 inside zones, the same number of lowest-opacity Gaussians pruned;
  nothing at all when no zone is found.
- **Localized reset (C5)**, every `reset_every` steps in (500, 15000): Gaussians in the camera→zone
  cone in front of a zone with opacity ≥ the zone's 70%-rank opacity → opacity 0.01.
- Global 3DGS opacity reset every 3000 (< 15000) kept; 3DGS loss/LRs/SH schedule unchanged.

| method | reset_every | angle | source |
|---|---|---|---|
| `lpm` | 100 | 45° | `arguments/__init__.py` defaults, README outdoor |
| `lpm-indoor` | 200 | 90° | README note for indoor scenes |

All other values in `ss_lpm/density.py:DEFAULTS` (sources in `../paper.md` §3).

## Requirements
LightGlue source at `~/models/lightglue/src` (override `LPM_LIGHTGLUE`), weights
`superpoint_v1.pth`, `superpoint_lightglue_v0-1_arxiv.pth` in `$TORCH_HOME/hub/checkpoints`
(linked from `~/models/lightglue/weights`). SuperPoint weights: Magic Leap non-commercial licence.

## Fidelity gaps
1. Training images reach the density only through renders: pairs are built lazily at a view's
   first LPM step from neighbours already seen (topped up later) instead of offline for all views.
2. Runtime order: optimizer step before density (official densifies before the step).
3. Runtime: white background, random view sampling with replacement, gsplat rasterizer.
4. The paper's "add points at the zone centre" is not in the official code → not implemented.
5. Indoor setting: README (200, 90°) vs the paper's reset-interval ablation favouring 100 indoors.
6. Rays use our K (with cx, cy); official uses a symmetric projection.

## Commands
```bash
pip install -e .
ss-verify lpm --device cuda
ss-train lpm --data ~/datasets/mipnerf360/native/bicycle --output runs/x --recipe 3dgs --num-gaussians 10000000
ss-train lpm-indoor --data ~/datasets/mipnerf360/native/room --output runs/y --recipe 3dgs --num-gaussians 10000000
```
