# splatstudio-method-ags-mesh

Native SplatStudio port of **AGS-Mesh** (Ren et al., 3DV 2025, arXiv 2411.19271), Splatfacto
variant ("Splatfacto + Ours"), following the official `ags-mesh` method in
github.com/maturk/dn-splatter @ 97588b4. Methods: `ags-mesh`, control `ags-mesh-rgb`.

## Mechanism → code
| contribution | code | defaults (source) |
|---|---|---|
| C1 DNC: sensor-depth normals (Open3D KNN 200) vs Omnidata, reject > 10° | `dnc.py:dnc_frame` (offline, `tools/prepare_dnc.py`), `losses.py:create_depth` | encoded-space angle + JPEG mask (scripts/depth_to_normal.py), switch ns ≥ 7000, λ_d 0.2, EdgeAwareLogL1 |
| C2 ANR: rendered-depth normals vs prior, edge mask < 15k, angle mask ≥ 15k | `renderer.py:surface_normals`, `losses.py:create_normal` | λ_n 0.1 for ns > 7000, 0.1 rad, edges thr 0.01 (regularization_strategy.py) |
| Gaussian-normal L1 (DN-Splatter normals) | `renderer.py`, `losses.py:normal_terms` | unmasked, λ_n |
| scale loss | `losses.py:create_scale` | 1.0 |
| base: Splatfacto density, sensor-depth TSDF init | `density.py`, `sidecar.py` (copied from ss_dnsplatter) | as DN-Splatter |

## Data
```bash
PYTHONPATH=$HOME/models/dtueval-pylib:$HOME/models/dpt/pylib:$HOME/models/omnidata/pylib \
  CUDA_VISIBLE_DEVICES=<gpu> tools/ssenv python reproductions/ags-mesh/tools/prepare_mushroom.py --room <room>
PYTHONPATH=$HOME/models/dtueval-pylib tools/ssenv python reproductions/ags-mesh/tools/prepare_dnc.py --room <room>
```
(or `tools/prepare_rooms.sh <gpu> room...`). Train with `--data ~/datasets/mushroom/native/<room>
--recipe splatstudio --num-gaussians 10000000 --max-gaussians 12000000 --steps 30000`.

## Fidelity gaps
- Official Splatfacto code detaches the depth for the ANR surface normals (term has no gradient);
  ported as is (`surface_grad: false`); the paper's formulation is `surface_grad: true`.
- Code vs paper: EdgeAwareLogL1 (paper L1), τ_N 0.1 rad (paper 10°), DNC angle between encoded
  normals; code used.
- No meshing (adaptive TSDF / IsoOctree) and no mesh metrics; no 2DGS variant.
- gsplat 1.5.3 (official 1.0), regenerated TSDF init and Omnidata normals, LPIPS-VGG.
