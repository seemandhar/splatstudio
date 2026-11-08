# splatstudio-method-vod3dgs

VoD-3DGS: View-opacity-Dependent 3D Gaussian Splatting (Nowak, Jarosz, Chin), https://arxiv.org/abs/2501.17978.
No official code was released (checked 2026-11-07); this port follows the paper text.

Methods: `vod3dgs-s` (VoD-3DGS[S], reset keeps the smallest eigenvalue) and `vod3dgs-l` ([L], largest).

| contribution | code | value (source) |
|---|---|---|
| Eq. 9 opacity σ(γ + ωᵀŜω), field `vod_S` (6) init 0 | `renderer.py`, `vod.py:quad_form` | – |
| Eq. 10 prune on max opacity over training views | `density.py:VoDStrategy._prune_gs` | τ = 0.005 (3DGS) |
| Eq. 11 eigen reset of Ŝ at each opacity reset (3000…12000) | `density.py:reset_vod_S` | variant S / L |
| Eq. 12 L_vc, second view ∝ SfM covisibility | `loss.py`, `covis.py` | weight 1 (Eq. 13) |
| γ and Ŝ learning rate lr_γ/4 | `density.py:initialize`, FieldSpec | 0.0125 (§5.1) |

Covisibility files (`ss_vod3dgs/covis_data/<scene>.npz`) are built from COLMAP `sparse/0`:
`python -m ss_vod3dgs.covis --sparse <scene>/sparse/0 --name <scene>`; shipped for the 9 Mip-NeRF 360,
2 T&T and 2 Deep Blending scenes. The loss refuses to run if no file matches the training cameras.

Fidelity gaps: SfM covisibility instead of LightGlue matches; γ is also reset (paper ambiguous);
means detached in L_vc and Ŝ Adam moments zeroed at reset (unspecified); gsplat rasterizer and
SplatStudio runtime instead of the Inria code.

```bash
tools/ssenv ss-train vod3dgs-s --data ~/datasets/tandt_db/native/truck --output runs/x --steps 30000 --recipe 3dgs --num-gaussians 10000000
```
