# splatstudio-method-splatfields

SplatFields: Neural Gaussian Splats for Sparse 3D and 4D Reconstruction (Mihajlovic et al.,
ECCV 2024, arXiv 2409.11211) — native SplatStudio port of the **static** branch
(SplatFields3D). Network code adapted from the official repo
https://github.com/markomih/SplatFields (MIT), see `ss_splatfields/network.py`.
Paper fidelity unverified.

## Methods
| method | what |
|---|---|
| `splatfields-3dgs` | stage 1: official `train.py --is_static` 3DGS (mask loss 0.1, densify every 100 from 500, last refine 39900 (official saves before the 40000 densify), no opacity reset, xyz LR 1.6e-4·5 → 1.6e-6·5 over 30k). The paper's 3DGS row and the source of the SplatFields cloud. |
| `splatfields` | stage 2: tri-plane CNN (3 × diffusers-style VAE decoders, fixed 8×20×20 noise → 16×160×160) + MLP fields (deform residual, rgb, scale = MLP + base, opacity, rotation) predict all splats from detached anchors; photometric + mask (0.1) + splat-norm (0.01); network Adam 8e-4 → 1.6e-6 over 40k, eps 1e-15; DefaultStrategy on the anchors; bake to plain 3DGS at `total_steps` = 40000. |

Global learned state: the density owns the network and its Adam (`network.ACTIVE`, as in the
eagles port); the renderer predicts the splats through it while training; at the last step
the predictions are baked into means / scales / quats / opacities / sh0 (shN = 0), so
validation, the checkpoint and `ss-eval` evaluate the network-free model (the paper also
discards the networks). Network weights are also saved in `density_state` for provenance.

## Data
Blender few-view scenes built by `reproductions/splatfields/tools/`:
`select_views.py` (official KMeans view choice, scikit-learn 1.0.2 venv) →
`prepare_blender_fewview.py` (images, splits, hull init, `masks.npz`) → stage-1 runs →
`build_stage2.sh` / `make_stage2.py` (stage-1 cloud, hull-filtered, ≤100k).
The mask loss reads `<data>/masks.npz` (data root from `SPLATFIELDS_DATA` or ss-train's
`--data`).

## Commands
```bash
tools/ssenv ss-train splatfields-3dgs --data ~/datasets/nerf_synthetic/fewview6/chair --output R1 --steps 40000 --recipe 3dgs --num-gaussians 10000000
tools/ssenv python -I reproductions/splatfields/tools/make_stage2.py --stage1-data ~/datasets/nerf_synthetic/fewview6/chair --run R1 --raw ~/datasets/nerf_synthetic/raw --out ~/datasets/nerf_synthetic/fewview6-sf
tools/ssenv ss-train splatfields --data ~/datasets/nerf_synthetic/fewview6-sf/chair --output R2 --steps 40000 --recipe 3dgs --num-gaussians 10000000
tools/ssenv ss-eval --run R2 --split test
```
`--steps` must equal the density's `total_steps` (40000) or nothing is baked; for short runs
use `smoke_stage2.py`.

## Fidelity gaps
gsplat vs ingra14m rasterizer; float vs 8-bit-PNG metrics; KMeans views from sklearn 1.0.2
(authors' exact version unknown); runtime steps per-Gaussian optimizers before densification
(stage 1); SH degree schedule one step earlier; optimizer step also on the last iteration;
stage-1 stochasticity; mask lookup via sidecar + argv; 4D branch and DTU protocol not ported.
