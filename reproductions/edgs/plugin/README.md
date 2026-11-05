# splatstudio-method-edgs

EDGS: Eliminating Densification for Efficient Convergence of 3DGS (arXiv 2504.13204, CVPR 2026).
Native port of https://github.com/CompVis/EDGS. Paper fidelity unverified.

## Install
```bash
tools/ssenv pip install -q einops kornia timm loguru h5py
tools/ssenv pip install -q --no-deps ~/models/roma/RoMa-3701174   # RoMa commit 370117431ffc (EDGS submodule pin)
cd reproductions/edgs/plugin && ../../../tools/ssenv pip install -e .
```
Weights: `~/models/roma/roma_outdoor.pth` (sha256 c7a45c80…), `~/models/roma/dinov2_vitl14_pretrain.pth` (d5383ea8…).
RoMa is installed in the `splatstudio` env (no torch/numpy changes needed).

## Stage 1: correspondence initialization (training views only)
```bash
tools/ssenv ss-edgs-init --source ~/datasets/mipnerf360/native/garden \
    --output ~/datasets/edgs-init/mipnerf360/garden --matches-per-ref 25000   # T&T/DB: 15000
```
Defaults follow the official `full_eval.py`: 180 reference views (k-means on W2C), 3 nearest
neighbours, RoMa outdoor, τ_corr 0.05, τ_proj 0.01 (L1 NDC), scale 0.0005·dist(ref cam), opacity 0.5.
Output: `transforms.json` + hard-linked `images/` (same dataset hash), `points.npz`,
`edgs_init.npz` (log scales, opacity logits, pinv SH rest, points hash), `edgs_init.json` (timings).

## Stage 2: training
```bash
tools/ssenv ss-train edgs --data ~/datasets/edgs-init/mipnerf360/garden --output runs/x \
    --steps 30000 --recipe splatstudio --num-gaussians 10000000 --max-gaussians 12000000
```
Use `--recipe splatstudio`: `--recipe 3dgs` OOMs (runtime 3-NN scale init on ~4M points);
`edgs-density` restores 3DGS's camera-extent means LR itself. `--num-gaussians` must be ≥ the init point count (all points kept, so the hash lookup of
`edgs_init.npz` succeeds; otherwise training fails loudly).

| method | SH rest init |
|---|---|
| `edgs` | DC = reference colour; bands 1–3 = constrained ridge fit (λ=1/16) of the neighbour residual (paper Sec. 3.5, composition is our documented deviation) |
| `edgs-nosh` | zero (released code; paper ablation "w/o SH init") |

`edgs-density` (official `trainer.py`, `no_densify=True`): no clone/split/reset; prune opacity<0.005
every step before 15000; opacity logit += log 0.99 every 10 steps before 15000; opacity LR 0.025;
means LR = 3DGS schedule evaluated at max(step, 8000).

## Fidelity gaps
White background and with-replacement view sampling (runtime); SH-init composition and ridge
λ=1/16 are a documented deviation (paper: plain pinv, no consistency rule or clamp); RoMa `match()` zeroes out-of-image flow certainty; pixel-centre reference coords;
true principal point in NDC; preprocessing time is not part of `training_seconds`.
