# splatstudio-method-wildgaussians

Native SplatStudio port of **WildGaussians: 3D Gaussian Splatting in the Wild** (Kulhanek et al.,
NeurIPS 2024, [arXiv 2407.08447](https://arxiv.org/abs/2407.08447)), following the official code
([jkulhanek/wild-gaussians](https://github.com/jkulhanek/wild-gaussians), MIT). Paper fidelity
unverified.

| method | schedule | appearance |
|---|---|---|
| `wildgaussians` | NeRF On-the-go (30k, `nerfonthego.yml`) | on (paper Tab. 1/7 "Ours") |
| `wildgaussians-noapp` | NeRF On-the-go | off (released `nerfonthego.yml`) |
| `wildgaussians-pt` | Photo Tourism (200k, `phototourism.yml`) | on |

## Mechanism → code
- Appearance (Eq. 4–5): `appearance.py` (MLP 59→128→128→6, ×0.01, offset/C0 on DC, mul on all SH;
  Fourier per-Gaussian embeddings, field `wg_embedding`), `renderer.py` (raw + toned gsplat passes).
- Objective (Eq. 6/11): `losses.py::wg-color` — D-SSIM on raw, L1 on toned, × uncertainty mask.
- Uncertainty (Eq. 8–10): `uncertainty.py`, `dino.py` (DINOv2 ViT-S/14-reg block 5, own
  implementation; weights `~/models/dinov2/dinov2_vits14_reg4_pretrain.pth`, sha256 checked),
  `losses.py::wg-uncertainty` (publishes `outputs["wg_mask"]`; must precede `wg-color`).
- Per-image state: `bank.py` (image embeddings, MLP, uncertainty head, DINO cache, own Adam),
  created by `density.py`, saved in `density_state["bank"]`.
- Base: Mip-Splatting 3D filter + 2D filter 0.1 (`geometry.py`, gsplat antialiased), GOF abs-grad
  densification with never-reset statistics, sky sphere, official LR schedule (`density.py`).
- Evaluation: `python -m ss_wildgaussians.evaluate --run RUN` (On-the-go: zero embedding;
  Photo Tourism: NeRF-W left-half embedding optimisation, right-half metrics; NerfBaselines
  metrics with LPIPS-Alex). `ss-eval` cannot see the bank and reports raw (untoned) renders.

## Data
`reproductions/wildgaussians/tools/prepare_wild.py` builds `~/datasets/onthego-undist/native/<scene>`
and `~/datasets/phototourism/native/<scene>` (on /bigdata).

## Commands
```bash
ss-train wildgaussians --data ~/datasets/onthego-undist/native/patio-high --output RUN \
  --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
python -m ss_wildgaussians.evaluate --run RUN
ss-train wildgaussians-pt --data ~/datasets/phototourism/native/brandenburg-gate --output RUN \
  --steps 200000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```

## Fidelity gaps
See `reproductions/wildgaussians/plan.md` (view sampling with replacement, optimizer/density order,
exact KNN, gsplat rasterizer, sky logit 20 vs +inf, single seed, paper-vs-code choices).
