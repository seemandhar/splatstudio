# splatstudio-method-pup-3dgs

Native SplatStudio port of **PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian
Splatting** (Hanson, Tu, et al., CVPR 2025, [arXiv 2406.10219](https://arxiv.org/abs/2406.10219),
[official code](https://github.com/j-alex-hanson/gaussian-splatting-pup)). Paper fidelity unverified.

## Methods
| name | pipeline |
|---|---|
| `pup-3dgs` | v3/CVPR: 3DGS 30k → prune 80% → refine 5k → prune 50% → refine 5k (90%); run with `--steps 40000` |
| `pup-3dgs-v1` | arXiv v1: 66% then 66% (88.44%); `--steps 40000` |

## Mechanism (single native run)
`ss_pup3dgs.density.PUPDensity` wraps SplatStudio's 3DGS density (gsplat `DefaultStrategy` +
opacity-reset fix) and:
1. Steps 1–30000: unchanged 3DGS; the means LR is reset to `lr0·0.01^(s/30000)`, i.e. the
   schedule of a 30k run, so pretraining equals `ss-train gsplat --steps 30000`.
2. After steps 30000 and 35000: per-Gaussian Fisher F_i = Σ_views Σ_pixels Σ_channels g gᵀ,
   g = ∂I/∂(mean_i, log-scale_i), rendered at ceil(H/4)×ceil(W/4) (4×4 patches), with the
   official CUDA kernel; score U_i = log|det F_i| (`log(svdvals).sum()`), prune the lowest
   80% / 50% with the official percentile rule, then clear Adam state (fresh optimizer).
3. Refinement LRs as in `prune_finetune.py`: means `get_expon_lr_func(1.6e-4·s, 1.6e-6·s,
   max_steps=35000)` at lr_iter 30001…35000 (restarted in round 2), other fields ×0.95 per
   400 iterations (torch `ExponentialLR`).

## Config (`pup-density`, paper/code values)
`base_steps` 30000; `prune_steps` [30000, 35000]; `prune_ratios` [0.8, 0.5]
(`scripts/full_pruning_pipeline.sh`); `fisher_resolution` 4 (`--fisher_resolution 4`);
`finetune_lr_start` 30000 / `finetune_lr_max_steps` 35000 (`--first_iter`/`--position_lr_max_steps`);
`lr_final_ratio` 0.01 (1.6e-6/1.6e-4); `exp_gamma` 0.95 / `exp_every` 400 (`prune_finetune.py`);
`reset_optimizer` true; `background` white (our training background; official: black).

## CUDA
`ss_pup3dgs/csrc/` vendors `rasterization_and_pup_fisher` (commit eaef2fa) by the PUP authors —
the Inria 3DGS rasterizer with `renderFisherCUDA` (derived from FisherRF) — and glm
(MIT, `third_party/glm`). Licence: Inria Gaussian-Splatting licence, non-commercial research
and evaluation use (`csrc/LICENSE-3DGS.md`). JIT-built on first use by
`torch.utils.cpp_extension.load` (run through `tools/ssenv`; ~1 min).

## Probes
- `fisher-oracle`: CUDA Fisher vs per-pixel autograd through gsplat on 3 low-res views
  (rel. err ≈ 5e-4); also confirms the probe would detect a linear-scale or full-resolution Fisher.
- `prune-schedule`: two prune rounds on a 400-Gaussian scene: survivors = highest float64
  `slogdet`, official counts, unseen Gaussians pruned, Adam reset, LRs equal an independent
  replay with torch `ExponentialLR`.

## Fidelity gaps
Base model is the native gsplat 3DGS recipe (not the authors' checkpoints); Fisher is computed
with the official 3DGS-rasterizer kernel on a gsplat-trained model (alpha clamp 0.99 vs 0.999,
near plane 0.2 vs 0.01); 5000 vs 4999 effective refine updates per round; views sampled with
replacement; white background. See `../plan.md`.

## Commands
```bash
tools/ssenv pip install -e reproductions/pup-3dgs/plugin
tools/ssenv ss-verify pup-3dgs --device cuda
tools/ssenv ss-train pup-3dgs --data ~/datasets/mipnerf360/native/bicycle --output runs/x \
    --steps 40000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```
