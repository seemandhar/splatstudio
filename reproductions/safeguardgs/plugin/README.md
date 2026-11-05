# splatstudio-method-safeguardgs

Native SplatStudio port of **SafeguardGS: 3D Gaussian Primitive Pruning While Avoiding
Catastrophic Scene Destruction** (Lee, Zhang, Fan; WACV 2026, arXiv
[2405.17793](https://arxiv.org/abs/2405.17793)), after the official code
https://github.com/ASU-ESIC-FAN-Lab/SafeguardGS (commit `539ca22e`). Paper fidelity unverified.

## Mechanism
Vanilla 3DGS for 30k steps (SplatStudio gsplat density + opacity-reset fix) with **one pruning
event during training at step 20000**: every training view is rendered against its ground truth,
each ray ranks its Gaussians by `Score_SG_36 = CS1(c_gt, c_i)·α_i·T_i`
(`CS1 = 1 − ‖c_gt − c_i‖₁/3`), the top-1 are kept, and the union over views survives
(`ops.remove`, survivors keep Adam moments).

| config (`sg-density`) | value | source |
|---|---|---|
| `prune_steps` | [20000] | paper §4.1; `benchmark/benchmark.yaml` |
| `top_k` | 1 | paper §3.4/§4.3; `benchmark.yaml` |
| `score_function` | 0x24 (= SG_36; 0x04 = SG_4 / EfficientGS) | paper Tab. 1, supp. §6; `arguments/__init__.py` |
| `selection` | `"tile"` (official kernel) / `"pixel"` (paper text) | see below |
| `p_dist_coef`, `c_dist_coef` | 1.0, 1.0 | `arguments/__init__.py` |

**Official selection semantics** (`forward.cu:renderCUDA_topk_color`, `markingTopKMasks`): per
16×16 tile, each pixel thread contributes one key per 256-Gaussian batch — the score of the
*last* contributing Gaussian of that batch (later contributors overwrite the slot) — and the k
best keys per tile are kept. With k = 1 that is ≤ 1 Gaussian per tile per view, which is what
gives the reported ≈ 89 % pruning. `selection: "pixel"` is the paper-text per-ray top-k
(ablation).

## CUDA
`csrc/sg_topk.cu` adapts the official `compute_score` and `renderCUDA_topk_color` (Inria 3DGS
licence, non-commercial research; `csrc/LICENSE-3DGS.md`) to gsplat 1.5.3 tile lists with gsplat
numerics; the tile-traversal layout follows the EfficientGS / Mini-Splatting plugins in this
workspace. Every tile writes its own padded slot range (the official kernel races across tiles in
partial last batches). The stable sort and per-tile top-k run in torch (`core.tile_topk_keep`).
JIT-built on first use (`tools/ssenv`).

## Fidelity gaps
- Official cross-tile slot race / always-kept Gaussian 0 not reproduced (deterministic).
- gsplat alpha numerics (cap 0.999, `next_T <= 1e-4`) instead of 3DGS (0.99, `< 1e-4`).
- Resolution: images_4 / images_2 here; the released sweep uses 3DGS's 1600-px auto-downscale of
  `images/` (paper: "original resolutions"). Survivor counts scale with the tile count.
- Native runtime: white background (score unaffected), sampling with replacement, density after
  the optimizer step of step 20000 (official: before), valid-conv SSIM, `lpips` LPIPS.
- Only dataset averages are published.

## Commands
```bash
tools/ssenv pip install -e reproductions/safeguardgs/plugin
tools/ssenv ss-inspect safeguardgs
tools/ssenv ss-verify safeguardgs --device cuda
tools/ssenv ss-train safeguardgs --data ~/datasets/mipnerf360/native/garden --output runs/x \
    --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```
