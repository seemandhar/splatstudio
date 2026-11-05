# splatstudio-method-trimgs — Trim3DGS (arXiv 2406.07499)

Native SplatStudio port of **Trim 3D Gaussian Splatting for Accurate Geometry Representation**
(Fan, Yang et al. 2024), 3DGS-based variant, following the official code
https://github.com/YuxueYang1204/TrimGS (`Trim3DGS/`, `scripts/run_Mipnerf360.py`) and its
rasterizer https://github.com/Abyssaledge/diff-gaussian-rasterization. Paper fidelity unverified.

Methods: `trimgs` (normal weight 0.1, Mip-NeRF 360 outdoor), `trimgs-indoor` (0.01, indoor).
Train **37000 steps** = 30000 flat-3DGS pre-training + 7000 Trim fine-tune. `--steps 30000`
yields our flat-3DGS pre-train, *assumed* comparable to the paper's "3DGS*" (see notes).

| mechanism | code | official source |
|---|---|---|
| flat Gaussians (scale axis 0 = 1e-8, normal = R[:,0]) | `renderer.py`, `core.flat_scales` | `scene/gaussian_model.py:FlattenGaussianModel` |
| two stages, fresh Adam, constant fine-tune xyz LR | `density.py` | `train.py`, `tune.py`, run script cmdline |
| contribution α^(2γ)T^(2−2γ)/|P_k| (custom CUDA on gsplat tiles, 3σ radius) | `csrc/trim_contrib.cu`, `kernels.py` | `forward.cu:356` |
| multi-view list (official record insertion, K=5), 10% trim every 1000 | `density.contribution`, `core.official_topk_insert` | `tune.py:prune_low_contribution_gaussians` |
| scale-driven split N=3, s/2.4, radii>100, culling box, opacity<0.005 | `core.split_n`, `density._densify` | `densify_and_scale_split`, `culling` |
| normal loss + kNN smoothness | `losses.py`, `core` | `tune.py:normal_regularization` |

Config values are in `density.DEFAULTS` / `losses.DEFAULTS` (sources in `../paper.md`).

Fidelity gaps: white background (runtime) vs black; gsplat rasterizer numerics for training
(the contribution pass uses the official 0.99 clamp / 3σ tiles); pre-training uses gsplat's
DefaultStrategy (as our baseline) instead of 3DGS's exact densifier; random view sampling;
training-view order for the record list is a seeded shuffle, not the official one; density runs
after the optimizer step. Geometry evaluation (DTU/TnT) not available.

```bash
tools/ssenv pip install -e reproductions/trimgs/plugin
tools/ssenv ss-verify trimgs --device cuda
tools/ssenv ss-train trimgs --data ~/datasets/mipnerf360/native/garden --output runs/x --steps 37000 --recipe 3dgs --num-gaussians 10000000
tools/ssenv python reproductions/trimgs/plugin/smoke_schedule.py trimgs <data> <out>   # compressed schedule
```

## Documentation notes (post-review)
- **Assumption, not established:** we treat our 30k flat pre-train (`trimgs --steps 30000`) as a stand-in for the paper's "3DGS*". The paper calls 3DGS* "our re-implementation" and says it fine-tunes from 3DGS pretrained models. The official scripts pre-train with `FlattenGaussianModel` (flat 3DGS). Our pre-train uses gsplat `DefaultStrategy` (as in our baseline), not the 3DGS densifier, so it may differ from the authors' 3DGS*.
- **Contribution weight deviation:** paper Eq. 5 uses α^γ·T^(1−γ). The official kernel computes α^(2γ)·T^(2−2γ) (the paper's term squared; at γ=0.5 this is α·T). We follow the code.
- **Top-5 views quirk depends on view order:** the official list only inserts a view that beats the current maximum and is initialised with K copies of the first view. It therefore holds the last K record-breaking values, not a true top-K, and the result depends on the order the training views are visited. The official order comes from Scene's seeded `random.shuffle`. Ours is a `random.Random(0)` shuffle of SplatStudio's train list, so the per-Gaussian values (and which Gaussians get trimmed) can differ from the official run.
