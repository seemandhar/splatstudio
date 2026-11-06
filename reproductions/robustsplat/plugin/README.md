# splatstudio-method-robustsplat

Native SplatStudio port of **RobustSplat: Decoupling Densification and Dynamics for
Transient-Free 3DGS** (Fu et al., ICCV 2025, arXiv 2506.02751), following the official code
github.com/fcyycf/RobustSplat @ a130281 (code values override the paper where they differ).

Methods: `robustsplat` (full), `robustsplat-nodg` (ablation 3DGS+Mask+MB),
`robustsplat-nomb` (ablation 3DGS+Mask+DG).

## Mechanism → components
| contribution | component | key values (source) |
|---|---|---|
| Delayed Gaussian growth (C1, §4.3, supp.) | `rs-density` | densify 10000 < t < 20000 every 100, grad 2e-4, percent_dense 0.01, prune σ < 0.005, no size pruning, opacity reset min(σ,0.01) at 15000/18000, opacity lr 0.025 (code) |
| Scale-cascaded mask bootstrapping (C2, §4.4) | `rs-mask-mlp` | t < 20000 (code `bootstrap_iter`; paper text implies 10000): residual from images_{4f} render vs GT, DINOv2 16×16 (224²); then training-res residual, 36×36 (504²) |
| Mask MLP + supervision (Eq. 3–6, supp. S1) | `rs-masked-photometric`, `rs-mask-mlp` | 384–16–1 MLP, Adam 1e-3; mask > 0.25, 7×7 erosion; λ_res = λ_cos = 0.5; τ 0.6 / 0.8; 10000-bin histogram, decay 0.95 |
| Early mask regularisation (Eq. 7–8) | `rs-mask-mlp` | λ_reg 2.0, β_reg 2000; MLP paused 300 steps after t ≡ 0 mod 3000 for t ≥ 15000 (code) |

Renderer `rs-renderer`: gsplat, black background, near plane 0.2 (Inria). DINOv2 ViT-S/14-reg
(`~/models/dinov2/dinov2_vits14_reg4_pretrain.pth`, sha256-checked) re-implemented in `dino.py`;
parity with timm `vit_small_patch14_reg4_dinov2` ≤ 7e-5 (probe `dino-parity`).

## Data
Scenes must be prepared with `reproductions/robustsplat/tools/prepare_robustsplat.py` from the
authors' release (hf.co/datasets/fcy99/RobustSplat-data); it adds the `coarse/` views and
per-frame `coarse_*` intrinsics. Registered datasets: `robustsplat-onthego` (patio ×4,
patio-high ×8), `robustsplat-robustnerf` (android, crab2 ×8).

```bash
tools/ssenv ss-train robustsplat --data ~/datasets/robustsplat/native/patio-high \
  --output runs/x --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run runs/x --split test
```

## Fidelity gaps
Runtime order (optimizer step before densification), view sampling with replacement, SH degree
increments one step later, evaluation SSIM (valid conv) and LPIPS (`lpips` VGG) from the runtime,
constant `get_regularizer` term omitted (no gradient in the official code), histogram with
`torch.histc` on GPU (official `torch.histogram` on CPU; bin-edge ties only), single seed.
