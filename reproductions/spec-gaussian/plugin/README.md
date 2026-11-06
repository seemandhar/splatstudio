# splatstudio-method-spec-gaussian

Native SplatStudio port of **Spec-Gaussian: Anisotropic View-Dependent Appearance for 3D
Gaussian Splatting** (Yang et al., NeurIPS 2024, arXiv 2402.15870), following the official
real-scene code path (github.com/ingra14m/Specular-Gaussians @0b40ce3, MIT): `train.py
--is_real --asg_degree 12 [--is_indoor]`, which produced the paper's "Ours" rows.

## Mechanism
- Per-Gaussian feature `spec_feat` (12, lr 0.0025, init 0).
- Global decoder (process-local bank, own Adam eps 1e-15, lr linear 0.0025 → 0.000125 / 30k):
  Θ = Linear(12 → 8·4) gives ξ∈R², λ, μ for N = 8 ASGs on fixed anisotropic axes
  (`init_predefined_omega(2,4)`); κ = ⊕ ξ·max(d·z,0)·exp(−softplus(λ−1)(d·x)² − softplus(μ−1)(d·y)²);
  Ψ = MLP([κ, d, PE₂(d)] (31) → 32 → 3), indoor variant has one more 32-wide hidden layer.
- Colour `clamp_min(SH(d)+0.5, 0) + c_s`, passed to gsplat as precomputed colours; `c_s = 0`
  until step 3000.
- Densification: 3DGS ADC on absolute (L1) screen gradients (gsplat absgrad = official
  `fabsf` kernel), τ_g = 0.0005; opacity reset every 3000; xyz lr with spatial_lr_scale 5.
- The decoder is packed into the `spec_blob` field (width 4) after every step, so ss-eval renders
  from the checkpoint without the bank.

## Variants
| method | decoder | resolution coarse-to-fine | scenes |
|---|---|---|---|
| `spec-gaussian` | outdoor | off (official) | bicycle flowers garden stump treehill |
| `spec-gaussian-indoor` | indoor | off (official) | room counter kitchen bonsai |
| `spec-gaussian-c2f`, `spec-gaussian-indoor-c2f` | as above | paper §3.3: 1/4 → 1 over 5k | paper-text ablation |

## Fidelity gaps
White background; no 3DGS 20-px screen-size prune (gsplat default, as in our baseline);
step-then-densify order; SH degree steps at 1001/2001/3001; views sampled with replacement;
reflection direction / normals and the paper-text N=32 / f∈R²⁴ / 3×64 decoder are not used
because the official real-scene code does not use them; resolution c2f off in the main variants
(absent from official `train.py`).

## Commands
```bash
tools/ssenv pip install -e reproductions/spec-gaussian/plugin
tools/ssenv ss-verify spec-gaussian --device cuda
tools/ssenv ss-train spec-gaussian --data ~/datasets/mipnerf360/pilr/garden --output runs/x \
  --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-train spec-gaussian-indoor --data ~/datasets/mipnerf360/pilr/counter ...
```
