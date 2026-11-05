# splatstudio-method-contextgs

A native SplatStudio port of **ContextGS: Compact 3D Gaussian Splatting with Anchor Level Context Model** (Wang et
al., NeurIPS 2024, arXiv 2405.20721). Written from the paper and the official code (github.com/wyf0912/ContextGS,
commit eff8d19). Paper fidelity is unverified.

The Scaffold-GS / HAC infrastructure (anchor fields, MLP bank, `visible_filter` prefilter, anchor growing and pruning,
encode → decode → bake, MLP weights carried to ss-eval in a per-anchor blob field, vendored CUDA arithmetic coder) is
**copied with attribution** from the HAC native port in this workspace (`reproductions/hac/plugin`). Nothing is
imported from it.

## Mechanism → components

- **Anchors as the population.** Fields: `means` (lr 0), `cgs_feat [50]`, `cgs_offset [10,3]`, `cgs_scaling [6]`,
  `cgs_mask [10,1]`, `cgs_hyper [12]` (hyperprior latent), `cgs_blob [8]`. `cgs-renderer` decodes K=10 neural Gaussians
  per visible anchor with the Scaffold MLPs and rasterizes them with gsplat.
- **Global state.** `model.ContextBank` (own Adam) holds the render MLPs, the factorized hyperprior (`FactorizedPrior`,
  a port of compressai's `EntropyBottleneck`) and the three per-level context MLPs. `cgs-density` creates and steps
  it, and the renderer reads it through `model.ACTIVE`.
- **Anchor partitioning (§4.1, Eq. 4–6), `levels.py`.** Anchors are voxelized at κ_k·ε. A binary search once finds
  the κ_k that give |V^{k+1}|/|V^k| ≈ 0.2. The minimum-index anchor of each voxel represents it, and the levels are
  made disjoint.
- **Anchor-level context + hyperprior (§4.2, Eq. 7–9).** Levels are coded 2 → 1 → 0.
  - The MLP of level k maps `[parent x, f, l (already quantized); ẑ]` (top level: `[x; ẑ]`) to μ, σ and the
    adaptive Q of f, l and O.
  - From step 10001 this runs on all anchors every step, with uniform-noise quantization and 15%-sampled bits.
  - The rate loss is `λ_e · bit_per_param + 5e-4 · mean σ(mask)`.
- **Bitstream (App. A).**
  - At `total_steps`, the hyperprior symbols are coded under the factorized CDF. f, l and O are coded per level under
    the predicted Gaussians. Masks are coded as Bernoulli, anchors as 16-bit integers, MLPs as fp32.
  - Decoding uses only the streams, and is checked to be lossless.
  - The decoded attributes are then baked into the fields, and the render MLPs into `cgs_blob`.
  - Sizes go to `checkpoint["density_state"]["contextgs"]["encoded"]`. Run `python report_size.py <run>` to read them.

## Config (paper/code values)
All values are in `density.DEFAULTS`, with their sources given in comments:
- voxel 0.001 (mip360), K=3, τ=0.2, h_c=4;
- Q0 = 1 / 0.001 / 0.2; noise phase (3000, 10000]; context from step 10001; 15% sample;
- the Scaffold growing schedule;
- every LR schedule.

λ_e = 0.004 (the low-rate rows) is set in `ss_contextgs.method(lmbda=...)`.

## Commands
```bash
tools/ssenv pip install -e reproductions/contextgs/plugin
tools/ssenv ss-verify contextgs --device cuda
tools/ssenv ss-train contextgs --data ~/datasets/mipnerf360/w1600/garden --output runs/x --steps 30000 \
  --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run runs/x --split test
tools/ssenv python reproductions/contextgs/plugin/report_size.py runs/x
# diagnostic: every phase compressed into N steps (bake at N)
tools/ssenv python smoke_schedule.py <data> <output> 2000
```
`--steps` must equal `total_steps` (30000). A shorter run never bakes, and its validation then refuses to render.

## Fidelity gaps
See `../plan.md`. In brief:
- gsplat replaces the Scaffold rasterizer fork; its `visible_filter` is re-implemented.
- The background is white here; the official mip360 runs use black.
- View sampling and SSIM follow the runtime.
- The update order differs.
- The global MLPs travel to ss-eval in a per-anchor field.
- The HAC CUDA coder replaces torchac and compressai rANS. The probability models are the same.
- Only the mip360 voxel size is registered.
- Two official index behaviours are reproduced verbatim, because the reported numbers come from them:
  - the context uses the parent's position;
  - the level-1 context is gathered in original-index order.
