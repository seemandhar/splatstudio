# splatstudio-method-compact-3dgs

Native SplatStudio port of **Compact 3D Gaussian Representation for Radiance Field** (Lee, Rho, Sun,
Ko, Park — CVPR 2024, arXiv 2311.13681). Official code: https://github.com/maincold2/Compact-3DGS
(read, not imported; Inria 3DGS licence). Paper fidelity is unverified.

Methods: `compact-3dgs` (Ours+PP, the evaluated default) and `compact-3dgs-nopp` (Ours).

## Mechanism → components
- **Volume mask (C1, Eq. 1–3)** — field `c3dgs_mask` (init 1, lr 0.01). `compact-renderer` multiplies scale
  and opacity by the straight-through mask `sg(1[σ(m)>0.01]−σ(m))+σ(m)`; loss `mask-loss` = 5e-4·mean σ(m);
  `compact-density` prunes σ(m) ≤ 0.01 at every densification step (500<it<15000, every 100) and every 1000
  steps afterwards, and at the end (official `densify_and_prune` / `mask_prune` / `final_prune`).
- **Neural view-dependent colour (C2, Eq. 7–8)** — `model.CompactBank` (owned by the density, read by the
  renderer through `model.ACTIVE`, EAGLES DecoderBank pattern): tcnn-convention hash grid (16 levels × 2 features,
  2^19, base 16, ×1.447; CUDA kernel in `csrc/hashgrid.cu`, conventions adapted from tiny-cuda-nn, BSD-3) at
  the contracted, detached position; tcnn SH degree-3 (9-dim) encoding of the view direction (with the official
  2d−1 input quirk); tcnn-style 2×64 ReLU MLP without biases (inputs padded with ones). The output is the SH-0
  coefficient (gsplat `sh_degree=0`). Net Adam lr 0.01 (eps 1e-15), warm-up ×0.01→1 over 100 steps, ×0.33 at
  5k/15k/25k, no step at the last iteration.
- **Geometry codebook (C3, Eq. 4–6)** — two 6-stage × 64-code residual VQs (port of vector-quantize-pytorch
  1.8.1 as configured officially: K-means init with 1 iteration, in-place Adam(1e-4) codebook step on the mse in
  every training forward, re-quantise, straight-through). They are applied to exp(scale) and the normalised
  quaternion for steps 29001–30000.
- **Storage / PP (C4)** — at step 30000 the density encodes the model into a byte container (fp16 xyz, R-VQ
  indices + fp32 codebooks, 8-bit min–max opacity and hash with |θ|≤0.1 pruning, Huffman with dahuffman's code
  lengths, fp16 MLP). It then decodes the container and replaces the parameters with the decoded model
  (Reduced-3DGS pattern).
  - Eval-time state reaches `ss-eval` in Gaussian fields: `c3dgs_feature` holds the hash features at the
    decoded positions (the official `precompute()`), and `c3dgs_blob` holds the MLP weights (HAC pattern).
  - Sizes are computed with the official `final_prune` formulas (MB = 1e6 B). Both "Ours" and "Ours+PP" are
    reported, together with the actual container bytes. Read them with `python report_size.py <run>`.
  - G-PCC is not used by this paper (only by its journal extension); positions are fp16.

## Commands
```bash
tools/ssenv pip install -e reproductions/compact-3dgs/plugin
tools/ssenv ss-verify compact-3dgs --device cuda
tools/ssenv ss-train compact-3dgs --data ~/datasets/mipnerf360/native/garden --output runs/x --steps 30000 \
  --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run runs/x --split test
tools/ssenv python reproductions/compact-3dgs/plugin/report_size.py runs/x
# diagnostic: every phase compressed into N steps
tools/ssenv python reproductions/compact-3dgs/plugin/smoke_schedule.py <data> <output> 2000
```
`--steps` must equal the density's `total_steps` (30000). Other run lengths raise at step 1, because the encode
and bake would never happen.

## Fidelity gaps (see ../plan.md)
- white background, random view sampling and valid-SSIM (runtime);
- gsplat rasteriser;
- fp32 training of the hash grid and MLP (tcnn uses fp16);
- update order (the runtime steps the per-Gaussian Adam before densification, and at step 30000);
- MLP init approximates tcnn's;
- global state reaches ss-eval through `c3dgs_feature` / `c3dgs_blob`;
- Morton sorting before encoding is omitted (it only reorders and affects no reported number).
