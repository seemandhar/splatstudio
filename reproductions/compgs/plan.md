# CompGS native plan

Method `compgs` (module `ss_compgs`), plus diagnostic `compgs-smoke` (compressed schedule for the
2000-step smoke test only). One native 30000-step run = official train.py schedule.

## Components
| kind | name | factory | notes |
|---|---|---|---|
| renderer | compgs-renderer | ss_compgs.renderer:create_renderer | gsplat Renderer; while a VQ state is active renders `x - x.detach() + C[idx]` (STE) for sh0, shN, scales(log), quats(raw) — C1 |
| density | compgs-density | ss_compgs.density:create_density | wraps SplatStudio `Density` (3DGS + opacity-reset fix, unchanged until 15000); opacity prune σ<0.005 at 15000<step≤20000, step%1000==0 (C2); k-means assignment / per-step centroid averaging (C1); decode-and-bake + size report at total_steps (C3) |
| loss | photometric | splatstudio.losses:create_photometric | unchanged L1 + 0.2 D-SSIM |
| loss | compgs-opacity-reg | ss_compgs.losses:create_opacity_reg | λ Σ sigmoid(o), 15000 ≤ step ≤ 20000 (C2) |

No new FieldSpecs. VQ state (codebooks, indices) lives in a module-global `ACTIVE` shared by the
density (owner) and renderer, and is stored in `density.state` for the checkpoint; after the bake,
params hold the decoded model, so `ss-eval` (fresh process, no VQ state) renders the decoded model.

## Step mapping (runtime: render(step) → backward → opt.step → after_backward(step))
`after_backward(step)` prepares quantisation for render(step+1) — official does it at the top of
iteration it=step+1 on the post-step params, so identical:
- it = step+1 > vq_start: if it % freq(it) == 1 → k-means assignment (iters Lloyd + final assign),
  else centroid averaging with cached indices. freq(it) = 100 (5000 for it > total-5000).
- step == total_steps: bake params ← C[idx] (centroids used for render(total_steps); no further
  update, as official saves them), deactivate VQ, record size.
- prune happens before the assignment in the same call (prune at 20000 → assign for 20001).

## Probes
| probe | checks (fails if mechanism removed) |
|---|---|
| kmeans-official | assign/update against an independent brute-force Lloyd implementation; empty cluster → 0; warm start |
| ste-render | renderer output equals render of decoded C[idx] params; grads on raw params equal grads w.r.t. quantised values (STE) |
| schedule | assignment its = {vq_start+1, +101, …, 24901, 25001}; centroid updates on all other its; prune its {16k..20k}; reg window |
| opacity-reg | loss = λ Σσ inside window, 0 outside; gradient = λ σ(1-σ) |
| pipeline | tiny CUDA run through the density: prune, assign, bake → params exactly on codebooks, ≤K distinct rows, size formula |

## Fidelity gaps
1. Runtime protocol: white background, random-with-replacement views, gsplat rasteriser (shared baseline gaps).
2. Centroid averaging via index_add instead of the official padded gather (same math, float summation order).
3. Prune after (not before) the optimizer step of the same iteration (one Adam step on pruned Gaussians; negligible).
4. Last iteration: runtime still steps means/opacity at 30000 (official skips the last step); decoded attributes are baked from the centroids used at it 30000 like the official.
5. Ambiguous hyper-parameters (VQ start 20K vs 15K, Lloyd iters 1/5/10, SH codebook 4096/512) follow the paper / the headline-matching table; configurable.
6. Size: paper formula (fp32 pos+opacity, log2 K-bit indices, RLE on rotation index, fp32 codebooks) computed analytically; not written as the official file set.
