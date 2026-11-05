# Compact-3DGS (Lee et al., CVPR 2024) native plan

Method `compact-3dgs`, module `ss_compact_3dgs`, plugin `reproductions/compact-3dgs/plugin/`.
Precedents copied with attribution, never imported: EAGLES `DecoderBank` + process-local slot
(`reproductions/eagles`), HAC hash-grid CUDA + global weights carried to ss-eval in a per-Gaussian field
(`reproductions/hac/plan.md`, design note), Reduced-3DGS final encode → decode → replace parameters.

## Feasibility
Feasible natively with the same workaround HAC documents (global learned state reaches ss-eval through
Gaussian-owned fields):

1. **Global learned state** (hash grid 12.6 M params, MLP, two R-VQs with their own Adam) lives in
   `model.CompactBank` (nn.Module), created and stepped by `compact-density`; the renderer reads it from
   the process-local slot `model.ACTIVE["bank"]` (EAGLES pattern). Net Adam: lr 0.01, eps 1e-15, closed-form
   ChainedScheduler(LinearLR 0.01→1 over 100, MultiStepLR 0.33 @ 5k/15k/25k) set before every step.
2. **ss-eval.** `runtime.evaluate()` rebuilds components and only hands `checkpoint["params"]` to the
   renderer. At `total_steps` the density bakes the *decoded* model into params:
   - base fields: fp16 positions, `log|R-VQ scale|`, R-VQ rotation, `logit(decoded opacity)`;
   - `c3dgs_feature [32]`: hash-grid features at contract(fp16 position) computed from the **decoded**
     hash table (8-bit + prune under PP) — this is exactly the official inference path
     (`GaussianModel.precompute()` → `render(itr=-1)` uses `pc._feature`);
   - `c3dgs_blob [8]`: header + the MLP weights (fp16-rounded, as stored) in the first ⌈6984/8⌉ rows.
   The renderer, finding no live bank, renders `mlp(cat[feature, SH9(2d−1)])` from these fields.
   The hash table itself (12.6 M floats) cannot fit in an [N,8] blob and is not needed at test time;
   storage counts it exactly as the paper does. Same representation abuse as HAC (gap G6).
3. **Mask, pruning** are per-Gaussian (`c3dgs_mask` field) + `ops.remove`, fully native.

`feasible_natively: true`.

## Components
| component | kind | factory | requires → provides | config (paper/code) |
|---|---|---|---|---|
| compact-renderer | renderer | ss_compact_3dgs.renderer:create_renderer | state.gaussian3d, camera.pinhole → render.rgb, render.depth.expected_z, gradient.screen.signed | mask_threshold 0.01, tcnn_direction_quirk true |
| compact-density | density | ss_compact_3dgs.density:create_density | gradient.screen.signed | base 3DGS (grow 2e-4, refine 500/100/15000, reset 3000), mask_threshold 0.01, mask_prune_every 1000, total_steps 30000, rvq_iter 29000, rvq C 64 L 6, codebook lr 1e-4, kmeans_iters 1, hash (16, 2, 2^19, 16, 1.447), mlp (64, 2), net_lr 0.01, warm-up 100 ×0.01, milestones [5000,15000,25000] ×0.33, comp (PP) true, hash_prune 0.1 |
| photometric | loss | splatstudio.losses:create_photometric | render.rgb | ssim 0.2 |
| mask-loss | loss | ss_compact_3dgs.losses:create_mask_loss | render.rgb | λ_m 5e-4 |

## Fields
| field | shape | lr | init | notes |
|---|---|---|---|---|
| c3dgs_mask | () | 0.01 | 1.0 | official `_mask` (Adam with the 3DGS group) |
| c3dgs_feature | (32,) | 1e-12 (never has a gradient) | 0 | baked hash features (eval only) |
| c3dgs_blob | (8,) | 1e-12 (never has a gradient) | 0 | baked MLP weights (eval only) |
Base `sh0/shN` exist but are unused (no gradient). Densification ops copy all fields.

## Renderer (training, bank live, not baked) — official `gaussian_renderer/__init__.py`
```
σm = sigmoid(mask); M = sg(1[σm>0.01] − σm) + σm                      # Eq. 1
if bank.step > 29000: s = RVQ_s(exp(scales)); q = RVQ_r(normalize(quats))   # Eq. 4–5 (+ in-place codebook Adam, STE)
else: s = exp(scales); q = normalize(quats)
ŝ = s·M, ô = sigmoid(opac)·M                                           # Eq. 2
feat = HashGrid(contract(means.detach())/4 + 0.5)                      # Eq. 7–8, tcnn conventions, CUDA
dir = (means − c)/||·||;  enc = SH_tcnn_deg3(2·dir − 1)                # 9 dims, official quirk
sh0 = MLP(cat[feat, enc]) → gsplat sh_degree 0 (rgb = max(C0·sh0+0.5, 0))
```
Baked / ss-eval: `ŝ = exp(scales)`, `q`, `ô = sigmoid(opac)` as stored, `feat = c3dgs_feature`, MLP from blob.

## Density (after each runtime optimizer step)
1. net Adam step (skipped at total_steps, official), lr for the next step.
2. 3DGS density (`splatstudio.backends.gsplat.Density`, opacity-reset fix).
3. mask prune σ(m) ≤ 0.01: at refine steps (500 < step < 15000, step % 100 == 0) and every 1000 steps
   from 15000.
4. step == total_steps: final prune → encode (fp16 xyz, R-VQ indices + codebooks, 8-bit opacity, 8-bit +
   pruned hash, Huffman, fp16 MLP) into a byte container → decode the container → bake (above).
   Stored: `density_state["compact3dgs"]["encoded"]` with the paper formula sizes (Ours and Ours+PP) and the
   actual container bytes. `report_size.py` writes `<run>/compact_size.json`.

## Probes
| probe | fails if… |
|---|---|
| hash-grid (CUDA) | CUDA grid ≠ independent torch reference of tcnn grid_scale/resolution/dense-index/coherent-prime-hash/trilinear (values, param gradients); level table ≠ tcnn param count 6,299,960×2 |
| neural-color (CUDA) | renderer colour ≠ C0·MLP(cat[hash(contract(p)), SH9(2d−1)])+0.5 computed independently; contraction ≠ paper Eq. 8; colour not view dependent; tcnn SH ≠ closed-form real SH table |
| volume-mask (CUDA) | masked Gaussian (σ<0.01) contributes; mask gradient ≠ STE analytic (∂L/∂ŝ·s + ∂L/∂ô·o)σ'(m); mask loss ≠ λ mean σ; prune schedule ≠ official |
| rvq | R-VQ ≠ independent loop (K-means 1 iter init, nearest code, residual cascade), STE grad ≠ L× upstream, in-place Adam update ≠ manual Adam on mse, R-VQ active only for step > 29000 |
| huffman-codec | code lengths ≠ dahuffman port & textbook Huffman cost; canonical encode/decode not lossless; post_quant ≠ official (min/max before prune) |
| net-schedule | lr closed form ≠ torch ChainedScheduler replay; mask prune / R-VQ / encode steps ≠ official loop |
| bake-eval (CUDA) | after encode→decode→bake on a fixture, a fresh bank-free renderer (checkpoint round trip) ≠ baked render; storage formula ≠ official final_prune; un-baked state renders |
| train-render-gradient (CUDA) | live-bank render lacks finite non-zero screen gradients, hash/MLP gradients |

## Fidelity gaps
1. **G1 Background / sampling / SSIM**: white bg, random-with-replacement views, valid SSIM (runtime); official black bg. Not closable without runtime change.
2. **G2 Rasterizer**: gsplat vs 3DGS diff-gaussian-rasterization (no 2D screen-size pruning in gsplat default; shared with baseline).
3. **G3 Precision**: tcnn trains hash grid + MLP in fp16 (fp16 params); here fp32 params/compute. Stored model is fp16/8-bit as in the paper.
4. **G4 Update order**: runtime steps per-Gaussian Adam before density; official densifies first and skips *all* optimizer steps at 30000 (only the net step is skipped here).
5. **G5 Resolution/GPU**: our images_4/images_2 (3DGS convention, same as paper); A6000 vs A100.
6. **G6 Global-state transport**: MLP in `c3dgs_blob`, baked hash features in `c3dgs_feature`.
7. **G7 tcnn MLP padding**: first-layer bias = 7 padded ones of tcnn (exact functional equivalent); init approximates tcnn's xavier on padded shapes; storage uses tcnn's padded count 8192 (paper formula).
8. **G8 Run length**: bake happens at `total_steps`; a shorter `--steps` raises at step 1 (run-length check copied from Reduced-3DGS); `smoke_schedule.py` compresses the schedule for diagnostics.
9. **G9 Codec container**: paper size = formula (tables excluded); actual container uses canonical Huffman with dahuffman's code lengths (identical payload bits) + tables; reported separately.
