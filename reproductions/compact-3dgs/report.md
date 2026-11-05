# Compact 3D Gaussian Representation (Lee et al., CVPR 2024): native SplatStudio reproduction

**Outcome: matched.** The decoded Ours+PP model is within ±0.5 dB of the paper on all 3 Mip-NeRF 360 scenes:
bicycle −0.10, garden −0.37, bonsai −0.04 dB.

- **Storage:** our models keep 7–35 % more Gaussians than the paper's, so storage is larger. Ours+PP is
  54.0 / 43.3 / 19.7 MB against the paper's 42.4 / 43.3 / 16.4 MB.
- **Main gaps:**
  - fp32 hash grid and MLP (tcnn trains in fp16);
  - white background and gsplat rasterizer (runtime);
  - global state reaches ss-eval through Gaussian fields.

A matched PSNR is not proof of fidelity. All deviations are listed below.

- **Paper:** arXiv 2311.13681.
- **Official code:** https://github.com/maincold2/Compact-3DGS. It was read, not imported; the pinned
  vector-quantize-pytorch 1.8.1, dahuffman 0.4.1 and tiny-cuda-nn semantics were ported.
- **Plugin:** `plugin/` (method `compact-3dgs`, Ours+PP; also `compact-3dgs-nopp`).
- **Runs:**
  - this method: `runs/compact-3dgs-mipnerf360/compact-3dgs/mipnerf360-{bicycle,garden,bonsai}/`;
  - baseline: `runs/baseline3dgs-mipnerf360-30000/gsplat/`.
- **Protocol:** 30k steps, `--recipe 3dgs`, all SfM points, llffhold 8. Images are images_4 for outdoor
  scenes and images_2 for indoor scenes. Test-split metrics come from **ss-eval on the decoded model**:
  encode → bytes → decode → replace the parameters.

## Results (test split, decoded Ours+PP)

| scene | PSNR ours | PSNR paper PP | Δ | SSIM ours / paper | LPIPS ours / paper | #Gauss ours / paper | train time |
|---|---|---|---|---|---|---|---|
| bicycle | 24.63 | 24.73 | −0.10 | 0.732 / 0.722 | 0.290 / 0.284 | 2.99M / 2.22M (+35 %) | 3h11m |
| garden | 26.35 | 26.72 | −0.37 | 0.835 / 0.831 | 0.172 / 0.158 | 2.36M / 2.21M (+7 %) | 2h53m |
| bonsai | 31.94 | 31.98 | −0.04 | 0.935 / 0.937 | 0.267 / 0.193 | 0.78M / 0.60M (+30 %) | 1h59m |

The paper's no-PP ("Ours") PSNR is 24.77 / 26.81 / 32.08. The paper's PP costs 0.04–0.10 dB.

- **SSIM** is close to the paper's. It is not strictly comparable, because the runtime uses valid-convolution
  SSIM.
- **LPIPS** is worse, most on bonsai (+0.074). The gsplat baseline shows the same pattern: our LPIPS-VGG
  setup on images_2 indoor scenes reads higher than the published numbers (see `_calibration`).
- **Training time** was measured on shared, fully loaded A6000s. It is far slower than the paper's
  24–45 min on an A100. The main costs are the fp32 hash grid with atomic-add backward on 12.6 M
  parameters every step, and the MLP over all Gaussians.

### Relative to the 3DGS baseline

| scene | ours − our gsplat | paper PP − paper 3DGS* | paper no-PP − 3DGS* |
|---|---|---|---|
| bicycle | 24.63 − 24.92 = **−0.29** | 24.73 − 25.10 = −0.37 | −0.33 |
| garden | 26.35 − 26.89 = **−0.55** | 26.72 − 27.25 = −0.53 | −0.44 |
| bonsai | 31.94 − 31.89 = **+0.05** | 31.98 − 32.16 = −0.18 | −0.08 |

The quality cost relative to the baseline matches the paper's own 3DGS* rows within 0.23 dB on every scene
(paper 3DGS* = the authors' retrained 3DGS). Our gsplat baseline has 6.59M / 4.76M / 1.37M Gaussians.
Ours has 45 % / 50 % / 57 % of those; the paper has 39 % / 39 % / 48 % of its 3DGS* counts.

## Storage

MB = 10^6 bytes, by the official `final_prune` formulas. Ours+PP excludes Huffman tables, as the paper does.
"Container" is the real encoded byte stream that was decoded for evaluation.

| scene | Ours (no PP) ours / paper | Ours+PP ours / paper | container (real) | baseline 3DGS* (paper) |
|---|---|---|---|---|
| bicycle | 76.05 / 62.99 | **53.99 / 42.42** | 53.995 MB | 1350.78 MB |
| garden | 65.37 / 62.78 | **43.35 / 43.26** | 43.351 MB | 1331.33 MB |
| bonsai | 38.50 / 35.44 | **19.67 / 16.40** | 19.678 MB | 295.08 MB |

### Ours+PP breakdown (MB)

| scene | position | opacity | scale | rotation | hash | MLP |
|---|---|---|---|---|---|---|
| bicycle | 17.94 | 2.55 | 12.45 | 13.40 | 7.64 | 0.016 |
| garden | 14.17 | 2.12 | 9.81 | 10.54 | 6.69 | 0.016 |
| bonsai | 4.69 | 0.70 | 3.26 | 3.48 | 7.53 | 0.016 |

Paper Mip-NeRF 360 average for comparison: position 8.3, opacity 1.2, scale 5.9, rotation 6.2, hash 7.4, MLP 0.016.

The per-Gaussian terms scale with N, so the storage excess follows the Gaussian excess:

- bicycle: +35 % Gaussians, +27 % storage;
- bonsai: +30 % Gaussians, +20 % storage;
- garden: +7 % Gaussians, +0.2 % storage.

The fixed terms match the paper:

- hash 25.2 MB uncompressed, 6.7–7.6 MB after PP (paper 7.4);
- MLP 0.016 MB;
- codebooks 4.6 KB + 6.1 KB.

The real container is within 4 KB of the formula (table and header overhead). The repository's newer
`point_cloud_pp.npz` sizes (Morton sort + npz compression) are smaller than the paper's numbers and were
not targeted.

**G-PCC:** not used. The CVPR paper and the official code store positions as fp16; G-PCC position coding
appears only in the journal extension (arXiv 2408.03822). The reported sizes are therefore the paper's own
non-G-PCC numbers, so no external binary was needed.

### Mask pruning

| scene | peak N | N at 15k (densify end) | final N | removed after 15k | cumulative mask-pruned rows* |
|---|---|---|---|---|---|
| bicycle | 4.96M | 4.91M | 2.99M | 39 % | 6.18M |
| garden | 4.20M | 4.05M | 2.36M | 42 % | 4.27M |
| bonsai | 1.36M | 1.33M | 0.78M | 41 % | 1.01M |

\* The cumulative count sums every pruning event, including Gaussians pruned during densification that
had been created by clones and splits.

The final prune at step 30000 removed nothing: masks had already been pruned at step 30000 by the regular
1000-step schedule. The final count is 60 % / 56 % / 57 % of the peak.

**Likely cause of the extra Gaussians: unverified.** Candidates include:

- update order: the runtime steps the per-Gaussian Adam before densification;
- gsplat's densification statistics and the absent 2D screen-size pruning (gsplat default; shared with the
  baseline, which also has more Gaussians than 3DGS);
- the white background;
- fp32 rather than fp16 colour-field gradients.

λ_m was not tuned. It stays at the paper's 5e-4, because tuning would be hyper-parameter fishing.

## Mechanisms and probes (`evidence/verify.json`: declared_checks_passed)

| contribution | implementation | probes |
|---|---|---|
| C1 volume mask (Eq. 1–3) | `c3dgs_mask` field (init 1, lr 0.01); straight-through mask on scale and opacity (ε 0.01); `mask-loss` 5e-4·mean σ; prune at densify steps (500<it<15k, every 100) and every 1000 afterwards | volume-mask (masked Gaussians contribute nothing; STE gradient = analytic; loss value/gradient; schedule), net-schedule, state-transitions |
| C2 neural colour (Eq. 7–8) | `CompactBank` (owned by the density, read via `model.ACTIVE`): tcnn-convention hash grid (16 levels × 2 features, 2^19, base 16, ×1.447; custom CUDA) at contract(means.detach()); tcnn SH degree-3 encoding (9 dims, official 2d−1 quirk); 2×64 ReLU MLP, no bias, inputs padded with ones; output SH0; Adam 0.01 with warm-up and ×0.33 at 5k/15k/25k | hash-grid (CUDA = independent int64 reference, values and gradients; 6,299,960×2 parameters), neural-color, net-schedule, train-render-gradient |
| C3 R-VQ (Eq. 4–6) | 6 stages × 64 codes for exp(scale) and the normalised quaternion, steps 29001–30000; 1-iteration K-means init; in-place Adam(1e-4) codebook step; straight-through (L× gradient, as in vector-quantize-pytorch) | rvq (K-means, cascade, Adam step, STE, last-1000-iteration window, renderer uses VQ geometry) |
| C4 PP / storage | fp16 xyz; R-VQ indices + fp32 codebooks; 8-bit min-max opacity; 8-bit hash with \|θ\|≤0.1 pruning; Huffman with dahuffman-port code lengths (canonical container); fp16 MLP → bytes → decode → replace the parameters; hash features at the decoded positions → `c3dgs_feature`; MLP → `c3dgs_blob` | huffman-codec (optimal cost incl. EOF, lossless, post_quant), bake-eval (fresh bank-free renderer after a checkpoint round trip = baked render; size formulas) |

Mutation checks:

- turning off the SH quirk fails neural-color;
- forcing the mask to 1 fails volume-mask;
- `rvq_iter=0` now fails rvq (that probe was strengthened after it initially passed).

Smoke test (2000-step compressed schedule, garden): 23.02 PSNR against 22.97 for gsplat. The
reloaded-checkpoint evaluation equals the validation PSNR. Details are in `evidence/test.json`.

## Fidelity gaps

1. **Precision.** tcnn trains the hash grid and MLP with fp16 parameters and computes in fp16. Here
   parameters and compute are fp32. Stored values are fp16/8-bit as in the paper, so storage is unaffected;
   the training dynamics differ slightly.
2. **Quaternion normalisation.** Training and the encoder normalise q before R-VQ, as the official
   `get_rotation` does. The VQ output (the sum of codes) is not renormalised before rasterisation, in either
   version: gsplat and the 3DGS rasteriser both normalise internally, so this is equivalent. The stored
   rotation is the raw code sum, as in the official `final_prune`.
   - Scales are stored as log|ŝ| of the activated VQ scale (exact, because the rasteriser uses s²).
   - Decoded opacity is stored as logit(o), clamped to [1e-7, 1−1e-7].
3. **Runtime differences.**
   - White background (official: black).
   - Views are sampled at random with replacement.
   - SSIM uses valid convolution.
   - Rasteriser: gsplat, not the 3DGS rasteriser (default: no 2D screen-size pruning).
   - Update order: the per-Gaussian Adam steps before densification and also at step 30000. Only the net
     step is skipped at 30000, as in the official code.
4. **Global-state transport.** The MLP weights travel in the `c3dgs_blob` Gaussian field and the baked hash
   features in `c3dgs_feature`. These are eval-only and never optimised; the HAC precedent is in its
   `plan.md`. The hash table itself is counted in storage but not shipped to ss-eval: test rendering uses
   precomputed features, exactly as the official `precompute()` path does.
5. **tcnn MLP initialisation.** Xavier-uniform on tcnn's padded shapes, with the input padded by ones; the
   forward function is identical to tcnn's. The SH and hash conventions were verified against tcnn sources.
6. **Morton sort omitted.** It only reorders and affects no reported number.
7. **Hardware.** A6000s, shared and loaded, vs the paper's A100. Training-time and FPS comparisons are
   meaningless here.
8. **Run length.** Encode and bake happen only at `total_steps` (30000); other lengths raise at step 1.
9. **Coverage.** Only 3 of 9 Mip-NeRF 360 scenes were run, so there is no dataset-average claim. Tanks &
   Temples and Deep Blending were not run.

## Reproduce
```bash
cd /home/sejain/splatify
tools/ssenv pip install -e reproductions/compact-3dgs/plugin
tools/ssenv ss-verify compact-3dgs --device cuda --output reproductions/compact-3dgs/evidence/verify.json
for s in bicycle garden bonsai; do
  tools/ssenv ss-train compact-3dgs --data ~/datasets/mipnerf360/native/$s --output runs/c3/$s --steps 30000 \
    --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
  tools/ssenv ss-eval --run runs/c3/$s --split test
  tools/ssenv python reproductions/compact-3dgs/plugin/report_size.py runs/c3/$s   # -> compact_size.json
done
# (benchmark actually run via tools/bench.py with bench_request.json)
tools/ssenv python reproductions/compact-3dgs/plugin/smoke_schedule.py ~/datasets/mipnerf360/native/garden /tmp/smoke 2000
```
