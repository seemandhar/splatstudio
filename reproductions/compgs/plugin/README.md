# splatstudio-method-compgs

Native SplatStudio port of **CompGS: Smaller and Faster Gaussian Splatting with Vector
Quantization** (Navaneet, Meibodi, Koohpayegani, Pirsiavash — ECCV 2024, arXiv 2311.18159),
ported from the official code https://github.com/UCDvision/compact3d (MIT). Paper fidelity unverified.

Variant: **CompGS 32K** with opacity regularisation (supp. per-scene table).

| mechanism | implementation | value (source) |
|---|---|---|
| 3DGS training | unchanged SplatStudio `Density` (gsplat DefaultStrategy + opacity reset fix) | paper §4 |
| opacity L1 | `compgs-opacity-reg`: λ Σ sigmoid(o), 15000 ≤ it ≤ 20000 | λ=1e-7 (§4, train_kmeans.py:150) |
| opacity prune | σ<0.005 every 1000 its, 15000<it≤20000 (`ops.remove`) | train_kmeans.py:203 |
| k-means VQ | 4 codebooks DC 4096 / SH 4096 / log-scale 32768 / raw quat 32768; STE in the renderer | §4, kmeans_quantize.py |
| schedule | VQ from it 20001; assignment (5 Lloyd its) when it%100==1 until 25001, centroid averaging every other it | §4, Table perf_vs_time, train_kmeans.py:124 |
| decoded eval | at 30000 params ← C[idx]; ss-eval renders the decoded model | §3 |
| size | `report_size.py <run>`: paper formula + official on-disk formula | §3 |

Ambiguities (paper vs released run.sh): VQ start 20K vs 15K, Lloyd iterations 1 (text) / 5
(table matching headline) / 10 (run.sh), SH codebook 4096 vs 512 — paper values used, all are
config keys of `compgs-density`.

Fidelity gaps: white background / random view sampling / gsplat rasteriser (runtime);
index_add centroid averaging (same math); prune after rather than before the optimizer step;
means/opacity get one extra Adam step at 30000; size computed analytically (no files written).

```bash
tools/ssenv ss-train compgs --data ~/datasets/mipnerf360/native/garden --output runs/x --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run runs/x --split test
tools/ssenv python reproductions/compgs/plugin/report_size.py runs/x
```
`--steps` must equal `total_steps` (30000); a guard raises otherwise. `compgs-smoke` is a
2000-step diagnostic schedule only.
