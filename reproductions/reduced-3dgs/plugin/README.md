# splatstudio-method-reduced-3dgs

Native SplatStudio port of **Reducing the Memory Footprint of 3D Gaussian Splatting**
(Papantonakis, Kopanas, Kerbl, Lanvin, Drettakis — Proc. ACM CGIT 7(1), I3D 2024,
doi:10.1145/3651282; project page https://repo-sam.inria.fr/fungraph/reduced_3dgs/; official
code https://github.com/graphdeco-inria/reduced-3dgs @ dc905c1). Paper fidelity unverified.

Method name: `reduced-3dgs` (official `full_eval.py` configuration `full_final`).

## Mechanism → code
| paper | code |
|---|---|
| §4.1 redundancy score: min pixel footprint over training views, sphere r = a·√3/2, 30-NN ellipsoid test (axes + r), min propagation | `redundancy.py` (`redundancy_scores`), CUDA exact kNN `csrc/knn.cu` (official simple-knn `boxKnn2`) |
| §4.1 prune: candidates score > max(μ+σ, 3) below median opacity, plus opacity < min(q₀.₀₃, 0.05); every 1000 its ≤ 27k, not on reset steps < 15k | `redundancy.mercy_mask`, `density.schedule_events` |
| §4.1 opacity L1 (0.001), dead-point prune σ(o) < 1/255 after 15k / at 7k, 30k | `losses.OpacityL1`, `density.py` |
| §4.2 Eq. 1–2 transmittance-weighted colour mean/std, band distances; culling at 15k (std < 0.04 → 0 bands; distance < 6√3/255 → 2 or 1 bands) | `shcull.py`, CUDA `csrc/transmittance.cu` (ΣT, #pixels on gsplat tile lists) |
| §4.2 SH sparsity (0.1; official CUDA backward λ/(45V)·sign) | `losses.SHSparsity` |
| per-Gaussian band count (`_degrees`) | field `sh_degree`; `renderer.py` masks the dropped coefficients |
| §4.3 256-entry 1-D K-means codebooks (opacity, scale, rot-re, rot-im, DC, 15 SH-rest), float16 positions + codebook entries | `codec.py` (K-means assignment kernel `csrc/kmeans.cu` from the official code) |
| decoded evaluation | at the final step `density._encode` writes the official `point_cloud_quantised_half.ply` byte layout (plyfile, in memory), decodes it as `load_ply(quantised, half_float)` does and replaces the parameters, so the checkpoint, `validation.json` and `ss-eval` measure the decoded model |
| reported size ("Mem", MB) | `codec.paper_memory_bytes` = `generate_results.py:memory_results`, MiB; plus the real PLY byte count; stored in the checkpoint at `density_state.reduced3dgs.encoded` and printed in the training log |

## Config defaults (sources)
`full_eval.py` `full_final` + `arguments/__init__.py`: lambda_sh_sparsity 0.1, cull_SH 15000,
std_threshold 0.04, cdist_threshold 6 (×√3/255), mercy_points / prune_dead_points on,
lambda_alpha_regul 0.001, mercy_type redundancy_opacity_opacity, lambda_mercy 1,
mercy_minimum 3, mercy_interval 10 × 100, box_size 1, knn 30; 3DGS defaults otherwise.

## Fidelity gaps
1. Native runtime runs density ops after the optimizer step (official: before).
2. gsplat rasterizer numerics (alpha clamp 0.999) vs Inria (0.99); per-Gaussian degree is a
   coefficient mask (identical colours, zero gradient for the dropped coefficients).
3. Shared baseline differences: white background, sampling with replacement, gsplat
   DefaultStrategy without the 20-px screen-size prune.
4. Decoded logit(1)/log(0) codebook entries (±inf) are clamped to ±65504 (renders identically).
5. The encoded PLY is not written into the run directory (the density has no output path); its
   byte length and sha256 are recorded.
6. The official K-means assignment kernel stages centres only from threads with a valid value,
   so the last partial block can read uninitialised centres; the vendored kernel stages all.

## Commands
```bash
tools/ssenv pip install -e reproductions/reduced-3dgs/plugin
tools/ssenv ss-inspect reduced-3dgs
tools/ssenv ss-verify reduced-3dgs --device cuda
tools/ssenv ss-train reduced-3dgs --data ~/datasets/mipnerf360/native/garden --output runs/x \
    --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run runs/x --split test          # decoded model
tools/ssenv python reproductions/reduced-3dgs/plugin/report_size.py runs/x   # encoded MB
tools/ssenv python reproductions/reduced-3dgs/plugin/smoke_schedule.py <data> <out>  # 2k-step all-events smoke
```
The training length must equal `total_steps` (default 30000) for the encode step to run.
