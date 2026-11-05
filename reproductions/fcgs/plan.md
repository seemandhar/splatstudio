# FCGS — native SplatStudio plan

Method names: `fcgs` (λ=1e-4, "Ours-highrate"), `fcgs-lowrate` (λ=16e-4, "Ours-lowrate").
Module `ss_fcgs`. Plugin: `reproductions/fcgs/plugin/`.

FCGS has no training of its own: a pretrained network (official release weights, SHA256
pinned) compresses a finished vanilla 3DGS. Retraining the codec on DL3DV-GS (6.7 K scenes,
60 GPU-days just for the 3DGS corpus) is out of scope and not needed because the weights are
released. The native method therefore = baseline gsplat 3DGS recipe (30 K, opacity reset) +
a post-training codec step inside the density component (same pattern as reduced-3dgs).

## Components
| component | kind | factory | notes |
|---|---|---|---|
| `gsplat` | renderer | splatstudio.backends.gsplat | unchanged |
| `fcgs-density` | density | `ss_fcgs.density:create_density` | wraps baseline `Density` (config `base` = baseline config); at `step == total_steps` (30000) runs `codec.encode_decode` and swaps every param for its decoded value (Adam moments and densification stats reset; training is over) |
| `photometric` | loss | splatstudio.losses | unchanged |

Config (`fcgs-density`): `total_steps 30000`, `lmd 1e-4|16e-4`, `encode true`,
`xyz_coder auto`, `weights_dir null` (→ `~/models/fcgs`), `base {...}`.
A run-length check at step 1 (from the means LR schedule) refuses runs whose length differs
from `total_steps` (the encode would never run).

No new FieldSpecs. Decoding keeps N; Gaussian order is permuted (official sort + seeded
shuffle), harmless for rendering.

## Codec (official code, vendored)
`fcgs_model.py`, `entropy_models.py`, `encodings_cuda.py`, `grid_utils.py` copied from the
official repo with import changes only; CUDA `gridcreater`, `gridencoder`, `freqencoder`
(official FCGS) and `arithmetic` (identical in FCGS and HAC; copied from the HAC plugin) are
JIT-built by `kernels.py`. `codec.py` reproduces `encode_single_scene.py` +
`decode_single_scene.py`: 1 M-Gaussian steps, `determ=1`, chunk sizes [2e6,1e6,1e6], real
bitstream files written to a temp dir and decoded back from disk; decode == encoder-side
reconstruction is checked. Size = Σ bits returned by `compress` / (8·2^20) — exactly the
paper's "MB" (MiB; verified against the CSV/table: 67.16e6 B mean = 64.05).

Positions: official 16-bit voxelisation + GPCC (`tmc3`, official flags) when the binary is
available ($FCGS_TMC3 or PATH). **tmc3 is not installed and building it was not permitted in
this session**, so `xyz_coder: auto` currently selects a substitute lossless coder
(Morton octree occupancy + LZMA). Decoded geometry is bit-identical either way (lossless on
the same grid); only the xyz part of the size differs. The report stores `xyz_coder_used`,
the xyz MB, and attribute-only MB, so the size can be compared component-wise
(paper DL3DV: xyz = 5.92 bits/coordinate ≈ 17.8 bits/Gaussian).

## Probes
| probe | contribution | what can fail it |
|---|---|---|
| official-weights | C1–C4 | weights not the official release |
| grid-context (CUDA) | C2 | grid creator / interpolator ≠ independent Eq. 4/5 torch reference |
| gmm-cdf (CUDA) | C4 | coder CDF ≠ softmax-mixture of Normal CDFs (Eq. 7); lossy decode; bits ≫ ideal |
| mem-and-quantisation (CUDA) | C1, C3, C4 | real encode→decode on a synthetic scene: m=1 colour ≠ g_s(round(g_a(x))), m=0 / geometry ≠ Q(x) with learned steps, positions off 16-bit grid, decode ≠ encoder reconstruction, attributes unchanged |
| context-models (CUDA) | C2, C3 | full-context geometry code length ≥ hyperprior-only ideal bits |
| xyz-lossless | C4 | octree coder not lossless (duplicates) / no compression |
| schedule | C4 | codec not run exactly once at total_steps |

## Offline tool
`plugin/apply_fcgs.py --run <run> --lmd ...` applies FCGS to any finished gsplat run
(e.g. the shared `baseline3dgs-*` checkpoints, which is literally the paper's protocol:
compress an existing 30 K 3DGS) and evaluates the decoded model with SplatStudio's metric
code; used for the smoke evidence and for the other λ values without retraining.

## Benchmark
Per-scene numbers exist (official `results/*.csv`): Mip-NeRF 360 bicycle, garden (outdoor),
counter (indoor); 30 K steps; method `fcgs` (+ optional `fcgs-lowrate`). Compare decoded
PSNR drop vs our own 3DGS (baseline3dgs-mipnerf360-30000 or the same run's pre-encode
model) with the paper's drop (FCGS-highrate − authors' 3DGS: Mip-NeRF 360 avg −0.13 dB).

## Fidelity gaps
1. **GPCC unavailable → substitute xyz coder** (size only; decoded model identical). Expected
   effect: xyz bits larger than GPCC (≈ +several MB on 5 M-Gaussian scenes). Closable: build
   MPEG TMC13 (`tmc3`) and set $FCGS_TMC3; `apply_fcgs.py --xyz-coder gpcc` re-measures.
2. **Base 3DGS differs**: our gsplat 3DGS (white background, gsplat rasteriser, fewer
   Gaussians: garden 4.76 M vs 5.62 M in the paper) vs official 3DGS. FCGS's pretrained
   networks (mask MLP, autoencoder) were trained on official-3DGS parameter statistics;
   gsplat parameters follow the same conventions (logit opacity, log scale, raw wxyz quat,
   SH layout), but the distribution may differ slightly. Affects absolute PSNR and size;
   compare deltas.
3. **Evaluation**: SplatStudio metrics (white background, no 8-bit rounding of renders,
   valid-SSIM) vs official validate script (black background, 8-bit rounding, same-SSIM).
4. Codec training (DL3DV-GS) not reproduced — official weights used (by design).
5. Decoded-model rendering uses gsplat instead of the official 3DGS rasteriser.
