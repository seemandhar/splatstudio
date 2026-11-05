# splatstudio-method-c3dgs

Native SplatStudio port of **Compressed 3D Gaussian Splatting for Accelerated Novel View
Synthesis** (Niedermayr, Stumpfegger, Westermann, CVPR 2024, arXiv 2401.02436), written against
the paper and the official code https://github.com/KeKsBoTer/c3dgs (Gaussian-Splatting licence;
the `weighted_distance` CUDA kernel is copied in `ss_c3dgs/csrc/` with its licence). Paper fidelity
unverified.

## Mechanism (one 35000-step run)
| steps | what | code |
|---|---|---|
| 1–30000 | unchanged SplatStudio 3DGS density, means LR of a 30000-step run (= `baseline3dgs-*-30000`) | `density.py`, `schedule.py` (from pup-3dgs) |
| 30000 | C1 sensitivity S(p) = sum_i abs(dE_i/dp) / sum_i P_i (unclamped colours, black bg), prune S_colour = 0 | `sensitivity.py` |
| 30000 | C2 sensitivity-weighted mini-batch k-means: 48-dim SH (4096, 100 it, 2^18 batch) and trace-normalised covariance (4096, 800 it, 2^20 batch), keep S > 6e-7 / 3e-6, eigh split | `vq.py` |
| 30001–35000 | C3 QA fine-tune: torch FakeQuantize qint8 (opacity, normalised scale, log eta, rotation, SH codebooks), float16 positions, official LRs, means frozen (official LR 0), fresh Adam, last bank step skipped | `bank.py`, `renderer.py` |
| 35000 | C4 Morton sort, official `save_npz` layout, `np.savez_compressed`; size = bytes / 2**20; npz decoded into the base fields → ss-eval measures the decoded model | `codec.py` |

Config keys (all with paper/code defaults) are in `ss_c3dgs/density.py:DEFAULTS`.

## Commands
```bash
tools/ssenv ss-train c3dgs --data ~/datasets/mipnerf360/native/garden --output runs/x --steps 35000 \
    --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run runs/x --split test
tools/ssenv python reproductions/c3dgs/plugin/report_size.py runs/x [--export]
```
`c3dgs-smoke` = 1500 base + 500 fine-tune steps (diagnostic only).

## Fidelity gaps
1. Base model is our gsplat 3DGS (white background, gsplat rasteriser), not the 3DGS authors'
   released reconstructions → compare deltas to our baseline.
2. Fine-tune over a white background (official Mip-NeRF 360 runs: black); sensitivity uses black.
3. The runtime steps opacity once more at step 35000 (official: no optimizer step at the last
   fine-tune iteration); bank tensors skip it.
4. Random view sampling with replacement; `index_add_` instead of torch_scatter.
5. ss-eval renders the decoded npz; the paper's compress.py evaluates the in-memory fake-quantised
   model (identical up to rotation stored as Q(normalize(r)) vs rendered normalize(FQ(r))).
6. WebGPU hardware renderer and its speed claim are not reproduced.
