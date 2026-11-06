# splatstudio-method-fisherrf

FisherRF (Jiang, Lei, Daniilidis, ECCV 2024, arXiv 2311.17874) as a native SplatStudio plugin,
ported from the Gaussian-splatting code github.com/JiangWenPL/FisherRF @ b747328.

## Mechanism
- **Fisher statistic rendering** (`fisher.py`, `csrc/`): the official modified Inria rasterizer
  (`diff/`, Gaussian-Splatting licence, non-commercial research) JIT-built; its fused backward
  accumulates the square of every per-pixel gradient, giving diag(J^T J) for means, SH,
  scales, opacities (rotation filtered out, official default). Fed activated tensors, so the raw
  log-scale / opacity-logit entries carry one exp / sigmoid' factor (official numerics).
- **EIG selection** (`selection.py`): score = sum_k H_cand[k] / (H_train[k] + 1e-6) (Eq. 5/8);
  greedy batch (Alg. 1); official random selector `random.Random(0).shuffle(sorted pool)`.
- **Active loop** (`active.py`, shared by renderer/density/loss): training index 0 + 3
  farthest cameras as initial views; at it = 100*sum|S| (seq1) acquire, then restore the initial
  Gaussians and fresh optimizers (official `init.ckpt` reload); xyz LR and densification follow
  cur_iter = it - base_iter; SH +1 every 5000 absolute iterations, 0 after restore; budget 20.
  Non-selected training frames drawn by `ss-train` are skipped (no gradient, no Adam step); the
  candidate pool is exactly the training split (test/validation never visible).

## Variants
`fisherrf`, `fisherrf-random` (seq1, Table 2), `fisherrf-batch`, `fisherrf-batch-random`
(seq4, Table 3), `fisherrf-v10`, `fisherrf-v10-random` (Table 4). Defaults = Blender scripts:
30000 effective iterations, white background, densify until 10000, min opacity 0.005.

## Running
`--steps` counts runtime loop iterations, most of which are skipped frames. Seq1 on a 100-view
pool needs ~218k on average; use `--steps 230000` (~9 sigma margin; a run that ends before the 30000 effective
iterations fails loudly at evaluation). Seq4 needs ~210k (230000 works), v10 ~336k (use 360000).

    tools/ssenv ss-train fisherrf --data ~/datasets/nerf_synthetic/fisherrf/lego \
        --output runs/x --steps 230000 --recipe 3dgs --num-gaussians 10000000

Selections are printed as `{"fisherrf_selection": ...}` lines and stored in the checkpoint
(`density_state["fisherrf"]`).

## Fidelity gaps
View sampling with replacement inside the selected set (official: per-epoch stack); gsplat
rasterizer for training (Fisher uses the official kernel); no 3DGS 20-px screen-size pruning;
RNG differs (so selected views differ even with the same algorithm); Mip-NeRF 360 setting not
supported by default (official black background, 20000 iterations).
