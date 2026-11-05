---
name: splat-bench
description: Benchmark one or more installed SplatStudio methods on Mip-NeRF 360 / Tanks&Temples / Deep Blending with the 3DGS protocol, against the shared gsplat baseline and optionally a paper's reported numbers.
argument-hint: <method...> [--dataset mipnerf360] [--scenes ...] [--steps 30000] [--paper reproductions/<slug>/paper_results.json] [--exp name]
---

# /splat-bench

Arguments: `$ARGUMENTS`

1. Read `/home/sejain/splatify/CLAUDE.md`. Confirm each method is installed:
   `tools/ssenv ss-methods` (names in `methods`).
2. Defaults: dataset `mipnerf360`, all scenes, 30000 steps, exp name
   `bench-<methods joined by +>-<dataset>-<steps>`.
3. Ensure the shared baseline `baseline3dgs-<dataset>-<steps>` exists for the scenes (launch it if
   not — see `/reproduce` stage 0).
4. Delegate to agent `splat-benchmarker` with: experiment names (new + baseline), methods,
   dataset, scenes, steps, paper file (if any), output dir `results/<exp>/` under the
   workspace root (or `reproductions/<slug>/results/` when a paper file is given).
5. Present the table and gaps. If the user asked whether the method "matches", apply the
   matched / matched-modulo-offset / not-matched definitions from `/reproduce` stage 7.
