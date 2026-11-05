---
name: reproduce-batch
description: Work through the 100-paper plan in id order, reproducing each paper with the /reproduce pipeline (one paper at a time, benchmarks of paper N overlapping with the build of paper N+1), keeping a resumable queue and progress table. Use when the user asks to reproduce many/all papers from the list, or says "continue" on a running batch.
argument-hint: [--ids 1-10 | --next 10] [--tolerance 0.5]
---

# /reproduce-batch

Arguments: `$ARGUMENTS`

Queue file: `reproductions/batch.json`
`{"queue": [ids...], "active": {"build": id|null, "bench": [ids...]}, "done": {id: outcome}, "rules": {...}}`.
Progress table: `python3 tools/progress.py` (writes `reproductions/PROGRESS.md`).

## Rules (from the user; do not relax without asking)
- Order: `reproductions/batch.json` queue — new papers (#26–#100) first, then the 23 existing Nerfstudio ports (#1–#23) and native configs (#24–#25). Skip ids already having an `outcome` in `reproductions/*/state.json`.
- Tolerance ±0.5 dB PSNR. On a miss: run the gap analysis; if it yields a concrete,
  paper/official-code-grounded fix that is likely to close the gap, do up to **2 fix
  iterations** (coder → tester → re-benchmark only the missed scenes); otherwise report and move
  on. Never tune on test images; label every post-hoc change in the report.
- Paper reports per-scene numbers → run only those scenes (≤3 per dataset, mixed
  outdoor/indoor), compare per scene. Averages only → all scenes of that dataset.
- No ablations unless the user asks.
- Custom CUDA kernels are in scope (nvcc 12.1 + gcc 11 in the env).
- Existing native configs (`gsplat` = 3DGS #24, `gsplat-mcmc` = #25) only need a benchmark
  against their paper, not a new plugin. Existing Nerfstudio ports (#1–#23) are prior art
  under `agentic-pipeline/splatify/base-code/gs_template/methods/` — the native plugin is new.
- Papers whose protocol needs data/infrastructure that is absent (e.g. LLFF few-view,
  DUSt3R/teacher weights, mesh evaluation, multi-scale eval): try to obtain it; if it is not
  obtainable in reasonable time, mark `blocked` with the exact missing piece and move on.

## Per-paper loop (lean version of /reproduce)
For paper id N with workspace `reproductions/<slug>/`:
1. **Build** (one agent, sequential stages in one context — the stages share most of the
   reading): give it `.claude/agents/splat-paper-reader.md`, `splat-planner.md`,
   `splat-coder.md`, `splat-tester.md` as stage instructions, the plan entry, the rules above,
   and the scene rule. It stops after the smoke test and reports the chosen benchmark
   (dataset, scenes, steps, methods).
2. **Review** (separate agent, `splat-reviewer.md`, one round). Critical/major issues go back to
   the build agent once (SendMessage), then proceed regardless; leftovers go in the report.
3. **Benchmark**: `python3 tools/bench.py launch --exp <slug>-<dataset> --methods <name>
   --dataset <d> --scenes <...> --per-gpu 2` (+ the shared `baseline3dgs-<d>-30000` for missing
   scenes). Freeze the plugin first (no edits until its runs are evaluated — runs hash the
   source). Start building paper N+1 while N trains.
4. **Report**: when N's runs finish, `bench.py summarize --exp <slug>-<dataset> --paper
   reproductions/<slug>/paper_results.json` (and the baseline experiment), then resume N's
   build agent to produce `report.md` (if the harness blocks the subagent's file write, it returns
   the markdown as text and you save it) (table, outcome, deviations, short gap analysis if
   missed) and set `state.json` `outcome`. Run `python3 tools/progress.py`.
5. Tell the user one line per finished paper: id, name, ours vs paper, outcome.

At most one paper in Build/Review at a time; at most two papers benchmarking at once.
