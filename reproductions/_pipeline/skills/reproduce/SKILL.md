---
name: reproduce
description: Reproduce a Gaussian-splatting paper as a native SplatStudio plugin and try to match its reported PSNR — parse paper, plan, implement, review, test, benchmark against the paper protocol, analyze gaps and iterate. Use when the user asks to reproduce/replicate/implement a 3DGS paper (arXiv id, title, or 100-paper-plan id) in SplatStudio.
argument-hint: <arxiv-id | title | plan#id> [--datasets mipnerf360,tandt,db] [--scenes a b c] [--quick] [--steps 30000] [--tolerance 0.3] [--iterations 2]
---

# /reproduce — paper → native SplatStudio plugin → matched numbers

Arguments: `$ARGUMENTS`

You are the orchestrator. Subagents do the work; you route artifacts between them, enforce
the gates below, and keep `state.json` current so a run can resume. Read
`/home/sejain/splatify/CLAUDE.md` first. Paths below are relative to `/home/sejain/splatify`.

## Defaults
- `--datasets mipnerf360` (add `tandt,db` only if the user asks or the paper's headline is there)
- `--scenes`: decided **after Stage 1** from `paper_results.json` (an explicit `--scenes`
  always wins):
  - paper has **per-scene** numbers for the dataset → run only a representative subset with
    reported values: up to 3 scenes, mixing outdoor and indoor (mipnerf360 default
    `garden bicycle counter`, filtered to scenes the paper reports). Compare per scene; extend
    to more scenes only if the subset result is ambiguous (some scenes within tolerance, others
    not) or the user asks.
  - paper has **averages only** → all scenes of the dataset are required, because a subset mean
    is not comparable with the paper's mean. Tell the user this before launching (cost estimate:
    scenes × ~1 h GPU each for method plus baseline).
  - `--quick` → the 3-scene subset regardless; with averages-only papers it can only report
    the baseline-adjusted delta, labeled "indicative, subset".
  Launch the baseline (Stage 0.4) only for the scenes chosen here, so it waits for Stage 1.
- `--steps 30000` (or the paper's own iteration count), `--tolerance 0.5` dB PSNR,
  `--iterations 2` gap-fix rounds, used only when the gap analysis gives a concrete,
  paper-grounded fix likely to close the gap; otherwise report the miss and move on.
- Custom CUDA is in scope: the env has CUDA 12.1 `nvcc` + gcc 11 (see CLAUDE.md); a
  paper that needs a custom rasterizer/kernel gets one built inside its plugin.

## Stage 0 — resolve and set up
1. Resolve the paper: `plan#<n>` → `100-paper-plan/papers_100.json` entry `id == n`
   (title, `source_url`, `code_url`, mechanism, integration notes). A title → search that file
   first, then the web. Pick `slug` = short kebab name (e.g. `fregs`).
2. Workspace `W=reproductions/<slug>`; create it with `source/ reviews/ evidence/ results/`.
   If `W/state.json` exists, **resume** from the first stage not marked done.
3. Environment check (fail fast, then suggest `/splat-setup`):
   `tools/ssenv python -c "import torch,gsplat,splatstudio,lpips;assert torch.cuda.is_available();print(gsplat.__version__)"`
   and `python3 tools/bench.py prepare --dataset <d>` for each dataset (no-op when ready).
4. Baseline (run this step right after Stage 1, once the scene set is decided): experiment
   `baseline3dgs-<dataset>-<steps>` holds shared `gsplat --recipe 3dgs` runs, reused across papers.
   If it is missing scenes you chose, launch them (relaunching the same experiment adds only
   missing jobs) so they overlap with stages 2–5:
   `python3 tools/bench.py launch --exp baseline3dgs-<dataset>-<steps> --methods gsplat --dataset <dataset> --scenes <scenes> --steps <steps>`
   (if a scheduler for it is already running, leave it alone).

`state.json`: `{"paper": ..., "slug": ..., "plan_id": <int|null>, "args": {...},
"stages": {"read": "done", ...}, "iteration": 0, "history": [...], "outcome": null}` —
update after every stage; set `outcome` (`matched` / `matched-modulo-offset` / `not-matched` /
`blocked`) in Stage 8.

## Stage 1 — read  → agent `splat-paper-reader`
Prompt with: paper identifier, plan entry (if any), workspace absolute path.
Gate: `W/paper.md` and `W/paper_results.json` exist; `paper_results.json` has numbers for at
least one chosen dataset. If the paper reports nothing on the chosen datasets, tell the user
which datasets it does report and stop.

## Stage 2 — plan  → agent `splat-planner`
Gate: `W/plan.json` with `feasible_natively`. If `false`, stop and present the required
runtime extension to the user (do not hack around it without approval).

## Stage 3 — implement  → agent `splat-coder`
Prompt with workspace path, method name and module from the plan.
Gate: `tools/ssenv ss-inspect <name>` prints `"issues": []`.

## Stage 4 — review loop (≤2 rounds)  → agent `splat-reviewer`, fixes by `splat-coder`
Pass the coder the exact review file path. Gate: latest review `approved: true`, or after
2 rounds only minor issues remain (record the remaining ones in `state.json`).

## Stage 5 — test  → agent `splat-tester`
Gate: `W/evidence/test.json` `passed: true`. On `blocking` runtime limitations, stop and report.

## Stage 6 — benchmark  → agent `splat-benchmarker`
Benchmark the main method only. Ablation variants the plan defines (e.g. `<name>-fr`) are not
run unless the user asks or the gap analyst needs one to test a specific hypothesis.
Experiment `<slug>-i<iteration>-<dataset>`: methods `<name>` only (baseline comes from
`baseline3dgs-<dataset>-<steps>`; tell the benchmarker to summarize both, and to wait for the
baseline experiment too). Results go to `W/results/<exp>/`.
Then compute the combined comparison yourself with bench.py summarize on each experiment, or
a short script reading both `summary.json` files:
- `abs_gap` (mean, or per scene for subsets), `baseline_offset`, `delta_gap` as defined in
  `.claude/agents/splat-gap-analyst.md`.

## Stage 7 — decide
- **Matched**: |mean abs_gap| ≤ tolerance (full dataset) or every scene |abs_gap| ≤ tolerance
  (subset). → Stage 8.
- **Matched modulo runtime offset**: |delta_gap| ≤ tolerance but abs_gap is not, and the
  baseline offset explains it. → Stage 8, labeled as such.
- **Not matched**, `--iterations 0` (default) → run agent `splat-gap-analyst` once (analysis
  only, no ablation runs) so the report explains the gap, then Stage 8 as "not matched".
- **Not matched** and iterations remain → agent `splat-gap-analyst` with all paths; if it
  recommends code changes → `splat-coder` (with the analysis file) → `splat-tester` →
  Stage 6 with `iteration+1`. If it says protocol offset only → Stage 8.
- Out of iterations → Stage 8 as "not matched", with the analyst's best explanation.

## Stage 8 — report
Write `W/report.md`: paper and link; outcome label; per-scene table (ours method, ours gsplat,
paper method, paper 3DGS) with means; gaps; Gaussian counts and training time; mechanism
probes and their measured evidence (`W/evidence/verify.json`); every fidelity gap and review
issue left open; iteration history; exact reproduce commands (`pip install -e W/plugin`,
`bench.py launch ...`). Follow the honesty rules in CLAUDE.md.
Then give the user a ≤15-line summary with the table of means and the outcome.

## Rules for you as orchestrator
- Give every subagent absolute paths and the specific files to read; they do not see this
  conversation.
- Do not write plugin code yourself — send it to `splat-coder`. You may read anything.
- Long waits belong to `splat-benchmarker`; do not poll in your own context.
- Never delete runs or results from earlier iterations.
