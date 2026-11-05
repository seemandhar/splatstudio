# Batch build brief (stages: read → plan → implement → smoke test)

You are reproducing one paper from `/home/sejain/splatify/100-paper-plan/papers_100.json` as a
native SplatStudio plugin. Read `/home/sejain/splatify/CLAUDE.md` first. Then execute these
stage instruction files **in order**, each fully, ignoring their YAML frontmatter:

1. `/home/sejain/splatify/.claude/agents/splat-paper-reader.md`
2. `/home/sejain/splatify/.claude/agents/splat-planner.md`
3. `/home/sejain/splatify/.claude/agents/splat-coder.md`
4. `/home/sejain/splatify/.claude/agents/splat-tester.md`

Workspace: `/home/sejain/splatify/reproductions/<slug>/` (create `source/ reviews/ evidence/
results/`; pick a short kebab `slug` = method name). Create `state.json` immediately:
`{"paper": <title>, "slug": <slug>, "plan_id": <id>, "stages": {}, "outcome": null}` and mark
each stage `"done"` as you finish it.

## Batch rules (from the user)
- Target: match the paper's PSNR within **±0.5 dB**. No hyper-parameter fishing on test images.
- **Scenes**: if the paper reports per-scene numbers (main text or supplementary), benchmark only
  those reported scenes, preferably 2 (one outdoor + one indoor for Mip-NeRF 360; ≤3 max), match
  them per scene, and move on — pick ≤3
  reported scenes per dataset (mix outdoor/indoor for Mip-NeRF 360), preferring datasets we have
  (`mipnerf360`, `tandt`, `db`, see CLAUDE.md). If it reports averages only, compare a complete
  average on the cheapest dataset that has one: prefer `tandt` (2 scenes) and/or `db` (2 scenes);
  otherwise a Mip-NeRF 360 outdoor (5) or indoor (4) group average if the paper reports groups;
  use all 9 Mip-NeRF 360 scenes only if that is the only average the paper gives. Never compare
  a subset mean to a full-dataset average. Shared gsplat baselines exist for all
  scenes: `baseline3dgs-mipnerf360-30000`, `baseline3dgs-tandt-30000`, `baseline3dgs-db-30000`.
- Use the paper's iteration count and protocol where the runtime allows; record differences.
- **Custom CUDA is in scope.** The env has CUDA 12.1 `nvcc` and gcc 11
  (`tools/ssenv` sets them up; build with `torch.utils.cpp_extension` inside the plugin,
  JIT or setup.py). Prefer adapting the official CUDA code (license permitting, cite it) over
  re-deriving kernels. Do not modify SplatStudio core or gsplat; everything lives in the plugin.
- **Existing prior art**: plan ids 1–23 have Nerfstudio ports in
  `/home/sejain/splatify/agentic-pipeline/splatify/base-code/gs_template/methods/` — read the
  matching one for understanding, but the native plugin is written fresh against the paper and
  official code. Built-in native configs `gsplat-absgrad`, `gsplat-aa`, `gsplat-mcmc`,
  `gsplat-surfel` may be reused as components where the paper truly matches them.
- If the paper needs data or infrastructure that is not available (LLFF few-view protocol,
  DUSt3R/teacher weights, mesh/geometry evaluation, multi-scale evaluation, …): try to obtain it
  (download datasets/weights into `~/datasets/` or `~/models/`); if impossible in reasonable time,
  set `state.json` `"outcome": "blocked"` with `"blocked_reason"`, write a short `report.md`, stop.
- Keep smoke tests short (≤2000 steps, one scene). Never launch the full benchmark yourself.

## When you finish
Write `<workspace>/bench_request.json`:
`{"method": <name>, "datasets": {"mipnerf360": ["garden", ...]}, "steps": 30000,
  "per_scene_reported": true|false, "extra": "<extra ss-train args or empty>"}`
and report in ≤12 lines: slug, method name, official code used?, contributions → components,
custom CUDA?, per-scene numbers available?, chosen benchmark, smoke-test result, top fidelity gaps.
